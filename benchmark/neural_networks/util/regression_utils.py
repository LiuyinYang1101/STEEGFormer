# Codes written by Liuyin Yang (liuyin.yang@kuleuven.be)
# All rights reserved.
# --------------------------------------------------------
from collections import OrderedDict
import os
from pathlib import Path
import pickle
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models.layers import trunc_normal_
from timm.data import Mixup
from timm.models import create_model

#all benchmark models
from models import models_vit_eeg
from models.conformer import Conformer
from models.eegnet import EEGNet
from models.deepconvnet import DeepConvNet
from models.ctnnet import EEGTransformer as CTNNet
import models.labram
from models.labram import load_state_dict
from models.EEGPT import LitEEGPTModel
from models.biot import BIOTClassifier
from models.bendr import BendrClassifier
from models.cbramod import CBraModClassifier
from models.fbssvepdnn import SSVEPDNN
import numpy as np

from util.eeg_downstream_dataset import UpperLimbDataset, ErrorDataset, InnerSpeechDataset, BinocularSSVEPDataset, BCI2aDataset, AlzheimerDataset, DTUDataset
import util.lr_decay as lrd
import util.misc as misc
from util.regression_loss import tildeq_loss, simple_regression_loss
from util.data_transform import standardize_per_channel_per_trial, LabramDataTransformerWithChannelSelection, EEGPTDataTransformerWithChannelSelection, BIOTDataTransformer, BENDRDataTransformer, CBraModDataTransformer, ViTDataTransformerWithChannelSelection
from util.labram_optim import create_labram_optimizer, get_parameter_groups, LayerDecayValueAssigner
from typing import Type, Any
import yaml
import wandb
from wandb import Api
def run_exists_online(
    api: wandb.Api,
    project_path: str,
    group_name: str,
    fold: int,
    job_type: str,
    downstream_task: str,
    model_name: str,
    subject_of_interest: str,
    optimizer_spec: str,
    evaluation_scheme: str
) -> bool:
    """
    Implements:
    - Always skip if any matching train/fine-tune job is running or pending.
    - For leave-one-out-finetuning: skip only if BOTH train AND fine-tune jobs are finished.
    - For other schemes: skip if train job is finished.
    """
    # Helper to check job states
    def get_run_states(jobtype):
        filters = {
            "group": group_name,
            "config.fold": fold,
            "jobType": jobtype,
            "config.downstream_task": downstream_task,
            "config.model": model_name,
            "config.subject_of_interest": subject_of_interest,
            "config.optimizer_spec": optimizer_spec,
        }
        runs = api.runs(project_path, filters=filters)
        states = set(run.state for run in runs)
        return states

    train_states = get_run_states("train")
    finetune_states = get_run_states("fine-tune")

    # Always skip if any are running/pending
    if "running" in train_states or "pending" in train_states:
        print("a job is actively running in the training stage")
        return True
    if "running" in finetune_states or "pending" in finetune_states:
        print("a job is actively running in the fine-tune stage")
        return True

    if evaluation_scheme == "leave-one-out-finetuning":
        # Only skip if BOTH are finished
        return "finished" in train_states and "finished" in finetune_states
    else:
        # Other schemes: skip if train is finished
        return "finished" in train_states


def run_exists_offline(log_dir: str, group_name: str, fold: int) -> bool:
    """
    Return True if local log_dir has a non-empty folder for this (group_name, fold).
    """
    if log_dir is None:
        return False

    this_run_fold_dir = os.path.join(log_dir, group_name, f"fold{fold}")
    return os.path.isdir(this_run_fold_dir) and bool(os.listdir(this_run_fold_dir))


def should_skip_run(
    api: wandb.Api,
    project_path: str,
    log_dir: str,
    group_name: str,
    fold: int,
    job_type: str,
    downstream_task: str,
    model_name: str,
    subject_of_interest: str,
    optimizer_spec: str,
    evaluation_scheme: str
) -> bool:

    skip_online = run_exists_online(
        api,
        project_path,
        group_name,
        fold,
        job_type,
        downstream_task,
        model_name,
        subject_of_interest,
        optimizer_spec,
        evaluation_scheme
    )


    skip_offline = run_exists_offline(log_dir, group_name, fold)

    return skip_online or skip_offline
        
class ExperimentRunSplit(object):
    """Holds the train-valid-test splits for different runs in this experiment.
    """
    def __init__(self, evaluation_scheme):
        self.train_runs = []
        self.finetune_runs = []
        self.test_runs = [] 
        self.evaluation_scheme = evaluation_scheme

    def get_number_of_runs(self):
        return len(self.train_runs)
    
    def get_evaluation_scheme(self):
        return self.evaluation_scheme
    
    def add_runs(self, train_subs, finetune_subs, test_subs):
        #train and test must not be empty
        assert len(train_subs)>0, "Zero subjects in the training set"
        assert len(test_subs)>0, "Zero subjects in the training set"
        self.train_runs.append(train_subs)
        self.finetune_runs.append(finetune_subs)
        self.test_runs.append(test_subs)

        
    def get_run(self, run_idx):
        return self.train_runs[run_idx], self.finetune_runs[run_idx], self.test_runs[run_idx]
    
    def get_run_description(self, run_idx):
        if self.evaluation_scheme == "population":
            num_train = len(self.train_runs[run_idx])
            return f"population_sub-all_{num_train}"
        elif self.evaluation_scheme == "leave-one-out-finetuning":
            num_train = len(self.train_runs[run_idx])
            leaveout_subject = self.finetune_runs[run_idx]
            return f"leave_out_sub-{leaveout_subject[0]}"
        elif self.evaluation_scheme == "per-subject":
            this_sub = self.train_runs[run_idx]
            return f"per_subject_sub-{this_sub[0]}"

def get_dataset_file_extention(downstream_task):
    if downstream_task in ["dtu", "inner_speech", "error", "upper_limb_motorexecution","upper_limb_motorimagination","labram_upper_limb_motorexecution","labram_upper_limb_motorimagination"]:
        return ".h5"
    elif downstream_task in ["binocular_ssvep", "bci_iv2a", "alzheimer"]:
        return ".pkl"
    
    
def split_recordings_for_evaluation(args):
    """
    For all subjects data, based on the evaluation scheme, get the corresponding lists as the train-finetune-test splits
    """
    # get all recordings name as the subject name
    ext = get_dataset_file_extention(args.downstream_task)
    files = [
        os.path.join(root, name)
        for root, dirs, names in os.walk(args.dataset_folder)
        for name in names
        if name.endswith(ext)
    ]
    subject_names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    experiment_run_split = ExperimentRunSplit(args.evaluation_scheme)
    # check the evaluation scheme
    if args.evaluation_scheme == "population": # train: all subjects, finetune: NA, test: all subjects
        experiment_run_split.add_runs(subject_names,[],subject_names)
    elif args.evaluation_scheme == "leave-one-out-finetuning": # train: all subjects-finetune_subject, finetune: finetune_subject, test: all subjects (leave-one-sub-out)
        for i in range(len(subject_names)):
            test_subject = subject_names[i]
            train_subjects = subject_names[:i] + subject_names[i+1:]
            experiment_run_split.add_runs(train_subjects,[test_subject],subject_names) 
    elif args.evaluation_scheme == "per-subject": # train: each subjects, finetune: na, test: all subjects
        for i in range(len(subject_names)):
            this_subjects = subject_names[i]
            experiment_run_split.add_runs([this_subjects],[],subject_names)
    else:
        try:
            raise Exception('Not defined evaluation scheme!')
        except Exception as error:
            print('Caught this error: ' + repr(error))

    return experiment_run_split

def _load_subject_dataset(subject_name: str, datset_dir: str, fold: int, train_flag: bool, file_extention: str, customDatasetClass: Type, **extra_init_kwargs: Any):
    """
    Load a single subject's (file_extention) file into a datset class.
    """
    filepath = os.path.join(datset_dir, f"{subject_name}"+file_extention)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No {file_extention} file for subject '{subject_name}' at {filepath}")
    
    return customDatasetClass(filepath, fold=fold, train=train_flag, **extra_init_kwargs)

def _load_datasets_from_list(subject_names, datset_dir: str, fold: int, train_flag: bool, file_extention: str, customDatasetClass: Type, **extra_init_kwargs: Any):
    """
    Give a list of file name (subject name) to load and return a list of datasets
    """
    # Load each subject's dataset
    datasets = []
    for subj in subject_names:
        ds = _load_subject_dataset(subj, datset_dir, fold, train_flag=train_flag, 
                                   file_extention=file_extention, customDatasetClass=customDatasetClass, **extra_init_kwargs)
        datasets.append(ds)
    return datasets


def get_upper_limb_dataset(args, fold, this_run_split, transform, chan_info):
    trainset, finetuneset, testset = this_run_split
    # downstream task keywork
    task = args.downstream_task.split("_")[-1]
    # Load each group
    train_datasets = _load_datasets_from_list(trainset, args.dataset_folder, fold, train_flag=True, file_extention=".h5", 
                                              customDatasetClass=UpperLimbDataset, classification_task=task, data_length=int(args.downstream_task_t*args.model_downstream_task_fs),
                                              class_label=args.class_label, transform=transform, chan_info=chan_info)
    finetune_datasets = _load_datasets_from_list(finetuneset, args.dataset_folder, fold, train_flag=True, file_extention=".h5", 
                                                 customDatasetClass=UpperLimbDataset, classification_task=task, data_length=int(args.downstream_task_t*args.model_downstream_task_fs),
                                                 class_label=args.class_label, transform=transform, chan_info=chan_info)
    test_datasets = _load_datasets_from_list(testset, args.dataset_folder, fold, train_flag=False,file_extention=".h5", 
                                             customDatasetClass=UpperLimbDataset, classification_task=task, data_length=int(args.downstream_task_t*args.model_downstream_task_fs),
                                             class_label=args.class_label, transform=transform, chan_info=chan_info)
    return train_datasets, finetune_datasets, test_datasets


def get_error_dataset(args, fold, this_run_split, transform, chan_info):
    trainset, finetuneset, testset = this_run_split
    # downstream task keywork
    task = "error"
    # Load each group
    train_datasets = _load_datasets_from_list(trainset, args.dataset_folder, fold, train_flag=True, file_extention=".h5", 
                                              customDatasetClass=ErrorDataset, classification_task=task, data_length=int(args.downstream_task_t*args.model_downstream_task_fs),
                                              class_label=args.class_label, transform=transform, chan_info=chan_info)
    finetune_datasets = _load_datasets_from_list(finetuneset, args.dataset_folder, fold, train_flag=True, file_extention=".h5", 
                                                 customDatasetClass=ErrorDataset, classification_task=task, data_length=int(args.downstream_task_t*args.model_downstream_task_fs),
                                                 class_label=args.class_label, transform=transform, chan_info=chan_info)
    test_datasets = _load_datasets_from_list(testset, args.dataset_folder, fold, train_flag=False,file_extention=".h5", 
                                             customDatasetClass=ErrorDataset, classification_task=task, data_length=int(args.downstream_task_t*args.model_downstream_task_fs),
                                             class_label=args.class_label, transform=transform, chan_info=chan_info)
    return train_datasets, finetune_datasets, test_datasets

def get_inner_speech_dataset(args, fold, this_run_split, transform, chan_info):
    trainset, finetuneset, testset = this_run_split
    # downstream task keywork
    task = "inner_speech"
    # Load each group
    train_datasets = _load_datasets_from_list(trainset, args.dataset_folder, fold, train_flag=True, file_extention=".h5", 
                                              customDatasetClass=InnerSpeechDataset, classification_task=task, data_length=int(args.downstream_task_t*args.model_downstream_task_fs),
                                              class_label=args.class_label, transform=transform, chan_info=chan_info)
    finetune_datasets = _load_datasets_from_list(finetuneset, args.dataset_folder, fold, train_flag=True, file_extention=".h5", 
                                                 customDatasetClass=InnerSpeechDataset, classification_task=task, data_length=int(args.downstream_task_t*args.model_downstream_task_fs),
                                                 class_label=args.class_label, transform=transform, chan_info=chan_info)
    test_datasets = _load_datasets_from_list(testset, args.dataset_folder, fold, train_flag=False,file_extention=".h5", 
                                             customDatasetClass=InnerSpeechDataset, classification_task=task, data_length=int(args.downstream_task_t*args.model_downstream_task_fs),
                                             class_label=args.class_label, transform=transform, chan_info=chan_info)
    return train_datasets, finetune_datasets, test_datasets

def get_binocular_ssvep_dataset(args, fold, this_run_split, transform, chan_info):
    trainset, finetuneset, testset = this_run_split
    # Load each group
    train_datasets = _load_datasets_from_list(trainset, args.dataset_folder, fold, train_flag=True, file_extention=".pkl", 
                                              customDatasetClass=BinocularSSVEPDataset, classification_task="async",
                                              class_label=args.class_label, transform=transform, chan_info=chan_info)
    
    finetune_datasets = _load_datasets_from_list(finetuneset, args.dataset_folder, fold, train_flag=True, file_extention=".pkl", 
                                                 customDatasetClass=BinocularSSVEPDataset, classification_task="async",
                                                 class_label=args.class_label, transform=transform, chan_info=chan_info)
    # sync
    test_datasets_sync = _load_datasets_from_list(testset, args.dataset_folder, fold, train_flag=False,file_extention=".pkl", 
                                             customDatasetClass=BinocularSSVEPDataset, classification_task="sync", 
                                             class_label=args.class_label, transform=transform, chan_info=chan_info)
    # async
    test_datasets_async = _load_datasets_from_list(testset, args.dataset_folder, fold, train_flag=False,file_extention=".pkl", 
                                             customDatasetClass=BinocularSSVEPDataset, classification_task="async", 
                                             class_label=args.class_label, transform=transform, chan_info=chan_info)
    
    test_datasets = test_datasets_sync + test_datasets_async
    
    return train_datasets, finetune_datasets, test_datasets

def get_bci_2a_dataset(args, fold, this_run_split, transform, chan_info):
    trainset, finetuneset, testset = this_run_split
    # downstream task keywork
    task = "bci_iv2a"
    # Load each group
    train_datasets = _load_datasets_from_list(trainset, args.dataset_folder, fold, train_flag=True, file_extention=".pkl", 
                                              customDatasetClass=BCI2aDataset, classification_task=task, data_length=int(args.downstream_task_t*args.model_downstream_task_fs),
                                              class_label=args.class_label, transform=transform, chan_info=chan_info)
    finetune_datasets = _load_datasets_from_list(finetuneset, args.dataset_folder, fold, train_flag=True, file_extention=".pkl", 
                                                 customDatasetClass=BCI2aDataset, classification_task=task, data_length=int(args.downstream_task_t*args.model_downstream_task_fs),
                                                 class_label=args.class_label, transform=transform, chan_info=chan_info)
    test_datasets = _load_datasets_from_list(testset, args.dataset_folder, fold, train_flag=False,file_extention=".pkl", 
                                             customDatasetClass=BCI2aDataset, classification_task=task, data_length=int(args.downstream_task_t*args.model_downstream_task_fs),
                                             class_label=args.class_label, transform=transform, chan_info=chan_info)
    return train_datasets, finetune_datasets, test_datasets


def get_alzheimer_dataset(args, fold, this_run_split, transform, chan_info):
    trainset, finetuneset, testset = this_run_split
    # downstream task keywork
    task = "alzheimer"
    # Load each group
    train_datasets = _load_datasets_from_list(trainset, args.dataset_folder, fold, train_flag=True, file_extention=".pkl", 
                                              customDatasetClass=AlzheimerDataset, classification_task=task, data_length=int(args.downstream_task_t*args.model_downstream_task_fs),
                                              class_label=args.class_label, transform=transform, chan_info=chan_info)
    finetune_datasets = []#_load_datasets_from_list(finetuneset, args.dataset_folder, fold, train_flag=True, file_extention=".pkl", 
                          #                       customDatasetClass=AlzheimerDataset, classification_task=task, data_length=int(args.downstream_task_t*args.model_downstream_task_fs),
                          #                       class_label=args.class_label, transform=transform, chan_info=chan_info)
                          # no finetune sets for this task: always leave-one-out
    test_datasets = _load_datasets_from_list(testset, args.dataset_folder, fold, train_flag=False,file_extention=".pkl", 
                                             customDatasetClass=AlzheimerDataset, classification_task=task, data_length=int(args.downstream_task_t*args.model_downstream_task_fs),
                                             class_label=args.class_label, transform=transform, chan_info=chan_info)
    return train_datasets, finetune_datasets, test_datasets


def get_dtu_dataset(args, fold, this_run_split, transform, chan_info):
    trainset, finetuneset, testset = this_run_split
    # downstream task keywork
    task = "dtu"
    
    # Load each group
    train_datasets = _load_datasets_from_list(trainset, args.dataset_folder, fold, train_flag=True, file_extention=".h5", 
                                              customDatasetClass=DTUDataset, regression_task=task, segment_time=args.segment_time, hop_time=args.hop_time, new_fs= args.model_downstream_task_fs,
                                              class_label=None, transform=transform, chan_info=chan_info)
    finetune_datasets = _load_datasets_from_list(finetuneset, args.dataset_folder, fold, train_flag=True, file_extention=".h5", 
                                                 customDatasetClass=DTUDataset, regression_task=task,segment_time=args.segment_time, hop_time=args.hop_time, new_fs=args.model_downstream_task_fs,
                                              class_label=None, transform=transform, chan_info=chan_info)
    
    test_datasets = _load_datasets_from_list(testset, args.dataset_folder, fold, train_flag=False,file_extention=".h5", 
                                             customDatasetClass=DTUDataset, regression_task=task, segment_time=args.downstream_task_t, new_fs=args.model_downstream_task_fs,
                                              class_label=None, transform=transform, chan_info=chan_info)
    return train_datasets, finetune_datasets, test_datasets



def get_dataset(args, fold, this_run_split):
    # determine if any data transformation is needed
    transform = None
    if args.model_data_transform == "z-score":
        transform = standardize_per_channel_per_trial
    if args.model == "labram":
        transform = LabramDataTransformerWithChannelSelection(divisor=args.downstream_task_labram_divisor, channel_idx=args.labram_channels_to_keep, original_sfreq=args.downstream_task_fs)
    if args.model == "eegpt":
        transform = EEGPTDataTransformerWithChannelSelection(divisor=args.downstream_task_eegpt_divisor, channel_idx=args.eegpt_channels_to_keep, original_sfreq=args.downstream_task_fs)
    if args.model == "biot":
        transform = BIOTDataTransformer(original_sfreq=args.downstream_task_fs)
    if args.model == "bendr":
        transform = BENDRDataTransformer(original_sfreq=args.downstream_task_fs)
    if args.model == "cbramod":
        transform = CBraModDataTransformer(divisor=args.downstream_task_labram_divisor, original_sfreq=args.downstream_task_fs)
    if "vit" in args.model:
        transform = ViTDataTransformerWithChannelSelection(channel_idx=args.vit_channels_to_keep, original_sfreq=args.downstream_task_fs, new_sfreq=128)
    # determine if any channel info is needed
    chan_info = get_model_channel_info(args)
    if args.downstream_task in ["upper_limb_motorexecution","upper_limb_motorimagination","labram_upper_limb_motorexecution","labram_upper_limb_motorimagination"]:
        train_datasets, finetune_datasets, test_datasets = get_upper_limb_dataset(args, fold, this_run_split, transform, chan_info)
    elif args.downstream_task == "error":
        train_datasets, finetune_datasets, test_datasets = get_error_dataset(args, fold, this_run_split, transform, chan_info)
    elif args.downstream_task == "inner_speech":
        train_datasets, finetune_datasets, test_datasets = get_inner_speech_dataset(args, fold, this_run_split, transform, chan_info)
    elif args.downstream_task == "binocular_ssvep":
        train_datasets, finetune_datasets, test_datasets = get_binocular_ssvep_dataset(args, fold, this_run_split, transform, chan_info)
    elif args.downstream_task == "bci_iv2a":
        train_datasets, finetune_datasets, test_datasets = get_bci_2a_dataset(args, fold, this_run_split, transform, chan_info)
    elif args.downstream_task == "alzheimer":
        train_datasets, finetune_datasets, test_datasets = get_alzheimer_dataset(args, fold, this_run_split, transform, chan_info)
    elif args.downstream_task == "dtu":
        train_datasets, finetune_datasets, test_datasets = get_dtu_dataset(args, fold, this_run_split, transform, chan_info)
    return train_datasets, finetune_datasets, test_datasets

def construct_mixup(args):
    mixup_fn = Mixup(
                        mixup_alpha   = 0.2,
                        cutmix_alpha  = 0.0,
                        prob          = args.mix_up,
                        switch_prob   = 0.0,
                        mode          = 'batch',
                        label_smoothing = args.smoothing,
                        num_classes     = args.nb_classes,
                    )
    return mixup_fn

def get_loss_criterion(args):
    if args.downstream_task == "dtu":
        criterion = simple_regression_loss
        if args.wandb_project == "debug":
            criterion = simple_regression_loss
    else:
        criterion = SoftTargetCrossEntropy()
    #if args.smoothing > 0.:
    #    criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    #else:
    #    criterion = torch.nn.CrossEntropyLoss()
    print("criterion = %s" % str(criterion))
    return criterion

def create_optimizer(args, model):
    spec       = args.optimizer_spec
    model_name = args.model.lower()
    lr         = args.lr
    wd         = args.weight_decay

    # ----------------------------------------------------------------------------
    # 1) "linear_prob" spec: freeze all â†’ unfreeze a small head â†’ single AdamW
    # ----------------------------------------------------------------------------
    if spec == "linear_prob":
        for p in model.parameters(): p.requires_grad = False

        if "vit" in model_name:
            mods = [model.head]
            if hasattr(model, "attnpool"):
                mods.append(model.attnpool)
        elif model_name == "labram":
            mods = [model.head]
        elif model_name == "eegpt":
            mods = [model.chan_conv, model.linear_probe1, model.linear_probe2]
        elif model_name == "biot":
            mods = [model.chan_conv, model.classifier]
        elif model_name == "bendr":
            mods = [model.chan_conv, model.linear_probe, model.scale_param]
        elif model_name == "cbramod":
            mods = [model.feed_forward]
        else:
            mods = []
    
        for obj in mods:
            if isinstance(obj, torch.nn.Parameter):
                obj.requires_grad = True
            else:
                for p in obj.parameters():
                    p.requires_grad = True
    
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    # ----------------------------------------------------------------------------
    # 2) "finetune" spec: either LRD for ViT/labram, or plain AdamW
    # ----------------------------------------------------------------------------
    elif spec == "finetune":
        # 2a) visionâ€transformers get layer-wise LR decay
        if "vit" in model_name:
            param_groups = lrd.param_groups_lrd(
                model, wd,
                no_weight_decay_list=model.no_weight_decay(),
                layer_decay=args.layer_decay
            )
            optimizer = torch.optim.AdamW(param_groups, lr=lr)

        # 2b) labram also uses its own LRD via get_parameter_groups
        elif "labram" in model_name:
            num_layers = model.get_num_layers()
            if args.layer_decay < 1.0:
                scales = [args.layer_decay ** (num_layers - i)
                          for i in range(num_layers + 1)]
                assigner = LayerDecayValueAssigner(scales)
                print("Assigned layer scales:", scales)
            else:
                assigner = None

            skip = set(model.no_weight_decay() or [])
            print(f"[finetune|labram] skipping weight decay on {skip}")

            param_groups = get_parameter_groups(
                model,
                weight_decay=wd,
                skip_list=skip,
                get_num_layer=(assigner.get_layer_id if assigner else None),
                get_layer_scale=(assigner.get_scale if assigner else None)
            )
            optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=0.0)

        # 2c) all other models just fine-tune the whole thing normally
        else:
            print(f"[finetune|{model_name}] simple AdamW")
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # ----------------------------------------------------------------------------
    # 3) default fallback: full-model AdamW
    # ----------------------------------------------------------------------------
    else:
        print(f"[default] optimizer AdamW over all parameters")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # optional: print total params / trainable params
    print_size(model)
    return optimizer


def get_model(args):
    if "vit" in args.model:
        model = models_vit_eeg.__dict__[args.model](
                    num_classes=args.nb_classes,
                    drop_rate = args.drop_rate,
                    drop_path_rate=args.drop_path,
                    attn_drop_rate=args.attn_drop_rate,
                    proj_drop_rate=args.proj_drop_rate,
                    global_pool=args.global_pool,
                )

        # load pre-trained model if needed
        if args.vit_pretrained_model_dir:
            checkpoint = torch.load(args.vit_pretrained_model_dir, map_location='cpu')
            print("Load pre-trained checkpoint from: %s" % args.vit_pretrained_model_dir)
            checkpoint_model = checkpoint['model']
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    del checkpoint_model[k]
            # load pre-trained model
            msg = model.load_state_dict(checkpoint_model, strict=False)
            # manually initialize fc layer
            #trunc_normal_(model.head.weight, std=2e-5)

        if args.model_adaptation == "batchNorm":
            # hack: revise model's head with BN
            print("uses batch normalization")
            model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
        return model
    
    elif args.model == "conformer":
        model = Conformer(num_channel=args.downstream_task_num_chan, data_length=int(args.downstream_task_t*args.downstream_task_fs), emb_size=40, depth=6, n_classes=args.nb_classes)
        return model
    
    elif args.model == "eegnet":
        model = EEGNet(no_spatial_filters=4, no_channels=args.downstream_task_num_chan, no_temporal_filters=8, temporal_length_1=int(args.downstream_task_fs/2), temporal_length_2=int(args.downstream_task_fs/128)*16, window_length=int(args.downstream_task_t*args.downstream_task_fs), num_class=args.nb_classes, drop_out_ratio=0.50, pooling2=int(args.downstream_task_fs/32), pooling3=8)
        return model
    
    elif args.model == "deepconvnet":
        model = DeepConvNet(number_channel=args.downstream_task_num_chan, nb_classes=args.nb_classes, dropout_rate=0.5, 
                            sampling_rate=int(args.downstream_task_fs), data_length=int(args.downstream_task_t*args.downstream_task_fs))
        return model
    
    elif args.model == "ctnnet":
        model = CTNNet(heads=4, emb_size=40, depth=6, number_class=args.nb_classes, number_channel=args.downstream_task_num_chan,
                       data_length=int(args.downstream_task_t*args.downstream_task_fs), sampling_rate=int(args.downstream_task_fs))
        return model
    
    elif args.model == "ssvepdnn":
        model = SSVEPDNN(no_fb=7, no_channels=args.downstream_task_num_chan, no_combined_channels=280, drop_out_ratio_1=0.2, drop_out_ratio_2=0.9, input_length=int(args.downstream_task_t*args.downstream_task_fs), num_class=args.nb_classes)
        return model
    
    elif args.model == "labram":
        model = create_model(
        'labram_base_patch200_200',
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_block_rate=None,
        use_mean_pooling=True,
        init_scale=0.001,
        use_rel_pos_bias=False,
        use_abs_pos_emb=True,
        init_values=0.1,
        qkv_bias=False,
    )
        checkpoint = torch.load(args.labram_pretrained_model_dir, map_location='cpu')
        model_key = 'model|module'
        #print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        if (checkpoint_model is not None):
            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('student.'):
                    new_dict[key[8:]] = checkpoint_model[key]
                else:
                    pass
            checkpoint_model = new_dict

        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

        load_state_dict(model, checkpoint_model, prefix='')
        return model
    
    elif args.model=="eegpt":
        model = LitEEGPTModel(load_path=args.eegpt_pretrained_model_dir,chans_num=len(args.eegpt_channels_to_keep), num_class=args.nb_classes, data_length=int(args.model_downstream_task_fs*args.downstream_task_t))
        return model
    
    elif args.model=="biot":
        model = BIOTClassifier(input_eeg_channel=args.downstream_task_num_chan, emb_size=256, heads=8, depth=4, n_classes=args.nb_classes,
                               n_fft=200, hop_length=100, n_channels=18)
        model.biot.load_state_dict(torch.load(args.biot_pretrained_model_dir))
        return model
    
    elif args.model=="bendr":
        model = BendrClassifier(num_class=args.nb_classes, num_channels=args.downstream_task_num_chan, data_length=int(args.model_downstream_task_fs*args.downstream_task_t), pre_trained_model_path=args.bendr_pretrained_model_dir)
        return model
    
    elif args.model=="cbramod":
        model = CBraModClassifier(num_class=args.nb_classes, num_channel=args.downstream_task_num_chan, data_length=int(args.model_downstream_task_fs*args.downstream_task_t), pretrained_dir=args.cbramod_pretrained_model_dir)
        return model

def construct_data_loaders(args, trainsets, finetunesets, testsets):
    """
    Given lists of datasets for training, finetuning, and testing, construct
    both per-dataset DataLoaders and â€œbigâ€ concatenated DataLoaders
    using args.train_batch_size, args.finetune_batch_size, etc.

    Returns:
        train_loaders:      list of DataLoader over each trainset
        finetune_loaders:   list of DataLoader over each finetuneset (or [] if none)
        test_loaders:       list of DataLoader over each testset
        big_train_loader:   DataLoader over ConcatDataset(trainsets)
        big_finetune_loader:DataLoader over ConcatDataset(finetunesets), or None if finetunesets is empty
    """
    # safe defaults
    num_workers      = getattr(args, 'num_workers', 0)
    pin_memory       = getattr(args, 'pin_memory', getattr(args, 'pin_mem', False))
    train_bs         = args.train_batch_size
    finetune_bs      = getattr(args, 'finetune_batch_size', train_bs)
    test_bs          = getattr(args, 'test_batch_size', 32)
    if args.downstream_task == "dtu":
        if_shuffle = True
    else:
        if_shuffle = True
    def make_loader(dataset, batch_size, shuffle):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )

    # --- per-dataset loaders
    train_loaders = [make_loader(ds, train_bs, shuffle=if_shuffle) for ds in trainsets]
    finetune_loaders = [make_loader(ds, finetune_bs, shuffle=if_shuffle) for ds in finetunesets] if finetunesets else []
    test_loaders = [make_loader(ds, test_bs, shuffle=False) for ds in testsets]

    # --- big concatenated loaders
    big_train_loader = make_loader(ConcatDataset(trainsets), train_bs, shuffle=if_shuffle)
    big_finetune_loader = (
        make_loader(ConcatDataset(finetunesets), finetune_bs, shuffle=if_shuffle)
        if finetunesets
        else None
    )
    big_test_loader = make_loader(ConcatDataset(testsets), test_bs, shuffle=False)
    

    return train_loaders, finetune_loaders, test_loaders, big_train_loader, big_finetune_loader, big_test_loader

 
def get_all_dataset_names(data_folder):
    # List all files ending with .h5
    with open(Path(data_folder)/"sub_chan_train_idx.pkl", 'rb') as file:
        all_senloc = pickle.load(file)
    return list(all_senloc.keys())

def prepare_args_for_phase(args,phase="train"):
    if phase=="train":
        args.epochs = args.train_epochs
        # model settings
        args.drop_rate = args.train_drop_rate
        args.drop_path = args.train_drop_path
        args.attn_drop_rate = args.train_attn_drop_rate
        args.proj_drop_rate = args.train_proj_drop_rate
        #set the training warmup
        args.warmup_epochs = args.train_warmup_epochs
        #set the training accum_iter
        args.accum_iter = args.train_accum_iter
        #batch size
        args.batch_size = args.train_batch_size
    elif phase=="finetune":
        args.epochs = args.finetune_epochs
        # model settings
        args.drop_rate = args.finetune_drop_rate
        args.drop_path = args.finetune_drop_path
        args.attn_drop_rate = args.finetune_attn_drop_rate
        args.proj_drop_rate = args.finetune_proj_drop_rate
        #set the training warmup
        args.warmup_epochs = args.finetune_warmup_epochs
        #set the training accum_iter
        args.accum_iter = args.finetune_accum_iter
        #batch size
        args.batch_size = args.finetune_batch_size
    return args

def print_size(net):
    """
    Print the total and trainable (requires_grad) number of parameters of a network.
    """
    # Ensure we have a valid PyTorch module
    if net is None or not isinstance(net, torch.nn.Module):
        return

    # Total parameters
    total_params = sum(p.numel() for p in net.parameters())
    # Trainable (unfrozen) parameters
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    # Print both counts in millions
    print(f"{net.__class__.__name__} Total Parameters:      {total_params / 1e6:.6f}M", flush=True)
    print(f"{net.__class__.__name__} Trainable Parameters:  {trainable_params / 1e6:.6f}M", flush=True)


def get_downstream_task_info(args):
    with open(args.dataset_yaml, 'r') as f:
        dataset_yaml = yaml.safe_load(f)
    if args.downstream_task == "dtu":
        # for the regression
        args.dataset_folder = dataset_yaml[args.downstream_task]['data_dir']
        args.downstream_task_t = dataset_yaml[args.downstream_task]['task_time']
        args.downstream_task_fs = dataset_yaml[args.downstream_task]['fs']
        args.downstream_task_fold = dataset_yaml[args.downstream_task]['fold']
        args.downstream_task_num_chan = dataset_yaml[args.downstream_task]['n_channels']
        args.downstream_task_chan_name = dataset_yaml[args.downstream_task]['chan_names']
        args.downstream_task_labram_divisor = dataset_yaml[args.downstream_task]['labram_divisor']
        args.downstream_task_eegpt_divisor = dataset_yaml[args.downstream_task]['eegpt_divisor']
        args.segment_time = dataset_yaml[args.downstream_task]['segment_time']
        args.hop_time = dataset_yaml[args.downstream_task]['hop_time']
        args.pred_length = dataset_yaml[args.downstream_task]['pred_length']
               
    else:
        args.dataset_folder = dataset_yaml[args.downstream_task]['data_dir']
        args.downstream_task_t = dataset_yaml[args.downstream_task]['task_time']
        args.downstream_task_fs = dataset_yaml[args.downstream_task]['fs']
        args.downstream_task_fold = dataset_yaml[args.downstream_task]['fold']
        args.downstream_task_num_chan = dataset_yaml[args.downstream_task]['n_channels']
        args.downstream_task_chan_name = dataset_yaml[args.downstream_task]['chan_names']
        args.downstream_task_labram_divisor = dataset_yaml[args.downstream_task]['labram_divisor']
        args.downstream_task_eegpt_divisor = dataset_yaml[args.downstream_task]['eegpt_divisor']
    with open(args.downstream_task_yaml, 'r') as f:
        task_yaml = yaml.safe_load(f)
    # 1) pull out just the label mapping (drop "num_classes")
    label_mapping = {k: v for k, v in task_yaml[args.downstream_task].items() if k != "num_classes"}

    # 2) invert it and build your list of class names in index order
    num = task_yaml[args.downstream_task]["num_classes"]
    label_to_name = {label: name for name, label in label_mapping.items()}
    class_names = [label_to_name[i] for i in range(num)]
    
    args.nb_classes = num
    args.class_label = label_mapping
    args.class_names = class_names
    
    return args


def prepared_downstream_task_for_model(args):
    if args.model=="labram":
        args.model_downstream_task_fs = 200
        labram_channels = ['FP1', 'FPZ', 'FP2', 'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', 'F9', 'F7', 'F5', 'F3', \
                           'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T9',\
                           'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10','TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',\
                           'TP10', 'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', \
                           'PO6', 'PO8', 'PO10', 'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', 'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', 'CFC1', \
                           'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', 'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', 'T1', 'T2', \
                           'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h', "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", \
                           "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"]
        downstream_channels = args.downstream_task_chan_name
        labram_channels_lower = [ch.lower() for ch in labram_channels]
        
        channels_keep = []
        labram_channel_idx = [0] #cls token
        # Filter standard channels that exist in montage
        for ch_idx, ch in enumerate(downstream_channels):
            ch_lower = ch.lower()
            if ch_lower in labram_channels_lower:
                # Exact match
                labram_channel_idx.append( labram_channels_lower.index(ch_lower) +1)
                channels_keep.append(ch_idx)
        print("keep ", len(channels_keep), "channels for labram")
        args.labram_channel_info = labram_channel_idx
        args.labram_channels_to_keep = channels_keep
        args.half_precision =True
    elif args.model=="eegpt":
        args.model_downstream_task_fs = 256
        downstream_channels = args.downstream_task_chan_name
        eegpt_channels = [      'FP1', 'FPZ', 'FP2', 
                        "AF7", 'AF3', 'AF4', "AF8", 
            'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
        'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 
            'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
        'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
             'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
                      'PO7', "PO5", 'PO3', 'POZ', 'PO4', "PO6", 'PO8', 
                               'O1', 'OZ', 'O2', ]

        eegpt_channels_lower = [ch.lower() for ch in eegpt_channels]

        channels_keep = []
        eegpt_channel_idx = []
        # Filter standard channels that exist in montage
        for ch_idx, ch in enumerate(downstream_channels):
            ch_lower = ch.lower()
            if ch_lower in eegpt_channels_lower:
                # Exact match
                eegpt_channel_idx.append( eegpt_channels_lower.index(ch_lower))
                channels_keep.append(ch_idx)
        print("keep ", len(channels_keep), "channels for EEGPT")
        args.eegpt_channel_info = eegpt_channel_idx
        args.eegpt_channels_to_keep = channels_keep
        args.half_precision =True
    elif args.model=="biot":
        args.model_downstream_task_fs = 200
        args.half_precision =False
    elif args.model=="bendr":
        args.model_downstream_task_fs = 256
        args.half_precision =True
    elif args.model=="cbramod":
        args.model_downstream_task_fs = 200
        args.half_precision =False
    elif "vit" in args.model:
        args.model_downstream_task_fs = 128
        with open("/vsc-hard-mounts/leuven-data/343/vsc34340/new_eeg_mae/senloc_file/sen_chan_idx.pkl", "rb") as f:
            data = pickle.load(f)
            # build a lowercase lookup
            lower_map = {k.lower(): v for k, v in data['channels_mapping'].items()}
            chan_idx = []
            channels_keep = []
            for ch_idx, ch in enumerate(args.downstream_task_chan_name):
                key = ch.lower()
                if key not in lower_map:
                    print(f"error! Unknown channel {ch} found in this dataset for the eegvit")
                else:
                    chan_idx.append(lower_map[key])
                    channels_keep.append(ch_idx)
            args.vit_channel_info = chan_idx
            args.vit_channels_to_keep = channels_keep
            print("keep ", len(channels_keep), "channels for ViT model")
            args.half_precision =True
    else:
        args.model_downstream_task_fs = args.downstream_task_fs
        args.half_precision =True
        
    if args.downstream_task == "dtu":
        args.nb_classes = int(args.pred_length)
    return args

def get_model_channel_info(args):
    # this function will return the necessary channel info based on current model and downstream dataset
    if "vit" in args.model: #needs channel indices
        chan_idx = np.array(args.vit_channel_info)
        return torch.from_numpy(chan_idx).type(torch.IntTensor)
    elif args.model == "labram":
        chan_idx = np.array(args.labram_channel_info)
        return torch.from_numpy(chan_idx).type(torch.IntTensor)
    elif args.model == "eegpt":
        chan_idx = np.array(args.eegpt_channel_info)
        return torch.from_numpy(chan_idx).type(torch.IntTensor)
    else:
        return None
        