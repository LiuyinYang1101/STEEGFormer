# Codes written by 
# All rights reserved.
# --------------------------------------------------------

import yaml
import os
from typing import Type, Any, Dict
import torch
import numpy as np
from collections import defaultdict
from data_transform import standardize_per_channel_per_trial
from downstream_eeg_dataset import UpperLimbDataset, ErrorDataset, InnerSpeechDataset, BinocularSSVEPDataset, BCI2aDataset, AlzheimerDataset, DTUDataset
from models.EEG_decoding_pipelines import (logvar_lda, csp_lda, csp_svm, fbcsp_lda, fbcsp_svm, cov_en, mdm, fgmdm, rbp_rf, rbp_svm, rbp_knn, rbp_lightGBM,
                                            xdawn_lda, xdawncov_mdm, xdawncov_ts_svm, erpcov_mdm, dcpm)
from models.ssvep_decoders import pipe_trca, pipe_cca

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


def get_downstream_task_info(args):
    with open(args.dataset_yaml, 'r') as f:
        dataset_yaml = yaml.safe_load(f)
    args.dataset_folder = dataset_yaml[args.downstream_task]['data_dir']
    args.downstream_task_t = dataset_yaml[args.downstream_task]['task_time']
    args.downstream_task_fs = dataset_yaml[args.downstream_task]['fs']
    args.fold = dataset_yaml[args.downstream_task]['fold']
    args.downstream_task_num_chan = dataset_yaml[args.downstream_task]['n_channels']
    args.downstream_task_chan_name = dataset_yaml[args.downstream_task]['chan_names']
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

def get_dataset_file_extention(downstream_task):
    if downstream_task in ["dtu", "inner_speech", "error", "upper_limb_motorexecution","upper_limb_motorimagination"]:
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
                                              customDatasetClass=UpperLimbDataset, classification_task=task, data_length=int(args.downstream_task_t*args.downstream_task_fs),
                                              class_label=args.class_label, transform=transform, chan_info=chan_info)
    finetune_datasets = _load_datasets_from_list(finetuneset, args.dataset_folder, fold, train_flag=True, file_extention=".h5",
                                                 customDatasetClass=UpperLimbDataset, classification_task=task, data_length=int(args.downstream_task_t*args.downstream_task_fs),
                                                 class_label=args.class_label, transform=transform, chan_info=chan_info)
    test_datasets = _load_datasets_from_list(testset, args.dataset_folder, fold, train_flag=False,file_extention=".h5",
                                             customDatasetClass=UpperLimbDataset, classification_task=task, data_length=int(args.downstream_task_t*args.downstream_task_fs),
                                             class_label=args.class_label, transform=transform, chan_info=chan_info)
    return train_datasets, finetune_datasets, test_datasets


def get_error_dataset(args, fold, this_run_split, transform, chan_info):
    trainset, finetuneset, testset = this_run_split
    # downstream task keywork
    task = "error"
    # Load each group
    train_datasets = _load_datasets_from_list(trainset, args.dataset_folder, fold, train_flag=True, file_extention=".h5",
                                              customDatasetClass=ErrorDataset, classification_task=task, data_length=int(args.downstream_task_t*args.downstream_task_fs),
                                              class_label=args.class_label, transform=transform, chan_info=chan_info)
    finetune_datasets = _load_datasets_from_list(finetuneset, args.dataset_folder, fold, train_flag=True, file_extention=".h5",
                                                 customDatasetClass=ErrorDataset, classification_task=task, data_length=int(args.downstream_task_t*args.downstream_task_fs),
                                                 class_label=args.class_label, transform=transform, chan_info=chan_info)
    test_datasets = _load_datasets_from_list(testset, args.dataset_folder, fold, train_flag=False,file_extention=".h5",
                                             customDatasetClass=ErrorDataset, classification_task=task, data_length=int(args.downstream_task_t*args.downstream_task_fs),
                                             class_label=args.class_label, transform=transform, chan_info=chan_info)
    return train_datasets, finetune_datasets, test_datasets

def get_inner_speech_dataset(args, fold, this_run_split, transform, chan_info):
    trainset, finetuneset, testset = this_run_split
    # downstream task keywork
    task = "inner_speech"
    # Load each group
    train_datasets = _load_datasets_from_list(trainset, args.dataset_folder, fold, train_flag=True, file_extention=".h5",
                                              customDatasetClass=InnerSpeechDataset, classification_task=task, data_length=int(args.downstream_task_t*args.downstream_task_fs),
                                              class_label=args.class_label, transform=transform, chan_info=chan_info)
    finetune_datasets = _load_datasets_from_list(finetuneset, args.dataset_folder, fold, train_flag=True, file_extention=".h5",
                                                 customDatasetClass=InnerSpeechDataset, classification_task=task, data_length=int(args.downstream_task_t*args.downstream_task_fs),
                                                 class_label=args.class_label, transform=transform, chan_info=chan_info)
    test_datasets = _load_datasets_from_list(testset, args.dataset_folder, fold, train_flag=False,file_extention=".h5",
                                             customDatasetClass=InnerSpeechDataset, classification_task=task, data_length=int(args.downstream_task_t*args.downstream_task_fs),
                                             class_label=args.class_label, transform=transform, chan_info=chan_info)
    return train_datasets, finetune_datasets, test_datasets

def get_binocular_ssvep_dataset(args, fold, this_run_split, transform, chan_info):
    trainset, finetuneset, testset = this_run_split
    # Load each group
    train_datasets = _load_datasets_from_list(trainset, args.dataset_folder, fold, train_flag=True, file_extention=".pkl",
                                              customDatasetClass=BinocularSSVEPDataset, classification_task="sync",
                                              class_label=args.class_label, transform=transform, chan_info=chan_info)
   
    finetune_datasets = _load_datasets_from_list(finetuneset, args.dataset_folder, fold, train_flag=True, file_extention=".pkl",
                                                 customDatasetClass=BinocularSSVEPDataset, classification_task="sync",
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
                                              customDatasetClass=BCI2aDataset, classification_task=task, data_length=int(args.downstream_task_t*args.downstream_task_fs),
                                              class_label=args.class_label, transform=transform, chan_info=chan_info)
    finetune_datasets = _load_datasets_from_list(finetuneset, args.dataset_folder, fold, train_flag=True, file_extention=".pkl",
                                                 customDatasetClass=BCI2aDataset, classification_task=task, data_length=int(args.downstream_task_t*args.downstream_task_fs),
                                                 class_label=args.class_label, transform=transform, chan_info=chan_info)
    test_datasets = _load_datasets_from_list(testset, args.dataset_folder, fold, train_flag=False,file_extention=".pkl",
                                             customDatasetClass=BCI2aDataset, classification_task=task, data_length=int(args.downstream_task_t*args.downstream_task_fs),
                                             class_label=args.class_label, transform=transform, chan_info=chan_info)
    return train_datasets, finetune_datasets, test_datasets


def get_alzheimer_dataset(args, fold, this_run_split, transform, chan_info):
    trainset, finetuneset, testset = this_run_split
    # downstream task keywork
    task = "alzheimer"
    # Load each group
    train_datasets = _load_datasets_from_list(trainset, args.dataset_folder, fold, train_flag=True, file_extention=".pkl",
                                              customDatasetClass=AlzheimerDataset, classification_task=task, data_length=int(args.downstream_task_t*args.downstream_task_fs),
                                              class_label=args.class_label, transform=transform, chan_info=chan_info)
    finetune_datasets = []#_load_datasets_from_list(finetuneset, args.dataset_folder, fold, train_flag=True, file_extention=".pkl",
                          #                       customDatasetClass=AlzheimerDataset, classification_task=task, data_length=int(args.downstream_task_t*args.downstream_task_fs),
                          #                       class_label=args.class_label, transform=transform, chan_info=chan_info)
                          # no finetune sets for this task: always leave-one-out
    test_datasets = _load_datasets_from_list(testset, args.dataset_folder, fold, train_flag=False,file_extention=".pkl",
                                             customDatasetClass=AlzheimerDataset, classification_task=task, data_length=int(args.downstream_task_t*args.downstream_task_fs),
                                             class_label=args.class_label, transform=transform, chan_info=chan_info)
    return train_datasets, finetune_datasets, test_datasets


def get_dtu_dataset(args, fold, this_run_split, transform, chan_info):
    trainset, finetuneset, testset = this_run_split
    # downstream task keywork
    task = "dtu"
    # Load each group
    train_datasets = _load_datasets_from_list(trainset, args.dataset_folder, fold, train_flag=True, file_extention=".h5",
                                              customDatasetClass=DTUDataset, regression_task=task, segment_time=3, new_fs= args.downstream_task_fs,
                                              class_label=None, transform=transform, chan_info=chan_info)
    finetune_datasets = _load_datasets_from_list(finetuneset, args.dataset_folder, fold, train_flag=True, file_extention=".h5",
                                                 customDatasetClass=DTUDataset, regression_task=task,segment_time=3,new_fs=args.downstream_task_fs,
                                              class_label=None, transform=transform, chan_info=chan_info)
    test_datasets = _load_datasets_from_list(testset, args.dataset_folder, fold, train_flag=False,file_extention=".h5",
                                             customDatasetClass=DTUDataset, regression_task=task, segment_time=3, new_fs=args.downstream_task_fs,
                                              class_label=None, transform=transform, chan_info=chan_info)
    return train_datasets, finetune_datasets, test_datasets

def get_dataset(args, fold, this_run_split):
    # determine if any data transformation is needed
    transform = None
    if args.model_data_transform == "z-score":
        transform = standardize_per_channel_per_trial
    # determine if any channel info is needed
    chan_info = None
    if args.downstream_task in ["upper_limb_motorexecution","upper_limb_motorimagination"]:
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


def stack_by_class_numpy(dataset) -> Dict[Any, np.ndarray]:
    """
    Given a torch.utils.data.Dataset where dataset[i] -> (data, label),
    return a dict mapping each unique label -> stacked NumPy array of all data
    with that label.
   
    The returned array for label L has shape:
        (num_samples_for_L, *data.shape)
    """
    buckets = defaultdict(list)

    # 1) collect into Python lists
    for i in range(len(dataset)):
        data, label = dataset[i]
        # convert to numpy
        if isinstance(data, torch.Tensor):
            arr = data.detach().cpu().numpy()
        else:
            arr = np.asarray(data)
        buckets[label].append(arr)

    # 2) stack each list into a single NumPy array
    result = {}
    for label, arrs in buckets.items():
        result[label] = np.stack(arrs, axis=0)

    return result


def get_ML_models(args):
    if (args.downstream_task == 'upper_limb_motorexecution' or
        args.downstream_task == 'upper_limb_motorimagination' or
        args.downstream_task == 'inner_speech' or
        args.downstream_task == 'bci_iv2a'):
    
        if args.model == 'csp_lda':
            pipeline = csp_lda(fs = args.downstream_task_fs)
        
        elif args.model == 'csp_svm':
            pipeline = csp_svm(fs = args.downstream_task_fs)
        
        elif args.model == 'fbcsp_lda':
            pipeline = fbcsp_lda(fs = args.downstream_task_fs)
        
        elif args.model == 'fbcsp_svm':
            pipeline = fbcsp_svm(fs = args.downstream_task_fs)
        
        elif args.model == 'cov_en':
            pipeline = cov_en(fs = args.downstream_task_fs)
            
        elif args.model == 'mdm':
            pipeline = mdm(fs = args.downstream_task_fs)
            
        elif args.model == 'fgmdm':
            pipeline = fgmdm(fs = args.downstream_task_fs)
            
            
    elif args.downstream_task == 'alzheimer':
        if args.model == 'rbp_rf':
            pipeline = rbp_rf(fs = args.downstream_task_fs)
        
        elif args.model == 'rbp_svm':
            pipeline = rbp_svm(fs = args.downstream_task_fs)
        
        elif args.model == 'rbp_knn':
            pipeline = rbp_knn(fs = args.downstream_task_fs)
        
        elif args.model == 'rbp_lightGBM':
            pipeline = rbp_lightGBM(fs = args.downstream_task_fs)
            
    elif args.downstream_task == 'error':
        if args.model == 'xdawn_lda':
            pipeline = xdawn_lda(fs = args.downstream_task_fs)
        
        elif args.model == 'xdawncov_mdm':
            pipeline = xdawncov_mdm(fs = args.downstream_task_fs)
        
        elif args.model == 'xdawncov_ts_svm':
            pipeline = xdawncov_ts_svm(fs = args.downstream_task_fs)
        
        elif args.model == 'erpcov_mdm':
            pipeline = erpcov_mdm(fs = args.downstream_task_fs)    
        
        elif args.model == 'dcpm':
            pipeline = dcpm(fs = args.downstream_task_fs)   
            
    elif args.downstream_task == 'binocular_ssvep':
        if args.model == 'trca':
            pipeline = pipe_trca(fs = args.downstream_task_fs)
        elif args.model == 'cca':
            pipeline = pipe_cca(fs = args.downstream_task_fs)
            
    
    CSV_PATH = f"results/{args.model}_{args.downstream_task}_{args.evaluation_scheme}.csv"

    return pipeline, CSV_PATH
    
    