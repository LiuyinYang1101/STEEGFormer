import wandb
import argparse
from wandb_engine_finetune_eeg import run_phase
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import timm
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.eeg_dataset import MultiEEGDataLoader
from util.utils import get_downstream_task_info, get_model, split_recordings_for_evaluation, get_dataset, construct_data_loaders, get_loss_criterion, prepare_args_for_phase, construct_mixup, prepared_downstream_task_for_model, should_skip_run


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for EEG classification', add_help=False)
    # Wandb parameters
    parser.add_argument('--wandb_project', default='debug', type=str,
                        help='Name of the Wandb project')
    parser.add_argument('--wandb_entity', default='liuyin_yang-ku-leuven', type=str,
                        help='Wandb account')
    parser.add_argument('--wandb_log_dir', default='/lustre1/project/stg_00160/wandb_log_bci_iv2a', type=str,
                        help='Wandb log directory')
    parser.add_argument('--wandb_log_every', type=int, default=5,
                        help="only log to W&B every N epochs")
    
    # Model parameters
    parser.add_argument('--model', default='vit_base_patch32', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model_data_transform', default='None', type=str,
                        help='data transformation')
    parser.add_argument('--pretrained_model_dir', default='', type=str,
                        help='pretrained model path')
    parser.add_argument('--vit_pretrained_model_dir', default='/lustre1/project/stg_00160/new_eeg_mae/experiment3_small/checkpoint-336.pth', type=str,
                        help='vit pretrained model path')
    parser.add_argument('--labram_pretrained_model_dir', default='/vsc-hard-mounts/leuven-data/343/vsc34340/LaBraM-main/checkpoints/labram-base.pth', type=str,
                        help='labram pretrained model path')
    parser.add_argument('--eegpt_pretrained_model_dir', default='/lustre1/project/stg_00160/eegpt/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt', type=str,
                        help='eegpt pretrained model path')
    parser.add_argument('--biot_pretrained_model_dir', default='/vsc-hard-mounts/leuven-data/343/vsc34340/BIOT-main/pretrained-models/EEG-six-datasets-18-channels.ckpt', type=str,
                        help='biot pretrained model path')
    parser.add_argument('--bendr_pretrained_model_dir', default='/lustre1/project/stg_00160/bendr/encoder.pt', type=str,
                        help='bendr pretrained model path')
    parser.add_argument('--cbramod_pretrained_model_dir', default='/lustre1/project/stg_00160/cbramod/pretrained_weights.pth', type=str,
                        help='cbramod pretrained model path')
    
    # STEEGFormer parameters
    parser.add_argument('--global_pool', default='avg', type=str,
                        help='pooling method for the classification head')
    parser.add_argument('--drop_rate', type=float, default=0.1,
                        help='head drop rate (default: 0.1)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.1,
                        help='Attention drop rate (default: 0.1)')
    parser.add_argument('--proj_drop_rate', type=float, default=0.1,
                        help='MLP drop rate (default: 0.1)')
    parser.add_argument('--train_drop_rate', type=float, default=0.1,
                        help='head drop rate in the training stage')
    parser.add_argument('--train_drop_path', type=float, default=0.1,
                        help='Drop path rate in the training stage')
    parser.add_argument('--train_attn_drop_rate', type=float, default=0.1,
                        help='Attention drop rate in the training stage')
    parser.add_argument('--train_proj_drop_rate', type=float, default=0.1,
                        help='MLP drop rate in the training stage')
    parser.add_argument('--finetune_drop_rate', type=float, default=0.1,
                        help='head drop rate in the finetuning stage')
    parser.add_argument('--finetune_drop_path', type=float, default=0.1,
                        help='Drop path rate in the finetuning stage')
    parser.add_argument('--finetune_attn_drop_rate', type=float, default=0.1,
                        help='Attention drop rate in the finetuning stage')
    parser.add_argument('--finetune_proj_drop_rate', type=float, default=0.1,
                        help='MLP drop rate in the finetuning stage')
    parser.add_argument('--model_adaptation', type=str, default="",
                        help='Apply any model adaptations')
    
    # Training parameters
    parser.add_argument('--train_batch_size', default=16, type=int,
                        help='Batch size per GPU used in the training stage')
    parser.add_argument('--finetune_batch_size', default=8, type=int,
                        help='Batch size per GPU used in the finetuning stage')                    
    parser.add_argument('--train_epochs', default=100, type=int,
                        help='number of training epochs')
    parser.add_argument('--finetune_epochs', default=50, type=int,
                        help='number of finetuning epochs')
    parser.add_argument('--train_accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations in the training stage')
    parser.add_argument('--finetune_accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations in the finetuning stage')


    # Optimizer parameters
    parser.add_argument('--optimizer_spec', default='linear_prob', type=str,
                        help='model optimizer settings')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=3e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--train_warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR in the training stage')
    parser.add_argument('--finetune_warmup_epochs', type=int, default=3, metavar='N',
                        help='epochs to warmup LR in the finetuning stage')

    # Model training parameters
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--mix_up', type=float, default=0.9,
                        help='Mixup probability (default: 0.9)')
    
    # Exp parameters
    parser.add_argument('--dataset_yaml', default="/vsc-hard-mounts/leuven-data/343/vsc34340/new_eeg_mae/util/dataset_specs.yaml", type=str,
                        help='dataset yaml with dataset specs')
    parser.add_argument('--downstream_task_yaml', default="/vsc-hard-mounts/leuven-data/343/vsc34340/new_eeg_mae/util/downstream_task_specs.yaml", type=str,
                        help='dataset yaml with dataset specs')
    parser.add_argument('--downstream_task', default="finger_classification", type=str,
                        help='which downstream task to benchmark')
    parser.add_argument('--evaluation_scheme', default="population", type=str,
                        help='which training-finetuning-testing scheme to use')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    return parser


def main_train(args):
    device = torch.device(args.device)
    global_out_put_dir = args.output_dir
    # job name
    job_name = args.evaluation_scheme
    # initialize the downstream task info
    args = get_downstream_task_info(args)
    # initialize the model related downstream task specs
    args = prepared_downstream_task_for_model(args)
    # get the train-finetune-test runs for this experiment
    experiment_run_split = split_recordings_for_evaluation(args)
    print(f"There are {experiment_run_split.get_number_of_runs()} of runs in this experiment!", flush= True)
    mixup_fn = construct_mixup(args) #controls mix_up for data augmentation and label smoothing
    
    # Initialize the W&B API (for online checks)
    api = wandb.Api()
    proj_path = f"{args.wandb_entity}/{args.wandb_project}"
    
    for run_idx in range(experiment_run_split.get_number_of_runs()):
        # initialize the run name
        wandb_group_name = experiment_run_split.get_run_description(run_idx)
        current_run_sub_of_interested = wandb_group_name.split("sub-")[-1]
        args.current_run_sub_of_interested = current_run_sub_of_interested
        this_run_split = experiment_run_split.get_run(run_idx)
        # fix the seed for reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Start run {run_idx}, {wandb_group_name}")
        # k fold cross-validation
        for fold in range(args.downstream_task_fold):
            this_run_name = f"fold{fold}_{args.model}_{args.optimizer_spec}_{wandb.util.generate_id()}"
            
            # SKIP TRAIN PHASE?
            if should_skip_run(
                api,
                proj_path,
                args.log_dir,
                wandb_group_name,
                fold,
                job_type="train",
                downstream_task=args.downstream_task,
                model_name=args.model,
                subject_of_interest=current_run_sub_of_interested,
                optimizer_spec=args.optimizer_spec,
                evaluation_scheme = args.evaluation_scheme
            ):
                print(f"⏭  Skipping train for {this_run_name} fold {fold} (already done).", flush=True)
                continue
            else:   
                print(f"⏭Not skipping train for {this_run_name} fold {fold} (not done).", flush=True)
                #continue
                
            args = prepare_args_for_phase(args,phase="train")
            # check if this run and fold have already been done
            #if args.log_dir is not None:
            #    this_run_fold_dir = f"{args.log_dir}/{this_run_name}/fold{fold}"
                #if this_run_fold_dir and os.path.exists(this_run_fold_dir):
                #    print(f"Skipping existing run {this_run_name} fold {fold}", flush=True)
                #    continue
            # exp_out log writer
            if args.log_dir is not None:
                this_run_fold_dir = f"{args.log_dir}/{wandb_group_name}/fold{fold}"
                os.makedirs(this_run_fold_dir, exist_ok=True)
                log_writer = SummaryWriter(log_dir=this_run_fold_dir)
            else:
                this_run_fold_dir = None
                log_writer = None
                
            # get model
            model = get_model(args)
            model.cuda()

            # get the training set, finetune set and the test set
            trainsets, finetunesets, testsets = get_dataset(args, fold, this_run_split)

            # construct the dataloader
            train_loaders, finetune_loaders, test_loaders, big_train_loader, big_finetune_loader = construct_data_loaders(args, trainsets, finetunesets, testsets)


            # the loss function
            criterion = get_loss_criterion(args)
            
            # initialize a fresh W&B run for this fold
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                group=wandb_group_name,
                dir=args.wandb_log_dir,
                name=this_run_name,
                job_type="train",
                reinit=True,
                config={
                  **vars(args),
                  "experiment_type": args.evaluation_scheme,         # population / per-subject / leave-one-out
                  "stage":           "train",       # explicit in config too, if you want
                  "fold":            fold,
                  "subject_of_interest": current_run_sub_of_interested,
                },
            )
            # watch model parameters & gradients
            wandb.watch(model, log="all", log_freq=100)
            
            # ——— 1) initial training ———
            prefix_train = f"{this_run_name}/fold{fold}/training"
            train_loader = MultiEEGDataLoader(train_loaders, distributed=args.distributed)
            test_loader_whole = MultiEEGDataLoader(test_loaders, distributed=args.distributed)
            run_phase(
                model, big_train_loader, test_loader_whole, test_loaders, prefix_train, args,
                device, criterion, NativeScaler, log_writer=log_writer,
                epochs=args.train_epochs, log_dir=this_run_fold_dir,
                mixup_fn= mixup_fn, wandb_log_freq=args.wandb_log_every
            )
            
            # finish the training W&B run
            wandb.finish()
            
            # ——— 2) fine‐tuning ———
            if finetunesets:
                # set the finetuning phase args settings
                args = prepare_args_for_phase(args,phase="finetune")

                base_state = {k: v.cpu() for k, v in model.state_dict().items()}
                prefix_ft = f"{this_run_name}/fold{fold}/finetune"
                # all finetune data combined
                combined_ft = MultiEEGDataLoader(finetune_loaders, distributed=args.distributed)
                # reinitialize the model
                model = get_model(args)
                model.load_state_dict(base_state)
                model.cuda()
                
                # Start a new W&B run for finetuning
                run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    group=wandb_group_name,
                    dir=args.wandb_log_dir,
                    name=this_run_name,
                    job_type="fine-tune",
                    reinit=True,
                    config={
                      **vars(args),
                      "experiment_type": args.evaluation_scheme,         # population / per-subject / leave-one-out
                      "stage":           "fine-tune",       # explicit in config too, if you want
                      "fold":            fold,
                      "subject_of_interest": current_run_sub_of_interested,
                    },
                )

                wandb.watch(model, log="all", log_freq=100)
                
                run_phase(
                    model, big_finetune_loader, test_loader_whole, test_loaders, prefix_ft, args,
                    device, criterion, NativeScaler, log_writer=log_writer,
                    epochs=args.finetune_epochs, log_dir = this_run_fold_dir,
                    mixup_fn= mixup_fn, wandb_log_freq=args.wandb_log_every
                )
                # finish the finetuning W&B run
                wandb.finish()
                del combined_ft, base_state
            del model, train_loader, test_loader_whole, train_loaders, finetune_loaders, test_loaders, trainsets, finetunesets, testsets
            torch.cuda.empty_cache()
            if args.log_dir is not None:
                log_writer.close()
    
def main(args):
    args.local_rank = 0
    args.rank = 0
    args.gpu = 0
    args.distributed = False
        
    main_train(args)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)