# Adapted from the MAE implementation by xxx
# --------------------------------------------------------
# References:
# MAE:  https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import os, sys
print(f"[{os.uname().nodename}] DDP script starting, PID {os.getpid()}, SLURM_PROCID={os.environ.get('SLURM_PROCID')}", flush=True)
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset
from torch.utils.data import random_split
import argparse
import datetime
import json
import numpy as np
import time
from pathlib import Path
import builtins
import logging

import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import timm
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.misc import print_size
from util.eeg_dataset import get_pretrain_dataset, MultiDatasetWrapper, MultiDatasetBatchSampler, InterleavedDistributedBatchSampler

import models_mae_eeg

from engine_pretrain_eeg import train_one_epoch, test_one_epoch
import wandb  


def get_batch_size(args,num_channel):
    this_batch_size = args.batch_size
    if args.model =="mae_vit_base_patch16":
        if num_channel >64:
            this_batch_size = 54
        elif num_channel >60:
            this_batch_size = 112
        else:
            pass
    elif args.model =="mae_vit_large_patch16":
        if num_channel >64:
            this_batch_size = 36
        elif num_channel >60:
            this_batch_size = 70
        else:
            pass
    else:
        pass
    return this_batch_size
    
class MultiEEGDataLoader:
    def __init__(self, dataloaders, distributed=False):
        self.dataloaders = dataloaders
        self.iterators = [iter(dl) for dl in dataloaders]
        self.done_dataloaders = [False] * len(dataloaders)
        self.distributed = distributed
        self.num_dataloaders = len(dataloaders)
        self.current_dataloader = 0
        
    def set_epoch(self, epoch):
        for dl in self.dataloaders:
            if self.distributed:
                if isinstance(dl.sampler, DistributedSampler):
                    dl.sampler.set_epoch(epoch)

    def __iter__(self):
        self.iterators = [iter(dl) for dl in self.dataloaders]
        self.done_dataloaders = [False] * len(self.dataloaders)
        self.current_dataloader = 0
        return self

    def __next__(self):
        if all(self.done_dataloaders):
            raise StopIteration

        while all(self.done_dataloaders)==False:
            if not self.done_dataloaders[self.current_dataloader]:
                try:
                    batch = next(self.iterators[self.current_dataloader])
                    self.current_dataloader = (self.current_dataloader + 1) % self.num_dataloaders
                    return batch
                except StopIteration:
                    self.done_dataloaders[self.current_dataloader] = True

            self.current_dataloader = (self.current_dataloader + 1) % self.num_dataloaders

        raise StopIteration

    def __len__(self):
        return sum(len(dl) for dl in self.dataloaders)
        
def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--batch_size', default=26, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--accum_iter', default=3, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=256, type=int,
                        help='eeg input length')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=0.0003, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--config_path', default='global_setting_conv_st_mae.json', type=str,
                        help='dataset config path')

    parser.add_argument('--output_dir', default='MAE_large_output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--resume', default='MAE_large_output/checkpoint-96.pth',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    # DDP configs:
    parser.add_argument('--world_size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist_url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='local rank for distributed training')

    #Wandb configs:
    parser.add_argument('--wandb_entity', default="USER", type=str,
                        help='Wandb entity')
    parser.add_argument('--wandb_project', default="eeg-mae-pretrain", type=str,
                        help='Wandb project')
    parser.add_argument('--wandb_log_dir', default="./wandb_log", type=str,
                        help='Wandb log directory')

    return parser


def mae_train(args):
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        device = torch.device('cuda', args.gpu)
    else:
        device = torch.device(args.device)
        
    model = models_mae_eeg.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    if args.rank==0:
        print_size(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=False)
        else:
            # This else clause can be removed because in your previous setup, args.gpu should always be set if distributed
            raise ValueError("GPU must be specified for distributed training")
    
        model_without_ddp = model.module
    else:
        global_rank = 0
        args.rank = 0
        model.to(device)
        model_without_ddp = model

    # Optionally, you can check if the model is properly assigned to the right device
    print(f"Model is assigned to device: {next(model.parameters()).device}")
    print(f"[Rank {args.rank}] using backend:", dist.get_backend(), flush=True)
        
    ### optimizer ###
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    loss_scaler = NativeScaler()
    
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Synchronize and log
    if args.distributed:
        dist.barrier()
        if args.rank == 0:
            print("All processes synchronized",flush=True)

    with open(args.config_path) as fp:
        config = json.load(fp)
        dataset_config = config["dataset_config"]
        train_datasets_to_use = dataset_config["dataset_to_use"]
        valid_datasets_to_use = dataset_config["valid_dataset_to_use"]

    train_sets = [get_pretrain_dataset(dataset) for dataset in train_datasets_to_use]
    valid_sets = [get_pretrain_dataset(dataset) for dataset in valid_datasets_to_use]

    # ratio for training vs validation
    train_ratio = 0.05
    train_splits = []
    valid_splits = []
    train_batch_sizes = []
    valid_batch_sizes = []
    g = torch.Generator().manual_seed(seed)
    for ds in train_sets:
        total_len = len(ds)
        n_train = int(total_len * train_ratio)
        n_valid = total_len - n_train
        # random_split returns two Subset objects
        train_ds, valid_ds = random_split(ds, [n_train, n_valid], generator=g)

        train_splits.append(train_ds)
        valid_splits.append(valid_ds)
        train_batch_sizes.append(get_batch_size(args,ds.num_channel))
        valid_batch_sizes.append(get_batch_size(args,ds.num_channel))
        #train_splits.append((train_ds,ds.num_channel))
        #valid_splits.append((valid_ds,ds.num_channel))

    for ds in valid_sets:
        valid_splits.append(ds)
        valid_batch_sizes.append(get_batch_size(args,ds.num_channel))
        #valid_splits.append((ds,ds.num_channel))

    # test for single dataloader approach
    concat_ds_train = ConcatDataset(train_splits)
    concat_ds_valid = ConcatDataset(valid_splits)

    train_sampler = InterleavedDistributedBatchSampler(
                datasets=train_splits,
                batch_sizes=train_batch_sizes,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=True,      # usually you want to shuffle each sub-dataset
                drop_last=False,
                seed=seed            # or any fixed seed
            )

    valid_sampler = InterleavedDistributedBatchSampler(
                datasets=valid_splits,
                batch_sizes=valid_batch_sizes,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=False,      # usually you want to shuffle each sub-dataset
                drop_last=False,
                seed=seed            # or any fixed seed
            )

    data_loader_train = torch.utils.data.DataLoader(
                    concat_ds_train,
                    batch_sampler=train_sampler,
                    num_workers=args.num_workers,           # pick something like 0.7 × #CPUs_per_task (e.g. 8 if you have 12 CPUs)
                    pin_memory=True,
                    persistent_workers=True,
                )

    data_loader_valid = torch.utils.data.DataLoader(
                    concat_ds_valid,
                    batch_sampler=valid_sampler,
                    num_workers=args.num_workers,           # pick something like 0.7 × #CPUs_per_task (e.g. 8 if you have 12 CPUs)
                    pin_memory=True,
                    persistent_workers=True,
                )    
    
    
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    
    if args.rank==0:
        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)
    
        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

    if args.rank == 0 and args.log_dir is not None:
        print("total training example: ", data_loader_train.__len__())
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
        # Use wandb_id.txt in output_dir to track the run
        wandb_id_path = os.path.join(args.output_dir, "wandb_id.txt")
        if os.path.exists(wandb_id_path):
            with open(wandb_id_path, "r") as f:
                wandb_id = f.read().strip()
            resume_mode = "allow"
        else:
            wandb_id = wandb.util.generate_id()
            with open(wandb_id_path, "w") as f:
                f.write(wandb_id)
            resume_mode = None  # force new run
            
        wandb.init(
            id=wandb_id,
            resume=resume_mode,
            entity=args.wandb_entity, 
            project=args.wandb_project, 
            name=os.path.basename(args.output_dir.rstrip("/")),
            config=vars(args),
            dir=args.wandb_log_dir )
        # watch model parameters & gradients
        wandb.watch(model, log="all", log_freq=100)
    
    else:
        log_writer = None
        
    for epoch in range(args.start_epoch, args.epochs):
        #if args.distributed:
        #    batch_sampler.set_epoch(epoch)
        if args.distributed:
            train_sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)
            #data_loader_train.set_epoch(epoch)
            #data_loader_valid.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args)

        valid_stats = test_one_epoch(
            model,data_loader_valid,
            device, epoch, log_writer=log_writer,
            args=args)
        
        #print("test epoch is done", flush=True)
        if args.output_dir and (epoch % 2 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'valid_{k}': v for k, v in valid_stats.items()},
                        'epoch': epoch,}

        if args.rank == 0:
            wandb.log(log_stats, step=epoch)
            
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    # Finish wandb at end of training
    if args.rank == 0 and wandb.run is not None:
        wandb.finish()
    
def main(args):
    # Print the environment variables for debugging
    print("Environment Variables:")
    print("SLURM_PROCID:", os.environ.get("SLURM_PROCID"))

    # DDP setting
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()
    #print(args.world_size, ngpus_per_node, args.distributed, flush=True)

    if args.distributed:
        if args.local_rank != -1:  # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ:  # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
            # Set local rank based on the GPU assigned
            args.local_rank = args.gpu
        else:
            raise ValueError("Unable to determine rank and gpu assignment.")
        
        if args.gpu >= torch.cuda.device_count():
            raise ValueError(f"Assigned GPU {args.gpu} exceeds available GPU count {torch.cuda.device_count()}.")

        print(f"Rank: {args.rank}, Local Rank: {args.local_rank}, GPU: {args.gpu}")
        
    mae_train(args)

if __name__ == "__main__":
    args = get_args_parser()
    main(args.parse_args())
