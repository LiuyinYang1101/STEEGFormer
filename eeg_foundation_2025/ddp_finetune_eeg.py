import argparse
import os
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
import pickle

import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.misc import print_size
from utils.new_utils import get_model, get_dataset, get_dataloader, get_dataloader_both, get_optimizer, get_loss, log_training_results, setup_experiment_dirs, freeze_for_finetune

from engine_finetune_eeg import train_one_epoch, evaluate
import wandb

def get_args_parser():
    parser = argparse.ArgumentParser('Regression training for NIPS25 challenge', add_help=False)
    parser.add_argument('--challenge', default='challenge1', type=str, metavar='CHALLENGE',
                        help='Which challenge to train')
    
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=4, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    # Foundation model parameters
    parser.add_argument('--finetune_layers', default=12, type=int, metavar='MODEL_LAYER',
                        help='How many layers to finetune')
    parser.add_argument('--head_method', default='avg', type=str, metavar='MODEL_HEAD',
                        help='Name of the classification head')
    parser.add_argument('--head_dropout', type=float, default=0.0, metavar='HDO',
                     help='Drop out using in the last head layer')
    parser.add_argument('--vit_pretrained_model_dir', default='/lustre1/scratch/343/vsc34340/pre_train_model/checkpoint-210.pth', type=str, metavar='PRETRAINED_MODEL_DIR',
                        help='Directory of the pretrained model')
    # --- LoRA args ---
    grp = parser.add_argument_group("LoRA")
    grp.add_argument('--use_lora', action='store_true',
                     help='Enable LoRA on ViT transformer blocks (head stays frozen).')
    grp.add_argument('--lora_r', type=int, default=8, metavar='RANK',
                     help='LoRA rank r.')
    grp.add_argument('--lora_alpha', type=int, default=16, metavar='ALPHA',
                     help='LoRA scaling alpha.')
    grp.add_argument('--lora_dropout', type=float, default=0.0, metavar='P',
                     help='LoRA dropout on the residual path (0 disables).')
    grp.add_argument('--lora_targets', type=str,
                     default='attn.qkv,attn.proj,mlp.fc1,mlp.fc2',
                     help='Comma-separated target linears inside each block.')
    # Which blocks to LoRA: all or just the last N (analogous to finetune_layers)
    grp.add_argument('--lora_last_n', type=int, default=None, metavar='N',
                     help='If set, apply LoRA only to the last N transformer blocks.')
    # Train bias? Train LayerNorms?
    try:
        # Python 3.9+ (you’re on 3.11) supports BooleanOptionalAction
        from argparse import BooleanOptionalAction
        grp.add_argument('--lora_train_bias', default=False, action=BooleanOptionalAction,
                         help='Whether to train biases inside LoRA-wrapped linears.')
        grp.add_argument('--lora_train_norms', default=True, action=BooleanOptionalAction,
                         help='Whether to keep LayerNorm params trainable.')
    except Exception:
        grp.add_argument('--lora_train_bias', action='store_true', help='Train biases.')
        grp.add_argument('--no-lora_train_bias', dest='lora_train_bias', action='store_false')
        grp.add_argument('--lora_train_norms', action='store_true', default=True, help='Train norms.')
        grp.add_argument('--no-lora_train_norms', dest='lora_train_norms', action='store_false')

    # Optional niceties
    grp.add_argument('--lora_wd_zero', action='store_true',
                     help='Force weight decay = 0 for LoRA params (A/B).')
    grp.add_argument('--lora_merge_on_eval', action='store_true',
                     help='Merge LoRA into base weights before eval/inference export.')
    grp.add_argument('--lora_lr_scale', type=float, default=1.0, metavar='SCALE',
                 help='Multiply LR for LoRA params (e.g., 2.0). Default 1.0.')
    grp.add_argument('--lora_no_lrd', action='store_true',
                 help='Disable layer-wise LR decay on LoRA params only.')

    # Optimizer parameters
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

    # Augmentation parameters
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
    parser.add_argument('--local_dataset_dir', default='/dodrio/scratch/projects/2025_500/data',
                        help='local dataset dir')
    parser.add_argument('--output_dir', default='/lustre1/scratch/343/vsc34340/MAE_finetune_output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--eval_weights_path', default='/dodrio/scratch/projects/2025_500/shared/challenge_1/exp_output/exp/vit_large_patch16_20251008_003418/checkpoint-best.pth',
                        help='path to the pre-trained model')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    # DDP configs:
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='local rank for distributed training')

    # Wandb parameters
    parser.add_argument('--wandb_entity', default='liuyin_yang-ku-leuven', type=str,
                        help='Wandb account')
    parser.add_argument('--wandb_log_dir', default='/lustre1/scratch/343/vsc34340/wandb_log', type=str,
                        help='Wandb log directory')
    parser.add_argument('--wandb_log_every', type=int, default=1,
                        help="only log to W&B every N epochs")
    return parser

def log_training_results_prefixed(args, train_stats, valid_stats, test_stats, epoch, log_writer, prefix: str = ""):
    """
    Wrap your existing log_training_results with a prefix for TB/scalar keys.
    If your original logger already supports a prefix/tag, just pass it through.
    """
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"

    # Option A: your logger already supports a 'prefix' or 'tag' kwarg
    try:
        # If it supports prefix/tag, prefer that (best)
        return log_training_results(args, train_stats, valid_stats, test_stats, epoch, log_writer, prefix=prefix)
    except TypeError:
        pass

    # Option B: fall back — clone stats dicts with prefixed keys and call as-is
    def _prefixed(d):
        return {prefix + k: v for k, v in d.items()}
    return log_training_results(args, _prefixed(train_stats), _prefixed(valid_stats), _prefixed(test_stats), epoch, log_writer)

def model_training(args):
    mixup_fn = None
    # --- DDP setup ---
    torch.cuda.set_device(args.gpu)
    # then later in model_training():
    if args.distributed:
        # device already set in main()
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        device = torch.device("cuda", args.gpu)
    else:
        device = torch.device(args.device)

    # --- Build model (identical on all ranks) ---
    model = get_model(args).to(device)
    
    # --- Freeze ONCE (pre-DDP), depending on LoRA or classic finetune ---
    is_vit = "vit" in str(args.model).lower()
    lora_on = bool(getattr(args, "use_lora", False))
    lora_injected = bool(getattr(model, "_lora_injected", False))
    already_frozen = bool(getattr(args, "already_frozen", False))

    if is_vit and (not lora_on) and (not lora_injected) and (not already_frozen):
        # Classic finetune path: unfreeze last N blocks + head/norm
        freeze_for_finetune(model, getattr(args, "finetune_layers", 0))
        args.already_frozen = True  # so get_optimizer won't run freezer again

    # --- Wrap with DDP if needed ---
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu,
            find_unused_parameters=False  # keep False; you froze pre-DDP
        )
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # --- Sanity prints ---
    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    if getattr(args, "rank", 0) == 0:
        print_size(model_without_ddp)
        print(f"Model device: {next(model_without_ddp.parameters()).device}")
        print(f"Trainable params: {n_parameters/1e6:.2f}M")
        print(f"Ready to train for {args.challenge}")

    # --- Optimizer (must NOT change requires_grad here) ---
    optimizer = get_optimizer(args, model_without_ddp)

    # Optional: assert optimizer params == trainable params expected by DDP
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    ddp_ids = {id(p) for p in model_without_ddp.parameters() if p.requires_grad}
    assert opt_ids == ddp_ids, (
        f"Optimizer has {len(opt_ids)} params, but {len(ddp_ids)} are trainable in the model."
    )
    
    # get the train-valid-test dataset
    if args.challenge == "both":
        train_set, valid_ds1, valid_ds2, test_ds1, test_ds2 = get_dataset(args)
        print(len(train_set),len(valid_ds1),len(valid_ds2),len(test_ds1),len(test_ds2))
        # get the dataloader 
        train_loader, valid_loader1, valid_loader2, test_loader1, test_loader2 = get_dataloader_both(args, train_set, valid_ds1, valid_ds2, test_ds1, test_ds2)
    else:
        train_set, valid_set, test_set = get_dataset(args)
        print(len(train_set),len(valid_set),len(test_set))
        # get the dataloader 
        train_loader, valid_loader, test_loader = get_dataloader(args, train_set, valid_set, test_set)


    # get the loss criterion
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    loss_scaler = NativeScaler(enabled=True, use_bfloat16=use_bf16)
    
    criterion = get_loss(args, mixup_fn)
    if args.rank==0:
        print("criterion = %s" % str(criterion))
    

    if args.eval:
        try:
            # PyTorch 2.x: safer loading
            checkpoint = torch.load(args.eval_weights_path, map_location="cpu", weights_only=False)
        except TypeError:
            # PyTorch <2.0
            checkpoint = torch.load(args.eval_weights_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        model.eval()
        valid_stats = evaluate(valid_loader, model, loss_scaler, device)
        test_stats = evaluate(test_loader, model, loss_scaler, device)
        valid_norm = valid_stats.get("norm_rmse", None)
        test_norm = test_stats.get("norm_rmse", None)
        print(
                f"Valid MSE: {valid_stats['mse']:.4f} | "
                f"Test MSE: {test_stats['mse']:.4f} | "
                f"Valid RMSE: {valid_stats['rmse']:.4f} | "
                f"Test RMSE: {test_stats['rmse']:.4f} | "
                f"Valid norm RMSE: {valid_norm:.4f} | "
                f"Test norm RMSE: {test_norm:.4f} | ",
                flush=True
            )
        exit(0)
        
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # then set experiment directories
    args = setup_experiment_dirs(args, model=model)
    # Synchronize and log
    if args.distributed:
        dist.barrier()
        if args.rank == 0:
            print("All processes synchronized",flush=True)

    # --- TensorBoard writer (rank 0 only) ---
    log_writer = None #SummaryWriter(log_dir=args.log_dir) if (misc.is_main_process() and args.log_dir) else None
    # --- W&B: use the exp folder name as run name; point dir to args.log_dir ---
    exp_name = os.path.basename(os.path.normpath(args.log_dir)) if args.log_dir else "run"
    this_run_name = getattr(args, "run_name", exp_name)  # or keep "test" if you prefer
    wandb_group_name = f"group_{time.strftime('%Y%m%d')}"
    run = None
    if misc.is_main_process():
        if args.challenge == "challenge1":
            wandb_project = "NIPSChallenge1New"
        elif args.challenge == "challenge2":
            wandb_project = "NIPSChallenge2New"
        else:
            wandb_project = "NIPSChallengeBothNew"
        run = wandb.init(project=wandb_project,entity=getattr(args, "wandb_entity", None),
            group=wandb_group_name,dir=args.log_dir,                 # <— same experiment folder
            name=this_run_name, job_type="train", reinit=True,
            config={**vars(args), "experiment_type": "challenge_1"},
            mode="online" if getattr(args, "wandb_online", True) else "offline",
        )
        wandb.watch(model.module if getattr(args, "distributed", False) else model, log="all", log_freq=100)
        
    min_valid_norm_rmse  = float("inf")
    min_test_norm_rmse   = float("inf")
    best_valid_epoch     = -1

    min_valid_norm_rmse2 = float("inf")
    min_test_norm_rmse2  = float("inf")
    best_valid_epoch2    = -1

    for epoch in range(args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            
        if args.rank==0:
            if args.output_dir:
                misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,loss_scaler=loss_scaler, epoch="last")

        train_stats = train_one_epoch(
            model, criterion, train_loader,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )

        if args.challenge == "both":
            # Evaluate both challenges explicitly
            valid_stats  = evaluate(valid_loader1, model, loss_scaler, device)  # ch1
            test_stats   = evaluate(test_loader1,  model, loss_scaler, device)
            valid_stats2 = evaluate(valid_loader2, model, loss_scaler, device)  # ch2
            test_stats2  = evaluate(test_loader2,  model, loss_scaler, device)
        else:
            valid_stats = evaluate(valid_loader, model, loss_scaler, device)
            test_stats  = evaluate(test_loader,  model, loss_scaler, device)

        if args.rank == 0:
            # -------- challenge 1 bookkeeping --------
            valid_norm = valid_stats.get("norm_rmse")
            test_norm  = test_stats.get("norm_rmse")

            if valid_norm is not None and valid_norm <= min_valid_norm_rmse:
                min_valid_norm_rmse = valid_norm
                best_valid_epoch = epoch
                if args.output_dir:
                    misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp,
                                    optimizer=optimizer, loss_scaler=loss_scaler, epoch="best_ch1")

            if test_norm is not None:
                min_test_norm_rmse = min(min_test_norm_rmse, test_norm)

            print(
                f"[Challenge 1] "
                f"Valid MSE: {valid_stats['mse']:.4f} | Test MSE: {test_stats['mse']:.4f} | "
                f"Valid RMSE: {valid_stats['rmse']:.4f} | Test RMSE: {test_stats['rmse']:.4f} | "
                f"Valid norm RMSE: {valid_norm:.4f} | Test norm RMSE: {test_norm:.4f} | "
                f"Min Valid norm RMSE: {min_valid_norm_rmse:.4f} | Min Test norm RMSE: {min_test_norm_rmse:.4f} | "
                f"Best Valid Epoch: {best_valid_epoch}",
                flush=True
            )

            # prefix-log ch1
            log_training_results_prefixed(args, train_stats, valid_stats, test_stats, epoch, log_writer, prefix="ch1")

            # -------- challenge 2 bookkeeping (only when both) --------
            if args.challenge == "both":
                valid_norm2 = valid_stats2.get("norm_rmse")
                test_norm2  = test_stats2.get("norm_rmse")

                if valid_norm2 is not None and valid_norm2 <= min_valid_norm_rmse2:
                    min_valid_norm_rmse2 = valid_norm2
                    best_valid_epoch2 = epoch
                    #if args.output_dir:
                    #    misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp,
                    #                    optimizer=optimizer, loss_scaler=loss_scaler, epoch="best_ch2")

                if test_norm2 is not None:
                    min_test_norm_rmse2 = min(min_test_norm_rmse2, test_norm2)

                print(
                    f"[Challenge 2] "
                    f"Valid MSE: {valid_stats2['mse']:.4f} | Test MSE: {test_stats2['mse']:.4f} | "
                    f"Valid RMSE: {valid_stats2['rmse']:.4f} | Test RMSE: {test_stats2['rmse']:.4f} | "
                    f"Valid norm RMSE: {valid_norm2:.4f} | Test norm RMSE: {test_norm2:.4f} | "
                    f"Min Valid norm RMSE: {min_valid_norm_rmse2:.4f} | Min Test norm RMSE: {min_test_norm_rmse2:.4f} | "
                    f"Best Valid Epoch: {best_valid_epoch2}",
                    flush=True
                )

                # prefix-log ch2
                log_training_results_prefixed(args, train_stats, valid_stats2, test_stats2, epoch, log_writer, prefix="ch2")

            # -------- periodic checkpointing (fix) --------
            if (epoch + 1) % 10 == 0 and args.output_dir:
                misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp,
                                optimizer=optimizer, loss_scaler=loss_scaler, epoch=f"epoch_{epoch+1}")
        
    # save the last epoch model
    if args.rank==0:
        if args.output_dir:
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,loss_scaler=loss_scaler, epoch="last")


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
    else:
        args.local_rank = 0
        args.rank = 0
        args.gpu = 0
        
    model_training(args)

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)