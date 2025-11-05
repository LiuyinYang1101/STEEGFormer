from utils.models import EEGNet, EEGTransformer
from utils import models_vit_eeg
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
from timm.models.layers import trunc_normal_

import utils.misc as misc
import utils.lr_decay as lrd

import os
import json
import numpy as np
import wandb
from datetime import datetime
from pathlib import Path
import torch.distributed as dist

import math
import pickle
from utils.challenge_custom_dataset import EEGH5Dataset, EEGH5CropDataset, EEGTwoChallengeDataset
from peft import LoraConfig, get_peft_model, TaskType
from collections import Counter

def _collect_last_n_target_linear_names(vit, last_n, targets):
    """
    Build an explicit list of fully-qualified Linear module names to LoRA-wrap,
    but only in the last N transformer blocks.
    """
    names = []
    blocks = list(getattr(vit, "blocks", []))
    if last_n is not None:
        blocks = blocks[-int(last_n):]
    # we need fully-qualified names, so walk named_modules()
    wanted_suffixes = set(t.strip() for t in targets)
    for full_name, mod in vit.named_modules():
        # we only want linears inside the last-N blocks & matching our target suffix
        if not isinstance(mod, torch.nn.Linear):
            continue
        # match on common timm paths like "...blocks.<i>.attn.qkv", "...mlp.fc1", "...mlp.fc2"
        parts = full_name.split(".")
        if "blocks" in parts:
            # suffix to match against targets (e.g., "attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
            suffix = ".".join(parts[-2:]) if parts[-2] in {"qkv","proj","fc1","fc2"} else ".".join(parts[-3:])
            # keep only desired leaf names
            if any(suf in suffix for suf in wanted_suffixes):
                names.append(full_name)
    # dedupe while preserving order
    seen = set(); out = []
    for n in names:
        if n not in seen:
            out.append(n); seen.add(n)
    return out

def _collect_last_n_target_linear_fqns(vit, last_n, targets):
    """
    Return fully-qualified module names (strings) for target Linear layers
    but ONLY in the last N transformer blocks.
    targets is a tuple like ('attn.qkv','attn.proj','mlp.fc1','mlp.fc2').
    """
    fqns = []
    blocks = getattr(vit, "blocks", None)
    if blocks is None:
        return fqns
    total = len(blocks)
    start = 0 if last_n is None else max(0, total - int(last_n))
    want_attn_qkv = "attn.qkv" in targets
    want_attn_proj = "attn.proj" in targets
    want_mlp_fc1 = "mlp.fc1" in targets
    want_mlp_fc2 = "mlp.fc2" in targets

    for i in range(start, total):
        blk = blocks[i]
        # attn.qkv
        if want_attn_qkv and hasattr(blk, "attn") and isinstance(getattr(blk.attn, "qkv", None), nn.Linear):
            fqns.append(f"blocks.{i}.attn.qkv")
        # attn.proj
        if want_attn_proj and hasattr(blk, "attn") and isinstance(getattr(blk.attn, "proj", None), nn.Linear):
            fqns.append(f"blocks.{i}.attn.proj")
        # mlp.fc1
        if want_mlp_fc1 and hasattr(blk, "mlp") and isinstance(getattr(blk.mlp, "fc1", None), nn.Linear):
            fqns.append(f"blocks.{i}.mlp.fc1")
        # mlp.fc2
        if want_mlp_fc2 and hasattr(blk, "mlp") and isinstance(getattr(blk.mlp, "fc2", None), nn.Linear):
            fqns.append(f"blocks.{i}.mlp.fc2")
    return fqns

def _apply_peft_lora_to_vit(model, *, r, alpha, dropout, targets, last_n,
                            train_bias=False, train_norms=True, verbose=True):
    base = model

    # 1) Build exact list of modules to wrap (only last_n blocks)
    target_list = _collect_last_n_target_linear_fqns(base, last_n, targets)
    if not target_list:
        print("[PEFT] Warning: no Linear targets found; check --lora_targets / --lora_last_n.")

    # 2) Configure LoRA
    bias_mode = "lora_only" if train_bias else "none"
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,  # freezes base weights by default
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=target_list,   # fully qualified names
        bias=bias_mode,               # "none" | "all" | "lora_only"
    )

    # 3) Apply PEFT
    peft_model = get_peft_model(base, lora_cfg)

    # 4) Optionally let LayerNorms train (no LoRA on them)
    if train_norms:
        for m in peft_model.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad_(True)

    # 5) Ensure classification head is trainable with LoRA fine-tuning
    head = (getattr(peft_model, "cls_head", None)
            or getattr(peft_model, "head", None)
            or getattr(peft_model, "fc", None))
    if head is not None:
        for p in head.parameters():
            p.requires_grad_(True)

    # Book-keeping flags
    setattr(peft_model, "_lora_injected", True)
    setattr(peft_model, "_lora_last_n", int(last_n) if last_n is not None else None)

    # ---- Debug / verification prints ----
    if verbose:
        # 1) PEFT's own summary if available
        try:
            peft_model.print_trainable_parameters()
        except Exception:
            pass

        # 2) List every trainable tensor (name and size)
        trainable_params = []
        for n, p in peft_model.named_parameters():
            if p.requires_grad:
                trainable_params.append((n, p.numel()))
        total_train = sum(n for _, n in trainable_params)
        print(f"[PEFT] Trainable tensors: {len(trainable_params)} | Trainable params: {total_train:,}")
        for name, num in sorted(trainable_params):
            print(f"  - {name} ({num:,})")

        # 3) Compact per-module summary (which module *objects* have any trainable params)
        mod_counter = Counter()
        for mod_name, mod in peft_model.named_modules():
            # Only count module-local params (no recurse) to avoid double counting
            if any(p.requires_grad for p in mod.parameters(recurse=False)):
                mod_counter[type(mod).__name__] += 1
        if mod_counter:
            print("[PEFT] Modules with trainable params (type: count):")
            for t, c in sorted(mod_counter.items()):
                print(f"  - {t}: {c}")

        # 4) Show which target module names got LoRA
        if target_list:
            print("[PEFT] LoRA target modules (by name):")
            for t in target_list:
                print(f"  - {t}")

        # 5) Explicitly state head status
        print(f"[PEFT] Classification head trainable: {any(p.requires_grad for p in head.parameters()) if head else False}")

        # 6) Quick sanity check that base (non-LoRA) transformer weights are frozen
        # (they should be, since TaskType.FEATURE_EXTRACTION freezes base params)
        base_frozen_ok = True
        for n, p in peft_model.named_parameters():
            if ("lora_" not in n) and ("cls_head" not in n) and ("head" not in n) and ("fc" not in n):
                # ignore norms if you chose to train them
                if not train_norms and isinstance(_get_module_by_param_name(peft_model, n), nn.LayerNorm):
                    continue
                # if it's not LoRA nor head, it should be frozen
                if p.requires_grad:
                    base_frozen_ok = False
                    break
        print(f"[PEFT] Base transformer frozen (excluding LoRA/head/norms-by-choice): {base_frozen_ok}")

    return peft_model

def _get_module_by_param_name(model, param_name: str):
    """
    Best-effort helper to find the owning module of a parameter by name.
    Returns the module if found, else None. Used only for the quick sanity check.
    """
    # Strip the final ".weight"/".bias"/etc. to get the module path
    parts = param_name.split(".")
    if len(parts) <= 1:
        return None
    mod_path = parts[:-1]
    mod = model
    for p in mod_path:
        if not hasattr(mod, p):
            return None
        mod = getattr(mod, p)
    return mod

# A small helper for Welford accumulation
class RunningStat:
    """Compute running mean and variance using Welford's algorithm."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # sum of squares of differences to current mean

    def update(self, x):
        """
        x: a numpy array or torch tensor (1D) or flattenable
        Flattened to a vector of values; call update on each scalar or batch logic.
        Here we do batch logic: x is 1D or more, but we aggregate over all elements.
        """
        # convert to flat
        x = x.view(-1).cpu().numpy()
        for val in x:
            self.n += 1
            delta = val - self.mean
            self.mean += delta / self.n
            delta2 = val - self.mean
            self.M2 += delta * delta2

    def merge(self, other):
        """Merge another RunningStat into this one (parallel / distributed)"""
        if other.n == 0:
            return
        if self.n == 0:
            self.n = other.n
            self.mean = other.mean
            self.M2 = other.M2
            return
        # based on formulas to combine two Welford accumulators
        n1, mu1, M2_1 = self.n, self.mean, self.M2
        n2, mu2, M2_2 = other.n, other.mean, other.M2

        delta = mu2 - mu1
        n = n1 + n2
        # new mean
        new_mean = (n1 * mu1 + n2 * mu2) / n
        # new M2
        new_M2 = M2_1 + M2_2 + delta * delta * (n1 * n2) / n

        self.n = n
        self.mean = new_mean
        self.M2 = new_M2

    @property
    def variance(self):
        return (self.M2 / self.n) if self.n > 0 else 0.0

    @property
    def std(self):
        return math.sqrt(self.variance)


def reduce_scalar_tensor(val, average=True):
    """Helper: reduce a scalar float tensor across all ranks."""
    if not (dist.is_available() and dist.is_initialized()):
        return val
    t = torch.tensor(val, dtype=torch.float64, device='cuda')
    dist.all_reduce(t)
    if average:
        t /= dist.get_world_size()
    return t.item()
    
EPOCH_LEN_S = 2.0
SFREQ = 100 # by definition here

def setup_experiment_dirs(args, model=None):
    """
    Make args.output_dir and args.log_dir point to:
      <root>/exp/<modelName_YYYYmmdd_HHMMSS>/
    Creates the folders (rank 0), then syncs.
    """
    # Figure out a readable model name
    if model is not None:
        model_name = getattr(args, "model", None) or getattr(model, "model_name", None) or model.__class__.__name__
    else:
        model_name = getattr(args, "model", None) or "model"

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{model_name}_{date_str}"

    out_root = Path(getattr(args, "output_dir", "./output_dir") or "./output_dir")
    log_root = Path(getattr(args, "log_dir", str(out_root)) or str(out_root))

    args.output_dir = str(out_root / "exp" / exp_name)
    args.log_dir    = str(log_root  / "exp" / exp_name)

    # Create on main process, then barrier so others see it
    try:
        from util import misc  # or wherever misc.is_main_process lives
    except Exception:
        # fallback if misc isn't importable here; only rank 0 creates
        def is_main_process():
            return (not dist.is_available() or not dist.is_initialized() 
                    or dist.get_rank() == 0)
    else:
        is_main_process = misc.is_main_process

    if is_main_process():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    return args
    


def get_model(args, window_length=200, no_channels=129, num_class=1):
    if args.model == "eegnet":
        model = EEGNet(no_spatial_filters=4, no_channels=no_channels, no_temporal_filters=8, temporal_length_1=64, temporal_length_2=16, window_length=window_length, num_class=num_class, drop_out_ratio=0.50, pooling2=4, pooling3=8)
        return model
    elif args.model == "ctnet":
        model = EEGTransformer(heads=4, emb_size=40, depth=6, number_class=num_class, number_channel=no_channels,
                       data_length=int(window_length), sampling_rate=int(100))
        return model
    if "vit" in args.model:
        if args.challenge == "both":
            num_tasks = 2
        else:
            num_tasks = 1
        model = models_vit_eeg.__dict__[args.model](
            num_classes=num_class,
            num_tasks=num_tasks,
            num_tokens=int(window_length/100*128*no_channels/16),
            drop_path_rate=0.1,
            proj_drop_rate=0.00,
            global_pool=args.head_method,
            head_drop_out=args.head_dropout
        )
        if hasattr(model, "set_grad_checkpointing"):
            print("Enabling grad checkpointing")
            model.set_grad_checkpointing(True)
        # load pre-trained model if needed
        if args.vit_pretrained_model_dir:
            checkpoint = torch.load(args.vit_pretrained_model_dir, map_location="cpu", weights_only=False)
            print(f"Load pre-trained checkpoint from: {args.vit_pretrained_model_dir}")
    
            checkpoint_model = checkpoint.get("model", checkpoint)  # allow raw state_dict as well
            state_dict = model.state_dict()
    
            # (A) Remove keys that don't exist or have mismatched shapes (robust across modes)
            to_delete = []
            for k, v in checkpoint_model.items():
                if k not in state_dict or state_dict[k].shape != v.shape:
                    to_delete.append(k)
            if to_delete:
                print(f"Pruning {len(to_delete)} incompatible keys from checkpoint (e.g. different head/num_tokens).")
                for k in to_delete:
                    del checkpoint_model[k]
    
            # (B) Load backbone (and any compatible head parts) non-strictly
            msg = model.load_state_dict(checkpoint_model, strict=False)
            # print(msg)  # optional debugging
    
            # (C) Manually (re)initialize classifier layer(s)
            cls_head = getattr(model, "cls_head", None)
    
            # token/avg/all use .final; all_simple uses .final_simple
            if hasattr(cls_head, "final") and isinstance(cls_head.final, nn.Linear):
                trunc_normal_(cls_head.final.weight, std=2e-5)
                if cls_head.final.bias is not None:
                    nn.init.zeros_(cls_head.final.bias)
    
            if hasattr(cls_head, "final_simple") and isinstance(cls_head.final_simple, nn.Linear):
                trunc_normal_(cls_head.final_simple.weight, std=2e-5)
                if cls_head.final_simple.bias is not None:
                    nn.init.zeros_(cls_head.final_simple.bias)
    
            # (D) Initialize the per-token projection used before flattening
            mode = getattr(cls_head, "mode", "")
            per_token_linear = None
            if mode == "all" and hasattr(cls_head, "per_token") and isinstance(cls_head.per_token[0], nn.Linear):
                per_token_linear = cls_head.per_token[0]  # Linear(embed_dim -> 64)
            elif mode == "all_simple" and hasattr(cls_head, "per_token_simple") and isinstance(cls_head.per_token_simple, nn.Linear):
                per_token_linear = cls_head.per_token_simple  # Linear(embed_dim -> 64)
    
            if per_token_linear is not None:
                trunc_normal_(per_token_linear.weight, std=0.02)
                if per_token_linear.bias is not None:
                    nn.init.zeros_(per_token_linear.bias)

        if args.use_lora:
            targets = tuple(x.strip() for x in args.lora_targets.split(',') if x.strip())
            model = _apply_peft_lora_to_vit(
                model,
                r=args.lora_r,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
                targets=targets,
                last_n=args.lora_last_n,        # now respected
                train_bias=args.lora_train_bias,
                train_norms=args.lora_train_norms,
            )
            args.already_frozen = True
            
        return model




def get_dataset(args):
    if args.challenge == "challenge1":
        train_set = EEGH5Dataset("/dodrio/scratch/projects/2025_500/h5_challenge_datasets/challenge1/eeg_challenge1_dataset.h5", split="train")
        valid_set = EEGH5Dataset("/dodrio/scratch/projects/2025_500/h5_challenge_datasets/challenge1/eeg_challenge1_dataset.h5", split="valid")
        test_set = EEGH5Dataset("/dodrio/scratch/projects/2025_500/h5_challenge_datasets/challenge1/eeg_challenge1_dataset.h5", split="test")
        return train_set, valid_set, test_set
    elif args.challenge == "challenge2":
        train_set = EEGH5CropDataset("/dodrio/scratch/projects/2025_500/h5_challenge_datasets/challenge2/eeg_challenge2_dataset.h5",
                                     split="train",crop_size=200,seed=2025)
        valid_set = EEGH5CropDataset("/dodrio/scratch/projects/2025_500/h5_challenge_datasets/challenge2/eeg_challenge2_dataset.h5",
                                     split="valid",crop_size=200,seed=2025)
        test_set = EEGH5CropDataset("/dodrio/scratch/projects/2025_500/h5_challenge_datasets/challenge2/eeg_challenge2_dataset.h5",
                                    split="test",crop_size=200,seed=2025)
        return train_set, valid_set, test_set
    else:
        ch1_train = EEGH5Dataset("/dodrio/scratch/projects/2025_500/h5_challenge_datasets/challenge1/eeg_challenge1_dataset.h5", split="train")
        ch1_valid = EEGH5Dataset("/dodrio/scratch/projects/2025_500/h5_challenge_datasets/challenge1/eeg_challenge1_dataset.h5", split="valid")
        ch1_test  = EEGH5Dataset("/dodrio/scratch/projects/2025_500/h5_challenge_datasets/challenge1/eeg_challenge1_dataset.h5", split="test")

        ch2_train = EEGH5CropDataset("/dodrio/scratch/projects/2025_500/h5_challenge_datasets/challenge2/eeg_challenge2_dataset.h5",
                                     split="train",crop_size=200,seed=2025)
        ch2_valid = EEGH5CropDataset("/dodrio/scratch/projects/2025_500/h5_challenge_datasets/challenge2/eeg_challenge2_dataset.h5",
                                     split="valid",crop_size=200,seed=2025)
        ch2_test  = EEGH5CropDataset("/dodrio/scratch/projects/2025_500/h5_challenge_datasets/challenge2/eeg_challenge2_dataset.h5",
                                    split="test",crop_size=200,seed=2025)

        # Wrap
        train_ds = EEGTwoChallengeDataset(ch1_train, ch2_train)
        valid_ds1 = EEGTwoChallengeDataset(ch1_valid, None)
        valid_ds2 = EEGTwoChallengeDataset(None, ch2_valid)
        test_ds1  = EEGTwoChallengeDataset(ch1_test,  None)
        test_ds2  = EEGTwoChallengeDataset(None,  ch2_test)
        return train_ds, valid_ds1, valid_ds2, test_ds1, test_ds2

def get_dataloader_both(args, train_set, valid_set1, valid_set2, test_set1, test_set2):
    if args.distributed:
        num_tasks = args.world_size
        global_rank = args.rank

        sampler_train = torch.utils.data.DistributedSampler(train_set, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        train_loader = torch.utils.data.DataLoader(
                train_set, sampler=sampler_train,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                persistent_workers=True,
                pin_memory=args.pin_mem,
                prefetch_factor=2,
                drop_last=True)

        sampler_valid = torch.utils.data.DistributedSampler(valid_set1, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        valid_loader1 = torch.utils.data.DataLoader(
                valid_set1, sampler=sampler_valid,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                persistent_workers=True,
                pin_memory=args.pin_mem,
                prefetch_factor=2,
                drop_last=True)
        
        sampler_valid2 = torch.utils.data.DistributedSampler(valid_set2, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        valid_loader2 = torch.utils.data.DataLoader(
                valid_set2, sampler=sampler_valid2,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                persistent_workers=True,
                pin_memory=args.pin_mem,
                prefetch_factor=2,
                drop_last=True)

        sampler_test = torch.utils.data.DistributedSampler(test_set1, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        test_loader1 = torch.utils.data.DataLoader(
                test_set1, sampler=sampler_test,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                persistent_workers=True,
                pin_memory=args.pin_mem,
                prefetch_factor=2,
                drop_last=True)
        
        sampler_test2 = torch.utils.data.DistributedSampler(test_set2, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        test_loader2 = torch.utils.data.DataLoader(
                test_set2, sampler=sampler_test2,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                persistent_workers=True,
                pin_memory=args.pin_mem,
                prefetch_factor=2,
                drop_last=True)
 
    else:
        global_rank = 0
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle = True)
        valid_loader1 = torch.utils.data.DataLoader(valid_set1, batch_size=args.batch_size, shuffle = False)
        valid_loader2 = torch.utils.data.DataLoader(valid_set2, batch_size=args.batch_size, shuffle = False)
        test_loader1 = torch.utils.data.DataLoader(test_set1, batch_size=args.batch_size, shuffle = False)
        test_loader2 = torch.utils.data.DataLoader(test_set2, batch_size=args.batch_size, shuffle = False)
    return train_loader, valid_loader1, valid_loader2, test_loader1, test_loader2

def get_dataloader(args, train_set, valid_set, test_set):
    if args.distributed:
        num_tasks = args.world_size
        global_rank = args.rank

        sampler_train = torch.utils.data.DistributedSampler(train_set, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        train_loader = torch.utils.data.DataLoader(
                train_set, sampler=sampler_train,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                persistent_workers=True,
                pin_memory=args.pin_mem,
                prefetch_factor=2,
                drop_last=True)

        sampler_valid = torch.utils.data.DistributedSampler(valid_set, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        valid_loader = torch.utils.data.DataLoader(
                valid_set, sampler=sampler_valid,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                persistent_workers=True,
                pin_memory=args.pin_mem,
                prefetch_factor=2,
                drop_last=True)

        sampler_test = torch.utils.data.DistributedSampler(test_set, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        test_loader = torch.utils.data.DataLoader(
                test_set, sampler=sampler_test,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                persistent_workers=True,
                pin_memory=args.pin_mem,
                prefetch_factor=2,
                drop_last=True)
 
    else:
        global_rank = 0
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle = True)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle = False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle = False)
    return train_loader, valid_loader, test_loader

def _unfreeze_module(mod):
    if mod is None:
        return
    for p in mod.parameters():
        p.requires_grad = True

def freeze_for_finetune(model, n_last_layers: int | None):
    """
    Freeze all params, then unfreeze:
      - the last `n_last_layers` transformer blocks (model.blocks[-n_last_layers:])
      - the classification head (model.head / model.cls_head / model.fc)
      - the final norm(s) if present (model.norm / model.fc_norm)
      - the task embeddings if present (e.g., model.task_token_embed)
    """
    # If wrapped by DDP/DataParallel, operate on the real module
    base = model.module if hasattr(model, "module") else model

    # 1) freeze everything
    for p in base.parameters():
        p.requires_grad = False

    # 2) unfreeze last N transformer blocks
    unfrozen_block_indices = []
    if hasattr(base, "blocks") and base.blocks is not None:
        total = len(base.blocks)
        n = 0 if n_last_layers is None else max(0, min(int(n_last_layers), total))
        start = total - n
        for i in range(start, total):
            for p in base.blocks[i].parameters():
                p.requires_grad = True
            unfrozen_block_indices.append(i)
        print(f"[finetune] Unfrozen transformer blocks: {unfrozen_block_indices}")
    else:
        print("[finetune] Warning: model has no `blocks`; only head/norms will be unfrozen.")

    # 3) unfreeze the classification head
    head_modules = []
    for name in ("head", "cls_head", "fc"):
        if hasattr(base, name) and getattr(base, name) is not None:
            head_modules.append(name)
            _unfreeze_module(getattr(base, name))
    if not head_modules:
        print("[finetune] Warning: model has no classification head among {head, cls_head, fc}.")

    # 4) unfreeze the final normalization(s) if present
    _unfreeze_module(getattr(base, "fc_norm", None))
    _unfreeze_module(getattr(base, "norm", None))

    # 5) NEW — unfreeze task embeddings if present
    # Common names to probe; add more aliases if you use different naming.
    task_embed_attr_candidates = (
        "task_token_embed",   # from your updated ViT
        "task_embed",
        "task_embedding",
        "task_emb",
    )
    found_task_embed = False
    for attr in task_embed_attr_candidates:
        if hasattr(base, attr) and getattr(base, attr) is not None:
            _unfreeze_module(getattr(base, attr))
            print(f"[finetune] Unfrozen task embeddings: `{attr}`")
            found_task_embed = True
            break
    if not found_task_embed:
        print("[finetune] Note: no task embeddings found to unfreeze.")

    # summary
    n_trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in base.parameters())
    pct = 100.0 * (n_trainable / max(1, n_total))
    print(f"[finetune] Trainable params: {n_trainable:,} / {n_total:,} ({pct:.2f}%)")


def get_optimizer(args, model_without_ddp):
    """
    AdamW with layer-wise LR decay (LRD) for ViT-like models.
    LoRA-aware:
      - Skips classic finetune freezer if LoRA is enabled/injected.
      - Keeps only params with requires_grad=True.
      - Optional knobs for LoRA groups:
          --lora_wd_zero     -> WD=0 for lora_A/B
          --lora_lr_scale    -> multiply LR for LoRA groups (e.g., 2.0)
          --lora_no_lrd      -> neutralize layer-decay for LoRA groups
    Expects:
      args.model, args.weight_decay, args.layer_decay, args.batch_size, args.accum_iter, args.blr, args.lr (opt)
    """
    # Map param -> name (to detect LoRA params in groups)
    name_of = {p: n for n, p in model_without_ddp.named_parameters()}

    if 'vit' in str(args.model).lower():
        # Skip classic freezer if LoRA is active
        use_lora = bool(getattr(args, "use_lora", False))
        lora_injected = bool(getattr(model_without_ddp, "_lora_injected", False))
        if (not getattr(args, "already_frozen", False)) and (not use_lora) and (not lora_injected):
            freeze_for_finetune(model_without_ddp, getattr(args, "finetune_layers", 0))

        # Build param groups with LRD (timm-style helper)
        no_wd_list = []
        if hasattr(model_without_ddp, "no_weight_decay") and callable(model_without_ddp.no_weight_decay):
            no_wd_list = list(model_without_ddp.no_weight_decay())

        param_groups = lrd.param_groups_lrd(
            model_without_ddp,
            weight_decay=args.weight_decay,
            no_weight_decay_list=no_wd_list,
            layer_decay=args.layer_decay
        )

        # Keep only trainable params in each group
        filtered_groups = []
        for g in param_groups:
            trainable = [p for p in g["params"] if p.requires_grad]
            if trainable:
                g = {**g, "params": trainable}
                filtered_groups.append(g)
        param_groups = filtered_groups

        if not param_groups or all(len(g["params"]) == 0 for g in param_groups):
            raise ValueError("No trainable parameters found. Check LoRA injection / finetune_layers / freezing logic.")

        # Effective LR (derive from blr if lr not provided)
        world_size = getattr(misc, "get_world_size", lambda: 1)()
        accum_iter = int(getattr(args, "accum_iter", 1))
        eff_bs = int(args.batch_size) * accum_iter * world_size
        if getattr(args, "lr", None) is None:
            args.lr = float(args.blr) * eff_bs / 256.0

        # LoRA-specific knobs
        lora_lr_scale = float(getattr(args, "lora_lr_scale", 1.0))
        lora_wd_zero = bool(getattr(args, "lora_wd_zero", False))
        disable_lrd_on_lora = bool(getattr(args, "lora_no_lrd", False))

        # Adjust groups that contain LoRA params
        for g in param_groups:
            names = [name_of[p] for p in g["params"] if p in name_of]
            has_lora = any(("lora_A" in n or "lora_B" in n) for n in names)
            if has_lora:
                if lora_wd_zero:
                    g["weight_decay"] = 0.0
                if lora_lr_scale != 1.0:
                    g["lr"] = args.lr * lora_lr_scale
                if disable_lrd_on_lora and "lr_scale" in g:
                    g["lr_scale"] = 1.0  # neutralize layer-decay for LoRA group

        # Logs (rank 0)
        if getattr(args, "rank", 0) == 0:
            base_lr = args.lr * 256.0 / max(1, eff_bs)
            print(f"base lr: {base_lr:.2e}")
            print(f"actual lr: {args.lr:.2e}")
            print(f"accumulate grad iterations: {accum_iter}")
            print(f"effective batch size: {eff_bs}")
            n_trainable = sum(p.numel() for g in param_groups for p in g["params"])
            print(f"Trainable parameters (post-grouping): {n_trainable:,}")

        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    else:
        # Non-ViT path: plain AdamW over trainables, derive LR if needed
        world_size = getattr(misc, "get_world_size", lambda: 1)()
        eff_bs = int(args.batch_size) * int(getattr(args, "accum_iter", 1)) * world_size
        if getattr(args, "lr", None) is None:
            args.lr = float(args.blr) * eff_bs / 256.0
        if getattr(args, "rank", 0) == 0:
            print("base lr: %.2e" % (args.lr * 256 / eff_bs))
            print("actual lr: %.2e" % args.lr)
            print("accumulate grad iterations: %d" % int(getattr(args, "accum_iter", 1)))
            print("effective batch size: %d" % eff_bs)
        params = [p for p in model_without_ddp.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0.05))

    return optimizer

def get_loss(args, mixup_fn=None):
    criterion = torch.nn.MSELoss()

    return criterion


def _to_float(v):
    if isinstance(v, torch.Tensor):
        return v.detach().item()
    if isinstance(v, (np.generic,)):
        return float(v)
    return float(v) if isinstance(v, (int, float)) else v

def _flatten_stats(prefix, stats: dict):
    return {f"{prefix}_{k}": _to_float(v) for k, v in stats.items()}

def _log_perf_scalars(writer, split_name, stats, epoch):
    """Log selected metrics to TensorBoard under perf/* if present."""
    if writer is None or not stats:
        return
    for key in ("loss", "mse", "rmse", "norm_rmse", "acc1", "acc2"):
        if key in stats:
            writer.add_scalar(f"perf/{split_name}_{key}", _to_float(stats[key]), epoch)

def log_training_results(args, train_stats, valid_stats, test_stats, epoch, log_writer):
    # ---- TensorBoard ----
    _log_perf_scalars(log_writer, "train", train_stats, epoch)
    _log_perf_scalars(log_writer, "valid", valid_stats, epoch)
    _log_perf_scalars(log_writer, "test",  test_stats,  epoch)

    # ---- Build a flat dict for JSON/W&B ----
    log_stats = {
        **_flatten_stats("train", train_stats or {}),
        **_flatten_stats("valid", valid_stats or {}),
        **_flatten_stats("test",  test_stats  or {}),
        "epoch": int(epoch),
    }

    # (Optional) Mirror perf/* keys in W&B
    if misc.is_main_process() and wandb.run is not None:
        every = getattr(args, "wandb_log_every", 1)
        if (epoch % max(1, int(every))) == 0:
            wb_perf = {}
            for split_name, stats in (("train", train_stats), ("valid", valid_stats), ("test", test_stats)):
                if not stats:
                    continue
                for key in ("loss", "mse", "rmse", "norm_rmse", "acc1", "acc2"):
                    if key in stats:
                        wb_perf[f"perf/{split_name}_{key}"] = _to_float(stats[key])
            wandb.log({**log_stats, **wb_perf}, step=int(epoch))

    # ---- Local JSON log (only main process) ----
    if args.output_dir and misc.is_main_process():
        if log_writer is not None:
            log_writer.flush()
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "log"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")