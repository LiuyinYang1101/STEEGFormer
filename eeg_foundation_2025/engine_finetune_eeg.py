# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch
import torch.distributed as dist

from timm.data import Mixup
from contextlib import nullcontext

import utils.misc as misc
import utils.lr_sched as lr_sched
from utils.new_utils import RunningStat

import inspect

def _model_supports_task_index(model) -> bool:
    base = model.module if hasattr(model, "module") else model
    return getattr(base, "task_token_embed", None) is not None


def _unpack_batch(samples, device):
    """
    Accepts (eeg, target) or (eeg, target, task_idx) or dict-like.
    Returns (eeg, targets, task_idx_or_None) moved to device.
    """
    task_idx = None

    if isinstance(samples, (list, tuple)):
        if len(samples) < 2:
            raise ValueError(f"Batch must contain at least 2 items, got {len(samples)}")
        eeg     = samples[0].to(device, non_blocking=True)
        targets = samples[1].to(device, non_blocking=True).float()
        if len(samples) >= 3 and samples[2] is not None:
            task_idx = samples[2].to(device, non_blocking=True).long()

    elif isinstance(samples, dict):
        # Try common keys; adjust if you use other names
        eeg     = (samples.get("eeg", samples.get("x"))).to(device, non_blocking=True)
        targets = (samples.get("targets", samples.get("y"))).to(device, non_blocking=True).float()
        if "task_idx" in samples and samples["task_idx"] is not None:
            task_idx = samples["task_idx"].to(device, non_blocking=True).long()
    else:
        raise TypeError(f"Unsupported batch type: {type(samples)}")

    return eeg, targets, task_idx

def _first_tensor(x):
    if isinstance(x, (list, tuple)):
        x = x[0]
    elif isinstance(x, dict):
        for k in ("logits", "pred", "output", "out"):
            if k in x:
                x = x[k]
                break
    return x

def _to_1d_float(x, name: str, *, detach: bool = False):
    if not torch.is_tensor(x):
        # If it's not a tensor (e.g., numpy), create a tensor ON THE SAME DEVICE AS x if known,
        # else fall back to CUDA if available.
        device = None
        if torch.cuda.is_available():
            device = torch.device("cuda")
        x = torch.as_tensor(x, device=device)
    if detach:
        x = x.detach()
    x = x.to(dtype=torch.float32)
    while x.ndim > 1 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    if x.ndim == 2 and x.shape[1] > 1:
        raise ValueError(f"{name} must be scalar per sample for challenge2; got {tuple(x.shape)}")
    if x.ndim == 0:
        x = x.view(1)
    if x.ndim != 1:
        raise ValueError(f"Expected {name} to be 1-D (B,), got {tuple(x.shape)}")
    return x

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ", rank=args.rank)
    metric_logger.add_meter('lr',   misc.SmoothedValue(window_size=1,  fmt='{value:.6f}'))
    metric_logger.add_meter('loss', misc.SmoothedValue(window_size=10, fmt='{avg:.6f}'))
    metric_logger.add_meter('mse',  misc.SmoothedValue(window_size=10, fmt='{avg:.6f}'))

    header = f'Epoch: [{epoch}]'
    print_freq = 400
    accum_iter = args.accum_iter

    optimizer.zero_grad(set_to_none=True)
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    supports_task = _model_supports_task_index(model)

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Optionally advance LR only on update steps:
        if (data_iter_step % accum_iter) == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        eeg, targets, task_idx = _unpack_batch(samples, device)

        # Optional mixup (if used) — applies to (eeg, targets) only
        if mixup_fn is not None:
            eeg, targets = mixup_fn(eeg, targets)

        # DDP no_sync for gradient accumulation
        use_no_sync = hasattr(model, "no_sync") and ((data_iter_step + 1) % accum_iter != 0)
        ddp_ctx = model.no_sync() if use_no_sync else nullcontext()

        with ddp_ctx, loss_scaler.autocast():
            if supports_task and (task_idx is not None):
                outputs = model(eeg, task_index=task_idx)
            else:
                outputs = model(eeg)
            outputs = _first_tensor(outputs)

            preds = _to_1d_float(outputs, "outputs", detach=False)  # keep graph
            trues = _to_1d_float(targets, "targets", detach=True)   # labels detached
            if preds.shape[0] != trues.shape[0]:
                raise ValueError(f"Pred/target batch mismatch: preds={tuple(preds.shape)}, targets={tuple(trues.shape)}")

            loss = criterion(preds, trues)

        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad(set_to_none=True)

        # Logging
        metric_logger.update(loss=loss_value)

        diff = (preds.detach() - trues.detach()).to(torch.float32)
        batch_sse = diff.pow(2).sum().item()
        batch_n   = diff.numel()
        batch_mse = batch_sse / max(1, batch_n)
        metric_logger.meters['mse'].update(batch_mse, n=batch_n)

        max_lr = max(group["lr"] for group in optimizer.param_groups)
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr',   max_lr,            epoch_1000x)

    metric_logger.synchronize_between_processes()
    global_mse = metric_logger.mse.global_avg if 'mse' in metric_logger.meters else float('nan')
    rmse = math.sqrt(global_mse) if (math.isfinite(global_mse) and global_mse >= 0) else float('nan')

    if args.rank == 0:
        print("Averaged stats:", metric_logger)
        print(f"* RMSE {rmse:.6f}", flush=True)

    out = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    out['rmse'] = rmse
    return out


@torch.no_grad()
def evaluate(data_loader, model, loss_scaler, device):
    is_dist    = dist.is_available() and dist.is_initialized()
    rank       = dist.get_rank() if is_dist else 0
    world_size = dist.get_world_size() if is_dist else 1

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ", rank=rank)
    header = "Test:"
    metric_logger.add_meter('mse', misc.SmoothedValue(window_size=10, fmt='{avg:.6f}'))

    running_true  = RunningStat()
    running_sqerr = torch.zeros((), device=device, dtype=torch.float64)
    running_count = torch.zeros((), device=device, dtype=torch.int64)

    supports_task = _model_supports_task_index(model)

    for samples in metric_logger.log_every(data_loader, 100, header):
        eeg, targets, task_idx = _unpack_batch(samples, device)

        with loss_scaler.autocast():
            if supports_task and (task_idx is not None):
                outputs = model(eeg, task_index=task_idx)
            else:
                outputs = model(eeg)
        outputs = _first_tensor(outputs)

        preds = _to_1d_float(outputs, "outputs", detach=True)
        trues = _to_1d_float(targets, "targets", detach=True)
        if preds.shape[0] != trues.shape[0]:
            raise ValueError(f"Pred/target batch mismatch: preds={tuple(preds.shape)}, targets={tuple(trues.shape)}")

        valid_mask = torch.isfinite(trues)
        if not torch.all(valid_mask):
            preds = preds[valid_mask]
            trues = trues[valid_mask]

        diff      = (preds - trues).to(torch.float32)
        batch_sse = (diff * diff).sum().to(torch.float64)
        batch_n   = torch.tensor(diff.numel(), device=device, dtype=torch.int64)

        running_sqerr += batch_sse
        running_count += batch_n

        running_true.update(trues.detach().cpu())

        batch_mse = (batch_sse / batch_n.clamp_min(1)).detach()
        metric_logger.meters["mse"].update(float(batch_mse), n=int(batch_n))

        if not hasattr(metric_logger, "_dbg_printed"):
            shape_task = None if task_idx is None else tuple(task_idx.shape)
            print(f"[debug] eeg={tuple(eeg.shape)}, preds={tuple(preds.shape)}, targets={tuple(trues.shape)}, task_idx={shape_task}")
            metric_logger._dbg_printed = True

    metric_logger.synchronize_between_processes()

    if is_dist:
        t_sse = running_sqerr.clone()
        t_cnt = running_count.clone().to(torch.float64)
        dist.all_reduce(t_sse)
        dist.all_reduce(t_cnt)
        running_sqerr = t_sse
        running_count = t_cnt.to(torch.int64)

        local_stats = torch.tensor((running_true.n, running_true.mean, running_true.M2),
                                   dtype=torch.float64, device=device)
        stats_list = [torch.zeros_like(local_stats) for _ in range(world_size)]
        dist.all_gather(stats_list, local_stats)
        if rank == 0:
            merged = RunningStat()
            for st in stats_list:
                n_i, mu_i, M2_i = st.tolist()
                other = RunningStat()
                other.n = int(n_i); other.mean = mu_i; other.M2 = M2_i
                merged.merge(other)
            merged_stats = torch.tensor((merged.n, merged.mean, merged.M2),
                                        dtype=torch.float64, device=device)
        else:
            merged_stats = torch.zeros(3, dtype=torch.float64, device=device)
        dist.broadcast(merged_stats, src=0)
        n_all, mu_all, M2_all = merged_stats.tolist()
        running_true.n = int(n_all); running_true.mean = mu_all; running_true.M2 = M2_all

    final_sse = float(running_sqerr.detach().cpu())
    final_n   = int(running_count.detach().cpu())
    mse   = final_sse / max(final_n, 1)
    rmse  = math.sqrt(mse)
    sigma = running_true.std + 1e-8 if running_true.n >= 2 else float('nan')
    norm_rmse = (rmse / sigma) if isinstance(sigma, float) and math.isfinite(sigma) else float('nan')

    if rank == 0:
        print(f"* RMSE = {rmse:.6f}, norm RMSE = {norm_rmse:.6f}")

    out = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    out["rmse"] = rmse
    out["norm_rmse"] = norm_rmse
    return out

    
'''
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ", rank=args.rank)
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('mse', misc.SmoothedValue(window_size=10, fmt='{avg:.6f}'))
    
    header = f'Epoch: [{epoch}]'
    print_freq = 10
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if data_iter_step % accum_iter == 0:
            # Adjust LR schedule
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        eeg = samples[0].to(device, non_blocking=True)
        targets = samples[1].to(device, non_blocking=True).float()

        with torch.amp.autocast('cuda'):
            outputs = model(eeg)
            loss = criterion(outputs, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Logging metrics
        metric_logger.update(loss=loss_value)

        diff = (outputs.detach().float() - targets.float())
        batch_sse = diff.pow(2).sum().item()
        batch_n = diff.numel()
        batch_mse = batch_sse / max(1, batch_n)
        metric_logger.meters['mse'].update(batch_mse, n=batch_n)

        # log lr
        max_lr = max(group["lr"] for group in optimizer.param_groups)
        metric_logger.update(lr=max_lr)

        # Logging to tensorboard / writer
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    global_mse = metric_logger.mse.global_avg if 'mse' in metric_logger.meters else float('nan')
    rmse = math.sqrt(global_mse) if (math.isfinite(global_mse) and global_mse >= 0) else float('nan')

    if args.rank == 0:
        print("Averaged stats:", metric_logger)
        print(f"* RMSE {rmse:.6f}", flush=True)

    out = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    out['rmse'] = rmse
    # We don’t compute norm_rmse here (we do in evaluate)
    return out

@torch.no_grad()
def evaluate(data_loader, model, device):
    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    world_size = dist.get_world_size() if is_dist else 1

    model.eval()
    metric_logger = MetricLogger(delimiter="  ", rank=rank)
    header = "Test:"

    # Running stats
    running_true = RunningStat()  # your existing class
    # Keep accumulators on GPU to avoid host syncs.
    running_sqerr = torch.zeros((), device=device, dtype=torch.float64)
    running_count = torch.zeros((), device=device, dtype=torch.int64)

    for batch in metric_logger.log_every(data_loader, 100, header):
        eeg     = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True)

        # with torch.amp.autocast(device_type='cuda'):
        outputs = model(eeg)
        outputs = _first_tensor(outputs)

        # normalize to (B,) on GPU
        preds = _to_1d_float(outputs, "outputs")
        trues = _to_1d_float(targets, "targets")

        if preds.shape[0] != trues.shape[0]:
            raise ValueError(f"Pred/target batch mismatch: preds={tuple(preds.shape)}, targets={tuple(trues.shape)}")

        # optional mask
        valid_mask = torch.isfinite(trues)
        if not torch.all(valid_mask):
            preds = preds[valid_mask]
            trues = trues[valid_mask]

        # GPU math
        diff      = preds - trues                    # (B,)
        batch_sse = (diff * diff).sum().to(torch.float64)  # scalar on GPU
        batch_n   = torch.tensor(diff.numel(), device=device, dtype=torch.int64)

        running_sqerr += batch_sse
        running_count += batch_n

        # running_true likely expects CPU
        running_true.update(trues.detach().cpu())

        # batch mse (GPU) -> scalar to CPU for logging
        batch_mse = (batch_sse / batch_n.clamp_min(1)).detach()
        metric_logger.update(loss=float(batch_mse.cpu()), n=int(batch_n.cpu()))
        metric_logger.meters["mse"].update(float(batch_mse.cpu()), n=int(batch_n.cpu()))

        if not hasattr(metric_logger, "_dbg_printed"):
            print(f"[debug] eeg={tuple(eeg.shape)}, preds={tuple(preds.shape)}, targets={tuple(trues.shape)}")
            metric_logger._dbg_printed = True

    # sync meters across ranks
    metric_logger.synchronize_between_processes()

    # DDP reductions for accumulators and running_true
    if is_dist:
        # use same device as accumulators
        acc_device = running_sqerr.device
        t_sse = running_sqerr.clone()
        t_cnt = running_count.clone().to(torch.float64)  # reduce as float64
        dist.all_reduce(t_sse)
        dist.all_reduce(t_cnt)
        running_sqerr = t_sse
        running_count = t_cnt.to(torch.int64)

        # merge RunningStat
        local_stats = torch.tensor((running_true.n, running_true.mean, running_true.M2),
                                   dtype=torch.float64, device=acc_device)
        stats_list = [torch.zeros_like(local_stats) for _ in range(world_size)]
        dist.all_gather(stats_list, local_stats)
        # rank 0 merges, then broadcasts
        if rank == 0:
            merged = RunningStat()
            for st in stats_list:
                n_i, mu_i, M2_i = st.tolist()
                other = RunningStat()
                other.n = int(n_i); other.mean = mu_i; other.M2 = M2_i
                merged.merge(other)
            merged_stats = torch.tensor((merged.n, merged.mean, merged.M2),
                                        dtype=torch.float64, device=acc_device)
        else:
            merged_stats = torch.zeros(3, dtype=torch.float64, device=acc_device)
        dist.broadcast(merged_stats, src=0)
        n_all, mu_all, M2_all = merged_stats.tolist()
        running_true.n = int(n_all); running_true.mean = mu_all; running_true.M2 = M2_all

    # Final metrics (move only scalars to CPU)
    final_sse = float(running_sqerr.detach().cpu())
    final_n   = int(running_count.detach().cpu())
    mse   = final_sse / max(final_n, 1)
    rmse  = math.sqrt(mse)
    sigma = running_true.std + 1e-8
    norm_rmse = rmse / sigma

    if rank == 0:
        print(f"* RMSE = {rmse:.6f}, norm RMSE = {norm_rmse:.6f}")

    out = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    out["rmse"] = rmse
    out["norm_rmse"] = norm_rmse
    return out

@torch.no_grad()
def evaluate(data_loader, model, device):
    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    world_size = dist.get_world_size() if is_dist else 1

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ", rank=rank)
    header = 'Test:'

    running_true = RunningStat()
    running_sqerr = 0.0
    running_count = 0

    for batch in metric_logger.log_every(data_loader, 10, header):
        eeg = batch[0].to(device)
        targets = batch[1].float().to(device)

        #with torch.amp.autocast('cuda'):
        outputs = model(eeg)

        # Move to CPU for error and stats
        preds_cpu = outputs.detach().float().cpu()
        trues_cpu = targets.detach().float().cpu()

        # Compute diff, SSE, count
        diff = preds_cpu - trues_cpu
        batch_sse = (diff ** 2).sum().item()
        batch_n = diff.numel()

        running_sqerr += batch_sse
        running_count += batch_n
        running_true.update(trues_cpu)

        batch_mse = batch_sse / batch_n
        metric_logger.update(loss=batch_mse, n=batch_n)
        metric_logger.meters['mse'].update(batch_mse, n=batch_n)

    metric_logger.synchronize_between_processes()

    if is_dist:
        # Use a device that collective supports (likely GPU) for reductions
        collective_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        t_sse = torch.tensor(running_sqerr, dtype=torch.float64, device=collective_device)
        t_cnt = torch.tensor(running_count, dtype=torch.float64, device=collective_device)
        dist.all_reduce(t_sse)
        dist.all_reduce(t_cnt)
        running_sqerr = t_sse.item()
        running_count = int(t_cnt.item())

        # Merge RunningStat across ranks
        # Convert local stats to a tensor on collective_device
        local_stats = (running_true.n, running_true.mean, running_true.M2)
        stats_tensor = torch.tensor(local_stats, dtype=torch.float64, device=collective_device)
        stats_list = [torch.zeros_like(stats_tensor) for _ in range(world_size)]
        dist.all_gather(stats_list, stats_tensor)
        if rank == 0:
            merged = RunningStat()
            for st in stats_list:
                n_i, mu_i, M2_i = st.tolist()
                other = RunningStat()
                other.n = int(n_i)
                other.mean = mu_i
                other.M2 = M2_i
                merged.merge(other)
            merged_stats = torch.tensor((merged.n, merged.mean, merged.M2),
                                        dtype=torch.float64, device=collective_device)
        else:
            merged_stats = torch.zeros(3, dtype=torch.float64, device=collective_device)
        dist.broadcast(merged_stats, src=0)
        n_all, mu_all, M2_all = merged_stats.tolist()
        running_true.n = int(n_all)
        running_true.mean = mu_all
        running_true.M2 = M2_all

    mse = running_sqerr / running_count
    rmse = math.sqrt(mse)
    sigma = running_true.std + 1e-8
    norm_rmse = rmse / sigma

    if rank == 0:
        print(f"* RMSE = {rmse:.6f}, norm RMSE = {norm_rmse:.6f}")

    out = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    out['rmse'] = rmse
    out['norm_rmse'] = norm_rmse
    return out
'''