# Adapted from the MAE implementation by xxx
# --------------------------------------------------------
# References:
# MAE:  https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import math
import sys
import torch
from typing import Iterable
import util.misc as misc
import util.lr_sched as lr_sched

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    log_writer=None,
                    args=None):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ", rank=args.rank)
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 10
    accum_iter = args.accum_iter

    # Zero gradients at start of first accumulation block
    optimizer.zero_grad()

    if log_writer is not None and args.rank == 0:
        print(f'log_dir: {log_writer.log_dir}')

    for i, g in enumerate(optimizer.param_groups):
        print(f"group {i}: lr={g['lr']}, lr_scale={g.get('lr_scale',1)}", flush=True)

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # 1) Per-iteration LR schedule
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, (data_iter_step + 1) / len(data_loader) + epoch, args)
            current_lr = optimizer.param_groups[0]['lr']
            metric_logger.update(lr=current_lr)

        # 2) Fetch data to GPU
        eeg, sensloc = samples
        eeg = eeg.to(device, non_blocking=True)
        sensloc = sensloc.to(device, non_blocking=True)

        # 3) Forward pass under AMP
        with torch.amp.autocast(device_type="cuda"):
            loss, _, _ = model(eeg, sensloc, mask_ratio=args.mask_ratio)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            if args.rank == 0:
                print(f"Loss is {loss_value}, stopping training", flush=True)
            sys.exit(1)

        # 4) Gradient accumulation logic
        loss = loss / accum_iter
        is_last_step = ((data_iter_step + 1) % accum_iter == 0)

        if args.distributed:
            if not is_last_step:
                with model.no_sync():
                    loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=False)
            else:
                loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=True)
                optimizer.zero_grad()
        else:
            # Non-DDP path must also skip the step until the last micro-batch
            loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=is_last_step)
            if is_last_step:
                optimizer.zero_grad()

        # 5) Update logging meters
        metric_logger.update(loss=loss_value)

    # 6) End of epoch: gather & log
    if args.distributed:
        metric_logger.synchronize_between_processes()

    if args.rank == 0:
        print(f"Averaged stats: {metric_logger}", flush=True)
        if log_writer is not None:
            # Write a single scalar per epoch instead of per batch
            global_loss = metric_logger.meters['loss'].global_avg
            current_lr = optimizer.param_groups[0]["lr"]
            log_writer.add_scalar('train_loss', global_loss, epoch)
            log_writer.add_scalar('lr', current_lr, epoch)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_one_epoch(model: torch.nn.Module,
                   data_loader: Iterable,
                   device: torch.device,
                   epoch: int,
                   log_writer=None,
                   args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ", rank=args.rank)
    header = f'Epoch: [{epoch}]'
    print_freq = 1000

    if log_writer is not None and args.rank == 0:
        print(f'log_dir: {log_writer.log_dir}')

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        eeg, sensloc = samples
        eeg = eeg.to(device, non_blocking=True)
        # (Optional) If you don’t really need to clamp here, remove the line below
        # eeg = eeg.clamp(-500, 500)
        sensloc = sensloc.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda"):
            loss, _, _ = model(eeg, sensloc, mask_ratio=args.mask_ratio)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            if args.rank == 0:
                print(f"Loss is {loss_value}, stopping training", flush=True)
            sys.exit(1)

        metric_logger.update(loss=loss_value)

    # One final sync across GPUs
    if args.distributed:
        metric_logger.synchronize_between_processes()

    global_test_loss = metric_logger.meters['loss'].global_avg
    if log_writer is not None and args.rank == 0:
        log_writer.add_scalar('test_loss', global_test_loss, epoch)

    if args.rank == 0:
        print(f"Averaged stats: {metric_logger}", flush=True)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
