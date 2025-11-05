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
import numpy as np
import sys
from typing import Iterable, Optional
import re
import torch

from timm.data import Mixup
from timm.utils import accuracy
from timm.data.mixup import mixup_target

import util.misc as misc
import util.lr_sched as lr_sched
from util.utils import create_optimizer
import os, json
import wandb
from sklearn.metrics import cohen_kappa_score, roc_auc_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize


def wandb_log_stats(prefix: str,
                    epoch: int,
                    total_epoch: int,
                    train_stats: dict,
                    test_whole: dict,
                    individual: dict,
                    args,
                    n_parameters: int):
    """
    Logs to W&B:
      • confusion matrices (whole / own / leave-out / others) once at the final epoch
        according to args.evaluation_scheme
      • a per-epoch Table of (epoch, stage, test_subject, *metrics)
      • flat scalars: train_* and test_whole_* metrics
    """
    scheme = args.evaluation_scheme
    is_final = epoch == total_epoch - 1

    # --- 1) Confusion matrices ---
    if is_final:
        # A) Always log the whole-set CM
        y_true_wh = test_whole.get("labels")
        y_pred_wh = test_whole.get("preds")
        if y_true_wh is not None and y_pred_wh is not None:
            cm_wh = wandb.plot.confusion_matrix(
                preds=y_pred_wh, y_true=y_true_wh, class_names=args.class_names
            )
            wandb.log({f"{prefix}_confmat_whole": cm_wh}, step=epoch)

        # B) population → nothing else
        if scheme == "population":
            pass

        # C) per-subject → own + others
        elif scheme == "per-subject":
            train_subj = args.current_run_sub_of_interested
            # own
            own = individual.get(train_subj, {})
            yt_o, yp_o = own.get("labels"), own.get("preds")
            if yt_o is not None and yp_o is not None:
                cm_o = wandb.plot.confusion_matrix(
                    preds=yp_o, y_true=yt_o, class_names=args.class_names
                )
                wandb.log({f"{prefix}_confmat_{train_subj}": cm_o}, step=epoch)
            # others aggregate
            all_yt, all_yp = [], []
            for subj, stats in individual.items():
                if subj != train_subj:
                    all_yt += stats.get("labels", [])
                    all_yp += stats.get("preds", [])
            if all_yt:
                cm_oth = wandb.plot.confusion_matrix(
                    preds=all_yp, y_true=all_yt, class_names=args.class_names
                )
                wandb.log({f"{prefix}_confmat_others": cm_oth}, step=epoch)

        # D) leave-one-out-finetuning → loo + others
        elif scheme == "leave-one-out-finetuning":
            loo = args.current_run_sub_of_interested
            loo_stats = individual.get(loo, {})
            yt_l, yp_l = loo_stats.get("labels"), loo_stats.get("preds")
            if yt_l is not None and yp_l is not None:
                cm_l = wandb.plot.confusion_matrix(
                    preds=yp_l, y_true=yt_l, class_names=args.class_names
                )
                wandb.log({f"{prefix}_confmat_{loo}": cm_l}, step=epoch)
            all_yt, all_yp = [], []
            for subj, stats in individual.items():
                if subj != loo:
                    all_yt += stats.get("labels", [])
                    all_yp += stats.get("preds", [])
            if all_yt:
                cm_oth = wandb.plot.confusion_matrix(
                    preds=all_yp, y_true=all_yt, class_names=args.class_names
                )
                wandb.log({f"{prefix}_confmat_others": cm_oth}, step=epoch)

    # --- 2) Build dynamic per-subject Table ---
    # drop preds/labels from metrics
    metric_keys = [k for k in test_whole.keys() if k not in ("preds","labels")]
    cols = ["epoch", "stage", "test_subject"] + metric_keys
    table = wandb.Table(columns=cols)

    # whole row
    table.add_data(
        epoch, prefix, "whole",
        *[test_whole[k] for k in metric_keys]
    )
    # per-subject rows
    for subj, stats in individual.items():
        table.add_data(
            epoch, prefix, subj,
            *[stats.get(k) for k in metric_keys]
        )

    # --- 3) Flat train metrics ---
    flat = {f"{prefix}/train_{k}": v for k, v in train_stats.items()}

    # --- 4) Flat whole-test metrics ---
    for k, v in test_whole.items():
        if k not in ("preds", "labels"):
            flat[f"{prefix}/test_whole_{k}"] = v

    # --- 5) Flat individual metrics ---
    # e.g. for each subject and each metric k (except 'preds','labels')
    for subj, stats in individual.items():
        for k, v in stats.items():
            if k in ("preds", "labels"):
                continue
            flat[f"{prefix}/{subj}_{k}"] = v

    # --- 6) Final W&B log ---
    wandb.log({
        **flat,
        f"{prefix}_results": table,
        "epoch": epoch,
        "n_parameters": n_parameters,
    }, step=epoch)
    
    return flat
    
def build_log_stats(prefix, train_stats, test_whole, individual, epoch, n_parameters):
    """
    Merge train/test/individual dicts into one, keyed by prefix.
    """
    stats = {
        **{f"{prefix}/train_{k}": v for k, v in train_stats.items()},
        **{f"{prefix}/test_whole_{k}": v for k, v in test_whole.items()},
        **{f"{prefix}/{subj}_test_{k}": val
           for subj, subdict in individual.items()
           for k, val in subdict.items()},
        f"{prefix}/epoch": epoch,
        "n_parameters": n_parameters
    }
    return stats

def run_phase(model, train_loader, test_loader_whole, test_loaders, prefix, args,
              device, criterion, loss_scaler_class,
              log_writer=None, epochs=None, log_dir=None, mixup_fn=None, wandb_log_freq=5):
    """
    Runs train→eval for `epochs` and logs:
      • flat scalars (train loss/acc, global AUC/Kappa, etc.)
      • one wandb.Table per epoch with (epoch, stage, test_subject, *metrics)
      • confusion matrices (whole + individual) only at the final epoch
    """
    # 0) prepare
    optimizer    = create_optimizer(args, model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    loss_scaler  = loss_scaler_class()
    epochs       = epochs or args.train_epochs

    for epoch in range(epochs):
        # 1) train
        train_stats = train_one_epoch(
            model, criterion, train_loader,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad,
            mixup_fn=mixup_fn,
            log_writer=log_writer, args=args, tag_base=prefix
        )

        # 2) evaluate on whole
        test_whole = evaluate(test_loader_whole, model, device, args)

        # 3) per-subject eval
        individual = {}
        loaders = test_loaders if isinstance(test_loaders, list) else [test_loaders]
        if len(loaders) > 1:
            for loader in loaders:
                subj_name = loader.dataset.subjectName
                individual[subj_name] = evaluate(loader, model, device, args)

        # 4) TensorBoard logging (optional)
        if log_writer:
            tb_base = prefix + "/"
            # train scalars
            for k, v in train_stats.items():
                log_writer.add_scalar(f"{tb_base}train_{k}", v, epoch)
            # whole-set eval scalars
            for k, v in test_whole.items():
                if k not in ("preds", "labels"):
                    log_writer.add_scalar(f"{tb_base}test_whole_{k}", v, epoch)
            # per-subject eval scalars
            for subj, stats in individual.items():
                for k, v in stats.items():
                    if k not in ("preds", "labels"):
                        log_writer.add_scalar(f"{tb_base}{subj}_test_{k}", v, epoch)

        # 5) Wandb logging every 5 epochs (and always on the last one)
        flat = {}
        if epoch % wandb_log_freq==0 or epoch == epochs - 1:
            flat = wandb_log_stats(
                prefix=prefix,
                epoch=epoch,
                total_epoch=epochs,
                train_stats=train_stats,
                test_whole=test_whole,
                individual=individual,
                args=args,
                n_parameters=n_parameters,
            )

        # 9) JSON dump (optional)
        if log_dir and misc.is_main_process():
            fname = os.path.join(log_dir, f"log_{prefix.replace('/', '_')}")
            with open(fname, "a") as f:
                f.write(json.dumps(flat) + "\n")
                

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None, tag_base=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ", rank=args.rank)
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        #print(f"start {data_iter_step} iter", flush =True)
        if len(samples)==3:
            eeg = samples[0]
            label = samples[1]
            sensloc = samples[2] 
        elif len(samples)==2:
            eeg = samples[0]
            label = samples[1]
            sensloc = None
        if eeg.shape[0]==1:
            pass
        else:
            # we use a per iteration (instead of per epoch) lr scheduler
            if data_iter_step % accum_iter == 0:
                lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
            #print(eeg.shape,sensloc.shape,label.shape)
            eeg = eeg.to(device, non_blocking=True)
            targets = label.to(device, non_blocking=True)
            if sensloc is not None:
                sensloc = sensloc.to(device, non_blocking=True)
            #print(args.model, eeg.shape, sensloc.shape)
            if mixup_fn is not None and len(eeg) % 2 == 0:
                eeg, targets = mixup_fn(eeg, targets)
            else:
                # no mix → λ = 1.0, but still apply label smoothing
                lam = 1.0
                targets = mixup_target(
                    targets,
                    mixup_fn.num_classes,
                    lam,
                    mixup_fn.label_smoothing
                )

            if sensloc is not None:
                outputs = model(eeg,sensloc)
            else:
                outputs = model(eeg)
            loss = criterion(outputs, targets)
            hard_from_soft = targets.argmax(dim=1)        # shape (B,), dtype=int64
            acc1, acc2 = accuracy(outputs, hard_from_soft, topk=(1, 2))
            batch_size = eeg.shape[0]
            loss_value = loss.item()
            #print("loss=",loss_value, flush=True)
            if not math.isfinite(loss_value):
                print(eeg.shape, eeg, targets, outputs)
                print("Loss is {}, stopping training".format(loss_value), flush=True)
                sys.exit(1)

            loss /= accum_iter
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=False,
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            #print("loss scaler update", flush=True)
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

            if args.distributed:
                torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc2'].update(acc2.item(), n=batch_size)
            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(lr=max_lr)
            #print("metri logger update", flush=True)
            if args.distributed:
                loss_value_reduce = misc.all_reduce_mean(loss_value)
            else:
                loss_value_reduce = loss_value
            if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar(tag_base +'/step_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar(tag_base +'/lr', max_lr, epoch_1000x)
                #print("log writer update", flush=True)
            #print(f"end {data_iter_step} iter", flush =True)

    # gather the stats from all processes
    if args.distributed:
        metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger, flush=True)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(data_loader, model, device, args=None):
    """
    Evaluate `model` on `data_loader`, returning a dict of:
      { 'loss', 'acc1', 'acc2', 'kappa', 'auc', 'balanced_acc', 'preds', 'labels' }
    """
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    
    # set up metric logger
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('kappa',        misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('auc',          misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('balanced_acc', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    all_targets, all_probs = [], []
    for samples in metric_logger.log_every(data_loader, 100, 'Test:'):
        eeg, label = samples[:2]
        sensloc    = samples[2] if len(samples) == 3 else None
        eeg, targets = eeg.to(device), label.to(device)
        if sensloc is not None:
            sensloc = sensloc.to(device)

        outputs = model(eeg, sensloc) if sensloc is not None else model(eeg)
        loss = criterion(outputs, targets)
        
        batch_size = eeg.size(0)
        probs = torch.softmax(outputs, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

        acc1, acc2 = accuracy(outputs, targets, topk=(1, 2))
        metric_logger.update(loss=loss.item())
        metric_logger.update(acc1=acc1.item(), n=batch_size)
        metric_logger.update(acc2=acc2.item(), n=batch_size)

    # stack for final metrics
    all_probs   = np.vstack(all_probs).astype(np.float64)
    all_probs  /= all_probs.sum(axis=1, keepdims=True)
    all_targets = np.concatenate(all_targets)
    preds       = all_probs.argmax(axis=1)

    present_classes = np.unique(all_targets)
    # --- EARLY EXIT WHEN ONLY ONE CLASS ---
    if len(present_classes) < 2:
        #print("⚠ Only one class in y_true; setting kappa, auc and balanced_acc to NaN")
        kappa = float('nan')
        auc   = float('nan')
        bal_acc = float('nan')
        metric_logger.update(kappa=kappa, auc=auc, balanced_acc=bal_acc)
        return {
            **{k: m.global_avg for k, m in metric_logger.meters.items()},
            "preds": preds.tolist(),
            "labels": all_targets.tolist()
        }

    # otherwise do the regular multi‐class / binary AUC
    probs_present = all_probs[:, present_classes]
    if len(present_classes) == 2:
        y_true_bin = label_binarize(all_targets, classes=present_classes)
    else:
        y_true_bin = label_binarize(all_targets, classes=present_classes)

    kappa = cohen_kappa_score(all_targets, preds)
    try:
        #print(y_true_bin.shape, probs_present.shape)
        #print(y_true_bin, probs_present)
        if probs_present.shape[1] < 2:
            raise ValueError("Cannot compute AUC: less than 2 class probabilities present.")
        if len(present_classes) == 2:
            # Determine the class considered "positive"
            pos_class = present_classes[1]  # second class
            auc = roc_auc_score((all_targets == pos_class).astype(int), all_probs[:, pos_class])
        else:
            auc = roc_auc_score(y_true_bin, probs_present, multi_class='ovr')
    except Exception as e:
        print("❌ AUC computation failed:", e)
        auc = float('nan')

    bal_acc = balanced_accuracy_score(all_targets, preds)

    metric_logger.update(kappa=kappa, auc=auc, balanced_acc=bal_acc)
    return {
        **{k: m.global_avg for k, m in metric_logger.meters.items()},
        "preds": preds.tolist(),
        "labels": all_targets.tolist()
    }