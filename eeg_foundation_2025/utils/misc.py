import builtins
import datetime
import os
import glob
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.distributed as dist
from torch import inf
import numpy as np


def print_size(net):
    """
    Print the number of parameters of a network
    """
    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True)
        

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return

        # Always use CPU tensor or guard against CUDA‐only
        dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device=dev)
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        if len(self.deque) == 0:
            return 0.0
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        if len(self.deque) == 0:
            return 0.0
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        if self.count == 0:
            return 0.0
        else:
            return self.total / self.count

    @property
    def max(self):
        return max(self.deque) if len(self.deque) > 0 else 0.0

    @property
    def value(self):
        return self.deque[-1] if len(self.deque) > 0 else 0.0

    def __str__(self):
        m = self.median
        a = self.avg
        g = self.global_avg
        M = self.max
        v = self.value
        return self.fmt.format(
            median=m,
            avg=a,
            global_avg=g,
            max=M,
            value=v)


class MetricLogger(object):
    def __init__(self, delimiter="\t", rank=-1):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.rank = rank

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    if self.rank == 0:
                        print(log_msg.format(
                            i, len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB), flush=True)
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if self.rank == 0:
            print('{} Total time: {} ({:.4f} s / it)'.format(
                header, total_time_str, total_time / len(iterable)), flush=True)
        

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK']); world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    elif 'SLURM_PROCID' in os.environ and 'SLURM_NTASKS' in os.environ:
        rank = int(os.environ['SLURM_PROCID']); world_size = int(os.environ['SLURM_NTASKS'])
        # best effort local_rank; many Slurm setups export LOCAL_RANK as well
        local_rank = int(os.environ.get('LOCAL_RANK', rank % torch.cuda.device_count()))
    else:
        print('Not using distributed mode (single process).', flush=True)
        return

    assert torch.cuda.is_available(), "CUDA required for NCCL backend."
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=world_size, rank=rank)
    dist.barrier()
    
'''
class NativeScalerWithGradNormCount:
    
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler("cuda")

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        try:
            self._scaler.scale(loss).backward(create_graph=create_graph)
        except RuntimeError as e:
            print(f"[Rank {dist.get_rank()}] ERROR IN BACKWARD: {e}", flush=True)
            raise
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
'''
class NativeScalerWithGradNormCount:
    """
    AMP scaler that auto-detects the available API:
      - PyTorch >= 2.3: torch.amp.GradScaler(device="cuda"), torch.amp.autocast(...)
      - PyTorch <= 2.2: torch.cuda.amp.GradScaler(), torch.cuda.amp.autocast(...)
    """
    state_dict_key = "amp_scaler"

    def __init__(self, enabled: bool = True, use_bfloat16: bool = False):
        import torch
        self.enabled = enabled
        self.dtype = torch.bfloat16 if use_bfloat16 else torch.float16

        has_new_amp = hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler")
        if has_new_amp:
            # New API (2.3+)
            self._backend = "torch.amp"
            self._autocast_ctx = lambda: torch.amp.autocast(
                device_type="cuda", dtype=self.dtype, enabled=self.enabled
            )
            # device=... kw only exists in new API; pass it here
            self._scaler = torch.amp.GradScaler(device="cuda", enabled=self.enabled)
        else:
            # Old API (<= 2.2)
            self._backend = "torch.cuda.amp"
            from torch.cuda.amp import GradScaler as _OldGradScaler, autocast as _old_autocast
            self._autocast_ctx = lambda: _old_autocast(dtype=self.dtype, enabled=self.enabled)
            self._scaler = _OldGradScaler(enabled=self.enabled)

    # convenience: use `with scaler.autocast(): ...`
    def autocast(self):
        return self._autocast_ctx()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None,
                 create_graph=False, update_grad=True):
        if not self.enabled:
            # no AMP: standard backward/step
            loss.backward(create_graph=create_graph)
            norm = None
            if update_grad:
                if clip_grad is not None:
                    assert parameters is not None
                    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
                optimizer.step()
            return norm

        try:
            self._scaler.scale(loss).backward(create_graph=create_graph)
        except RuntimeError as e:
            # keep your nice rank-aware error message
            r = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            print(f"[Rank {r}] ERROR IN BACKWARD: {e}", flush=True)
            raise

        norm = None
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                if parameters is not None:
                    self._scaler.unscale_(optimizer)
                    norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    """
    Robust checkpoint saver for DDP + network FS:
      - rank-0 only writes (no barriers inside)
      - write to node-local tmp, copy to dest tmp, atomic rename to final path
      - legacy serializer (_use_new_zipfile_serialization=False)
      - optional DeepSpeed branch if loss_scaler is None
    """
    import os
    import tempfile
    import shutil
    from pathlib import Path
    import torch
    import torch.distributed as dist

    # --- helpers ---
    def _is_dist():
        return dist.is_available() and dist.is_initialized()

    def _is_main():
        return (not _is_dist()) or dist.get_rank() == 0

    def _atomic_torch_save(obj, final_path: Path):
        """
        Write on node-local TMPDIR -> copy to a temp in destination dir -> atomic rename.
        Avoids EXDEV while keeping final step atomic on the target filesystem.
        """
        final_path = Path(final_path)
        dest_dir = final_path.parent
        dest_dir.mkdir(parents=True, exist_ok=True)

        # 1) Write to node-local temp (fast)
        node_tmpdir = os.environ.get("TMPDIR") or os.environ.get("TMP") or "/tmp"
        fd_local, local_tmp = tempfile.mkstemp(dir=node_tmpdir, prefix=".ckpt_local_")
        os.close(fd_local)
        try:
            # Legacy serializer avoids zip writer quirks on some FS
            torch.save(obj, local_tmp, _use_new_zipfile_serialization=False)

            # 2) Copy to a temp file inside the destination directory (same FS as final)
            fd_dest, dest_tmp = tempfile.mkstemp(dir=str(dest_dir), prefix=".ckpt_dest_")
            os.close(fd_dest)
            try:
                shutil.copyfile(local_tmp, dest_tmp)
                # Ensure bytes are on disk before rename
                try:
                    with open(dest_tmp, "rb") as f:
                        os.fsync(f.fileno())
                except Exception:
                    pass
                # 3) Atomic rename within same filesystem
                os.replace(dest_tmp, str(final_path))
            except Exception:
                # best-effort cleanup of dest tmp
                try:
                    if os.path.exists(dest_tmp):
                        os.remove(dest_tmp)
                except Exception:
                    pass
                raise
        finally:
            # best-effort cleanup of local tmp
            try:
                if os.path.exists(local_tmp):
                    os.remove(local_tmp)
            except Exception:
                pass

    # ---- build final path ----
    output_dir = Path(getattr(args, "output_dir", "."))
    epoch_name = str(epoch)
    checkpoint_path = output_dir / f"checkpoint-{epoch_name}.pth"

    # ---- DeepSpeed-style branch (no scaler provided) ----
    # (No barriers here; only rank-0 invokes if present)
    if loss_scaler is None and hasattr(model, "save_checkpoint"):
        if _is_main():
            client_state = {"epoch": epoch}
            model.save_checkpoint(
                save_dir=str(output_dir),
                tag=f"checkpoint-{epoch_name}",
                client_state=client_state
            )
        return

    # ---- Standard PyTorch checkpoint ----
    # Keep args lightweight to avoid serialization issues
    safe_args = {}
    try:
        for k, v in vars(args).items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                safe_args[k] = v
    except Exception:
        pass

    to_save = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "scaler": loss_scaler.state_dict() if loss_scaler is not None else None,
        "args": safe_args,
    }

    if _is_main():
        _atomic_torch_save(to_save, checkpoint_path)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    resume_path = None
    if args.resume == "none":
        return

    if args.resume:
        if args.resume == "latest":
            pattern = os.path.join(args.output_dir, "checkpoint-*.pth")
            ckpts = glob.glob(pattern)
            if not ckpts:
                if args.rank == 0:
                    print(f"No checkpoints found in {args.output_dir}")
                return

            def parse_epoch(path):
                fname = os.path.basename(path)
                try:
                    return int(fname.split("-")[1].split(".pth")[0])
                except:
                    return -1

            resume_path = max(ckpts, key=parse_epoch)
            if args.rank == 0:
                print(f"Resuming from latest checkpoint: {resume_path}")

        elif args.resume.startswith("https://") or args.resume.startswith("http://"):
            resume_path = args.resume
            if args.rank == 0:
                print(f"Downloading checkpoint from URL: {resume_path}")
        else:
            resume_path = args.resume
            if args.rank == 0:
                print(f"Resuming from the given checkpoint path: {resume_path}")
    else:
        return

    if resume_path.startswith("http"):
        checkpoint = torch.hub.load_state_dict_from_url(
            resume_path, map_location="cpu", check_hash=True)
    else:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)

    model_without_ddp.load_state_dict(checkpoint["model"])
    if args.rank == 0:
        print(f"Loaded model weights from {resume_path}")

    if "optimizer" in checkpoint and "epoch" in checkpoint and not (hasattr(args, "eval") and args.eval):
        optimizer.load_state_dict(checkpoint["optimizer"])
        args.start_epoch = checkpoint["epoch"] + 1
        if "scaler" in checkpoint and loss_scaler is not None:
            loss_scaler.load_state_dict(checkpoint["scaler"])
        if args.rank == 0:
            print(f"Restored optimizer, scaler, and will start at epoch {args.start_epoch}")


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x, device='cuda', dtype=torch.float32)
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x