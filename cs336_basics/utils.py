from pathlib import Path
import os
from typing import BinaryIO, IO

import torch

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    model = _unwrap_model(model)
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_class': model.__class__.__name__,
        'model_config': getattr(model, 'config', {}),
    }
    # checkpoint.update(kwargs)
    if isinstance(out, (str, os.PathLike)):
        if not isinstance(out, (str)):
            out = str(out)
        # 确保父目录存在
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # 使用临时文件进行原子写入
            temp_path = out_path.with_suffix('.tmp')
            torch.save(checkpoint, temp_path)
            # 原子重命名（POSIX系统上保证原子性）
            temp_path.rename(out_path)
            print(f"Checkpoint saved successfully to {out_path}")
        except Exception as e:
            # 清理临时文件
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"Failed to save checkpoint: {e}")
    else:
        try:
            torch.save(checkpoint, out)
            if hasattr(out, 'flush'):
                out.flush()
            print(f"Checkpoint saved to file object")
        except Exception as e:
            raise RuntimeError(f"Failed to save checkpoint to file object: {e}")


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint = torch.load(src) #, map_location=device, **kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    # torch.compile wraps the original model and prefixes state_dict keys with _orig_mod.
    # vLLM expects the original key names, so always sync/save with the unwrapped module.
    return model._orig_mod if hasattr(model, "_orig_mod") else model

from datetime import datetime
import os
import time
def get_log_and_output_dir(output_dir, model_name):
    now = datetime.now()
    current_time = now.strftime("%m-%d-%H-%M-%S") + "-" + model_name

    if not os.path.exists(f"{output_dir}/{current_time}"):
        os.makedirs(f"{output_dir}/{current_time}")
    output_dir = f"{output_dir}/{current_time}"
    log = Log(f"{output_dir}/logs.txt")
    return log, output_dir

class Log:
    def __init__(self, log_path, time_key=True):
        self.path = log_path
        if time_key:
            self.path = self.path.replace(
                ".", "{}.".format(time.strftime("_%Y%m%d%H%M%S", time.localtime(time.time())))
            )
        print(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
            file=open(self.path, "a+"),
        )
        print("log path:", self.path)
        print("****************开始记录*********************", file=open(self.path, "a+"))

    def __call__(self, *content):
        t1 = time.strftime("%H:%M:%S", time.localtime(time.time()))
        print(*content)
        print(t1, content, file=open(self.path, "a+"))

    def clean(self):
        print(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
            file=open(self.path, "w"),
        )
        print("****************开始记录*********************", file=open(self.path, "a+"))


