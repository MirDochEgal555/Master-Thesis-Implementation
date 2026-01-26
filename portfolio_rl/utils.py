# portfolio_rl/utils.py
from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set seeds for python / numpy / torch.
    deterministic=True can reduce speed; use it only for debugging.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Note: deterministic algorithms can be slower and sometimes raise errors
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def set_cpu_threads(n: int = 1) -> None:
    """
    Prevent CPU oversubscription (very common on Intel laptops + PyTorch).
    Often improves speed dramatically.
    Call once at program start.
    """
    n = int(max(1, n))
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)

    torch.set_num_threads(n)
    torch.set_num_interop_threads(n)


def get_device(device: str = "cpu") -> torch.device:
    """
    device="cpu" or "cuda".
    """
    device = device.lower().strip()
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cpu")


@contextmanager
def timed(name: str, enabled: bool = True):
    """
    Small timing context manager.
    Usage:
        with timed("train_step"):
            ...
    """
    if not enabled:
        yield
        return
    t0 = time.perf_counter()
    yield
    dt = (time.perf_counter() - t0) * 1000.0
    print(f"[timer] {name}: {dt:.2f} ms")


@dataclass
class EWMA:
    """
    Exponential moving average tracker for scalars.
    """
    beta: float = 0.9
    value: Optional[float] = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = float(x)
        else:
            self.value = self.beta * self.value + (1.0 - self.beta) * float(x)
        return self.value
