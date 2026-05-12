"""Deterministic-ish seeding for reproducible experiments.

This module exposes :func:`set_seed`, which fixes the seed for Python's
``random`` module, NumPy, and PyTorch (CPU + all CUDA devices). It is the
first thing any experiment script / notebook should call.

Note
----
Full determinism on GPU is famously hard. We do *not* set
``torch.use_deterministic_algorithms(True)`` here because several pyvene /
HuggingFace ops do not have deterministic CUDA implementations and would
just raise. If you need bit-exact reproducibility, set the environment
variable ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` and enable deterministic
algorithms in your launch script.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 0, deterministic_cudnn: bool = True) -> int:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) to ``seed``.

    Parameters
    ----------
    seed:
        Integer seed. Negative values are coerced to ``abs(seed)``.
    deterministic_cudnn:
        If ``True`` (default), set ``torch.backends.cudnn.deterministic = True``
        and ``torch.backends.cudnn.benchmark = False``. This trades a bit of
        speed for substantially more reproducible runs on GPU.

    Returns
    -------
    int
        The seed that was actually applied. Useful when callers want to
        log it.
    """
    seed = int(abs(seed))

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def get_device(prefer: Optional[str] = None) -> torch.device:
    """Return the best available torch device.

    Parameters
    ----------
    prefer:
        If given (``"cpu"`` / ``"cuda"`` / ``"mps"`` / ``"cuda:0"``...), the
        function tries to honor that preference and falls back to CPU if
        the requested device is unavailable.
    """
    if prefer is not None:
        try:
            dev = torch.device(prefer)
        except (RuntimeError, ValueError):
            return torch.device("cpu")
        if dev.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        if dev.type == "mps" and not torch.backends.mps.is_available():
            return torch.device("cpu")
        return dev

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
