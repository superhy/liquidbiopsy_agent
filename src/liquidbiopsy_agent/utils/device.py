from __future__ import annotations

import platform

import torch


def resolve_torch_device(device: str = "auto") -> torch.device:
    """Resolve torch device with a cross-platform auto policy.

    Auto policy:
    - Windows/Linux: use the first available CUDA device when available.
    - macOS (Apple Silicon): use MPS when available.
    - Fallback: CPU.
    """
    requested = str(device).strip()
    if requested and requested.lower() != "auto":
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda:0")

    if platform.system() == "Darwin":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and bool(mps_backend.is_available()):
            return torch.device("mps")

    return torch.device("cpu")


def should_pin_memory(device: torch.device | str) -> bool:
    """Pin memory only when training/inference is on CUDA."""
    return torch.device(device).type == "cuda"
