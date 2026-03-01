"""Cross-modal learning components for tissue-image and liquid-biopsy signals."""

from .config import MultiModalConfig

__all__ = ["MultiModalConfig", "train_from_config"]


def train_from_config(*args, **kwargs):
    from .train import train_from_config as _train_from_config

    return _train_from_config(*args, **kwargs)
