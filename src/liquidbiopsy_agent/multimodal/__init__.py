"""Cross-modal learning components for tissue-image and liquid-biopsy signals."""

from .config import MultiModalConfig

__all__ = [
    "MultiModalConfig",
    "train_from_config",
    "encode_bed_folder_to_embeddings",
    "build_dna_foundation_encoder",
    "list_supported_dna_foundation_model_keys",
]

_SUPPORTED_DNA_FOUNDATION_MODEL_KEYS = (
    "ntv2",
    "dnabert2",
    "hyenadna",
    "caduceus",
    "epibert",
    "epcot",
    "enformer",
)


def train_from_config(*args, **kwargs):
    from .train import train_from_config as _train_from_config

    return _train_from_config(*args, **kwargs)


def encode_bed_folder_to_embeddings(*args, **kwargs):
    from .bed_embedding import encode_bed_folder_to_embeddings as _encode_bed_folder_to_embeddings

    return _encode_bed_folder_to_embeddings(*args, **kwargs)


def build_dna_foundation_encoder(*args, **kwargs):
    from .dna_foundation_encoders import build_dna_foundation_encoder as _build_dna_foundation_encoder

    return _build_dna_foundation_encoder(*args, **kwargs)


def list_supported_dna_foundation_model_keys():
    return _SUPPORTED_DNA_FOUNDATION_MODEL_KEYS
