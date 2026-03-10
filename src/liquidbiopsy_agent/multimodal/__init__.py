"""Cross-modal learning components for tissue-image and liquid-biopsy signals."""

from .config import MultiModalConfig

__all__ = [
    "MultiModalConfig",
    "train_from_config",
    "encode_bed_folder_to_embeddings",
    "encode_tcga_brca_wsi",
    "run_trident_uni_v2_patch_encoding",
    "run_tangle_slide_embedding",
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


def run_trident_uni_v2_patch_encoding(*args, **kwargs):
    from .wsi_encoding import run_trident_uni_v2_patch_encoding as _run_trident_uni_v2_patch_encoding

    return _run_trident_uni_v2_patch_encoding(*args, **kwargs)


def run_tangle_slide_embedding(*args, **kwargs):
    from .wsi_encoding import run_tangle_slide_embedding as _run_tangle_slide_embedding

    return _run_tangle_slide_embedding(*args, **kwargs)


def encode_tcga_brca_wsi(*args, **kwargs):
    from .wsi_encoding import encode_tcga_brca_wsi as _encode_tcga_brca_wsi

    return _encode_tcga_brca_wsi(*args, **kwargs)


def build_dna_foundation_encoder(*args, **kwargs):
    from .dna_foundation_encoders import build_dna_foundation_encoder as _build_dna_foundation_encoder

    return _build_dna_foundation_encoder(*args, **kwargs)


def list_supported_dna_foundation_model_keys():
    return _SUPPORTED_DNA_FOUNDATION_MODEL_KEYS
