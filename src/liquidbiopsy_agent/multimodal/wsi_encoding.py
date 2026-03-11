from __future__ import annotations

import importlib
import json
import os
import pickle
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from liquidbiopsy_agent.utils.device import resolve_torch_device
from liquidbiopsy_agent.utils.storage import get_models_root, resolve_data_path

TRIDENT_REPO_URL = "https://github.com/mahmoodlab/TRIDENT.git"
TANGLE_REPO_URL = "https://github.com/mahmoodlab/TANGLE.git"
UNI_V2_REPO_ID = "MahmoodLab/UNI2-h"
UNI_V2_CKPT_FILENAME = "pytorch_model.bin"
TANGLE_PRETRAINED_DRIVE_URL = "https://drive.google.com/drive/folders/1IKEuRULUz-Uvb8ZL8vvYw0Z49aD_Qp_4"


def _run_command(cmd: Sequence[str], *, cwd: Path | None = None) -> None:
    env = os.environ.copy()
    # Reduce AppleDouble metadata files creation on mounted volumes.
    env.setdefault("COPYFILE_DISABLE", "1")
    subprocess.run(list(cmd), check=True, cwd=str(cwd) if cwd is not None else None, env=env)


def _cleanup_appledouble_files(root_dir: Path) -> None:
    if not root_dir.exists():
        return
    for p in root_dir.rglob("._*"):
        if p.is_file():
            try:
                p.unlink()
            except OSError:
                pass


def _resolve_models_root(models_root: str | Path | None) -> Path:
    if models_root is None:
        root = get_models_root(must_exist=False)
    else:
        root = resolve_data_path(models_root, path_kind="models root", must_exist=False)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_repo_dir(path_value: str | Path | None, default_dir: Path) -> Path:
    if path_value is None:
        target = default_dir
    else:
        target = Path(path_value).expanduser().resolve()
    return target


def _ensure_git_repo(repo_dir: Path, repo_url: str, *, update: bool = False) -> Path:
    if (repo_dir / ".git").exists():
        _cleanup_appledouble_files(repo_dir)
        if update:
            _run_command(["git", "-C", str(repo_dir), "fetch", "--all", "--tags", "--prune"])
            _run_command(["git", "-C", str(repo_dir), "pull", "--ff-only"])
        _cleanup_appledouble_files(repo_dir / ".git")
        return repo_dir

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    _run_command(["git", "clone", "--depth", "1", repo_url, str(repo_dir)])
    _cleanup_appledouble_files(repo_dir / ".git")
    return repo_dir


def _prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def ensure_wsi_repositories(
    *,
    models_root: str | Path | None = None,
    trident_repo_dir: str | Path | None = None,
    tangle_repo_dir: str | Path | None = None,
    update_repos: bool = False,
    require_trident: bool = True,
    require_tangle: bool = True,
) -> dict[str, Path]:
    root = _resolve_models_root(models_root)
    third_party_root = root / "third_party"

    out: dict[str, Path] = {
        "models_root": root,
    }
    if require_trident:
        trident_dir = _resolve_repo_dir(trident_repo_dir, third_party_root / "TRIDENT")
        trident_dir = _ensure_git_repo(trident_dir, TRIDENT_REPO_URL, update=update_repos)
        out["trident_repo_dir"] = trident_dir
    if require_tangle:
        tangle_dir = _resolve_repo_dir(tangle_repo_dir, third_party_root / "TANGLE")
        tangle_dir = _ensure_git_repo(tangle_dir, TANGLE_REPO_URL, update=update_repos)
        out["tangle_repo_dir"] = tangle_dir
    return out


def ensure_uni_v2_checkpoint(
    *,
    models_root: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    allow_download: bool = True,
    hf_token: str | None = None,
) -> Path:
    if checkpoint_path is not None:
        ckpt = Path(checkpoint_path).expanduser().resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"UNI-V2 checkpoint not found: {ckpt}")
        return ckpt

    root = _resolve_models_root(models_root)
    target_dir = root / "uni_v2"
    target_dir.mkdir(parents=True, exist_ok=True)
    ckpt = target_dir / UNI_V2_CKPT_FILENAME
    if ckpt.exists():
        return ckpt

    if not allow_download:
        raise FileNotFoundError(
            f"UNI-V2 checkpoint missing at {ckpt}. Enable download or pass --uni_v2_ckpt_path."
        )

    token = hf_token or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=UNI_V2_REPO_ID,
            filename=UNI_V2_CKPT_FILENAME,
            local_dir=str(target_dir),
            token=token,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Failed to download UNI-V2 checkpoint from Hugging Face. "
            "Ensure you accepted model access terms and configured a valid token if required."
        ) from exc

    return Path(downloaded).expanduser().resolve()


def _discover_tangle_checkpoints(root_dir: Path) -> list[Path]:
    if not root_dir.exists():
        return []
    candidates: list[Path] = []
    for model_pt in root_dir.rglob("model.pt"):
        cfg = model_pt.parent / "config.json"
        if cfg.exists():
            candidates.append(model_pt.parent.resolve())
    return sorted(set(candidates))


def _select_tangle_checkpoint(
    candidates: Sequence[Path],
    *,
    preferred_keyword: str | None,
) -> Path:
    if not candidates:
        raise FileNotFoundError("No TANGLE checkpoint directories with model.pt + config.json were found.")

    if preferred_keyword:
        needle = preferred_keyword.lower()
        filtered = [p for p in candidates if needle in str(p).lower()]
        if filtered:
            return sorted(filtered)[0]

    brca_like = [p for p in candidates if "brca" in str(p).lower()]
    if brca_like:
        return sorted(brca_like)[0]

    return sorted(candidates)[0]


def ensure_tangle_checkpoint(
    *,
    models_root: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    pretrained_root_dir: str | Path | None = None,
    allow_download: bool = True,
    drive_folder_url: str = TANGLE_PRETRAINED_DRIVE_URL,
    preferred_keyword: str | None = "tangle_brca",
) -> Path:
    if checkpoint_dir is not None:
        target = Path(checkpoint_dir).expanduser().resolve()
        if not (target / "model.pt").exists() or not (target / "config.json").exists():
            raise FileNotFoundError(
                f"Invalid TANGLE checkpoint dir: {target}. Expected model.pt and config.json."
            )
        return target

    root = _resolve_models_root(models_root)
    if pretrained_root_dir is None:
        pretrained_root = root / "tangle_pretrained"
    else:
        pretrained_root = Path(pretrained_root_dir).expanduser().resolve()
    pretrained_root.mkdir(parents=True, exist_ok=True)

    candidates = _discover_tangle_checkpoints(pretrained_root)
    if not candidates and allow_download:
        try:
            import gdown

            gdown.download_folder(url=drive_folder_url, output=str(pretrained_root), quiet=False, remaining_ok=True)
        except Exception:
            # Some folders may fail while enough checkpoints are already downloaded.
            pass
        candidates = _discover_tangle_checkpoints(pretrained_root)
        if not candidates:
            raise RuntimeError(
                "Failed to download TANGLE pretrained checkpoints from Google Drive. "
                "Please pass --tangle_checkpoint_dir manually."
            )

    return _select_tangle_checkpoint(candidates, preferred_keyword=preferred_keyword)


def _import_trident_modules(trident_repo_dir: Path):
    _prepend_sys_path(trident_repo_dir)
    trident = importlib.import_module("trident")
    seg_load = importlib.import_module("trident.segmentation_models.load")
    patch_load = importlib.import_module("trident.patch_encoder_models.load")
    return trident, seg_load, patch_load


def _to_data_path(path_value: str | Path, *, path_kind: str, must_exist: bool) -> Path:
    return resolve_data_path(path_value, path_kind=path_kind, must_exist=must_exist)


def run_trident_uni_v2_patch_encoding(
    *,
    slides_dir: str | Path,
    job_dir: str | Path,
    device: str = "auto",
    models_root: str | Path | None = None,
    trident_repo_dir: str | Path | None = None,
    uni_v2_ckpt_path: str | Path | None = None,
    hf_token: str | None = None,
    allow_model_download: bool = True,
    reader_type: str = "openslide",
    segmenter: str = "hest",
    seg_conf_thresh: float = 0.5,
    remove_holes: bool = False,
    remove_artifacts: bool = True,
    remove_penmarks: bool = False,
    mag: float = 20.0,
    patch_size: int = 256,
    overlap: int = 0,
    min_tissue_proportion: float = 0.0,
    seg_batch_size: int = 16,
    feat_batch_size: int = 256,
    max_workers: int | None = None,
    skip_errors: bool = True,
    search_nested: bool = False,
    custom_list_of_wsis: str | Path | None = None,
) -> dict[str, Any]:
    resolved_device = resolve_torch_device(device)
    resolved_device_str = str(resolved_device)
    artifact_requested = bool(remove_artifacts or remove_penmarks)
    artifact_effective = artifact_requested

    # TRIDENT upstream currently routes artifact-removal segmentation through a CUDA default path.
    # On non-CUDA environments (MPS/CPU), force-disable this branch to prevent runtime failures.
    if artifact_requested and resolved_device.type != "cuda":
        warnings.warn(
            "TRIDENT artifact removal is auto-disabled on non-CUDA devices "
            f"(resolved device: {resolved_device_str}).",
            stacklevel=2,
        )
        remove_artifacts = False
        remove_penmarks = False
        artifact_effective = False

    models_root_path = _resolve_models_root(models_root)
    trident_home = models_root_path / "trident_cache"
    os.environ.setdefault("TRIDENT_HOME", str(trident_home))

    slides_path = _to_data_path(slides_dir, path_kind="WSI slides directory", must_exist=True)
    if not slides_path.is_dir():
        raise NotADirectoryError(f"WSI slides directory is not a folder: {slides_path}")
    _cleanup_appledouble_files(slides_path)

    job_path = _to_data_path(job_dir, path_kind="WSI job directory", must_exist=False)
    job_path.mkdir(parents=True, exist_ok=True)
    _cleanup_appledouble_files(job_path)

    repos = ensure_wsi_repositories(
        models_root=models_root_path,
        trident_repo_dir=trident_repo_dir,
        tangle_repo_dir=None,
        update_repos=False,
        require_trident=True,
        require_tangle=False,
    )
    trident_repo = repos["trident_repo_dir"]

    uni_ckpt = ensure_uni_v2_checkpoint(
        models_root=models_root_path,
        checkpoint_path=uni_v2_ckpt_path,
        allow_download=allow_model_download,
        hf_token=hf_token,
    )

    custom_list_path = None
    if custom_list_of_wsis is not None:
        custom_list_path = _to_data_path(
            custom_list_of_wsis,
            path_kind="custom WSI list csv",
            must_exist=True,
        )

    trident_mod, seg_load, patch_load = _import_trident_modules(trident_repo)
    Processor = trident_mod.Processor
    segmentation_model_factory = seg_load.segmentation_model_factory
    encoder_factory = patch_load.encoder_factory

    processor = Processor(
        job_dir=str(job_path),
        wsi_source=str(slides_path),
        skip_errors=skip_errors,
        custom_list_of_wsis=str(custom_list_path) if custom_list_path is not None else None,
        max_workers=max_workers,
        reader_type=reader_type,
        search_nested=search_nested,
    )

    seg_model = segmentation_model_factory(segmenter, confidence_thresh=seg_conf_thresh)
    if resolved_device.type != "cuda" and hasattr(seg_model, "precision"):
        seg_model.precision = torch.float32

    if remove_artifacts or remove_penmarks:
        artifact_model = segmentation_model_factory(
            "grandqc_artifact",
            remove_penmarks_only=(remove_penmarks and not remove_artifacts),
        )
        if resolved_device.type != "cuda" and hasattr(artifact_model, "precision"):
            artifact_model.precision = torch.float32
    else:
        artifact_model = None

    seg_device = "cpu" if segmenter == "otsu" else resolved_device_str

    contours_dir = processor.run_segmentation_job(
        segmentation_model=seg_model,
        seg_mag=seg_model.target_mag,
        holes_are_tissue=not remove_holes,
        batch_size=seg_batch_size,
        artifact_remover_model=artifact_model,
        device=seg_device,
    )

    coords_subdir = f"{mag}x_{patch_size}px_{overlap}px_overlap"
    coords_dir = processor.run_patching_job(
        target_magnification=mag,
        patch_size=patch_size,
        overlap=overlap,
        saveto=coords_subdir,
        min_tissue_proportion=min_tissue_proportion,
    )

    patch_encoder = encoder_factory("uni_v2", weights_path=str(uni_ckpt))
    if resolved_device.type != "cuda" and hasattr(patch_encoder, "precision"):
        # TRIDENT currently hardcodes autocast device_type='cuda' in patch extraction;
        # forcing float32 disables autocast so MPS/CPU paths remain usable.
        patch_encoder.precision = torch.float32

    patch_features_dir = processor.run_patch_feature_extraction_job(
        coords_dir=coords_subdir,
        patch_encoder=patch_encoder,
        device=resolved_device_str,
        saveas="h5",
        batch_limit=feat_batch_size,
    )
    _cleanup_appledouble_files(Path(patch_features_dir))

    processor.release()

    return {
        "device": resolved_device_str,
        "slides_dir": str(slides_path),
        "job_dir": str(job_path),
        "contours_dir": str(contours_dir),
        "coords_dir": str(coords_dir),
        "patch_features_dir": str(patch_features_dir),
        "uni_v2_ckpt": str(uni_ckpt),
        "trident_repo_dir": str(trident_repo),
        "artifact_removal_requested": artifact_requested,
        "artifact_removal_enabled": artifact_effective,
        "artifact_removal_auto_disabled": artifact_requested and not artifact_effective,
    }


def _import_tangle_modules(tangle_repo_dir: Path):
    _prepend_sys_path(tangle_repo_dir)
    mmssl_mod = importlib.import_module("core.models.mmssl")
    dataset_mod = importlib.import_module("core.dataset.dataset")
    return mmssl_mod, dataset_mod


def _restore_tangle_state_dict(model: torch.nn.Module, state_dict_obj: Any) -> torch.nn.Module:
    state_dict = state_dict_obj
    if isinstance(state_dict_obj, dict) and "state_dict" in state_dict_obj:
        state_dict = state_dict_obj["state_dict"]

    if not isinstance(state_dict, dict):
        raise TypeError("Invalid TANGLE checkpoint format: expected a state dict dictionary.")

    keys = list(state_dict.keys())
    has_module_prefix = any(k.startswith("module.") for k in keys)
    if has_module_prefix:
        cleaned = {k[len("module.") :]: v for k, v in state_dict.items()}
        model.load_state_dict(cleaned, strict=True)
    else:
        model.load_state_dict(state_dict, strict=True)
    return model


def _collate_slide_batch(batch):
    features, slide_ids = zip(*batch)
    return torch.stack(features, dim=0), list(slide_ids)


def _adapt_feature_dim(features: torch.Tensor, expected_dim: int, policy: str) -> torch.Tensor:
    observed_dim = int(features.shape[-1])
    if observed_dim == expected_dim:
        return features

    if observed_dim > expected_dim:
        if policy in {"truncate", "truncate_or_pad"}:
            return features[..., :expected_dim]
        raise ValueError(
            f"Observed patch feature dim {observed_dim} > expected {expected_dim}, "
            f"but policy='{policy}' does not allow truncation."
        )

    if policy in {"pad", "truncate_or_pad"}:
        pad_dim = expected_dim - observed_dim
        return torch.nn.functional.pad(features, (0, pad_dim), mode="constant", value=0.0)

    raise ValueError(
        f"Observed patch feature dim {observed_dim} < expected {expected_dim}, "
        f"but policy='{policy}' does not allow zero-padding."
    )


def _discover_patch_feature_files(patch_features_dir: Path, extension: str) -> list[Path]:
    return sorted(
        p
        for p in patch_features_dir.rglob(f"*{extension}")
        if p.is_file() and not p.name.startswith("._")
    )


def _iter_h5_datasets(group: h5py.Group, prefix: str = ""):
    for key, value in group.items():
        dataset_key = f"{prefix}{key}"
        if isinstance(value, h5py.Dataset):
            yield dataset_key, value
        elif isinstance(value, h5py.Group):
            yield from _iter_h5_datasets(value, prefix=f"{dataset_key}/")


def _find_h5_dataset_key(
    h5_file: h5py.File,
    *,
    preferred_names: Sequence[str],
    min_ndim: int = 1,
    min_last_dim: int | None = None,
    fallback_any: bool = True,
) -> str | None:
    datasets = list(_iter_h5_datasets(h5_file))
    if not datasets:
        return None

    lowered_names = tuple(name.lower() for name in preferred_names)
    for preferred in lowered_names:
        for key, dataset in datasets:
            key_l = key.lower()
            leaf_l = key_l.split("/")[-1]
            if key_l == preferred or leaf_l == preferred:
                if dataset.ndim < min_ndim:
                    continue
                if min_last_dim is not None and int(dataset.shape[-1]) < min_last_dim:
                    continue
                return key

    if fallback_any:
        for key, dataset in datasets:
            if dataset.ndim < min_ndim:
                continue
            if min_last_dim is not None and int(dataset.shape[-1]) < min_last_dim:
                continue
            return key

    return None


def _load_patch_feature_file(feature_file: Path) -> tuple[np.ndarray, np.ndarray | None, str, str | None]:
    with h5py.File(feature_file, "r") as h5_file:
        features_key = _find_h5_dataset_key(
            h5_file,
            preferred_names=("features", "feats", "embeddings", "patch_features"),
            min_ndim=2,
            min_last_dim=8,
        )
        if features_key is None:
            raise ValueError(f"No valid 2D feature dataset found in {feature_file}")
        features = np.asarray(h5_file[features_key], dtype=np.float32)

        coords_key = _find_h5_dataset_key(
            h5_file,
            preferred_names=("coords", "coordinates", "xy", "patch_coords"),
            min_ndim=2,
            min_last_dim=2,
            fallback_any=False,
        )
        coords: np.ndarray | None = None
        if coords_key is not None and coords_key != features_key:
            coords_arr = np.asarray(h5_file[coords_key])
            if coords_arr.ndim == 2 and int(coords_arr.shape[0]) == int(features.shape[0]):
                coords = coords_arr

    return features, coords, features_key, coords_key


def _safe_l2_normalize(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return features / norms


def _build_candidate_pool(
    n_tiles: int,
    *,
    max_input_tiles: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    all_indices = np.arange(n_tiles, dtype=np.int64)
    if max_input_tiles is None or n_tiles <= max_input_tiles:
        return all_indices
    sampled = rng.choice(all_indices, size=max_input_tiles, replace=False)
    sampled.sort()
    return sampled.astype(np.int64)


def _select_tiles_random(
    features: np.ndarray,
    *,
    top_k: int,
    rng: np.random.Generator,
    splice_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    _ = splice_alpha
    n_tiles = int(features.shape[0])
    k = min(int(top_k), n_tiles)
    selected = np.sort(rng.choice(n_tiles, size=k, replace=False).astype(np.int64))
    scores = np.ones(k, dtype=np.float32)
    return selected, scores


def _select_tiles_fps(
    features: np.ndarray,
    *,
    top_k: int,
    rng: np.random.Generator,
    splice_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    _ = splice_alpha
    norm_features = _safe_l2_normalize(features.astype(np.float32, copy=False))
    n_tiles = int(norm_features.shape[0])
    k = min(int(top_k), n_tiles)

    first_idx = int(rng.integers(0, n_tiles))
    selected = [first_idx]
    selected_scores = [1.0]
    picked = np.zeros(n_tiles, dtype=bool)
    picked[first_idx] = True

    min_cosine_dist = 1.0 - np.clip(norm_features @ norm_features[first_idx], -1.0, 1.0)
    min_cosine_dist[first_idx] = -np.inf

    while len(selected) < k:
        next_idx = int(np.argmax(min_cosine_dist))
        if picked[next_idx]:
            break
        selected.append(next_idx)
        selected_scores.append(float(min_cosine_dist[next_idx]))
        picked[next_idx] = True
        next_dist = 1.0 - np.clip(norm_features @ norm_features[next_idx], -1.0, 1.0)
        min_cosine_dist = np.minimum(min_cosine_dist, next_dist)
        min_cosine_dist[picked] = -np.inf

    return np.asarray(selected, dtype=np.int64), np.asarray(selected_scores, dtype=np.float32)


def _select_tiles_splice(
    features: np.ndarray,
    *,
    top_k: int,
    rng: np.random.Generator,
    splice_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    _ = rng
    alpha = float(np.clip(splice_alpha, 0.0, 1.0))
    norm_features = _safe_l2_normalize(features.astype(np.float32, copy=False))
    n_tiles = int(norm_features.shape[0])
    k = min(int(top_k), n_tiles)

    centroid = norm_features.mean(axis=0, keepdims=True)
    centroid = _safe_l2_normalize(centroid)[0]
    representativeness = np.clip(norm_features @ centroid, -1.0, 1.0)

    first_idx = int(np.argmax(representativeness))
    selected = [first_idx]
    selected_scores = [float(representativeness[first_idx])]
    picked = np.zeros(n_tiles, dtype=bool)
    picked[first_idx] = True

    novelty = 1.0 - np.clip(norm_features @ norm_features[first_idx], -1.0, 1.0)
    novelty[first_idx] = 0.0

    while len(selected) < k:
        combined = alpha * representativeness + (1.0 - alpha) * novelty
        combined[picked] = -np.inf
        next_idx = int(np.argmax(combined))
        if picked[next_idx]:
            break

        selected.append(next_idx)
        selected_scores.append(float(combined[next_idx]))
        picked[next_idx] = True

        next_dist = 1.0 - np.clip(norm_features @ norm_features[next_idx], -1.0, 1.0)
        novelty = np.minimum(novelty, next_dist)

    return np.asarray(selected, dtype=np.int64), np.asarray(selected_scores, dtype=np.float32)


TileSelectorFn = Callable[
    [np.ndarray, int, np.random.Generator, float],
    tuple[np.ndarray, np.ndarray],
]

_TILE_SELECTOR_REGISTRY: dict[str, TileSelectorFn] = {
    "splice": lambda feats, top_k, rng, alpha: _select_tiles_splice(
        feats, top_k=top_k, rng=rng, splice_alpha=alpha
    ),
    "fps": lambda feats, top_k, rng, alpha: _select_tiles_fps(
        feats, top_k=top_k, rng=rng, splice_alpha=alpha
    ),
    "random": lambda feats, top_k, rng, alpha: _select_tiles_random(
        feats, top_k=top_k, rng=rng, splice_alpha=alpha
    ),
}


def list_supported_tile_selection_methods() -> tuple[str, ...]:
    return tuple(sorted(_TILE_SELECTOR_REGISTRY.keys()))


def _build_slide_output_name(feature_file: Path, patch_features_root: Path, extension: str) -> tuple[str, str]:
    relative = feature_file.relative_to(patch_features_root)
    rel_str = relative.as_posix()
    if rel_str.endswith(extension):
        rel_trim = rel_str[: -len(extension)]
    else:
        rel_trim = relative.stem
    slide_id = Path(rel_trim).name
    output_name = rel_trim.replace("/", "__")
    return slide_id, output_name


def run_representative_tile_selection(
    *,
    patch_features_dir: str | Path,
    output_dir: str | Path,
    method: str = "splice",
    top_k: int = 32,
    extension: str = ".h5",
    max_input_tiles: int | None = 4096,
    splice_alpha: float = 0.7,
    random_seed: int = 0,
    tile_encoder_name: str = "uni_v2",
    tile_patch_size: int = 256,
) -> dict[str, Any]:
    patch_features_path = _to_data_path(
        patch_features_dir,
        path_kind="patch feature directory for tile selection",
        must_exist=True,
    )
    if not patch_features_path.is_dir():
        raise NotADirectoryError(f"Patch feature directory is not a folder: {patch_features_path}")

    out_dir = _to_data_path(output_dir, path_kind="tile selection output directory", must_exist=False)
    out_dir.mkdir(parents=True, exist_ok=True)

    method_l = str(method).strip().lower()
    if method_l not in _TILE_SELECTOR_REGISTRY:
        raise ValueError(
            f"Unsupported tile selection method '{method}'. "
            f"Choose from: {', '.join(list_supported_tile_selection_methods())}"
        )
    if int(top_k) <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if max_input_tiles is not None and int(max_input_tiles) <= 0:
        raise ValueError(f"max_input_tiles must be positive when set, got {max_input_tiles}")

    selector = _TILE_SELECTOR_REGISTRY[method_l]
    feature_files = _discover_patch_feature_files(patch_features_path, extension=extension)
    if not feature_files:
        raise ValueError(
            f"No patch feature files with extension '{extension}' found in: {patch_features_path}"
        )

    selected_dir = out_dir / "selected_tiles"
    selected_dir.mkdir(parents=True, exist_ok=True)
    packed_tile_features: dict[str, list[tuple[torch.Tensor, tuple[tuple[int, int], tuple[int, int]]]]] = {}

    records: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []

    # TODO(superhy): once data disk is mounted, run end-to-end checks on real TRIDENT+UNI H5s
    # and manually verify selected tiles are tissue-informative and non-redundant.
    for file_idx, feature_file in enumerate(feature_files):
        slide_id, output_name = _build_slide_output_name(feature_file, patch_features_path, extension)
        try:
            features, coords, features_key, coords_key = _load_patch_feature_file(feature_file)
            if features.ndim != 2:
                raise ValueError(f"Feature array must be 2D, got shape={features.shape}")
            n_tiles = int(features.shape[0])
            if n_tiles == 0:
                raise ValueError("Feature array is empty")

            rng = np.random.default_rng(int(random_seed) + file_idx)
            candidate_indices = _build_candidate_pool(
                n_tiles,
                max_input_tiles=max_input_tiles,
                rng=rng,
            )
            candidate_features = features[candidate_indices]
            local_selected, selected_scores = selector(
                candidate_features,
                int(top_k),
                rng,
                float(splice_alpha),
            )
            selected_indices = candidate_indices[local_selected]
            selected_features = features[selected_indices].astype(np.float32, copy=False)

            payload: dict[str, Any] = {
                "slide_id": slide_id,
                "source_file": str(feature_file),
                "method": method_l,
                "selected_indices": selected_indices.astype(np.int64, copy=False),
                "selection_scores": selected_scores.astype(np.float32, copy=False),
                "selected_features": selected_features,
            }
            if coords is not None:
                payload["selected_coords"] = coords[selected_indices]
            packed_entries: list[tuple[torch.Tensor, tuple[tuple[int, int], tuple[int, int]]]] = []
            for row_i, emb_vec in enumerate(selected_features):
                emb_tensor = torch.from_numpy(np.asarray(emb_vec, dtype=np.float32))
                if coords is not None:
                    raw_coord = coords[selected_indices[row_i]]
                    raw_coord = np.asarray(raw_coord).reshape(-1)
                    if raw_coord.shape[0] >= 4:
                        x1, y1, x2, y2 = [int(v) for v in raw_coord[:4].tolist()]
                    elif raw_coord.shape[0] >= 2:
                        x1, y1 = [int(v) for v in raw_coord[:2].tolist()]
                        x2 = int(x1 + int(tile_patch_size))
                        y2 = int(y1 + int(tile_patch_size))
                    else:
                        x1 = y1 = x2 = y2 = 0
                else:
                    x1 = y1 = x2 = y2 = 0
                packed_entries.append((emb_tensor, ((x1, y1), (x2, y2))))
            packed_tile_features[slide_id] = packed_entries

            npz_path = selected_dir / f"{output_name}.npz"
            np.savez_compressed(npz_path, **payload)

            records.append(
                {
                    "slide_id": slide_id,
                    "source_file": str(feature_file),
                    "features_key": features_key,
                    "coords_key": coords_key,
                    "n_tiles_total": n_tiles,
                    "n_tiles_candidate_pool": int(candidate_indices.shape[0]),
                    "n_tiles_selected": int(selected_indices.shape[0]),
                    "output_npz": str(npz_path),
                }
            )
        except Exception as exc:
            failed.append({"source_file": str(feature_file), "error": str(exc)})

    if not records:
        raise RuntimeError(
            "Representative tile selection failed for all slides. "
            f"First error: {failed[0]['error'] if failed else 'unknown'}"
        )

    records_df = pd.DataFrame(records)
    csv_path = out_dir / "selected_tiles_index.csv"
    parquet_path = out_dir / "selected_tiles_index.parquet"
    safe_selector = re.sub(r"[^A-Za-z0-9_.-]+", "-", method_l)
    safe_encoder = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(tile_encoder_name).strip().lower())
    packed_pt_path = out_dir / f"wsi_tile_features__selector-{safe_selector}__encoder-{safe_encoder}.pt"
    summary_path = out_dir / "summary.json"

    records_df.to_csv(csv_path, index=False)
    records_df.to_parquet(parquet_path, index=False)
    torch.save(packed_tile_features, packed_pt_path)

    summary = {
        "patch_features_dir": str(patch_features_path),
        "method": method_l,
        "tile_encoder_name": str(tile_encoder_name),
        "top_k": int(top_k),
        "extension": extension,
        "max_input_tiles": int(max_input_tiles) if max_input_tiles is not None else None,
        "splice_alpha": float(splice_alpha),
        "random_seed": int(random_seed),
        "tile_patch_size": int(tile_patch_size),
        "n_feature_files": len(feature_files),
        "n_slides_succeeded": len(records),
        "n_slides_failed": len(failed),
        "failed_examples": failed[:10],
        "selected_tiles_dir": str(selected_dir),
        "tile_features_pt": str(packed_pt_path),
        "index_csv": str(csv_path),
        "index_parquet": str(parquet_path),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def run_tangle_slide_embedding(
    *,
    patch_features_dir: str | Path,
    output_dir: str | Path,
    device: str = "auto",
    models_root: str | Path | None = None,
    tangle_repo_dir: str | Path | None = None,
    tangle_checkpoint_dir: str | Path | None = None,
    tangle_pretrained_root_dir: str | Path | None = None,
    allow_model_download: bool = True,
    tangle_drive_url: str = TANGLE_PRETRAINED_DRIVE_URL,
    tangle_checkpoint_keyword: str = "tangle_brca",
    extension: str = ".h5",
    batch_size: int = 1,
    num_workers: int = 0,
    feature_dim_policy: str = "truncate_or_pad",
    slide_encoder_name: str = "tangle",
    slide_mode_name: str = "slide_level",
) -> dict[str, Any]:
    if batch_size != 1:
        raise ValueError("TANGLE inference currently requires batch_size=1 due variable token counts per slide.")

    resolved_device = resolve_torch_device(device)
    resolved_device_str = str(resolved_device)

    patch_features_path = _to_data_path(
        patch_features_dir,
        path_kind="patch feature directory",
        must_exist=True,
    )
    if not patch_features_path.is_dir():
        raise NotADirectoryError(f"Patch feature directory is not a folder: {patch_features_path}")

    out_dir = _to_data_path(output_dir, path_kind="slide embedding output directory", must_exist=False)
    out_dir.mkdir(parents=True, exist_ok=True)

    models_root_path = _resolve_models_root(models_root)
    repos = ensure_wsi_repositories(
        models_root=models_root_path,
        trident_repo_dir=None,
        tangle_repo_dir=tangle_repo_dir,
        update_repos=False,
        require_trident=False,
        require_tangle=True,
    )
    tangle_repo = repos["tangle_repo_dir"]

    checkpoint_dir = ensure_tangle_checkpoint(
        models_root=models_root_path,
        checkpoint_dir=tangle_checkpoint_dir,
        pretrained_root_dir=tangle_pretrained_root_dir,
        allow_download=allow_model_download,
        drive_folder_url=tangle_drive_url,
        preferred_keyword=tangle_checkpoint_keyword,
    )

    _cleanup_appledouble_files(patch_features_path)

    mmssl_mod, dataset_mod = _import_tangle_modules(tangle_repo)
    MMSSL = mmssl_mod.MMSSL
    SlideDataset = dataset_mod.SlideDataset

    with (checkpoint_dir / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise TypeError(f"Invalid TANGLE config format in {checkpoint_dir / 'config.json'}")

    n_tokens_rna = int(config.get("rna_token_dim", 4999))
    model = MMSSL(config=config, n_tokens_rna=n_tokens_rna)
    ckpt_obj = torch.load(checkpoint_dir / "model.pt", map_location="cpu")
    model = _restore_tangle_state_dict(model, ckpt_obj)
    model.to(resolved_device)
    model.eval()

    dataset = SlideDataset(features_path=str(patch_features_path), extension=extension)
    if len(dataset) == 0:
        raise ValueError(
            f"No patch feature files with extension '{extension}' found in: {patch_features_path}"
        )

    policy = str(feature_dim_policy).strip().lower()
    allowed_policies = {"strict", "truncate", "pad", "truncate_or_pad"}
    if policy not in allowed_policies:
        raise ValueError(f"feature_dim_policy must be one of {sorted(allowed_policies)}, got '{feature_dim_policy}'")

    # Validate patch feature dimension against checkpoint config when available.
    sample_features, _ = dataset[0]
    expected_dim = config.get("embedding_dim", None)
    observed_dim = None
    feature_dim_adapted = False
    if sample_features.ndim >= 2:
        observed_dim = int(sample_features.shape[-1])

    if isinstance(expected_dim, int) and observed_dim is not None and observed_dim != int(expected_dim):
        if policy == "strict":
            raise ValueError(
                "Patch feature dimension mismatch between TRIDENT and TANGLE checkpoint: "
                f"observed={observed_dim}, expected={int(expected_dim)}. "
                "Use a compatible checkpoint or set feature_dim_policy "
                "to truncate/pad for compatibility."
            )
        feature_dim_adapted = True

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_slide_batch,
    )

    slide_ids: list[str] = []
    embeddings: list[np.ndarray] = []

    with torch.inference_mode():
        for features, batch_slide_ids in loader:
            features = features.to(resolved_device)
            if isinstance(expected_dim, int):
                features = _adapt_feature_dim(features, int(expected_dim), policy=policy)
            out = model.get_features(features)
            out_np = out.float().cpu().numpy()
            for i, slide_id in enumerate(batch_slide_ids):
                slide_ids.append(str(slide_id))
                embeddings.append(out_np[i])

    embeds_np = np.asarray(embeddings, dtype=np.float32)
    emb_dim = int(embeds_np.shape[1])
    emb_cols = [f"z_{d}" for d in range(emb_dim)]
    emb_df = pd.DataFrame(embeds_np, columns=emb_cols)
    emb_df.insert(0, "slide_id", slide_ids)

    parquet_path = out_dir / "slide_embeddings.parquet"
    csv_path = out_dir / "slide_embeddings.csv"
    pkl_path = out_dir / "slide_embeddings.pkl"
    safe_encoder = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(slide_encoder_name).strip().lower())
    safe_mode = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(slide_mode_name).strip().lower())
    slide_features_pt = out_dir / f"wsi_slide_features__encoder-{safe_encoder}__mode-{safe_mode}.pt"
    summary_path = out_dir / "summary.json"

    emb_df.to_parquet(parquet_path, index=False)
    emb_df.to_csv(csv_path, index=False)

    with pkl_path.open("wb") as f:
        pickle.dump({"slide_ids": slide_ids, "embeds": embeds_np}, f)
    slide_feature_payload = {
        str(slide_id): torch.from_numpy(np.asarray(embeds_np[idx], dtype=np.float32))
        for idx, slide_id in enumerate(slide_ids)
    }
    torch.save(slide_feature_payload, slide_features_pt)

    summary = {
        "device": resolved_device_str,
        "patch_features_dir": str(patch_features_path),
        "tangle_repo_dir": str(tangle_repo),
        "tangle_checkpoint_dir": str(checkpoint_dir),
        "n_slides": len(slide_ids),
        "embedding_dim": emb_dim,
        "slide_encoder_name": str(slide_encoder_name),
        "slide_mode_name": str(slide_mode_name),
        "patch_feature_dim_observed": observed_dim,
        "patch_feature_dim_expected": int(expected_dim) if isinstance(expected_dim, int) else None,
        "feature_dim_policy": policy,
        "feature_dim_adapted": feature_dim_adapted,
        "parquet": str(parquet_path),
        "csv": str(csv_path),
        "pkl": str(pkl_path),
        "slide_features_pt": str(slide_features_pt),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def encode_tcga_brca_wsi(
    *,
    slides_dir: str | Path = "TCGA-BRCA/slides",
    output_root: str | Path = "TCGA-BRCA/wsi_embeddings",
    device: str = "auto",
    models_root: str | Path | None = "models",
    trident_repo_dir: str | Path | None = None,
    tangle_repo_dir: str | Path | None = None,
    uni_v2_ckpt_path: str | Path | None = None,
    hf_token: str | None = None,
    tangle_checkpoint_dir: str | Path | None = None,
    tangle_pretrained_root_dir: str | Path | None = None,
    allow_model_download: bool = True,
    tangle_drive_url: str = TANGLE_PRETRAINED_DRIVE_URL,
    tangle_checkpoint_keyword: str = "tangle_brca",
    run_trident: bool = True,
    run_tangle: bool = True,
    run_tile_selection: bool = False,
    patch_features_dir: str | Path | None = None,
    reader_type: str = "openslide",
    segmenter: str = "hest",
    seg_conf_thresh: float = 0.5,
    remove_holes: bool = False,
    remove_artifacts: bool = True,
    remove_penmarks: bool = False,
    mag: float = 20.0,
    patch_size: int = 256,
    overlap: int = 0,
    min_tissue_proportion: float = 0.0,
    seg_batch_size: int = 16,
    feat_batch_size: int = 256,
    max_workers: int | None = None,
    skip_errors: bool = True,
    search_nested: bool = False,
    custom_list_of_wsis: str | Path | None = None,
    extension: str = ".h5",
    tangle_num_workers: int = 0,
    feature_dim_policy: str = "truncate_or_pad",
    tile_selection_method: str = "splice",
    tile_selection_top_k: int = 32,
    tile_selection_output_dir: str | Path | None = None,
    tile_selection_max_input_tiles: int | None = 4096,
    tile_selection_splice_alpha: float = 0.7,
    tile_selection_seed: int = 0,
    tile_encoder_name: str = "uni_v2",
    slide_encoder_name: str = "tangle",
    slide_mode_name: str = "slide_level",
) -> dict[str, Any]:
    out_root = _to_data_path(output_root, path_kind="WSI output root", must_exist=False)
    out_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "device": str(resolve_torch_device(device)),
        "output_root": str(out_root),
    }

    if run_trident:
        trident_result = run_trident_uni_v2_patch_encoding(
            slides_dir=slides_dir,
            job_dir=out_root / "trident_uni_v2",
            device=device,
            models_root=models_root,
            trident_repo_dir=trident_repo_dir,
            uni_v2_ckpt_path=uni_v2_ckpt_path,
            hf_token=hf_token,
            allow_model_download=allow_model_download,
            reader_type=reader_type,
            segmenter=segmenter,
            seg_conf_thresh=seg_conf_thresh,
            remove_holes=remove_holes,
            remove_artifacts=remove_artifacts,
            remove_penmarks=remove_penmarks,
            mag=mag,
            patch_size=patch_size,
            overlap=overlap,
            min_tissue_proportion=min_tissue_proportion,
            seg_batch_size=seg_batch_size,
            feat_batch_size=feat_batch_size,
            max_workers=max_workers,
            skip_errors=skip_errors,
            search_nested=search_nested,
            custom_list_of_wsis=custom_list_of_wsis,
        )
        summary["trident"] = trident_result
        effective_patch_features_dir = trident_result["patch_features_dir"]
    else:
        if patch_features_dir is None:
            raise ValueError("patch_features_dir must be provided when run_trident is False")
        effective_patch_features_dir = str(
            _to_data_path(patch_features_dir, path_kind="patch feature directory", must_exist=True)
        )

    if run_tile_selection:
        tile_selection_out_dir = (
            out_root / "tile_features"
            if tile_selection_output_dir is None
            else _to_data_path(
                tile_selection_output_dir,
                path_kind="tile selection output directory",
                must_exist=False,
            )
        )
        tile_selection_result = run_representative_tile_selection(
            patch_features_dir=effective_patch_features_dir,
            output_dir=tile_selection_out_dir,
            method=tile_selection_method,
            top_k=tile_selection_top_k,
            extension=extension,
            max_input_tiles=tile_selection_max_input_tiles,
            splice_alpha=tile_selection_splice_alpha,
            random_seed=tile_selection_seed,
            tile_encoder_name=tile_encoder_name,
            tile_patch_size=patch_size,
        )
        summary["tile_selection"] = tile_selection_result

    if run_tangle:
        tangle_result = run_tangle_slide_embedding(
            patch_features_dir=effective_patch_features_dir,
            output_dir=out_root / "slide_features",
            device=device,
            models_root=models_root,
            tangle_repo_dir=tangle_repo_dir,
            tangle_checkpoint_dir=tangle_checkpoint_dir,
            tangle_pretrained_root_dir=tangle_pretrained_root_dir,
            allow_model_download=allow_model_download,
            tangle_drive_url=tangle_drive_url,
            tangle_checkpoint_keyword=tangle_checkpoint_keyword,
            extension=extension,
            batch_size=1,
            num_workers=tangle_num_workers,
            feature_dim_policy=feature_dim_policy,
            slide_encoder_name=slide_encoder_name,
            slide_mode_name=slide_mode_name,
        )
        summary["tangle"] = tangle_result

    with (out_root / "pipeline_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
