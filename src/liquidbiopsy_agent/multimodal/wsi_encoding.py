from __future__ import annotations

import importlib
import json
import os
import pickle
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

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
    summary_path = out_dir / "summary.json"

    emb_df.to_parquet(parquet_path, index=False)
    emb_df.to_csv(csv_path, index=False)

    with pkl_path.open("wb") as f:
        pickle.dump({"slide_ids": slide_ids, "embeds": embeds_np}, f)

    summary = {
        "device": resolved_device_str,
        "patch_features_dir": str(patch_features_path),
        "tangle_repo_dir": str(tangle_repo),
        "tangle_checkpoint_dir": str(checkpoint_dir),
        "n_slides": len(slide_ids),
        "embedding_dim": emb_dim,
        "patch_feature_dim_observed": observed_dim,
        "patch_feature_dim_expected": int(expected_dim) if isinstance(expected_dim, int) else None,
        "feature_dim_policy": policy,
        "feature_dim_adapted": feature_dim_adapted,
        "parquet": str(parquet_path),
        "csv": str(csv_path),
        "pkl": str(pkl_path),
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

    if run_tangle:
        tangle_result = run_tangle_slide_embedding(
            patch_features_dir=effective_patch_features_dir,
            output_dir=out_root / "tangle_slide_embeddings",
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
        )
        summary["tangle"] = tangle_result

    with (out_root / "pipeline_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
