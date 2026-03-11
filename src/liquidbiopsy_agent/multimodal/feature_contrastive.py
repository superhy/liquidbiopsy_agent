from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from liquidbiopsy_agent.utils.device import resolve_torch_device, should_pin_memory
from liquidbiopsy_agent.utils.storage import get_data_root, resolve_data_path

from .config import MultiModalConfig
from .extractors import BloodFoundationExtractor
from .losses import SubtypeAwareContrastiveLoss
from .model import ProjectionHead


LOGGER = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_table(path: Path) -> pd.DataFrame:
    suffixes = [s.lower() for s in path.suffixes]
    if ".parquet" in suffixes or path.suffix.lower() in {".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _to_numpy_vector(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D feature vector, got shape={arr.shape}")
    if arr.size == 0:
        raise ValueError("Feature vector is empty")
    return arr


def _parse_diag_coord(value: Any) -> tuple[tuple[int, int], tuple[int, int]]:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        p1, p2 = value[0], value[1]
        if (
            isinstance(p1, (tuple, list))
            and isinstance(p2, (tuple, list))
            and len(p1) >= 2
            and len(p2) >= 2
        ):
            return (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))
    return (0, 0), (0, 0)


def _load_blood_feature_map(pt_path: Path) -> dict[str, np.ndarray]:
    payload = torch.load(pt_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"Blood feature PT must be a mapping, got {type(payload)}")
    out: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        out[str(key)] = _to_numpy_vector(value)
    if not out:
        raise ValueError(f"Blood feature PT is empty: {pt_path}")
    return out


def _load_slide_feature_map(pt_path: Path) -> dict[str, np.ndarray]:
    payload = torch.load(pt_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"Slide feature PT must be a mapping, got {type(payload)}")
    out: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        out[str(key)] = _to_numpy_vector(value)
    if not out:
        raise ValueError(f"Slide feature PT is empty: {pt_path}")
    return out


def _load_tile_feature_map(
    pt_path: Path,
) -> dict[str, list[tuple[np.ndarray, tuple[tuple[int, int], tuple[int, int]]]]]:
    payload = torch.load(pt_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"Tile feature PT must be a mapping, got {type(payload)}")

    out: dict[str, list[tuple[np.ndarray, tuple[tuple[int, int], tuple[int, int]]]]] = {}
    for slide_id, entries in payload.items():
        key = str(slide_id)
        if not isinstance(entries, Sequence):
            continue
        parsed: list[tuple[np.ndarray, tuple[tuple[int, int], tuple[int, int]]]] = []
        for item in entries:
            if not isinstance(item, (tuple, list)) or len(item) < 1:
                continue
            emb = _to_numpy_vector(item[0])
            coord = _parse_diag_coord(item[1]) if len(item) >= 2 else ((0, 0), (0, 0))
            parsed.append((emb, coord))
        if parsed:
            out[key] = parsed
    if not out:
        raise ValueError(f"Tile feature PT is empty or invalid: {pt_path}")
    return out


def _split_df(
    df: pd.DataFrame,
    *,
    split_col: str | None,
    patient_col: str | None,
    val_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if split_col and split_col in df.columns:
        train_df = df[df[split_col].astype(str).str.lower() == "train"].copy()
        val_df = df[df[split_col].astype(str).str.lower() == "val"].copy()
        if train_df.empty or val_df.empty:
            raise ValueError("Split column exists but train/val rows are missing.")
        return train_df, val_df

    rng = np.random.default_rng(seed)
    if patient_col and patient_col in df.columns:
        patients = df[patient_col].astype(str).drop_duplicates().tolist()
        rng.shuffle(patients)
        n_val = max(1, int(len(patients) * val_ratio))
        val_patients = set(patients[:n_val])
        is_val = df[patient_col].astype(str).isin(val_patients)
    else:
        idx = np.arange(len(df))
        rng.shuffle(idx)
        n_val = max(1, int(len(df) * val_ratio))
        is_val = np.zeros(len(df), dtype=bool)
        is_val[idx[:n_val]] = True

    train_df = df[~is_val].copy()
    val_df = df[is_val].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("Split produced empty train/val; adjust val_ratio or provide split column.")
    return train_df, val_df


@dataclass
class SlidePairSample:
    blood_sample_id: str
    slide_id: str
    label_text: str
    label_id: int
    blood_vec: np.ndarray
    tissue_vec: np.ndarray


@dataclass
class TilePairSample:
    blood_sample_id: str
    slide_id: str
    label_text: str
    label_id: int
    blood_vec: np.ndarray
    tile_vecs: np.ndarray
    tile_coords: list[tuple[tuple[int, int], tuple[int, int]]]


def _build_slide_samples(
    label_df: pd.DataFrame,
    *,
    blood_id_col: str,
    slide_id_col: str,
    label_col: str,
    label_to_id: Mapping[str, int],
    blood_map: Mapping[str, np.ndarray],
    slide_map: Mapping[str, np.ndarray],
) -> tuple[list[SlidePairSample], dict[str, int]]:
    samples: list[SlidePairSample] = []
    dropped = {"missing_blood": 0, "missing_slide": 0}
    for _, row in label_df.iterrows():
        blood_id = str(row[blood_id_col])
        slide_id = str(row[slide_id_col])
        label_text = str(row[label_col])
        blood_vec = blood_map.get(blood_id)
        if blood_vec is None:
            dropped["missing_blood"] += 1
            continue
        tissue_vec = slide_map.get(slide_id)
        if tissue_vec is None:
            dropped["missing_slide"] += 1
            continue
        samples.append(
            SlidePairSample(
                blood_sample_id=blood_id,
                slide_id=slide_id,
                label_text=label_text,
                label_id=int(label_to_id[label_text]),
                blood_vec=np.asarray(blood_vec, dtype=np.float32),
                tissue_vec=np.asarray(tissue_vec, dtype=np.float32),
            )
        )
    return samples, dropped


def _build_tile_samples(
    label_df: pd.DataFrame,
    *,
    blood_id_col: str,
    slide_id_col: str,
    label_col: str,
    label_to_id: Mapping[str, int],
    blood_map: Mapping[str, np.ndarray],
    tile_map: Mapping[str, list[tuple[np.ndarray, tuple[tuple[int, int], tuple[int, int]]]]],
) -> tuple[list[TilePairSample], dict[str, int]]:
    samples: list[TilePairSample] = []
    dropped = {"missing_blood": 0, "missing_slide": 0, "empty_tiles": 0}
    for _, row in label_df.iterrows():
        blood_id = str(row[blood_id_col])
        slide_id = str(row[slide_id_col])
        label_text = str(row[label_col])
        blood_vec = blood_map.get(blood_id)
        if blood_vec is None:
            dropped["missing_blood"] += 1
            continue
        tile_entries = tile_map.get(slide_id)
        if tile_entries is None:
            dropped["missing_slide"] += 1
            continue
        if not tile_entries:
            dropped["empty_tiles"] += 1
            continue
        tile_vecs = np.stack([np.asarray(item[0], dtype=np.float32) for item in tile_entries], axis=0)
        tile_coords = [item[1] for item in tile_entries]
        samples.append(
            TilePairSample(
                blood_sample_id=blood_id,
                slide_id=slide_id,
                label_text=label_text,
                label_id=int(label_to_id[label_text]),
                blood_vec=np.asarray(blood_vec, dtype=np.float32),
                tile_vecs=tile_vecs,
                tile_coords=tile_coords,
            )
        )
    return samples, dropped


class SlideFeatureDataset(Dataset):
    def __init__(self, rows: Sequence[SlidePairSample]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        return {
            "blood": torch.tensor(row.blood_vec, dtype=torch.float32),
            "tissue": torch.tensor(row.tissue_vec, dtype=torch.float32),
            "label": torch.tensor(row.label_id, dtype=torch.long),
            "label_text": row.label_text,
            "blood_sample_id": row.blood_sample_id,
            "slide_id": row.slide_id,
        }


class TileFeatureDataset(Dataset):
    def __init__(self, rows: Sequence[TilePairSample]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        return {
            "blood": torch.tensor(row.blood_vec, dtype=torch.float32),
            "tiles": torch.tensor(row.tile_vecs, dtype=torch.float32),
            "label": torch.tensor(row.label_id, dtype=torch.long),
            "label_text": row.label_text,
            "blood_sample_id": row.blood_sample_id,
            "slide_id": row.slide_id,
            "tile_coords": row.tile_coords,
        }


def _tile_collate_fn(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    blood = torch.stack([item["blood"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    max_tiles = max(int(item["tiles"].shape[0]) for item in batch)
    tile_dim = int(batch[0]["tiles"].shape[1])

    tiles = torch.zeros((len(batch), max_tiles, tile_dim), dtype=torch.float32)
    tile_mask = torch.zeros((len(batch), max_tiles), dtype=torch.bool)
    tile_counts = torch.zeros((len(batch),), dtype=torch.long)
    for i, item in enumerate(batch):
        n_tiles = int(item["tiles"].shape[0])
        tiles[i, :n_tiles, :] = item["tiles"]
        tile_mask[i, :n_tiles] = True
        tile_counts[i] = n_tiles

    return {
        "blood": blood,
        "tiles": tiles,
        "tile_mask": tile_mask,
        "tile_counts": tile_counts,
        "label": labels,
        "label_text": [item["label_text"] for item in batch],
        "blood_sample_id": [item["blood_sample_id"] for item in batch],
        "slide_id": [item["slide_id"] for item in batch],
    }


class SlideFeatureContrastiveModel(nn.Module):
    def __init__(
        self,
        *,
        blood_input_dim: int,
        tissue_input_dim: int,
        projection_dim: int,
        projection_hidden_dim: int,
        blood_hidden_dims: Sequence[int],
        tissue_hidden_dims: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.blood_encoder = BloodFoundationExtractor(
            input_dim=blood_input_dim,
            hidden_dims=[int(x) for x in blood_hidden_dims],
            output_dim=None,
            dropout=dropout,
            freeze=False,
        )
        self.tissue_encoder = BloodFoundationExtractor(
            input_dim=tissue_input_dim,
            hidden_dims=[int(x) for x in tissue_hidden_dims],
            output_dim=None,
            dropout=dropout,
            freeze=False,
        )
        self.blood_head = ProjectionHead(
            input_dim=int(self.blood_encoder.output_dim),
            hidden_dim=projection_hidden_dim,
            output_dim=projection_dim,
            dropout=dropout,
        )
        self.tissue_head = ProjectionHead(
            input_dim=int(self.tissue_encoder.output_dim),
            hidden_dim=projection_hidden_dim,
            output_dim=projection_dim,
            dropout=dropout,
        )

    def forward(self, blood: torch.Tensor, tissue: torch.Tensor) -> dict[str, torch.Tensor]:
        blood_feat = self.blood_encoder(blood)
        tissue_feat = self.tissue_encoder(tissue)
        blood_proj = F.normalize(self.blood_head(blood_feat), dim=-1)
        tissue_proj = F.normalize(self.tissue_head(tissue_feat), dim=-1)
        return {
            "blood_feat": blood_feat,
            "tissue_feat": tissue_feat,
            "blood_proj": blood_proj,
            "tissue_proj": tissue_proj,
        }


class TileFeatureContrastiveModel(nn.Module):
    def __init__(
        self,
        *,
        blood_input_dim: int,
        tile_input_dim: int,
        projection_dim: int,
        projection_hidden_dim: int,
        blood_hidden_dims: Sequence[int],
        tile_hidden_dims: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.blood_encoder = BloodFoundationExtractor(
            input_dim=blood_input_dim,
            hidden_dims=[int(x) for x in blood_hidden_dims],
            output_dim=None,
            dropout=dropout,
            freeze=False,
        )
        self.tile_encoder = BloodFoundationExtractor(
            input_dim=tile_input_dim,
            hidden_dims=[int(x) for x in tile_hidden_dims],
            output_dim=None,
            dropout=dropout,
            freeze=False,
        )
        tile_feat_dim = int(self.tile_encoder.output_dim)
        self.tile_score = nn.Linear(tile_feat_dim, 1)
        self.blood_head = ProjectionHead(
            input_dim=int(self.blood_encoder.output_dim),
            hidden_dim=projection_hidden_dim,
            output_dim=projection_dim,
            dropout=dropout,
        )
        self.tissue_head = ProjectionHead(
            input_dim=tile_feat_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=projection_dim,
            dropout=dropout,
        )

    def _aggregate_tiles(self, tiles: torch.Tensor, tile_mask: torch.Tensor) -> torch.Tensor:
        batch_size, max_tiles, tile_dim = tiles.shape
        flat_tiles = tiles.reshape(batch_size * max_tiles, tile_dim)
        flat_feat = self.tile_encoder(flat_tiles)
        tile_feat = flat_feat.reshape(batch_size, max_tiles, -1)

        scores = self.tile_score(tile_feat).squeeze(-1)
        scores = scores.masked_fill(~tile_mask, -1e4)
        weights = torch.softmax(scores, dim=1)
        weights = torch.where(tile_mask, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (weights.unsqueeze(-1) * tile_feat).sum(dim=1)

    def forward(self, blood: torch.Tensor, tiles: torch.Tensor, tile_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        blood_feat = self.blood_encoder(blood)
        tissue_feat = self._aggregate_tiles(tiles, tile_mask)
        blood_proj = F.normalize(self.blood_head(blood_feat), dim=-1)
        tissue_proj = F.normalize(self.tissue_head(tissue_feat), dim=-1)
        return {
            "blood_feat": blood_feat,
            "tissue_feat": tissue_feat,
            "blood_proj": blood_proj,
            "tissue_proj": tissue_proj,
        }


def _batch_similarity_gap(tissue_proj: torch.Tensor, blood_proj: torch.Tensor, labels: torch.Tensor) -> float:
    sim = tissue_proj @ blood_proj.T
    pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
    neg_mask = ~pos_mask
    pos = sim[pos_mask]
    neg = sim[neg_mask]
    if pos.numel() == 0 or neg.numel() == 0:
        return 0.0
    return float((pos.mean() - neg.mean()).detach().cpu().item())


def _run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: SubtypeAwareContrastiveLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    feature_mode: str,
    grad_clip_norm: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_gap = 0.0
    total_batches = 0
    total_valid_anchors = 0.0

    for batch in loader:
        blood = batch["blood"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        if feature_mode == "slide":
            tissue = batch["tissue"].to(device, non_blocking=True)
            batch_out = model(blood, tissue)
        else:
            tiles = batch["tiles"].to(device, non_blocking=True)
            tile_mask = batch["tile_mask"].to(device, non_blocking=True)
            batch_out = model(blood, tiles, tile_mask)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            loss, stats = criterion(batch_out["tissue_proj"], batch_out["blood_proj"], labels)
            if is_train:
                loss.backward()
                if grad_clip_norm > 0:
                    clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        total_loss += float(loss.detach().cpu().item())
        total_gap += _batch_similarity_gap(batch_out["tissue_proj"], batch_out["blood_proj"], labels)
        total_valid_anchors += float(stats["valid_anchors"])
        total_batches += 1

    if total_batches == 0:
        return {"loss": 0.0, "similarity_gap": 0.0, "valid_anchors": 0.0}
    return {
        "loss": total_loss / total_batches,
        "similarity_gap": total_gap / total_batches,
        "valid_anchors": total_valid_anchors / total_batches,
    }


@torch.no_grad()
def _export_projected_embeddings(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    feature_mode: str,
    path: Path,
) -> None:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        blood = batch["blood"].to(device, non_blocking=True)
        labels = batch["label"].cpu().numpy()
        if feature_mode == "slide":
            tissue = batch["tissue"].to(device, non_blocking=True)
            out = model(blood, tissue)
            tile_counts = None
        else:
            tiles = batch["tiles"].to(device, non_blocking=True)
            tile_mask = batch["tile_mask"].to(device, non_blocking=True)
            out = model(blood, tiles, tile_mask)
            tile_counts = batch["tile_counts"].cpu().numpy()

        tissue_proj = out["tissue_proj"].cpu().numpy()
        blood_proj = out["blood_proj"].cpu().numpy()
        for i in range(tissue_proj.shape[0]):
            row: dict[str, Any] = {
                "blood_sample_id": batch["blood_sample_id"][i],
                "slide_id": batch["slide_id"][i],
                "label_text": batch["label_text"][i],
                "label_id": int(labels[i]),
            }
            if tile_counts is not None:
                row["tile_count"] = int(tile_counts[i])
            for d, value in enumerate(tissue_proj[i]):
                row[f"tissue_z_{d}"] = float(value)
            for d, value in enumerate(blood_proj[i]):
                row[f"blood_z_{d}"] = float(value)
            rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def train_feature_contrastive_from_config(config_path: Path) -> dict[str, Any]:
    config = MultiModalConfig.load(config_path)
    seed = int(config.get("train.seed", 42))
    _set_seed(seed)

    feature_mode = str(config.get("data.feature_mode", "slide")).strip().lower()
    if feature_mode not in {"slide", "tile"}:
        raise ValueError("data.feature_mode must be either 'slide' or 'tile'")

    blood_pt_path = resolve_data_path(
        config.get("data.blood_features_pt"),
        path_kind="cfDNA features PT",
        must_exist=True,
    )
    labels_path = resolve_data_path(
        config.get("data.labels_table"),
        path_kind="cross-modal labels table",
        must_exist=True,
    )
    if feature_mode == "slide":
        tissue_pt_path = resolve_data_path(
            config.get("data.tissue_slide_features_pt"),
            path_kind="slide feature PT",
            must_exist=True,
        )
    else:
        tissue_pt_path = resolve_data_path(
            config.get("data.tissue_tile_features_pt"),
            path_kind="tile feature PT",
            must_exist=True,
        )

    labels_columns = dict(
        config.get(
            "data.labels_columns",
            {
                "blood_sample_id": "blood_sample_id",
                "slide_id": "slide_id",
                "her2_label": "her2_status",
                "patient_id": "patient_id",
                "split": "split",
            },
        )
    )
    blood_id_col = str(labels_columns.get("blood_sample_id", "blood_sample_id"))
    slide_id_col = str(labels_columns.get("slide_id", "slide_id"))
    label_col = str(labels_columns.get("her2_label", "her2_status"))
    patient_col = labels_columns.get("patient_id", None)
    split_col = labels_columns.get("split", None)

    labels_df = _read_table(labels_path)
    required_cols = [blood_id_col, slide_id_col, label_col]
    missing_cols = [c for c in required_cols if c not in labels_df.columns]
    if missing_cols:
        raise ValueError(f"labels table missing required columns: {missing_cols}")

    labels_df = labels_df.dropna(subset=[blood_id_col, slide_id_col, label_col]).copy()
    labels_df[blood_id_col] = labels_df[blood_id_col].astype(str)
    labels_df[slide_id_col] = labels_df[slide_id_col].astype(str)
    labels_df[label_col] = labels_df[label_col].astype(str)
    if labels_df.empty:
        raise ValueError("No usable rows found in labels table after dropna.")

    label_values = sorted(labels_df[label_col].unique().tolist())
    label_to_id = {str(label): idx for idx, label in enumerate(label_values)}

    blood_map = _load_blood_feature_map(blood_pt_path)
    if feature_mode == "slide":
        tissue_map = _load_slide_feature_map(tissue_pt_path)
    else:
        tissue_map = _load_tile_feature_map(tissue_pt_path)

    val_ratio = float(config.get("data.val_ratio", 0.2))
    train_df, val_df = _split_df(
        labels_df,
        split_col=str(split_col) if split_col else None,
        patient_col=str(patient_col) if patient_col else None,
        val_ratio=val_ratio,
        seed=seed,
    )

    # TODO(superhy): after mounting data disk, verify Her2 label joins against final cohort curation table.
    if feature_mode == "slide":
        train_rows, train_drop = _build_slide_samples(
            train_df,
            blood_id_col=blood_id_col,
            slide_id_col=slide_id_col,
            label_col=label_col,
            label_to_id=label_to_id,
            blood_map=blood_map,
            slide_map=tissue_map,
        )
        val_rows, val_drop = _build_slide_samples(
            val_df,
            blood_id_col=blood_id_col,
            slide_id_col=slide_id_col,
            label_col=label_col,
            label_to_id=label_to_id,
            blood_map=blood_map,
            slide_map=tissue_map,
        )
        if not train_rows or not val_rows:
            raise ValueError(
                "No usable slide-level pairs after label-feature join. "
                "Check blood_sample_id/slide_id keys and labels table."
            )
        train_dataset = SlideFeatureDataset(train_rows)
        val_dataset = SlideFeatureDataset(val_rows)
        collate_fn = None
        blood_input_dim = int(train_rows[0].blood_vec.shape[0])
        tissue_input_dim = int(train_rows[0].tissue_vec.shape[0])
    else:
        train_rows, train_drop = _build_tile_samples(
            train_df,
            blood_id_col=blood_id_col,
            slide_id_col=slide_id_col,
            label_col=label_col,
            label_to_id=label_to_id,
            blood_map=blood_map,
            tile_map=tissue_map,
        )
        val_rows, val_drop = _build_tile_samples(
            val_df,
            blood_id_col=blood_id_col,
            slide_id_col=slide_id_col,
            label_col=label_col,
            label_to_id=label_to_id,
            blood_map=blood_map,
            tile_map=tissue_map,
        )
        if not train_rows or not val_rows:
            raise ValueError(
                "No usable tile-level pairs after label-feature join. "
                "Check blood_sample_id/slide_id keys and tile feature package."
            )
        train_dataset = TileFeatureDataset(train_rows)
        val_dataset = TileFeatureDataset(val_rows)
        collate_fn = _tile_collate_fn
        blood_input_dim = int(train_rows[0].blood_vec.shape[0])
        tissue_input_dim = int(train_rows[0].tile_vecs.shape[1])

    batch_size = int(config.get("train.batch_size", 32))
    num_workers = int(config.get("train.num_workers", 0))
    device = resolve_torch_device(str(config.get("train.device", "auto")))
    pin_memory = should_pin_memory(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    projection_dim = int(config.get("model.projection_dim", 128))
    projection_hidden_dim = int(config.get("model.projection_hidden_dim", 256))
    dropout = float(config.get("model.dropout", 0.1))
    blood_hidden_dims = list(config.get("model.blood_hidden_dims", [256]))
    tissue_hidden_dims = list(config.get("model.tissue_hidden_dims", [256]))
    tile_hidden_dims = list(config.get("model.tile_hidden_dims", [256]))

    if feature_mode == "slide":
        model = SlideFeatureContrastiveModel(
            blood_input_dim=blood_input_dim,
            tissue_input_dim=tissue_input_dim,
            projection_dim=projection_dim,
            projection_hidden_dim=projection_hidden_dim,
            blood_hidden_dims=blood_hidden_dims,
            tissue_hidden_dims=tissue_hidden_dims,
            dropout=dropout,
        )
    else:
        model = TileFeatureContrastiveModel(
            blood_input_dim=blood_input_dim,
            tile_input_dim=tissue_input_dim,
            projection_dim=projection_dim,
            projection_hidden_dim=projection_hidden_dim,
            blood_hidden_dims=blood_hidden_dims,
            tile_hidden_dims=tile_hidden_dims,
            dropout=dropout,
        )
    model.to(device)

    criterion = SubtypeAwareContrastiveLoss(
        temperature=float(config.get("train.temperature", 0.07)),
        symmetric=True,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config.get("train.lr", 1e-3)),
        weight_decay=float(config.get("train.weight_decay", 1e-4)),
    )

    epochs = int(config.get("train.epochs", 20))
    grad_clip_norm = float(config.get("train.grad_clip_norm", 1.0))
    output_dir = resolve_data_path(
        config.get("train.output_dir", "experiments/multimodal_feature/her2_contrastive"),
        path_kind="feature-contrastive output directory",
        must_exist=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Using data root: %s", get_data_root())
    LOGGER.info("Using torch device: %s", device)
    LOGGER.info("Feature mode: %s", feature_mode)
    LOGGER.info("Training outputs: %s", output_dir)

    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            feature_mode=feature_mode,
            grad_clip_norm=grad_clip_norm,
        )
        val_metrics = _run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            feature_mode=feature_mode,
            grad_clip_norm=0.0,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_similarity_gap": train_metrics["similarity_gap"],
            "val_similarity_gap": val_metrics["similarity_gap"],
            "train_valid_anchors": train_metrics["valid_anchors"],
            "val_valid_anchors": val_metrics["valid_anchors"],
        }
        history.append(row)
        LOGGER.info(
            "epoch=%03d train_loss=%.4f val_loss=%.4f val_gap=%.4f",
            epoch,
            row["train_loss"],
            row["val_loss"],
            row["val_similarity_gap"],
        )
        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    history_path = output_dir / "training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    _export_projected_embeddings(
        model=model,
        loader=train_loader,
        device=device,
        feature_mode=feature_mode,
        path=output_dir / "embeddings_train.parquet",
    )
    _export_projected_embeddings(
        model=model,
        loader=val_loader,
        device=device,
        feature_mode=feature_mode,
        path=output_dir / "embeddings_val.parquet",
    )

    label_map_path = output_dir / "label_map.json"
    label_map_path.write_text(json.dumps(label_to_id, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = {
        "feature_mode": feature_mode,
        "blood_features_pt": str(blood_pt_path),
        "tissue_features_pt": str(tissue_pt_path),
        "labels_table": str(labels_path),
        "labels_columns": {
            "blood_sample_id": blood_id_col,
            "slide_id": slide_id_col,
            "her2_label": label_col,
            "patient_id": patient_col,
            "split": split_col,
        },
        "train_rows_input": int(len(train_df)),
        "val_rows_input": int(len(val_df)),
        "train_pairs_kept": int(len(train_dataset)),
        "val_pairs_kept": int(len(val_dataset)),
        "train_rows_dropped": train_drop,
        "val_rows_dropped": val_drop,
        "blood_input_dim": int(blood_input_dim),
        "tissue_input_dim": int(tissue_input_dim),
        "projection_dim": int(projection_dim),
    }
    metadata_path = output_dir / "data_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    checkpoint_path = output_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mode": feature_mode,
            "model_config": config.get("model", {}),
            "train_config": config.get("train", {}),
            "data_config": config.get("data", {}),
            "metadata": metadata,
            "label_to_id": label_to_id,
            "best_val_loss": best_val_loss,
        },
        checkpoint_path,
    )

    summary = {
        "feature_mode": feature_mode,
        "best_val_loss": float(best_val_loss),
        "epochs": int(epochs),
        "device": str(device),
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
        "data_metadata": str(metadata_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
