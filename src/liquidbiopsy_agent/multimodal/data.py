from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from liquidbiopsy_agent.utils.storage import ensure_within_data_root, resolve_data_path


def _read_table(path: Path) -> pd.DataFrame:
    suffixes = [s.lower() for s in path.suffixes]
    if ".parquet" in suffixes or path.suffix.lower() in {".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _resolve_path(path_value: str | Path, base_dir: Optional[Path] = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return ensure_within_data_root(path, path_kind="image path")
    if base_dir is not None:
        return ensure_within_data_root((base_dir / path).resolve(), path_kind="image path")
    return resolve_data_path(path, path_kind="image path", must_exist=False)


def _build_image_transform(image_size: int, augment: bool):
    try:
        from torchvision import transforms
    except ImportError as e:
        raise ImportError(
            "torchvision is required for image transforms. Install with pip install -e \".[multimodal]\""
        ) from e

    if augment:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@dataclass
class PairRow:
    patient_id: str
    blood_sample_id: str
    tissue_image_path: Path
    subtype_label: str


class TissueBloodPairDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[PairRow],
        blood_vectors: Mapping[str, np.ndarray],
        label_to_id: Mapping[str, int],
        image_size: int = 224,
        augment: bool = False,
    ) -> None:
        self.rows: List[PairRow] = list(rows)
        self.blood_vectors = blood_vectors
        self.label_to_id = label_to_id
        self.transform = _build_image_transform(image_size=image_size, augment=augment)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            from PIL import Image
        except ImportError as e:
            raise ImportError("Pillow is required for image loading.") from e

        row = self.rows[idx]
        with Image.open(row.tissue_image_path) as img:
            image = img.convert("RGB")
            image_tensor = self.transform(image)
        blood = torch.tensor(self.blood_vectors[row.blood_sample_id], dtype=torch.float32)
        label_id = self.label_to_id[row.subtype_label]
        return {
            "image": image_tensor,
            "blood": blood,
            "label": torch.tensor(label_id, dtype=torch.long),
            "label_text": row.subtype_label,
            "patient_id": row.patient_id,
            "blood_sample_id": row.blood_sample_id,
            "image_path": str(row.tissue_image_path),
        }


def _infer_blood_feature_columns(df: pd.DataFrame, id_col: str) -> List[str]:
    cols = []
    for col in df.columns:
        if col == id_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def _normalise_features(
    train_values: np.ndarray,
    full_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    normalised = (full_values - mean) / std
    return normalised, mean, std


def _build_pair_rows(
    pair_df: pd.DataFrame,
    pair_columns: Dict[str, str],
    image_base_dir: Optional[Path],
) -> List[PairRow]:
    patient_col = pair_columns["patient_id"]
    blood_col = pair_columns["blood_sample_id"]
    image_col = pair_columns["tissue_image_path"]
    label_col = pair_columns["subtype_label"]

    rows: List[PairRow] = []
    for _, r in pair_df.iterrows():
        image_path = _resolve_path(r[image_col], image_base_dir)
        rows.append(
            PairRow(
                patient_id=str(r[patient_col]),
                blood_sample_id=str(r[blood_col]),
                tissue_image_path=image_path,
                subtype_label=str(r[label_col]),
            )
        )
    return rows


def _split_pairs(
    pair_df: pd.DataFrame,
    patient_col: str,
    split_col: Optional[str],
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if split_col and split_col in pair_df.columns:
        train_df = pair_df[pair_df[split_col].astype(str).str.lower() == "train"].copy()
        val_df = pair_df[pair_df[split_col].astype(str).str.lower() == "val"].copy()
        if train_df.empty or val_df.empty:
            raise ValueError("When using split column, both train and val rows are required.")
        return train_df, val_df

    patients = pair_df[patient_col].astype(str).drop_duplicates().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    n_val = max(1, int(len(patients) * val_ratio))
    val_patients = set(patients[:n_val])
    is_val = pair_df[patient_col].astype(str).isin(val_patients)
    train_df = pair_df[~is_val].copy()
    val_df = pair_df[is_val].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("Split produced empty train/val set. Increase cohort size or adjust val_ratio.")
    return train_df, val_df


def build_dataloaders(
    pair_table_path: Path,
    blood_feature_table_path: Path,
    pair_columns: Dict[str, str],
    blood_id_col: str,
    blood_feature_cols: Optional[List[str]],
    batch_size: int,
    num_workers: int,
    image_size: int,
    val_ratio: float,
    seed: int,
    image_base_dir: Optional[Path] = None,
    split_col: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    pair_df = _read_table(pair_table_path)
    blood_df = _read_table(blood_feature_table_path)

    required_pair = {"patient_id", "blood_sample_id", "tissue_image_path", "subtype_label"}
    missing_keys = [k for k in required_pair if k not in pair_columns]
    if missing_keys:
        raise ValueError(f"pair_columns is missing keys: {missing_keys}")

    for key, col in pair_columns.items():
        if col not in pair_df.columns and key != "split":
            raise ValueError(f"Column '{col}' from pair_columns['{key}'] is not present in pair table")
    if blood_id_col not in blood_df.columns:
        raise ValueError(f"Blood id column '{blood_id_col}' not found in blood feature table")

    if not blood_feature_cols:
        blood_feature_cols = _infer_blood_feature_columns(blood_df, blood_id_col)
    if not blood_feature_cols:
        raise ValueError("No numeric blood features found. Provide blood_feature_cols explicitly.")

    train_df, val_df = _split_pairs(
        pair_df=pair_df,
        patient_col=pair_columns["patient_id"],
        split_col=split_col,
        val_ratio=val_ratio,
        seed=seed,
    )

    valid_blood = set(blood_df[blood_id_col].astype(str).tolist())
    for sub_df_name, sub_df in [("train", train_df), ("val", val_df)]:
        missing = ~sub_df[pair_columns["blood_sample_id"]].astype(str).isin(valid_blood)
        if missing.any():
            missing_ids = sorted(set(sub_df.loc[missing, pair_columns["blood_sample_id"]].astype(str).tolist()))
            raise ValueError(f"{sub_df_name} split has blood ids missing in feature table: {missing_ids[:10]}")

    train_blood_ids = set(train_df[pair_columns["blood_sample_id"]].astype(str).tolist())
    blood_numeric = blood_df[[blood_id_col] + blood_feature_cols].copy()
    blood_numeric[blood_id_col] = blood_numeric[blood_id_col].astype(str)
    blood_numeric[blood_feature_cols] = blood_numeric[blood_feature_cols].astype(np.float32)

    train_values = blood_numeric[blood_numeric[blood_id_col].isin(train_blood_ids)][blood_feature_cols].to_numpy()
    full_values = blood_numeric[blood_feature_cols].to_numpy()
    norm_values, mean, std = _normalise_features(train_values=train_values, full_values=full_values)
    blood_numeric.loc[:, blood_feature_cols] = norm_values

    blood_vectors: Dict[str, np.ndarray] = {}
    for _, row in blood_numeric.iterrows():
        blood_vectors[str(row[blood_id_col])] = row[blood_feature_cols].to_numpy(dtype=np.float32)

    all_labels = sorted(pair_df[pair_columns["subtype_label"]].astype(str).dropna().unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(all_labels)}

    train_rows = _build_pair_rows(train_df, pair_columns, image_base_dir)
    val_rows = _build_pair_rows(val_df, pair_columns, image_base_dir)

    for row in train_rows + val_rows:
        if not row.tissue_image_path.exists():
            raise FileNotFoundError(f"Tissue image path does not exist: {row.tissue_image_path}")

    train_dataset = TissueBloodPairDataset(
        rows=train_rows,
        blood_vectors=blood_vectors,
        label_to_id=label_to_id,
        image_size=image_size,
        augment=True,
    )
    val_dataset = TissueBloodPairDataset(
        rows=val_rows,
        blood_vectors=blood_vectors,
        label_to_id=label_to_id,
        image_size=image_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    metadata = {
        "blood_input_dim": len(blood_feature_cols),
        "label_to_id": label_to_id,
        "blood_feature_cols": blood_feature_cols,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "blood_normalisation": {
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
    }
    return train_loader, val_loader, metadata
