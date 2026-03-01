from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_

from .config import MultiModalConfig
from .data import build_dataloaders
from .extractors import BloodFoundationExtractor, build_tissue_extractor
from .losses import SubtypeAwareContrastiveLoss
from .model import CrossModalModel, ProjectionHead
from liquidbiopsy_agent.utils.storage import get_data_root, resolve_data_path


LOGGER = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_str: str) -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_model(config: MultiModalConfig, blood_input_dim: int) -> CrossModalModel:
    tissue_backbone = str(config.get("model.tissue_backbone", "resnet18"))
    tissue_pretrained = bool(config.get("model.tissue_pretrained", True))
    freeze_tissue = bool(config.get("model.freeze_tissue_extractor", True))
    tissue_extractor, tissue_feature_dim = build_tissue_extractor(
        backbone=tissue_backbone,
        pretrained=tissue_pretrained,
        freeze=freeze_tissue,
    )

    blood_hidden_dims = config.get("model.blood_hidden_dims", [256])
    if not isinstance(blood_hidden_dims, list):
        raise ValueError("model.blood_hidden_dims must be a list of integers")
    blood_output_dim = config.get("model.blood_output_dim", None)
    freeze_blood = bool(config.get("model.freeze_blood_extractor", False))
    dropout = float(config.get("model.dropout", 0.1))

    blood_extractor = BloodFoundationExtractor(
        input_dim=blood_input_dim,
        hidden_dims=[int(x) for x in blood_hidden_dims],
        output_dim=int(blood_output_dim) if blood_output_dim is not None else None,
        dropout=dropout,
        freeze=freeze_blood,
    )
    blood_feature_dim = int(blood_extractor.output_dim)

    projection_dim = int(config.get("model.projection_dim", 128))
    projection_hidden = int(config.get("model.projection_hidden_dim", 256))
    tissue_head = ProjectionHead(
        input_dim=tissue_feature_dim,
        hidden_dim=projection_hidden,
        output_dim=projection_dim,
        dropout=dropout,
    )
    blood_head = ProjectionHead(
        input_dim=blood_feature_dim,
        hidden_dim=projection_hidden,
        output_dim=projection_dim,
        dropout=dropout,
    )
    return CrossModalModel(
        tissue_extractor=tissue_extractor,
        tissue_head=tissue_head,
        blood_extractor=blood_extractor,
        blood_head=blood_head,
    )


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
    model: CrossModalModel,
    loader,
    criterion: SubtypeAwareContrastiveLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    grad_clip_norm: float,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_gap = 0.0
    total_batches = 0
    total_valid_anchors = 0.0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        blood = batch["blood"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            out = model(images, blood)
            loss, stats = criterion(out["tissue_proj"], out["blood_proj"], labels)
            if is_train:
                loss.backward()
                if grad_clip_norm > 0:
                    clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        total_loss += float(loss.detach().cpu().item())
        total_gap += _batch_similarity_gap(out["tissue_proj"], out["blood_proj"], labels)
        total_valid_anchors += stats["valid_anchors"]
        total_batches += 1

    if total_batches == 0:
        return {"loss": 0.0, "similarity_gap": 0.0, "valid_anchors": 0.0}
    return {
        "loss": total_loss / total_batches,
        "similarity_gap": total_gap / total_batches,
        "valid_anchors": total_valid_anchors / total_batches,
    }


@torch.no_grad()
def _export_embeddings(model: CrossModalModel, loader, device: torch.device, path: Path) -> None:
    model.eval()
    rows = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        blood = batch["blood"].to(device, non_blocking=True)
        labels = batch["label"].cpu().numpy()
        out = model(images, blood)
        tissue = out["tissue_proj"].cpu().numpy()
        blood_z = out["blood_proj"].cpu().numpy()
        for i in range(tissue.shape[0]):
            row: Dict[str, Any] = {
                "patient_id": batch["patient_id"][i],
                "blood_sample_id": batch["blood_sample_id"][i],
                "label_text": batch["label_text"][i],
                "label_id": int(labels[i]),
            }
            for d, value in enumerate(tissue[i]):
                row[f"tissue_z_{d}"] = float(value)
            for d, value in enumerate(blood_z[i]):
                row[f"blood_z_{d}"] = float(value)
            rows.append(row)
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def train_from_config(config_path: Path) -> Dict[str, Any]:
    config = MultiModalConfig.load(config_path)
    seed = int(config.get("train.seed", 42))
    _set_seed(seed)

    pair_table_path = resolve_data_path(
        config.get("data.pair_table"),
        path_kind="pair table",
        must_exist=True,
    )
    blood_table_path = resolve_data_path(
        config.get("data.blood_feature_table"),
        path_kind="blood feature table",
        must_exist=True,
    )
    pair_columns = dict(
        config.get(
            "data.pair_columns",
            {
                "patient_id": "patient_id",
                "blood_sample_id": "blood_sample_id",
                "tissue_image_path": "tissue_image_path",
                "subtype_label": "subtype_label",
            },
        )
    )
    split_col = config.get("data.split_col", None)
    batch_size = int(config.get("data.batch_size", 16))
    num_workers = int(config.get("data.num_workers", 0))
    image_size = int(config.get("data.image_size", 224))
    val_ratio = float(config.get("data.val_ratio", 0.2))
    blood_id_col = str(config.get("data.blood_id_col", "sample_id"))
    blood_feature_cols = config.get("data.blood_feature_cols", None)
    image_base_dir = config.get("data.image_base_dir", None)
    image_base = (
        resolve_data_path(image_base_dir, path_kind="image base directory", must_exist=True)
        if image_base_dir
        else None
    )

    train_loader, val_loader, metadata = build_dataloaders(
        pair_table_path=pair_table_path,
        blood_feature_table_path=blood_table_path,
        pair_columns=pair_columns,
        blood_id_col=blood_id_col,
        blood_feature_cols=blood_feature_cols,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        val_ratio=val_ratio,
        seed=seed,
        image_base_dir=image_base,
        split_col=split_col,
    )

    model = _build_model(config, metadata["blood_input_dim"])
    device = _resolve_device(str(config.get("train.device", "auto")))
    model.to(device)

    temperature = float(config.get("train.temperature", 0.07))
    criterion = SubtypeAwareContrastiveLoss(temperature=temperature, symmetric=True)

    lr = float(config.get("train.lr", 1e-3))
    weight_decay = float(config.get("train.weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    epochs = int(config.get("train.epochs", 20))
    grad_clip_norm = float(config.get("train.grad_clip_norm", 1.0))
    output_dir = resolve_data_path(
        config.get("train.output_dir", "experiments/multimodal"),
        path_kind="training output directory",
        must_exist=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Using data root: %s", get_data_root())
    LOGGER.info("Training outputs will be written to: %s", output_dir)

    history = []
    best_state = None
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=grad_clip_norm,
        )
        val_metrics = _run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
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

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    history_df = pd.DataFrame(history)
    history_path = output_dir / "training_history.csv"
    history_df.to_csv(history_path, index=False)

    label_map_path = output_dir / "label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(metadata["label_to_id"], f, indent=2, sort_keys=True)

    meta_path = output_dir / "data_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    checkpoint_path = output_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": config.get("model", {}),
            "train_config": config.get("train", {}),
            "data_config": config.get("data", {}),
            "metadata": metadata,
            "best_val_loss": best_val_loss,
        },
        checkpoint_path,
    )

    _export_embeddings(model, train_loader, device, output_dir / "embeddings_train.parquet")
    _export_embeddings(model, val_loader, device, output_dir / "embeddings_val.parquet")

    summary = {
        "best_val_loss": best_val_loss,
        "epochs": epochs,
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
