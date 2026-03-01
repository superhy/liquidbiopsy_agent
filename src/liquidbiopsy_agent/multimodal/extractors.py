from __future__ import annotations

from typing import Sequence, Tuple

import torch.nn as nn


def _freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def build_tissue_extractor(
    backbone: str = "resnet18",
    pretrained: bool = True,
    freeze: bool = True,
) -> Tuple[nn.Module, int]:
    try:
        from torchvision import models
    except ImportError as e:
        raise ImportError(
            "torchvision is required for tissue feature extraction. Install with pip install -e \".[multimodal]\""
        ) from e

    backbone = backbone.lower()
    if backbone == "resnet18":
        model = _load_resnet(models.resnet18, models.ResNet18_Weights.DEFAULT if pretrained else None)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif backbone == "resnet50":
        model = _load_resnet(models.resnet50, models.ResNet50_Weights.DEFAULT if pretrained else None)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif backbone == "efficientnet_b0":
        model = _load_efficientnet(
            models.efficientnet_b0,
            models.EfficientNet_B0_Weights.DEFAULT if pretrained else None,
        )
        feature_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unsupported tissue backbone: {backbone}")

    if freeze:
        _freeze_module(model)
    return model, feature_dim


def _load_resnet(builder, weights):
    try:
        return builder(weights=weights)
    except Exception:
        return builder(weights=None)


def _load_efficientnet(builder, weights):
    try:
        return builder(weights=weights)
    except Exception:
        return builder(weights=None)


class BloodFoundationExtractor(nn.Module):
    """Feature extractor for blood modality.

    This is intentionally separate from the projection head so the model keeps the
    two-stage structure: foundation features -> trainable encoder head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] | None = None,
        output_dim: int | None = None,
        dropout: float = 0.1,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims or [])
        dims = [input_dim, *hidden_dims]
        if output_dim is not None:
            dims.append(output_dim)

        if len(dims) <= 1:
            self.net = nn.Identity()
            self.output_dim = input_dim
        else:
            layers = []
            for in_dim, out_dim in zip(dims[:-1], dims[1:]):
                layers.append(nn.Linear(in_dim, out_dim))
                if out_dim != dims[-1]:
                    layers.append(nn.GELU())
                    layers.append(nn.Dropout(dropout))
            self.net = nn.Sequential(*layers)
            self.output_dim = dims[-1]

        if freeze:
            _freeze_module(self)

    def forward(self, x):
        return self.net(x)
