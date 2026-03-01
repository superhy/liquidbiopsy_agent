from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossModalModel(nn.Module):
    def __init__(
        self,
        tissue_extractor: nn.Module,
        tissue_head: nn.Module,
        blood_extractor: nn.Module,
        blood_head: nn.Module,
    ) -> None:
        super().__init__()
        self.tissue_extractor = tissue_extractor
        self.tissue_head = tissue_head
        self.blood_extractor = blood_extractor
        self.blood_head = blood_head

    def forward(self, image: torch.Tensor, blood: torch.Tensor) -> Dict[str, torch.Tensor]:
        tissue_feat = self.tissue_extractor(image)
        blood_feat = self.blood_extractor(blood)
        tissue_proj = F.normalize(self.tissue_head(tissue_feat), dim=-1)
        blood_proj = F.normalize(self.blood_head(blood_feat), dim=-1)
        return {
            "tissue_feat": tissue_feat,
            "blood_feat": blood_feat,
            "tissue_proj": tissue_proj,
            "blood_proj": blood_proj,
        }
