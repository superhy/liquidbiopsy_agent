from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def _directional_supervised_infonce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, int]:
    # logits: [batch, batch] between one modality and the other.
    same_class = labels.unsqueeze(1) == labels.unsqueeze(0)
    stabilised = logits - logits.max(dim=1, keepdim=True).values
    exp_logits = torch.exp(stabilised)
    pos_scores = (exp_logits * same_class).sum(dim=1)
    all_scores = exp_logits.sum(dim=1)
    valid = pos_scores > 0
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        return logits.new_zeros(()), 0
    loss = -torch.log((pos_scores[valid] + eps) / (all_scores[valid] + eps)).mean()
    return loss, valid_count


class SubtypeAwareContrastiveLoss(nn.Module):
    """Cross-modal supervised contrastive loss using molecular subtype labels."""

    def __init__(self, temperature: float = 0.07, symmetric: bool = True) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = temperature
        self.symmetric = symmetric

    def forward(
        self,
        tissue_embeddings: torch.Tensor,
        blood_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits = tissue_embeddings @ blood_embeddings.T / self.temperature
        tb_loss, tb_valid = _directional_supervised_infonce(logits, labels)
        bt_loss, bt_valid = _directional_supervised_infonce(logits.T, labels)

        if self.symmetric:
            valid_total = tb_valid + bt_valid
            if valid_total == 0:
                loss = logits.new_zeros(())
            elif tb_valid > 0 and bt_valid > 0:
                loss = 0.5 * (tb_loss + bt_loss)
            elif tb_valid > 0:
                loss = tb_loss
            else:
                loss = bt_loss
        else:
            loss = tb_loss
            valid_total = tb_valid

        stats = {
            "valid_anchors": float(valid_total),
            "loss_tb": float(tb_loss.detach().cpu().item()) if tb_valid > 0 else 0.0,
            "loss_bt": float(bt_loss.detach().cpu().item()) if bt_valid > 0 else 0.0,
        }
        return loss, stats
