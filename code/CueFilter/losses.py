"""
Loss functions for CueFilter pretraining and joint optimization.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for frame-level cue supervision under class imbalance."""

    def __init__(self, eps: float = 1.0):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        inter = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2 * inter + self.eps) / (union + self.eps)
        return 1.0 - dice


def cue_supervision_loss(
    p_cue: torch.Tensor,
    cue_labels: torch.Tensor,
    lambda_d: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """BCE + Dice supervision for frame-level cue probabilities."""
    bce = F.binary_cross_entropy(p_cue, cue_labels)
    dice = DiceLoss()(p_cue, cue_labels)
    total = bce + lambda_d * dice
    return total, {
        "cue_bce": bce.detach(),
        "cue_dice": dice.detach(),
        "cue_total": total.detach(),
    }


def budget_regularization(
    p_cue: torch.Tensor,
    cue_coverage: torch.Tensor,
) -> torch.Tensor:
    """Match average cue activation density to the empirical cue coverage ratio."""
    predicted_coverage = p_cue.mean(dim=1)
    return torch.abs(predicted_coverage - cue_coverage).mean()


def cuefilter_pretrain_loss(
    p_cue: torch.Tensor,
    cue_labels: torch.Tensor,
    cue_coverage: torch.Tensor,
    lambda_d: float = 0.5,
    lambda_b: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Stage-1 objective: cue supervision plus budget regularization."""
    cue_loss, metrics = cue_supervision_loss(p_cue, cue_labels, lambda_d=lambda_d)
    budget = budget_regularization(p_cue, cue_coverage)
    total = cue_loss + lambda_b * budget
    metrics.update(
        {
            "budget": budget.detach(),
            "loss": total.detach(),
        }
    )
    return total, metrics


def cuefilter_joint_loss(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    p_cue: torch.Tensor,
    cue_labels: torch.Tensor,
    cue_coverage: torch.Tensor,
    lambda_c: float = 0.5,
    lambda_d: float = 0.5,
    lambda_b: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Stage-2 objective: regression + cue supervision + budget regularization."""
    dep = F.mse_loss(y_hat, y_true)
    cue_loss, metrics = cue_supervision_loss(p_cue, cue_labels, lambda_d=lambda_d)
    budget = budget_regularization(p_cue, cue_coverage)
    total = dep + lambda_c * cue_loss + lambda_b * budget
    metrics.update(
        {
            "dep_mse": dep.detach(),
            "budget": budget.detach(),
            "loss": total.detach(),
        }
    )
    return total, metrics
