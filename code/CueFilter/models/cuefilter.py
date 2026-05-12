"""
CueFilter: a plug-and-play cue suppression front-end.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .components import DepthwiseSeparableConvBlock


class CueFilter(nn.Module):
    """
    Lightweight temporal scorer that predicts frame-level cue probabilities
    and suppresses cue-dominated regions in feature space.
    """

    def __init__(
        self,
        input_dim: int,
        n_blocks: int = 2,
        kernel_size: int = 5,
        groups: int = 8,
        alpha: float = 0.8,
        gamma: float = 0.2,
        cue_threshold: float = 0.5,
        renorm_eps: float = 1e-6,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.alpha = alpha
        self.gamma = gamma
        self.cue_threshold = cue_threshold
        self.renorm_eps = renorm_eps

        self.blocks = nn.ModuleList(
            [
                DepthwiseSeparableConvBlock(
                    channels=input_dim,
                    kernel_size=kernel_size,
                    groups=groups,
                )
                for _ in range(n_blocks)
            ]
        )
        self.scoring_head = nn.Linear(input_dim, 1)

        self.register_buffer("train_mean", torch.zeros(input_dim))
        self.register_buffer("train_std", torch.ones(input_dim))

    def set_feature_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Store training-set channel statistics for post-filter renormalization."""
        if mean.ndim != 1 or std.ndim != 1:
            raise ValueError("CueFilter feature stats must be 1D tensors")
        if mean.shape[0] != self.input_dim or std.shape[0] != self.input_dim:
            raise ValueError("CueFilter feature stats must match input_dim")
        self.train_mean = mean.detach().clone()
        self.train_std = std.detach().clone().clamp_min(self.renorm_eps)

    def compute_gate(
        self,
        p_cue: torch.Tensor,
        mode: str = "soft",
        alpha: Optional[float] = None,
        gamma: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> torch.Tensor:
        alpha = self.alpha if alpha is None else alpha
        gamma = self.gamma if gamma is None else gamma
        threshold = self.cue_threshold if threshold is None else threshold

        if mode == "soft":
            return torch.clamp(1.0 - alpha * p_cue, min=gamma, max=1.0)
        if mode == "binary":
            return torch.where(
                p_cue >= threshold,
                torch.full_like(p_cue, gamma),
                torch.ones_like(p_cue),
            )
        if mode == "identity":
            return torch.ones_like(p_cue)
        raise ValueError(f"Unsupported CueFilter inference mode: {mode}")

    def renormalize(self, filtered_features: torch.Tensor) -> torch.Tensor:
        """Match sample-wise channel statistics to the training-set feature scale."""
        sample_mean = filtered_features.mean(dim=1, keepdim=True)
        sample_std = filtered_features.std(dim=1, unbiased=False, keepdim=True)
        sample_std = sample_std.clamp_min(self.renorm_eps)

        target_mean = self.train_mean.to(filtered_features).view(1, 1, -1)
        target_std = self.train_std.to(filtered_features).view(1, 1, -1)
        return (filtered_features - sample_mean) / sample_std * target_std + target_mean

    def forward(
        self,
        features: torch.Tensor,
        mode: str = "soft",
        alpha: Optional[float] = None,
        gamma: Optional[float] = None,
        threshold: Optional[float] = None,
        renorm: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, F, D) frame sequence
            mode: soft | binary | identity

        Returns:
            Dict with cue probabilities, gate values, and filtered features.
        """
        if features.ndim != 3:
            raise ValueError(f"CueFilter expects (B, F, D) input, got {features.shape}")
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"CueFilter input_dim mismatch: expected {self.input_dim}, got {features.shape[-1]}"
            )

        x = features.transpose(1, 2)
        for block in self.blocks:
            x = block(x)

        context = x.transpose(1, 2)
        cue_logits = self.scoring_head(context).squeeze(-1)
        p_cue = torch.sigmoid(cue_logits)

        gate = self.compute_gate(
            p_cue,
            mode=mode,
            alpha=alpha,
            gamma=gamma,
            threshold=threshold,
        )
        filtered_features = gate.unsqueeze(-1) * features
        output_features = self.renormalize(filtered_features) if renorm else filtered_features

        return {
            "context_features": context,
            "cue_logits": cue_logits,
            "p_cue": p_cue,
            "gate": gate,
            "filtered_features": filtered_features,
            "renormed_features": output_features,
        }


class CueFilterBackbone(nn.Module):
    """Attach CueFilter in front of an arbitrary sequence-to-score backbone."""

    def __init__(self, cuefilter: CueFilter, backbone: nn.Module):
        super().__init__()
        self.cuefilter = cuefilter
        self.backbone = backbone

    def forward(self, features: torch.Tensor, **cuefilter_kwargs) -> Dict[str, torch.Tensor]:
        outputs = self.cuefilter(features, **cuefilter_kwargs)
        backbone_output = self.backbone(outputs["renormed_features"])

        if isinstance(backbone_output, dict):
            y_hat = backbone_output.get("y_hat")
            if y_hat is None:
                raise ValueError("Backbone dict outputs must include 'y_hat'")
            result = dict(backbone_output)
        else:
            y_hat = backbone_output
            result = {}

        if y_hat.ndim > 1 and y_hat.shape[-1] == 1:
            y_hat = y_hat.squeeze(-1)

        result["y_hat"] = y_hat
        result.update(outputs)
        return result
