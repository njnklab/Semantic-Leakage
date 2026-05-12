"""
Shared building blocks for CueFilter and reference downstream heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConvBlock(nn.Module):
    """Residual depthwise-separable temporal convolution block."""

    def __init__(self, channels: int, kernel_size: int = 5, groups: int = 8):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(groups, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x + residual


class AttentiveDepressionHead(nn.Module):
    """
    Lightweight reference regressor for joint training smoke tests.

    This head is not part of CueFilter itself; it simply provides a
    sequence-to-score backbone that accepts frame-level features.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, 1)
        self.out = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.proj(features)
        hidden = F.gelu(hidden)
        attn_logits = self.attn(hidden).squeeze(-1)
        attn = torch.softmax(attn_logits, dim=1)
        pooled = (attn.unsqueeze(-1) * hidden).sum(dim=1)
        return self.out(pooled).squeeze(-1)
