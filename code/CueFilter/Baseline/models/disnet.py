from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pad_to_multiple(x: torch.Tensor, mult_h: int, mult_w: int) -> torch.Tensor:
    pad_h = (mult_h - x.shape[-2] % mult_h) % mult_h
    pad_w = (mult_w - x.shape[-1] % mult_w) % mult_w
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h))


class LearnableFrequencyFilterbank(nn.Module):
    """
    A practical supervised LFB implementation inspired by DisNet.

    It applies Gaussian frequency-domain filters on the STFT power spectrum
    and a PCEN-style nonlinear transformation with learnable parameters.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        win_length: int = 400,
        hop_length: int = 160,
        num_filters: int = 128,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.num_filters = num_filters
        self.eps = eps
        self.num_bins = n_fft // 2 + 1

        self.register_buffer("window", torch.hann_window(win_length))
        self.register_buffer("bin_positions", torch.linspace(0.0, 1.0, self.num_bins))

        self.center_deltas = nn.Parameter(torch.zeros(num_filters))
        self.sigma_logits = nn.Parameter(torch.linspace(-2.2, -1.6, num_filters))
        self.gain_logits = nn.Parameter(torch.zeros(num_filters))
        self.exponent_logits = nn.Parameter(torch.zeros(num_filters))
        self.delta_logits = nn.Parameter(torch.zeros(num_filters))
        self.mu_logits = nn.Parameter(torch.full((num_filters,), -2.2))

    def _build_filterbank(self) -> torch.Tensor:
        deltas = F.softplus(self.center_deltas) + 1e-4
        centers = torch.cumsum(deltas, dim=0)
        centers = centers / centers[-1].clamp_min(1e-6)

        sigmas = 0.005 + 0.095 * torch.sigmoid(self.sigma_logits)
        distance = self.bin_positions.unsqueeze(0) - centers.unsqueeze(1)
        weights = torch.exp(-(distance.pow(2)) / (2.0 * sigmas.unsqueeze(1).pow(2)))
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return weights

    def _ema(self, x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        steps: List[torch.Tensor] = []
        prev = x[:, :, 0]
        steps.append(prev)
        for t in range(1, x.shape[-1]):
            prev = (1.0 - mu) * prev + mu * x[:, :, t]
            steps.append(prev)
        return torch.stack(steps, dim=-1)

    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        if audio.ndim != 3 or audio.shape[1] != 1:
            raise ValueError(f"LearnableFrequencyFilterbank expects (B, 1, T) audio, got {audio.shape}")

        waveform = audio.squeeze(1)
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(audio.device),
            center=True,
            return_complex=True,
        )
        power = spec.abs().pow(2).clamp_min(self.eps)

        filterbank = self._build_filterbank().to(audio.device)
        filtered = torch.einsum("kb,nbt->nkt", filterbank, power)

        mu = (0.001 + 0.499 * torch.sigmoid(self.mu_logits)).view(1, -1)
        smooth = self._ema(filtered, mu)
        gain = (0.5 + 0.5 * torch.sigmoid(self.gain_logits)).view(1, -1, 1)
        exponent = (0.3 + 0.7 * torch.sigmoid(self.exponent_logits)).view(1, -1, 1)
        delta = (1e-3 + F.softplus(self.delta_logits)).view(1, -1, 1)

        normalized = filtered / (self.eps + smooth).pow(gain)
        features = (normalized + delta).pow(exponent) - delta.pow(exponent)
        return {
            "lfb_features": torch.log1p(features.clamp_min(0.0)),
            "power_spectrum": power,
            "filterbank": filterbank,
        }


class HierarchicalExtractionBlock(nn.Module):
    """
    A lightweight HRE-inspired token block using self-attention, feed-forward
    transformation, and learned token sparsification.
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(round(embed_dim * mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.token_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_input = self.norm1(tokens)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))
        gate = torch.sigmoid(self.token_gate(self.norm3(tokens)))
        tokens = tokens * gate
        return tokens, gate.squeeze(-1)


@dataclass
class DisNetConfig:
    sample_rate: int = 16000
    n_fft: int = 1024
    win_length: int = 400
    hop_length: int = 160
    num_filters: int = 128
    stem_channels: int = 32
    downsample_stride: Tuple[int, int] = (2, 4)
    patch_size: Tuple[int, int] = (8, 8)
    embed_dim: int = 256
    depth: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    max_freq_patches: int = 32
    max_time_patches: int = 256
    temporal_hidden_dim: int = 128


class DisNetRegressor(nn.Module):
    """
    Supervised DisNet-style backbone for depression severity regression.

    The implementation keeps the paper's core ideas:
    1. learnable frequency-domain filterbank (LFB),
    2. patch-based hierarchical token extraction (HRE-inspired),
    3. interpretable patch gates and filterbank parameters.
    """

    def __init__(self, config: DisNetConfig | None = None):
        super().__init__()
        self.config = config or DisNetConfig()
        self.feature_dim = self.config.embed_dim

        self.lfb = LearnableFrequencyFilterbank(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            win_length=self.config.win_length,
            hop_length=self.config.hop_length,
            num_filters=self.config.num_filters,
        )

        self.stem = nn.Sequential(
            nn.Conv2d(1, self.config.stem_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.config.stem_channels),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=self.config.downsample_stride, stride=self.config.downsample_stride),
        )
        self.patch_embed = nn.Conv2d(
            self.config.stem_channels,
            self.config.embed_dim,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
        )
        self.freq_embed = nn.Parameter(
            torch.randn(self.config.max_freq_patches, self.config.embed_dim) * 0.02
        )
        self.time_embed = nn.Parameter(
            torch.randn(self.config.max_time_patches, self.config.embed_dim) * 0.02
        )
        self.hre_blocks = nn.ModuleList(
            [
                HierarchicalExtractionBlock(
                    embed_dim=self.config.embed_dim,
                    num_heads=self.config.num_heads,
                    mlp_ratio=self.config.mlp_ratio,
                    dropout=self.config.dropout,
                )
                for _ in range(self.config.depth)
            ]
        )
        self.sequence_norm = nn.LayerNorm(self.config.embed_dim)
        self.temporal_encoder = nn.GRU(
            input_size=self.config.embed_dim,
            hidden_size=self.config.temporal_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=self.config.dropout,
            bidirectional=True,
        )
        temporal_dim = self.config.temporal_hidden_dim * 2
        self.attention_pool = nn.Linear(temporal_dim, 1)
        self.regressor = nn.Sequential(
            nn.LayerNorm(temporal_dim),
            nn.Linear(temporal_dim, temporal_dim // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(temporal_dim // 2, 1),
        )

    def _tokenize(self, lfb_features: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = lfb_features.unsqueeze(1)
        x = self.stem(x)
        x = _pad_to_multiple(x, self.config.patch_size[0], self.config.patch_size[1])
        x = self.patch_embed(x)
        freq_patches, time_patches = x.shape[-2], x.shape[-1]
        if freq_patches > self.config.max_freq_patches or time_patches > self.config.max_time_patches:
            raise ValueError(
                "DisNet token grid exceeds positional embedding capacity: "
                f"freq_patches={freq_patches}, time_patches={time_patches}"
            )

        tokens = x.flatten(2).transpose(1, 2)
        pos = (
            self.freq_embed[:freq_patches].unsqueeze(1)
            + self.time_embed[:time_patches].unsqueeze(0)
        ).reshape(freq_patches * time_patches, self.config.embed_dim)
        tokens = tokens + pos.unsqueeze(0)
        return tokens, (freq_patches, time_patches)

    def encode(self, audio: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        lfb_outputs = self.lfb(audio)
        tokens, (freq_patches, time_patches) = self._tokenize(lfb_outputs["lfb_features"])

        gate_maps: List[torch.Tensor] = []
        for block in self.hre_blocks:
            tokens, gate = block(tokens)
            gate_maps.append(gate)

        batch_size = tokens.shape[0]
        token_grid = tokens.reshape(batch_size, freq_patches, time_patches, self.config.embed_dim)
        time_sequence = token_grid.mean(dim=1)
        time_sequence = self.sequence_norm(time_sequence)

        aux = {
            "lfb_features": lfb_outputs["lfb_features"],
            "filterbank": lfb_outputs["filterbank"],
            "token_grid_shape": torch.tensor([freq_patches, time_patches], device=audio.device),
        }
        if gate_maps:
            aux["patch_gates"] = torch.stack(gate_maps, dim=1)
        return time_sequence, aux

    def predict_from_sequence(self, time_sequence: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.temporal_encoder(time_sequence)
        attn = torch.softmax(self.attention_pool(encoded).squeeze(-1), dim=1)
        pooled = torch.sum(encoded * attn.unsqueeze(-1), dim=1)
        return self.regressor(pooled).squeeze(-1)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        time_sequence, _aux = self.encode(audio)
        return self.predict_from_sequence(time_sequence)
