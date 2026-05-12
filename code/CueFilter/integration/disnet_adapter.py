from __future__ import annotations

import torch

from ..Baseline.models import DisNetConfig, DisNetRegressor
from .common import TemporalBackboneAdapter


class DisNetAdapter(TemporalBackboneAdapter):
    """
    DisNet adapter exposing the HRE-derived temporal sequence after frequency
    aggregation and before the final temporal prediction head.
    """

    def __init__(self):
        super().__init__()
        self.backbone = DisNetRegressor(
            DisNetConfig(
                sample_rate=16000,
                n_fft=1024,
                win_length=400,
                hop_length=160,
                num_filters=128,
                embed_dim=256,
                depth=4,
                num_heads=8,
                temporal_hidden_dim=128,
            )
        )
        self.feature_dim = self.backbone.feature_dim

    def encode(self, audio: torch.Tensor):
        sequence_features, aux = self.backbone.encode(audio)
        return sequence_features, aux

    def predict_from_sequence(self, sequence_features: torch.Tensor) -> torch.Tensor:
        return self.backbone.predict_from_sequence(sequence_features)
