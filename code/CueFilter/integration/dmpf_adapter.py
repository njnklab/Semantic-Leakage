from __future__ import annotations

import torch

from ..Baseline.models import DMPFConfig, DMPFRegressor
from .common import TemporalBackboneAdapter


class DMPFAdapter(TemporalBackboneAdapter):
    """
    DMPF adapter exposing the concatenated multi-perspective temporal sequence
    before perspective pooling and graph-attention fusion.
    """

    def __init__(self):
        super().__init__()
        self.backbone = DMPFRegressor(
            DMPFConfig(
                sample_rate=16000,
                common_time_steps=64,
                perspective_seq_dim=128,
                perspective_out_dim=256,
                common_dim=128,
                fusion_hidden_dim=64,
                fusion_out_dim=16,
            )
        )
        self.feature_dim = self.backbone.feature_dim

    def encode(self, audio: torch.Tensor):
        sequence_features, aux = self.backbone.encode(audio)
        return sequence_features, aux

    def predict_from_sequence(self, sequence_features: torch.Tensor) -> torch.Tensor:
        return self.backbone.predict_from_sequence(sequence_features)
