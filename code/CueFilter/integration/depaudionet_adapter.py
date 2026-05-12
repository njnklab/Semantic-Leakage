from __future__ import annotations

import torch

from ..Baseline.models import DepAudioNetBackbone
from .common import TemporalBackboneAdapter


class DepAudioNetAdapter(TemporalBackboneAdapter):
    """
    Raw-audio DepAudioNet adapter with a clean temporal insertion point
    between the convolutional stack and the recurrent aggregation stage.
    """

    def __init__(self):
        super().__init__()
        self.backbone = DepAudioNetBackbone()
        self.feature_dim = self.backbone.feature_dim

    def encode(self, audio: torch.Tensor):
        return self.backbone.encode(audio), {}

    def predict_from_sequence(self, sequence_features: torch.Tensor) -> torch.Tensor:
        return self.backbone.predict_from_sequence(sequence_features)
