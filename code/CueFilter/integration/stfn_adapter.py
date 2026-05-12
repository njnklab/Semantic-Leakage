from __future__ import annotations

import torch

from ..Baseline.models import STFNModel
from .common import TemporalBackboneAdapter


class STFNAdapter(TemporalBackboneAdapter):
    """
    STFN adapter exposing the HCPC context sequence before the final
    prediction head.
    """

    def __init__(self):
        super().__init__()
        self.backbone = STFNModel(input_dim=1, dropout=0.3, prediction_steps=1)
        self.feature_dim = 256

    def encode(self, audio: torch.Tensor):
        x_vqwt = self.backbone.vqwtnet(audio)
        x_sf = self.backbone.sfnet(x_vqwt)
        context, predictions, targets = self.backbone.hcpcnet(x_sf)
        hcpc_loss = self.backbone.hcpcnet.compute_hcpc_loss(predictions, targets)
        return context, {"hcpc_loss": hcpc_loss}

    def predict_from_sequence(self, sequence_features: torch.Tensor) -> torch.Tensor:
        return self.backbone.prediction_net(sequence_features).squeeze(-1)
