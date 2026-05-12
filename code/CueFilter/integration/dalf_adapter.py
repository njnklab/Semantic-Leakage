from __future__ import annotations

import torch

from ..Baseline.models import DALFConfig, DALFNet
from .common import TemporalBackboneAdapter


class DALFAdapter(TemporalBackboneAdapter):
    """
    DALF adapter exposing the temporal hidden sequence immediately before
    global temporal averaging.
    """

    def __init__(self):
        super().__init__()
        self.backbone = DALFNet(
            DALFConfig(
                sample_rate=16000,
                num_filters=64,
                gabor_kernel=401,
                pool_kernel=401,
                pool_stride=160,
                meb_blocks=4,
                mssa_hidden=128,
                head_channels=128,
                dropout=0.2,
            )
        )
        self.feature_dim = 128

    def encode(self, audio: torch.Tensor):
        x = self.backbone.dfbl(audio)
        _h, skips = self.backbone.mssa(x)
        x_fuse = torch.stack(skips, dim=0).sum(dim=0)
        x_att = self.backbone.fa(x_fuse)
        z = self.backbone.head_pre(x_att)
        z = self.backbone.res1(z)
        z = self.backbone.res2(z)
        z = self.backbone.res3(z)
        sequence_features = z.transpose(1, 2)
        return sequence_features, {}

    def predict_from_sequence(self, sequence_features: torch.Tensor) -> torch.Tensor:
        z = sequence_features.transpose(1, 2)
        z = z.mean(dim=2)
        z = self.backbone.dropout(z)
        return self.backbone.out(z).squeeze(-1)
