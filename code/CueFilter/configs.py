"""
Configuration dataclasses for the CueFilter suppression front-end.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    """Shared data and featurization settings."""

    train_manifest_path: str = "data/train_manifest.json"
    val_manifest_path: str = "data/val_manifest.json"

    sample_rate: int = 16000
    frame_hop: int = 320
    frame_hop_sec: float = 0.02

    segment_frames: int = 1500
    segment_hop_frames: Optional[int] = None
    drop_last_segment: bool = True
    boundary_expand_frames: int = 2
    boundary_tolerance_sec: float = 1.0

    feature_source: str = "auto"  # auto | feature | audio
    n_mels: int = 64
    n_fft: int = 320
    win_length: int = 320

    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class CueFilterConfig:
    """CueFilter pretraining configuration."""

    input_dim: Optional[int] = None
    n_blocks: int = 2
    kernel_size: int = 5
    groups: int = 8

    alpha: float = 0.8
    gamma: float = 0.2
    cue_threshold: float = 0.5
    merge_gap_frames: int = 1
    renorm_eps: float = 1e-6

    lambda_d: float = 0.5
    lambda_b: float = 0.1

    lr: float = 5e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    max_epochs: int = 50
    patience: int = 10

    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "cuefilter_best.pt"


@dataclass
class CueFilterJointConfig(CueFilterConfig):
    """CueFilter + downstream model joint training configuration."""

    lambda_c: float = 0.5

    max_epochs: int = 80
    patience: int = 20
    checkpoint_name: str = "cuefilter_joint_best.pt"

    backbone_hidden_dim: int = 128
    backbone_dropout: float = 0.2
