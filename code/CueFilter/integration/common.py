from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class TemporalBackboneAdapter(nn.Module):
    """
    Standard adapter interface for attaching CueFilter to a temporal backbone.

    Each adapter exposes:
    1. a frame sequence before the backbone's temporal aggregation stage, and
    2. a prediction head that maps the filtered sequence back to a depression score.
    """

    feature_dim: int

    def encode(self, audio: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError

    def predict_from_sequence(self, sequence_features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        sequence_features, aux = self.encode(audio)
        y_hat = self.predict_from_sequence(sequence_features)
        if y_hat.ndim > 1 and y_hat.shape[-1] == 1:
            y_hat = y_hat.squeeze(-1)
        return {
            "sequence_features": sequence_features,
            "y_hat": y_hat,
            **aux,
        }


def project_spans_to_frames(
    cue_spans_sec: Sequence[Tuple[float, float]],
    num_frames: int,
    duration_sec: float,
    expansion_frames: int = 2,
    expansion_sec: Optional[float] = None,
) -> torch.Tensor:
    labels = torch.zeros(num_frames, dtype=torch.float32)
    if num_frames <= 0 or duration_sec <= 0:
        return labels

    effective_expansion_frames = max(0, int(expansion_frames))
    if expansion_sec is not None and expansion_sec > 0:
        frame_sec = float(duration_sec) / float(num_frames)
        sec_frames = int(math.floor(float(expansion_sec) / frame_sec + 1e-9))
        effective_expansion_frames = max(effective_expansion_frames, sec_frames)

    for start_sec, end_sec in cue_spans_sec:
        start_sec = max(0.0, float(start_sec))
        end_sec = min(float(duration_sec), float(end_sec))
        if end_sec <= start_sec:
            continue

        start_idx = int(start_sec / duration_sec * num_frames)
        end_idx = int(torch.ceil(torch.tensor(end_sec / duration_sec * num_frames)).item())
        start_idx = max(0, start_idx - effective_expansion_frames)
        end_idx = min(num_frames, end_idx + effective_expansion_frames)
        if end_idx > start_idx:
            labels[start_idx:end_idx] = 1.0

    return labels


def build_cue_supervision_batch(
    cue_spans_batch: Sequence[Sequence[Tuple[float, float]]],
    num_frames: int,
    durations_sec: Sequence[float],
    device: torch.device,
    expansion_frames: int = 2,
    expansion_sec: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = []
    coverages = []
    for spans, duration_sec in zip(cue_spans_batch, durations_sec):
        frame_labels = project_spans_to_frames(
            cue_spans_sec=spans,
            num_frames=num_frames,
            duration_sec=float(duration_sec),
            expansion_frames=expansion_frames,
            expansion_sec=expansion_sec,
        )
        labels.append(frame_labels)
        coverages.append(float(frame_labels.mean().item()) if num_frames > 0 else 0.0)

    cue_labels = torch.stack(labels, dim=0).to(device)
    cue_coverage = torch.tensor(coverages, dtype=torch.float32, device=device)
    return cue_labels, cue_coverage


def estimate_sequence_feature_stats(
    adapter: TemporalBackboneAdapter,
    loader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    adapter.eval()
    total_sum = None
    total_sq_sum = None
    total_frames = 0

    with torch.no_grad():
        for batch in loader:
            audios = batch["audio"].to(device, non_blocking=True)
            sequence_features, _ = adapter.encode(audios)
            batch_sum = sequence_features.sum(dim=(0, 1)).double()
            batch_sq_sum = sequence_features.pow(2).sum(dim=(0, 1)).double()

            if total_sum is None:
                total_sum = batch_sum
                total_sq_sum = batch_sq_sum
            else:
                total_sum += batch_sum
                total_sq_sum += batch_sq_sum
            total_frames += sequence_features.shape[0] * sequence_features.shape[1]

    if total_sum is None or total_sq_sum is None or total_frames == 0:
        raise ValueError("Unable to estimate feature statistics: no training frames were observed.")

    mean = total_sum / total_frames
    var = total_sq_sum / total_frames - mean.pow(2)
    std = var.clamp_min(1e-6).sqrt()
    return mean.float().cpu(), std.float().cpu()


def merge_metrics_rows(rows: List[Dict[str, object]]) -> str:
    import pandas as pd

    return pd.DataFrame(rows).to_markdown(index=False)
