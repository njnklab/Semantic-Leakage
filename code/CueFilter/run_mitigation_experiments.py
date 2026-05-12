"""
CueFilter plug-and-play mitigation experiments.

This runner trains:
1. a baseline temporal backbone on the preprocessed participant speech,
2. a CueFilter with frozen backbone features (stage 1), and
3. a jointly optimized CueFilter + backbone model (stage 2).

It then reports baseline-vs-CueFilter performance on:
    pre / cue-excluded / random
and summarizes leakage-sensitive gaps before and after mitigation.
"""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .Baseline.audio_views import DATASET_CONFIGS
    from .Baseline.experiment_utils import (
        RecordingSplitDataManager,
        finalize_predictions,
        format_mean_std,
        safe_r2_score,
        set_seed,
    )
    from .integration.common import build_cue_supervision_batch, estimate_sequence_feature_stats
    from .integration.data import CueAwareAudioDataset, CueAwarePreDataManager, cueaware_collate_fn
    from .integration.registry import build_adapter, list_supported_adapters
    from .losses import cuefilter_joint_loss, cuefilter_pretrain_loss
    from .models import CueFilter
    from .evaluate import _cue_metrics_from_arrays, cue_curve_points_from_arrays, extract_spans, match_spans
except ImportError:
    from CueFilter.Baseline.audio_views import DATASET_CONFIGS
    from CueFilter.Baseline.experiment_utils import (
        RecordingSplitDataManager,
        finalize_predictions,
        format_mean_std,
        safe_r2_score,
        set_seed,
    )
    from CueFilter.integration.common import build_cue_supervision_batch, estimate_sequence_feature_stats
    from CueFilter.integration.data import CueAwareAudioDataset, CueAwarePreDataManager, cueaware_collate_fn
    from CueFilter.integration.registry import build_adapter, list_supported_adapters
    from CueFilter.losses import cuefilter_joint_loss, cuefilter_pretrain_loss
    from CueFilter.models import CueFilter
    from CueFilter.evaluate import _cue_metrics_from_arrays, cue_curve_points_from_arrays, extract_spans, match_spans


@dataclass
class MitigationRunOutput:
    base_pre_mae: float
    base_pre_rmse: float
    base_pre_r2: float
    base_exc_mae: float
    base_rand_mae: float
    cf_pre_mae: float
    cf_pre_rmse: float
    cf_pre_r2: float
    cf_exc_mae: float
    cf_rand_mae: float
    n_before: float
    n_after: float
    g_before: float
    g_after: float
    stage1_frame_precision: float
    stage1_frame_recall: float
    stage1_frame_f1: float
    stage1_auc_roc: float
    stage1_auc_pr: float
    stage1_span_f1: float
    stage1_budget_mae: float
    stage2_frame_precision: float
    stage2_frame_recall: float
    stage2_frame_f1: float
    stage2_auc_roc: float
    stage2_auc_pr: float
    stage2_span_f1: float
    stage2_budget_mae: float


MITIGATION_DEFAULTS: Dict[str, Dict[str, float]] = {
    "DepAudioNet": {
        "batch_size": 8,
        "baseline_epochs": 80,
        "stage1_epochs": 30,
        "stage2_epochs": 60,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "baseline_patience": 12,
        "stage1_patience": 8,
        "stage2_patience": 12,
    },
    "DisNet": {
        "batch_size": 2,
        "baseline_epochs": 100,
        "stage1_epochs": 30,
        "stage2_epochs": 60,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "baseline_patience": 15,
        "stage1_patience": 8,
        "stage2_patience": 12,
    },
    "DMPF": {
        "batch_size": 4,
        "baseline_epochs": 60,
        "stage1_epochs": 25,
        "stage2_epochs": 50,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "baseline_patience": 10,
        "stage1_patience": 8,
        "stage2_patience": 10,
    },
    "DALF": {
        "batch_size": 8,
        "baseline_epochs": 80,
        "stage1_epochs": 30,
        "stage2_epochs": 60,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "baseline_patience": 12,
        "stage1_patience": 8,
        "stage2_patience": 12,
    },
    "STFN": {
        "batch_size": 4,
        "baseline_epochs": 80,
        "stage1_epochs": 30,
        "stage2_epochs": 60,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "baseline_patience": 12,
        "stage1_patience": 8,
        "stage2_patience": 12,
    },
}


def _with_model_defaults(args, model_name: str):
    effective = argparse.Namespace(**vars(args))
    defaults = MITIGATION_DEFAULTS.get(model_name, {})
    for key, value in defaults.items():
        if getattr(effective, key, None) is None:
            setattr(effective, key, value)
    return effective


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CueFilter mitigation experiments.")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_CONFIGS.keys()), default=["edaic"])
    parser.add_argument("--models", nargs="+", choices=list_supported_adapters(), default=list_supported_adapters())
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--cue-role", choices=["patient", "doctor", "all"], default="patient")
    parser.add_argument("--speech-scope", choices=["participant", "interviewer", "dialogue"], default="participant")
    parser.add_argument(
        "--shared-pretrain-datasets",
        nargs="+",
        choices=list(DATASET_CONFIGS.keys()),
        default=None,
        help="Optional pooled datasets used to train a shared stage-1 CueFilter prior.",
    )
    parser.add_argument(
        "--load-shared-cuefilter",
        type=str,
        default=None,
        help="Optional checkpoint path or template (supports {model} and {seed}) for loading shared CueFilter weights.",
    )
    parser.add_argument(
        "--save-shared-cuefilter",
        type=str,
        default=None,
        help="Optional checkpoint path or template (supports {model} and {seed}) for saving shared CueFilter weights.",
    )
    parser.add_argument("--segment-length", type=float, default=30.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--baseline-epochs", type=int, default=None)
    parser.add_argument("--stage1-epochs", type=int, default=None)
    parser.add_argument("--stage2-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--baseline-patience", type=int, default=None)
    parser.add_argument("--stage1-patience", type=int, default=None)
    parser.add_argument("--stage2-patience", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hcpc-weight", type=float, default=0.1)
    parser.add_argument("--lambda-c", type=float, default=0.5)
    parser.add_argument("--lambda-d", type=float, default=0.5)
    parser.add_argument("--lambda-b", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--cue-threshold", type=float, default=0.5)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--boundary-expand-frames", type=int, default=2)
    parser.add_argument(
        "--boundary-tolerance-sec",
        type=float,
        default=1.0,
        help="Boundary tolerance converted to each backbone frame grid; uses the largest whole-frame expansion within this value before normal grid quantization.",
    )
    parser.add_argument("--max-groups-per-split", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval-level", choices=["recording", "sample"], default="recording")
    parser.add_argument("--segment-first", action="store_true", help="Segment pre audio first, then apply cue removal within each 30-s sample.")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--curve-output-dir", type=str, default=None)
    parser.add_argument("--detail-output-dir", type=str, default=None)
    parser.add_argument("--info-only", action="store_true")
    return parser


def _compute_regression_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(labels - preds)))
    rmse = float(np.sqrt(np.mean((labels - preds) ** 2)))
    r2 = float(safe_r2_score(labels, preds))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def _forward_with_optional_cuefilter(
    adapter,
    audios: torch.Tensor,
    cuefilter: Optional[CueFilter] = None,
    gate_mode: str = "soft",
    use_renorm: bool = True,
):
    sequence_features, aux = adapter.encode(audios)
    cue_outputs = None
    pred_input = sequence_features
    if cuefilter is not None:
        cue_outputs = cuefilter(sequence_features, mode=gate_mode, renorm=use_renorm)
        pred_input = cue_outputs["renormed_features"]
    y_hat = adapter.predict_from_sequence(pred_input)
    if y_hat.ndim > 1 and y_hat.shape[-1] == 1:
        y_hat = y_hat.squeeze(-1)
    return y_hat, sequence_features, cue_outputs, aux


def _evaluate_variant(
    adapter,
    data_manager,
    device,
    cuefilter: Optional[CueFilter] = None,
    gate_mode: str = "soft",
    use_renorm: bool = True,
    return_predictions: bool = False,
    condition: Optional[str] = None,
) -> Dict[str, float]:
    loader = data_manager.get_loader("test", shuffle=False)
    adapter.eval()
    if cuefilter is not None:
        cuefilter.eval()

    preds_scaled: List[float] = []
    labels_scaled: List[float] = []
    sample_ids: List[str] = []
    rows: List[Dict[str, object]] = []

    with torch.no_grad():
        row_offset = 0
        for audios, labels, batch_sample_ids in loader:
            audios = audios.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).squeeze(-1)
            y_hat, _seq, _cue, _aux = _forward_with_optional_cuefilter(
                adapter,
                audios,
                cuefilter=cuefilter,
                gate_mode=gate_mode,
                use_renorm=use_renorm,
            )
            batch_preds = y_hat.cpu().numpy().tolist()
            batch_labels = labels.cpu().numpy().tolist()
            preds_scaled.extend(batch_preds)
            labels_scaled.extend(batch_labels)
            sample_ids.extend(list(batch_sample_ids))
            if return_predictions:
                batch_pred_raw = data_manager.inverse_transform(np.asarray(batch_preds, dtype=np.float32))
                batch_label_raw = data_manager.inverse_transform(np.asarray(batch_labels, dtype=np.float32))
                for local_idx, sample_id in enumerate(batch_sample_ids):
                    rows.append(
                        {
                            "condition": condition,
                            "sample_id": str(sample_id),
                            "segment_row": row_offset + local_idx,
                            "y_true": float(batch_label_raw[local_idx]),
                            "y_pred": float(batch_pred_raw[local_idx]),
                            "abs_error": float(abs(batch_label_raw[local_idx] - batch_pred_raw[local_idx])),
                        }
                    )
            row_offset += len(batch_sample_ids)

    preds_raw = data_manager.inverse_transform(np.asarray(preds_scaled))
    labels_raw = data_manager.inverse_transform(np.asarray(labels_scaled))
    preds_eval, labels_eval = finalize_predictions(preds_raw, labels_raw, sample_ids, eval_level=data_manager.eval_level)
    metrics = _compute_regression_metrics(preds_eval, labels_eval)
    if return_predictions:
        return metrics, rows
    return metrics


def _collect_stage1_arrays(
    cuefilter,
    adapter,
    loader,
    device,
    expansion_frames: int,
    expansion_sec: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    cuefilter.eval()
    adapter.eval()
    all_preds = []
    all_labels = []
    all_coverages = []
    all_gates = []
    sample_ids: List[str] = []
    durations: List[float] = []
    cue_spans: List[List[tuple]] = []

    with torch.no_grad():
        for batch in loader:
            audios = batch["audio"].to(device, non_blocking=True)
            sequence_features, _aux = adapter.encode(audios)
            cue_labels, cue_coverage = build_cue_supervision_batch(
                cue_spans_batch=batch["cue_spans_sec"],
                num_frames=sequence_features.shape[1],
                durations_sec=batch["duration_sec"],
                device=device,
                expansion_frames=expansion_frames,
                expansion_sec=expansion_sec,
            )
            outputs = cuefilter(sequence_features, mode="soft")
            all_preds.append(outputs["p_cue"].detach().cpu().numpy())
            all_labels.append(cue_labels.detach().cpu().numpy())
            all_coverages.append(cue_coverage.detach().cpu().numpy())
            all_gates.append(outputs["gate"].detach().cpu().numpy())
            sample_ids.extend([str(item) for item in batch["sample_id"]])
            durations.extend([float(item) for item in batch["duration_sec"]])
            cue_spans.extend([list(item) for item in batch["cue_spans_sec"]])

    return {
        "preds": np.concatenate(all_preds, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "coverages": np.concatenate(all_coverages, axis=0),
        "gates": np.concatenate(all_gates, axis=0),
        "sample_ids": np.asarray(sample_ids, dtype=object),
        "durations": np.asarray(durations, dtype=np.float32),
        "cue_spans": cue_spans,
    }


def _evaluate_stage1(
    cuefilter,
    adapter,
    loader,
    device,
    threshold: float,
    expansion_frames: int,
    expansion_sec: Optional[float] = None,
) -> Dict[str, float]:
    arrays = _collect_stage1_arrays(
        cuefilter=cuefilter,
        adapter=adapter,
        loader=loader,
        device=device,
        expansion_frames=expansion_frames,
        expansion_sec=expansion_sec,
    )
    preds = arrays["preds"]
    labels = arrays["labels"]
    coverages = arrays["coverages"]
    return _cue_metrics_from_arrays(preds, labels, coverages, threshold=threshold, merge_gap=1)


def _write_stage1_curves(
    curve_output_dir: Optional[str],
    dataset_name: str,
    model_name: str,
    seed: int,
    cue_role: str,
    speech_scope: str,
    arrays: Dict[str, np.ndarray],
) -> None:
    if not curve_output_dir:
        return

    output_dir = Path(curve_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset_name.replace('-', '').lower()}__{model_name}__seed{seed}"
    curves = cue_curve_points_from_arrays(arrays["preds"], arrays["labels"])

    roc = pd.DataFrame(curves["roc"], columns=["fpr", "tpr", "threshold"])
    roc.insert(0, "point_idx", np.arange(len(roc)))
    roc.insert(0, "SpeechScope", speech_scope)
    roc.insert(0, "CueRole", cue_role)
    roc.insert(0, "Seed", seed)
    roc.insert(0, "Model", model_name)
    roc.insert(0, "Dataset", dataset_name)
    roc.to_csv(output_dir / f"{stem}__roc_points.csv", index=False)

    pr = pd.DataFrame(curves["pr"], columns=["recall", "precision", "threshold"])
    pr.insert(0, "point_idx", np.arange(len(pr)))
    pr.insert(0, "SpeechScope", speech_scope)
    pr.insert(0, "CueRole", cue_role)
    pr.insert(0, "Seed", seed)
    pr.insert(0, "Model", model_name)
    pr.insert(0, "Dataset", dataset_name)
    pr.to_csv(output_dir / f"{stem}__pr_points.csv", index=False)


def _write_localization_details(
    detail_output_dir: Optional[str],
    dataset_name: str,
    model_name: str,
    seed: int,
    cue_role: str,
    speech_scope: str,
    threshold: float,
    arrays: Dict[str, object],
) -> None:
    if not detail_output_dir:
        return

    output_root = Path(detail_output_dir)
    stem = f"{dataset_name.replace('-', '').lower()}__{model_name}__seed{seed}"
    for subdir in ("frame_scores", "spans", "budget"):
        (output_root / subdir).mkdir(parents=True, exist_ok=True)

    preds = arrays["preds"]
    labels = arrays["labels"]
    gates = arrays["gates"]
    coverages = arrays["coverages"]
    sample_ids = arrays["sample_ids"]
    durations = arrays["durations"]

    frame_rows: List[Dict[str, object]] = []
    budget_rows: List[Dict[str, object]] = []
    pred_span_rows: List[Dict[str, object]] = []
    true_span_rows: List[Dict[str, object]] = []
    match_rows: List[Dict[str, object]] = []

    for segment_idx in range(preds.shape[0]):
        duration = float(durations[segment_idx])
        num_frames = int(preds.shape[1])
        frame_sec = duration / num_frames if num_frames > 0 else 0.0
        sample_id = str(sample_ids[segment_idx])
        binary_pred = preds[segment_idx] >= threshold

        for frame_idx in range(num_frames):
            frame_rows.append(
                {
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Seed": seed,
                    "CueRole": cue_role,
                    "SpeechScope": speech_scope,
                    "sample_id": sample_id,
                    "segment_idx": segment_idx,
                    "frame_idx": frame_idx,
                    "time_start_sec": frame_idx * frame_sec,
                    "time_end_sec": (frame_idx + 1) * frame_sec,
                    "y_true": int(labels[segment_idx, frame_idx] >= 0.5),
                    "p_cue": float(preds[segment_idx, frame_idx]),
                    "binary_pred": int(binary_pred[frame_idx]),
                    "gate": float(gates[segment_idx, frame_idx]),
                }
            )

        pred_spans = extract_spans(preds[segment_idx], threshold=threshold, merge_gap=1)
        true_spans = extract_spans(labels[segment_idx], threshold=0.5, merge_gap=1)
        matches = match_spans(pred_spans, true_spans, iou_threshold=0.5)

        budget_rows.append(
            {
                "Dataset": dataset_name,
                "Model": model_name,
                "Seed": seed,
                "CueRole": cue_role,
                "SpeechScope": speech_scope,
                "sample_id": sample_id,
                "segment_idx": segment_idx,
                "duration_sec": duration,
                "num_frames": num_frames,
                "true_cue_coverage": float(coverages[segment_idx]),
                "predicted_mean_pcue": float(preds[segment_idx].mean()),
                "predicted_binary_coverage": float(binary_pred.mean()),
                "budget_error": float(abs(preds[segment_idx].mean() - coverages[segment_idx])),
                "num_true_spans": len(true_spans),
                "num_pred_spans": len(pred_spans),
            }
        )

        pred_id_by_span = {}
        for pred_idx, (start, end) in enumerate(pred_spans):
            pred_id_by_span[(start, end)] = pred_idx
            span_scores = preds[segment_idx, start:end]
            pred_span_rows.append(
                {
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Seed": seed,
                    "sample_id": sample_id,
                    "segment_idx": segment_idx,
                    "pred_span_id": pred_idx,
                    "start_frame": start,
                    "end_frame": end,
                    "start_sec": start * frame_sec,
                    "end_sec": end * frame_sec,
                    "mean_p_cue": float(span_scores.mean()) if len(span_scores) else 0.0,
                    "max_p_cue": float(span_scores.max()) if len(span_scores) else 0.0,
                }
            )

        true_id_by_span = {}
        for true_idx, (start, end) in enumerate(true_spans):
            true_id_by_span[(start, end)] = true_idx
            true_span_rows.append(
                {
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Seed": seed,
                    "sample_id": sample_id,
                    "segment_idx": segment_idx,
                    "true_span_id": true_idx,
                    "start_frame": start,
                    "end_frame": end,
                    "start_sec": start * frame_sec,
                    "end_sec": end * frame_sec,
                }
            )

        for pred_span, true_span in matches:
            inter_start = max(pred_span[0], true_span[0])
            inter_end = min(pred_span[1], true_span[1])
            inter = max(0, inter_end - inter_start)
            union = (pred_span[1] - pred_span[0]) + (true_span[1] - true_span[0]) - inter
            match_rows.append(
                {
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Seed": seed,
                    "sample_id": sample_id,
                    "segment_idx": segment_idx,
                    "pred_span_id": pred_id_by_span.get(pred_span),
                    "true_span_id": true_id_by_span.get(true_span),
                    "iou": float(inter / union) if union > 0 else 0.0,
                    "pred_start_sec": pred_span[0] * frame_sec,
                    "pred_end_sec": pred_span[1] * frame_sec,
                    "true_start_sec": true_span[0] * frame_sec,
                    "true_end_sec": true_span[1] * frame_sec,
                }
            )

    pd.DataFrame(frame_rows).to_csv(output_root / "frame_scores" / f"{stem}__frame_scores.csv.gz", index=False)
    pd.DataFrame(budget_rows).to_csv(output_root / "budget" / f"{stem}__cue_budget_by_sample.csv", index=False)
    pd.DataFrame(pred_span_rows).to_csv(output_root / "spans" / f"{stem}__predicted_spans.csv", index=False)
    pd.DataFrame(true_span_rows).to_csv(output_root / "spans" / f"{stem}__true_spans.csv", index=False)
    pd.DataFrame(match_rows).to_csv(output_root / "spans" / f"{stem}__span_matches.csv", index=False)


def _write_prediction_details(
    detail_output_dir: Optional[str],
    relative_path: str,
    rows: List[Dict[str, object]],
) -> None:
    if not detail_output_dir or not rows:
        return
    output_path = Path(detail_output_dir) / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _evaluate_stage2_joint(
    cuefilter,
    adapter,
    loader,
    device,
    threshold: float,
    expansion_frames: int,
    inverse_transform_fn,
    eval_level: str = "recording",
    gate_mode: str = "soft",
    use_renorm: bool = True,
    expansion_sec: Optional[float] = None,
) -> Dict[str, float]:
    cuefilter.eval()
    adapter.eval()
    all_preds = []
    all_labels = []
    all_coverages = []
    all_y_hat = []
    all_y_true = []
    all_sample_ids = []

    with torch.no_grad():
        for batch in loader:
            audios = batch["audio"].to(device, non_blocking=True)
            y_true = batch["label"].to(device, non_blocking=True)
            sequence_features, _aux = adapter.encode(audios)
            cue_labels, cue_coverage = build_cue_supervision_batch(
                cue_spans_batch=batch["cue_spans_sec"],
                num_frames=sequence_features.shape[1],
                durations_sec=batch["duration_sec"],
                device=device,
                expansion_frames=expansion_frames,
                expansion_sec=expansion_sec,
            )
            cue_outputs = cuefilter(sequence_features, mode=gate_mode, renorm=use_renorm)
            y_hat = adapter.predict_from_sequence(cue_outputs["renormed_features"])
            if y_hat.ndim > 1 and y_hat.shape[-1] == 1:
                y_hat = y_hat.squeeze(-1)

            all_preds.append(cue_outputs["p_cue"].detach().cpu().numpy())
            all_labels.append(cue_labels.detach().cpu().numpy())
            all_coverages.append(cue_coverage.detach().cpu().numpy())
            all_y_hat.extend(y_hat.detach().cpu().numpy().tolist())
            all_y_true.extend(y_true.detach().cpu().numpy().tolist())
            all_sample_ids.extend(batch["sample_id"])

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    coverages = np.concatenate(all_coverages, axis=0)
    cue_metrics = _cue_metrics_from_arrays(preds, labels, coverages, threshold=threshold, merge_gap=1)

    y_hat_raw = inverse_transform_fn(np.asarray(all_y_hat, dtype=np.float32))
    y_true_raw = inverse_transform_fn(np.asarray(all_y_true, dtype=np.float32))
    y_hat_eval, y_true_eval = finalize_predictions(y_hat_raw, y_true_raw, all_sample_ids, eval_level=eval_level)
    reg = _compute_regression_metrics(y_hat_eval, y_true_eval)
    reg.update(cue_metrics)
    return reg


def _train_baseline_adapter(adapter, train_loader, val_loader, inverse_transform_fn, device, args):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=max(2, args.baseline_patience // 3))
    best_score = float("inf")
    best_state = None
    patience_counter = 0

    for _epoch in range(args.baseline_epochs):
        adapter.train()
        for batch in train_loader:
            audios = batch["audio"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad()
            y_hat, _seq, _cue, aux = _forward_with_optional_cuefilter(adapter, audios, cuefilter=None)
            loss = criterion(y_hat, labels)
            if "hcpc_loss" in aux:
                loss = loss + args.hcpc_weight * aux["hcpc_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=args.grad_clip)
            optimizer.step()

        val_metrics = _evaluate_adapter_on_cueaware_loader(
            adapter,
            val_loader,
            inverse_transform_fn,
            device,
            eval_level=args.eval_level,
        )
        scheduler.step(val_metrics["MAE"])
        if val_metrics["MAE"] < best_score:
            best_score = val_metrics["MAE"]
            best_state = copy.deepcopy(adapter.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.baseline_patience:
                break

    if best_state is not None:
        adapter.load_state_dict(best_state)
    return adapter


def _evaluate_adapter_on_cueaware_loader(adapter, loader, inverse_transform_fn, device, eval_level: str = "recording"):
    adapter.eval()
    preds_scaled = []
    labels_scaled = []
    sample_ids = []
    with torch.no_grad():
        for batch in loader:
            audios = batch["audio"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            y_hat, _seq, _cue, _aux = _forward_with_optional_cuefilter(adapter, audios, cuefilter=None)
            preds_scaled.extend(y_hat.detach().cpu().numpy().tolist())
            labels_scaled.extend(labels.detach().cpu().numpy().tolist())
            sample_ids.extend(batch["sample_id"])
    preds_raw = inverse_transform_fn(np.asarray(preds_scaled, dtype=np.float32))
    labels_raw = inverse_transform_fn(np.asarray(labels_scaled, dtype=np.float32))
    preds_eval, labels_eval = finalize_predictions(preds_raw, labels_raw, sample_ids, eval_level=eval_level)
    return _compute_regression_metrics(preds_eval, labels_eval)


def _train_stage1_cuefilter(cuefilter, adapter, train_loader, val_loader, device, args):
    feature_mean, feature_std = estimate_sequence_feature_stats(adapter, train_loader, device)
    cuefilter.set_feature_stats(feature_mean, feature_std)

    optimizer = torch.optim.AdamW(cuefilter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_f1 = -1.0
    best_state = None
    patience_counter = 0

    for _epoch in range(args.stage1_epochs):
        cuefilter.train()
        adapter.eval()
        for batch in train_loader:
            audios = batch["audio"].to(device, non_blocking=True)
            with torch.no_grad():
                sequence_features, _aux = adapter.encode(audios)
            cue_labels, cue_coverage = build_cue_supervision_batch(
                cue_spans_batch=batch["cue_spans_sec"],
                num_frames=sequence_features.shape[1],
                durations_sec=batch["duration_sec"],
                device=device,
                expansion_frames=args.boundary_expand_frames,
                expansion_sec=getattr(args, "boundary_tolerance_sec", None),
            )

            optimizer.zero_grad()
            outputs = cuefilter(sequence_features, mode="soft")
            loss, _metrics = cuefilter_pretrain_loss(
                outputs["p_cue"],
                cue_labels,
                cue_coverage,
                lambda_d=args.lambda_d,
                lambda_b=args.lambda_b,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cuefilter.parameters(), max_norm=args.grad_clip)
            optimizer.step()

        val_metrics = _evaluate_stage1(
            cuefilter=cuefilter,
            adapter=adapter,
            loader=val_loader,
            device=device,
            threshold=args.cue_threshold,
            expansion_frames=args.boundary_expand_frames,
            expansion_sec=getattr(args, "boundary_tolerance_sec", None),
        )
        if val_metrics["frame_f1"] > best_f1:
            best_f1 = val_metrics["frame_f1"]
            best_state = copy.deepcopy(cuefilter.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.stage1_patience:
                break

    if best_state is not None:
        cuefilter.load_state_dict(best_state)
    return cuefilter, _evaluate_stage1(
        cuefilter,
        adapter,
        val_loader,
        device,
        args.cue_threshold,
        args.boundary_expand_frames,
        getattr(args, "boundary_tolerance_sec", None),
    )


def _train_stage2_joint(
    cuefilter,
    adapter,
    train_loader,
    val_loader,
    inverse_transform_fn,
    device,
    args,
    gate_mode: str = "soft",
    use_renorm: bool = True,
):
    params = list(adapter.parameters()) + list(cuefilter.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=max(2, args.stage2_patience // 3))
    best_mae = float("inf")
    best_adapter_state = None
    best_cuefilter_state = None
    patience_counter = 0

    for _epoch in range(args.stage2_epochs):
        adapter.train()
        cuefilter.train()
        for batch in train_loader:
            audios = batch["audio"].to(device, non_blocking=True)
            y_true = batch["label"].to(device, non_blocking=True)
            sequence_features, aux = adapter.encode(audios)
            cue_labels, cue_coverage = build_cue_supervision_batch(
                cue_spans_batch=batch["cue_spans_sec"],
                num_frames=sequence_features.shape[1],
                durations_sec=batch["duration_sec"],
                device=device,
                expansion_frames=args.boundary_expand_frames,
                expansion_sec=getattr(args, "boundary_tolerance_sec", None),
            )

            optimizer.zero_grad()
            cue_outputs = cuefilter(sequence_features, mode=gate_mode, renorm=use_renorm)
            y_hat = adapter.predict_from_sequence(cue_outputs["renormed_features"])
            if y_hat.ndim > 1 and y_hat.shape[-1] == 1:
                y_hat = y_hat.squeeze(-1)
            loss, _metrics = cuefilter_joint_loss(
                y_hat=y_hat,
                y_true=y_true,
                p_cue=cue_outputs["p_cue"],
                cue_labels=cue_labels,
                cue_coverage=cue_coverage,
                lambda_c=args.lambda_c,
                lambda_d=args.lambda_d,
                lambda_b=args.lambda_b,
            )
            if "hcpc_loss" in aux:
                loss = loss + args.hcpc_weight * aux["hcpc_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=args.grad_clip)
            optimizer.step()

        val_metrics = _evaluate_stage2_joint(
            cuefilter=cuefilter,
            adapter=adapter,
            loader=val_loader,
            device=device,
            threshold=args.cue_threshold,
            expansion_frames=args.boundary_expand_frames,
            inverse_transform_fn=inverse_transform_fn,
            eval_level=args.eval_level,
            gate_mode=gate_mode,
            use_renorm=use_renorm,
            expansion_sec=getattr(args, "boundary_tolerance_sec", None),
        )
        scheduler.step(val_metrics["MAE"])
        if val_metrics["MAE"] < best_mae:
            best_mae = val_metrics["MAE"]
            best_adapter_state = copy.deepcopy(adapter.state_dict())
            best_cuefilter_state = copy.deepcopy(cuefilter.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.stage2_patience:
                break

    if best_adapter_state is not None:
        adapter.load_state_dict(best_adapter_state)
    if best_cuefilter_state is not None:
        cuefilter.load_state_dict(best_cuefilter_state)
    return cuefilter, adapter, _evaluate_stage2_joint(
        cuefilter=cuefilter,
        adapter=adapter,
        loader=val_loader,
        device=device,
        threshold=args.cue_threshold,
        expansion_frames=args.boundary_expand_frames,
        inverse_transform_fn=inverse_transform_fn,
        eval_level=args.eval_level,
        gate_mode=gate_mode,
        use_renorm=use_renorm,
        expansion_sec=getattr(args, "boundary_tolerance_sec", None),
    )


def _resolve_checkpoint_path(path_template: Optional[str], model_name: str, seed: int) -> Optional[Path]:
    if not path_template:
        return None
    return Path(path_template.format(model=model_name, seed=seed))


def _build_cueaware_loader(items: List[Dict], batch_size: int, num_workers: int, shuffle: bool, is_training: bool):
    dataset = CueAwareAudioDataset(items, is_training=is_training)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=cueaware_collate_fn,
    )


def _initialize_shared_cuefilter_if_needed(
    cuefilter: CueFilter,
    adapter,
    model_name: str,
    seed: int,
    device: torch.device,
    args,
) -> Optional[Dict[str, float]]:
    load_path = _resolve_checkpoint_path(args.load_shared_cuefilter, model_name, seed)
    if load_path is not None and load_path.exists():
        state = torch.load(load_path, map_location="cpu")
        cuefilter.load_state_dict(state)
        return {}

    if not args.shared_pretrain_datasets:
        return None

    pooled_train_items: List[Dict] = []
    pooled_val_items: List[Dict] = []
    for dataset_key in args.shared_pretrain_datasets:
        dm = CueAwarePreDataManager(
            dataset_key=dataset_key,
            segment_length=args.segment_length,
            sample_rate=args.sample_rate,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            random_state=seed,
            max_groups_per_split=args.max_groups_per_split,
            cue_role=args.cue_role,
            speech_scope=args.speech_scope,
        )
        pooled_train_items.extend(dm.get_items("train"))
        pooled_val_items.extend(dm.get_items("val"))

    if not pooled_train_items or not pooled_val_items:
        return None

    train_loader = _build_cueaware_loader(
        pooled_train_items,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        is_training=True,
    )
    val_loader = _build_cueaware_loader(
        pooled_val_items,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        is_training=False,
    )
    cuefilter, metrics = _train_stage1_cuefilter(
        cuefilter=cuefilter,
        adapter=adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        args=args,
    )

    save_path = _resolve_checkpoint_path(args.save_shared_cuefilter, model_name, seed)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cuefilter.state_dict(), save_path)

    return metrics


def run_single_mitigation(dataset_key: str, model_name: str, seed: int, args) -> MitigationRunOutput:
    args = _with_model_defaults(args, model_name)
    set_seed(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    cueaware_dm = CueAwarePreDataManager(
        dataset_key=dataset_key,
        segment_length=args.segment_length,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_state=seed,
        max_groups_per_split=args.max_groups_per_split,
        cue_role=args.cue_role,
        speech_scope=args.speech_scope,
    )
    shared_split_map = cueaware_dm.split_map

    if args.info_only:
        info = cueaware_dm.get_info()
        print(f"[{DATASET_CONFIGS[dataset_key].display_name} | {model_name}] {info}")
        return None

    pre_dm = RecordingSplitDataManager(
        dataset_key=dataset_key,
        variant="pre",
        segment_length=args.segment_length,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_state=seed,
        split_map=shared_split_map,
        cue_role=args.cue_role,
        speech_scope=args.speech_scope,
        eval_level=args.eval_level,
        segment_first=args.segment_first,
    )
    exc_dm = RecordingSplitDataManager(
        dataset_key=dataset_key,
        variant="cue-excluded",
        segment_length=args.segment_length,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_state=seed,
        split_map=shared_split_map,
        cue_role=args.cue_role,
        speech_scope=args.speech_scope,
        eval_level=args.eval_level,
        segment_first=args.segment_first,
    )
    rand_dm = RecordingSplitDataManager(
        dataset_key=dataset_key,
        variant="random",
        segment_length=args.segment_length,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_state=seed,
        split_map=shared_split_map,
        cue_role=args.cue_role,
        speech_scope=args.speech_scope,
        eval_level=args.eval_level,
        segment_first=args.segment_first,
    )

    baseline_adapter = build_adapter(model_name).to(device)
    baseline_adapter = _train_baseline_adapter(
        adapter=baseline_adapter,
        train_loader=cueaware_dm.get_loader("train", shuffle=True),
        val_loader=cueaware_dm.get_loader("val", shuffle=False),
        inverse_transform_fn=cueaware_dm.inverse_transform,
        device=device,
        args=args,
    )

    base_pre, base_pre_rows = _evaluate_variant(
        baseline_adapter, pre_dm, device, cuefilter=None, return_predictions=True, condition="baseline_original"
    )
    base_exc, base_exc_rows = _evaluate_variant(
        baseline_adapter, exc_dm, device, cuefilter=None, return_predictions=True, condition="baseline_cue_removed"
    )
    base_rand, base_rand_rows = _evaluate_variant(
        baseline_adapter, rand_dm, device, cuefilter=None, return_predictions=True, condition="baseline_random"
    )

    mitigation_adapter = copy.deepcopy(baseline_adapter).to(device)
    cuefilter = CueFilter(
        input_dim=mitigation_adapter.feature_dim,
        n_blocks=args.n_blocks,
        kernel_size=args.kernel_size,
        groups=args.groups,
        alpha=args.alpha,
        gamma=args.gamma,
        cue_threshold=args.cue_threshold,
    ).to(device)

    shared_stage1_metrics = _initialize_shared_cuefilter_if_needed(
        cuefilter=cuefilter,
        adapter=mitigation_adapter,
        model_name=model_name,
        seed=seed,
        device=device,
        args=args,
    )
    if shared_stage1_metrics is None:
        cuefilter, _stage1_val_metrics = _train_stage1_cuefilter(
            cuefilter=cuefilter,
            adapter=mitigation_adapter,
            train_loader=cueaware_dm.get_loader("train", shuffle=True),
            val_loader=cueaware_dm.get_loader("val", shuffle=False),
            device=device,
            args=args,
        )

    stage1_test_arrays = _collect_stage1_arrays(
        cuefilter=cuefilter,
        adapter=mitigation_adapter,
        loader=cueaware_dm.get_loader("test", shuffle=False),
        device=device,
        expansion_frames=args.boundary_expand_frames,
        expansion_sec=getattr(args, "boundary_tolerance_sec", None),
    )
    stage1_metrics = _cue_metrics_from_arrays(
        stage1_test_arrays["preds"],
        stage1_test_arrays["labels"],
        stage1_test_arrays["coverages"],
        threshold=args.cue_threshold,
        merge_gap=1,
    )
    _write_stage1_curves(
        curve_output_dir=args.curve_output_dir,
        dataset_name=DATASET_CONFIGS[dataset_key].display_name,
        model_name=model_name,
        seed=seed,
        cue_role=args.cue_role,
        speech_scope=args.speech_scope,
        arrays=stage1_test_arrays,
    )
    _write_localization_details(
        detail_output_dir=args.detail_output_dir,
        dataset_name=DATASET_CONFIGS[dataset_key].display_name,
        model_name=model_name,
        seed=seed,
        cue_role=args.cue_role,
        speech_scope=args.speech_scope,
        threshold=args.cue_threshold,
        arrays=stage1_test_arrays,
    )

    cuefilter, mitigation_adapter, stage2_metrics = _train_stage2_joint(
        cuefilter=cuefilter,
        adapter=mitigation_adapter,
        train_loader=cueaware_dm.get_loader("train", shuffle=True),
        val_loader=cueaware_dm.get_loader("val", shuffle=False),
        inverse_transform_fn=cueaware_dm.inverse_transform,
        device=device,
        args=args,
    )

    cf_pre, cf_pre_rows = _evaluate_variant(
        mitigation_adapter, pre_dm, device, cuefilter=cuefilter, return_predictions=True, condition="cuefilter_original"
    )
    cf_exc, cf_exc_rows = _evaluate_variant(
        mitigation_adapter, exc_dm, device, cuefilter=cuefilter, return_predictions=True, condition="cuefilter_cue_removed"
    )
    cf_rand, cf_rand_rows = _evaluate_variant(
        mitigation_adapter, rand_dm, device, cuefilter=cuefilter, return_predictions=True, condition="cuefilter_random"
    )

    prediction_rows: List[Dict[str, object]] = []
    for row in base_pre_rows + base_exc_rows + base_rand_rows + cf_pre_rows + cf_exc_rows + cf_rand_rows:
        enriched = {
            "Dataset": DATASET_CONFIGS[dataset_key].display_name,
            "Model": model_name,
            "Seed": seed,
            "CueRole": args.cue_role,
            "SpeechScope": args.speech_scope,
            **row,
        }
        prediction_rows.append(enriched)
    _write_prediction_details(
        args.detail_output_dir,
        f"mitigation_predictions/{DATASET_CONFIGS[dataset_key].display_name.replace('-', '').lower()}__{model_name}__seed{seed}.csv",
        prediction_rows,
    )

    return MitigationRunOutput(
        base_pre_mae=base_pre["MAE"],
        base_pre_rmse=base_pre["RMSE"],
        base_pre_r2=base_pre["R2"],
        base_exc_mae=base_exc["MAE"],
        base_rand_mae=base_rand["MAE"],
        cf_pre_mae=cf_pre["MAE"],
        cf_pre_rmse=cf_pre["RMSE"],
        cf_pre_r2=cf_pre["R2"],
        cf_exc_mae=cf_exc["MAE"],
        cf_rand_mae=cf_rand["MAE"],
        n_before=base_exc["MAE"] - base_pre["MAE"],
        n_after=cf_exc["MAE"] - cf_pre["MAE"],
        g_before=base_exc["MAE"] - base_rand["MAE"],
        g_after=cf_exc["MAE"] - cf_rand["MAE"],
        stage1_frame_precision=stage1_metrics["frame_precision"],
        stage1_frame_recall=stage1_metrics["frame_recall"],
        stage1_frame_f1=stage1_metrics["frame_f1"],
        stage1_auc_roc=stage1_metrics["auc_roc"],
        stage1_auc_pr=stage1_metrics["auc_pr"],
        stage1_span_f1=stage1_metrics["span_f1"],
        stage1_budget_mae=stage1_metrics["budget_mae"],
        stage2_frame_precision=stage2_metrics["frame_precision"],
        stage2_frame_recall=stage2_metrics["frame_recall"],
        stage2_frame_f1=stage2_metrics["frame_f1"],
        stage2_auc_roc=stage2_metrics["auc_roc"],
        stage2_auc_pr=stage2_metrics["auc_pr"],
        stage2_span_f1=stage2_metrics["span_f1"],
        stage2_budget_mae=stage2_metrics["budget_mae"],
    )


def main():
    args = create_parser().parse_args()
    summary_rows: List[Dict[str, object]] = []

    for dataset_key in args.datasets:
        dataset_name = DATASET_CONFIGS[dataset_key].display_name
        for model_name in args.models:
            run_outputs: List[MitigationRunOutput] = []
            for seed in args.seeds:
                print(f"Running mitigation | {dataset_name} | {model_name} | seed={seed}", flush=True)
                try:
                    result = run_single_mitigation(dataset_key, model_name, seed, args)
                except Exception as exc:
                    print(f"  failed: {exc}")
                    continue
                if result is not None:
                    run_outputs.append(result)

            if not run_outputs:
                continue

            summary_rows.append(
                {
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "CueRole": args.cue_role,
                    "SpeechScope": args.speech_scope,
                    "SharedCuePrior": bool(args.shared_pretrain_datasets or args.load_shared_cuefilter),
                    "EvalLevel": args.eval_level,
                    "SegmentFirst": args.segment_first,
                    "Baseline Original MAE": format_mean_std([r.base_pre_mae for r in run_outputs]),
                    "CueFilter Original MAE": format_mean_std([r.cf_pre_mae for r in run_outputs]),
                    "Baseline cue-removal increase": format_mean_std([r.n_before for r in run_outputs]),
                    "CueFilter cue-removal increase": format_mean_std([r.n_after for r in run_outputs]),
                    "Reduction": format_mean_std([r.n_before - r.n_after for r in run_outputs]),
                    "Baseline extra-over-random": format_mean_std([r.g_before for r in run_outputs]),
                    "CueFilter extra-over-random": format_mean_std([r.g_after for r in run_outputs]),
                    "Precision": format_mean_std([r.stage1_frame_precision for r in run_outputs]),
                    "Recall": format_mean_std([r.stage1_frame_recall for r in run_outputs]),
                    "Stage1 Frame F1": format_mean_std([r.stage1_frame_f1 for r in run_outputs]),
                    "AUC-ROC": format_mean_std([r.stage1_auc_roc for r in run_outputs]),
                    "AUC-PR": format_mean_std([r.stage1_auc_pr for r in run_outputs]),
                    "Span F1": format_mean_std([r.stage1_span_f1 for r in run_outputs]),
                    "BErr": format_mean_std([r.stage1_budget_mae for r in run_outputs]),
                    "Stage2 Precision": format_mean_std([r.stage2_frame_precision for r in run_outputs]),
                    "Stage2 Recall": format_mean_std([r.stage2_frame_recall for r in run_outputs]),
                    "Stage2 Frame F1": format_mean_std([r.stage2_frame_f1 for r in run_outputs]),
                    "Stage2 AUC-ROC": format_mean_std([r.stage2_auc_roc for r in run_outputs]),
                    "Stage2 AUC-PR": format_mean_std([r.stage2_auc_pr for r in run_outputs]),
                    "Stage2 Span F1": format_mean_std([r.stage2_span_f1 for r in run_outputs]),
                    "Stage2 BErr": format_mean_std([r.stage2_budget_mae for r in run_outputs]),
                    "BoundaryToleranceSec": args.boundary_tolerance_sec,
                    "Seeds": len(run_outputs),
                }
            )

    if not summary_rows:
        print("No mitigation results were produced.")
        return

    results_df = pd.DataFrame(summary_rows)
    print("\n" + results_df.to_markdown(index=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".csv":
            results_df.to_csv(output_path, index=False)
        else:
            output_path.write_text(results_df.to_markdown(index=False), encoding="utf-8")
        print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
