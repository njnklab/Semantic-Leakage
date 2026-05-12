"""
Evaluation utilities for CueFilter pretraining and joint suppression experiments.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def extract_spans(labels: np.ndarray, threshold: float = 0.5, merge_gap: int = 1) -> List[tuple]:
    """Convert frame probabilities or labels into merged contiguous spans."""
    binary = labels >= threshold
    spans = []
    start = None

    for idx, flag in enumerate(binary):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            spans.append((start, idx))
            start = None

    if start is not None:
        spans.append((start, len(binary)))

    if not spans:
        return spans

    merged = [spans[0]]
    for start, end in spans[1:]:
        if start - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def match_spans(
    pred_spans: List[tuple],
    true_spans: List[tuple],
    iou_threshold: float = 0.5,
) -> List[tuple]:
    """Greedy one-to-one span matching under an IoU threshold."""
    matches = []
    used_true = set()

    for pred_span in pred_spans:
        best_iou = 0.0
        best_true_idx = None

        for idx, true_span in enumerate(true_spans):
            if idx in used_true:
                continue

            inter_start = max(pred_span[0], true_span[0])
            inter_end = min(pred_span[1], true_span[1])
            if inter_start >= inter_end:
                continue

            inter = inter_end - inter_start
            union = (pred_span[1] - pred_span[0]) + (true_span[1] - true_span[0]) - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_true_idx = idx

        if best_iou >= iou_threshold and best_true_idx is not None:
            matches.append((pred_span, true_spans[best_true_idx]))
            used_true.add(best_true_idx)

    return matches


def _cue_metrics_from_arrays(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    all_coverages: np.ndarray,
    threshold: float = 0.5,
    merge_gap: int = 1,
) -> Dict[str, float]:
    preds_flat = all_preds.reshape(-1)
    labels_flat = all_labels.reshape(-1)

    binary_preds = (preds_flat >= threshold).astype(int)
    binary_labels = (labels_flat >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        binary_labels,
        binary_preds,
        average="binary",
        zero_division=0,
    )

    try:
        auc_roc = roc_auc_score(binary_labels, preds_flat)
        auc_pr = average_precision_score(binary_labels, preds_flat)
    except ValueError:
        auc_roc = 0.0
        auc_pr = 0.0

    span_precisions = []
    span_recalls = []
    span_f1s = []
    for preds, labels in zip(all_preds, all_labels):
        pred_spans = extract_spans(preds, threshold=threshold, merge_gap=merge_gap)
        true_spans = extract_spans(labels, threshold=0.5, merge_gap=merge_gap)
        matches = match_spans(pred_spans, true_spans, iou_threshold=0.5)

        tp = len(matches)
        fp = len(pred_spans) - tp
        fn = len(true_spans) - tp

        span_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        span_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        span_f1 = (
            2 * span_precision * span_recall / (span_precision + span_recall)
            if (span_precision + span_recall) > 0
            else 0.0
        )

        span_precisions.append(span_precision)
        span_recalls.append(span_recall)
        span_f1s.append(span_f1)

    predicted_coverages = all_preds.mean(axis=1)
    budget_mae = np.mean(np.abs(predicted_coverages - all_coverages))

    return {
        "frame_precision": float(precision),
        "frame_recall": float(recall),
        "frame_f1": float(f1),
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "span_precision": float(np.mean(span_precisions)) if span_precisions else 0.0,
        "span_recall": float(np.mean(span_recalls)) if span_recalls else 0.0,
        "span_f1": float(np.mean(span_f1s)) if span_f1s else 0.0,
        "budget_mae": float(budget_mae),
    }


def cue_curve_points_from_arrays(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Return ROC and PR curve points from frame-level scores and labels."""
    preds_flat = all_preds.reshape(-1)
    labels_flat = (all_labels.reshape(-1) >= 0.5).astype(int)

    if np.unique(labels_flat).size < 2:
        return {
            "roc": np.empty((0, 3), dtype=float),
            "pr": np.empty((0, 3), dtype=float),
        }

    fpr, tpr, roc_thresholds = roc_curve(labels_flat, preds_flat)
    precision, recall, pr_thresholds = precision_recall_curve(labels_flat, preds_flat)
    pr_thresholds = np.concatenate([pr_thresholds, [np.nan]])

    return {
        "roc": np.column_stack([fpr, tpr, roc_thresholds]),
        "pr": np.column_stack([recall, precision, pr_thresholds]),
    }


def evaluate_cuefilter(
    model,
    loader,
    device,
    threshold: float = 0.5,
    merge_gap: int = 1,
) -> Dict[str, float]:
    """Evaluate frame/span cue localization and budget calibration."""
    model.eval()
    all_preds = []
    all_labels = []
    all_coverages = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            cue_labels = batch["cue_labels"].to(device)
            cue_coverage = batch["cue_coverage"].to(device)

            outputs = model(features, mode="soft")
            all_preds.append(outputs["p_cue"].cpu().numpy())
            all_labels.append(cue_labels.cpu().numpy())
            all_coverages.append(cue_coverage.cpu().numpy())

    return _cue_metrics_from_arrays(
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_coverages, axis=0),
        threshold=threshold,
        merge_gap=merge_gap,
    )


def evaluate_cuefilter_joint(
    model,
    loader,
    device,
    threshold: float = 0.5,
    merge_gap: int = 1,
) -> Dict[str, float]:
    """Evaluate joint regression performance plus CueFilter localization quality."""
    model.eval()
    all_preds = []
    all_labels = []
    all_coverages = []
    all_y_true = []
    all_y_hat = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            cue_labels = batch["cue_labels"].to(device)
            cue_coverage = batch["cue_coverage"].to(device)
            y_true = batch["depression_score"].to(device)

            outputs = model(features, mode="soft")
            all_preds.append(outputs["p_cue"].cpu().numpy())
            all_labels.append(cue_labels.cpu().numpy())
            all_coverages.append(cue_coverage.cpu().numpy())
            all_y_true.append(y_true.cpu().numpy())
            all_y_hat.append(outputs["y_hat"].cpu().numpy())

    cue_metrics = _cue_metrics_from_arrays(
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_coverages, axis=0),
        threshold=threshold,
        merge_gap=merge_gap,
    )

    y_true = np.concatenate(all_y_true, axis=0)
    y_hat = np.concatenate(all_y_hat, axis=0)

    mae = np.mean(np.abs(y_hat - y_true))
    rmse = np.sqrt(np.mean((y_hat - y_true) ** 2))
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - np.sum((y_hat - y_true) ** 2) / denom if denom > 0 else 0.0

    cue_metrics.update(
        {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
        }
    )
    return cue_metrics
