from __future__ import annotations

import argparse
import copy
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.svm import SVR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .Baseline.audio_views import (
        DATASET_CONFIGS,
        annotation_path,
        cue_role_from_overlap,
        iter_output_samples,
        load_json,
        role_matches_filter,
    )
    from .Baseline.experiment_utils import RecordingSplitDataManager, finalize_predictions, format_mean_std, set_seed
    from .integration.common import build_cue_supervision_batch
    from .integration.data import CueAwarePreDataManager
    from .integration.registry import build_adapter, list_supported_adapters
    from .models import CueFilter
    from .run_mitigation_experiments import (
        _compute_regression_metrics,
        _initialize_shared_cuefilter_if_needed,
        _train_baseline_adapter,
        _train_stage1_cuefilter,
        _train_stage2_joint,
        _with_model_defaults,
    )
except ImportError:
    from CueFilter.Baseline.audio_views import (
        DATASET_CONFIGS,
        annotation_path,
        cue_role_from_overlap,
        iter_output_samples,
        load_json,
        role_matches_filter,
    )
    from CueFilter.Baseline.experiment_utils import RecordingSplitDataManager, finalize_predictions, format_mean_std, set_seed
    from CueFilter.integration.common import build_cue_supervision_batch
    from CueFilter.integration.data import CueAwarePreDataManager
    from CueFilter.integration.registry import build_adapter, list_supported_adapters
    from CueFilter.models import CueFilter
    from CueFilter.run_mitigation_experiments import (
        _compute_regression_metrics,
        _initialize_shared_cuefilter_if_needed,
        _train_baseline_adapter,
        _train_stage1_cuefilter,
        _train_stage2_joint,
        _with_model_defaults,
    )


LABEL_RANGES = {
    "edaic": 24.0,   # PHQ-8
    "cmdc": 27.0,   # PHQ-9 total
    "pdch": 52.0,   # HAMD-17 conventional maximum
    "mandic": 52.0, # HAMD-17 conventional maximum
}

FUNCTIONAL_METHODS = ["Backbone", "CueFilter", "Agent mask", "Random gate", "Uniform gate", "Shuffled gate"]
FEATURE_METHODS = [
    "Backbone feature",
    "Backbone+gate stats",
    "CueFilter feature",
    "Random-gated feature",
    "Agent-mask feature",
]
PROBE_FEATURE_METHODS = ["Backbone feature", "CueFilter feature", "Random-gated feature", "Agent-mask feature"]
RIDGE_ALPHAS = np.logspace(0, 7, 8)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paper-aligned CueFilter functional experiments.")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_CONFIGS.keys()), default=["edaic", "mandic"])
    parser.add_argument("--models", nargs="+", choices=list_supported_adapters(), default=list_supported_adapters())
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--cue-role", choices=["patient", "doctor", "all"], default="patient")
    parser.add_argument("--speech-scope", choices=["participant", "interviewer", "dialogue"], default="participant")
    parser.add_argument("--shared-pretrain-datasets", nargs="+", choices=list(DATASET_CONFIGS.keys()), default=None)
    parser.add_argument("--load-shared-cuefilter", type=str, default=None)
    parser.add_argument("--save-shared-cuefilter", type=str, default=None)
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
    parser.add_argument("--boundary-tolerance-sec", type=float, default=1.0)
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.0, 0.2, 0.5, 0.8, 1.0])
    parser.add_argument("--max-groups-per-split", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval-level", choices=["recording", "sample"], default="recording")
    parser.add_argument("--segment-first", action="store_true")
    parser.add_argument("--skip-frozen", action="store_true")
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--skip-strength", action="store_true")
    parser.add_argument(
        "--sequential-variant-load",
        action="store_true",
        help="Load pre/cue-excluded/random evaluation views one at a time. Useful for ManDIC memory pressure.",
    )
    parser.add_argument("--output-dir", type=str, default="CueFilter/results/formal_functional")
    parser.add_argument("--info-only", action="store_true")
    return parser


def _score_range(dataset_key: str) -> float:
    return LABEL_RANGES.get(dataset_key, 1.0)


def _normal(value: float, dataset_key: str) -> float:
    return float(value) / _score_range(dataset_key)


def _make_cueaware_data_manager(dataset_key: str, seed: int, args):
    return CueAwarePreDataManager(
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


def _make_recording_data_manager(dataset_key: str, variant: str, seed: int, split_map: Dict[str, str], args):
    return RecordingSplitDataManager(
        dataset_key=dataset_key,
        variant=variant,
        segment_length=args.segment_length,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_state=seed,
        split_map=split_map,
        cue_role=args.cue_role,
        speech_scope=args.speech_scope,
        max_groups_per_split=args.max_groups_per_split,
        eval_level=args.eval_level,
        segment_first=args.segment_first,
    )


def _make_data_managers(dataset_key: str, seed: int, args):
    cueaware_dm = _make_cueaware_data_manager(dataset_key, seed, args)
    split_map = cueaware_dm.split_map
    pre_dm = _make_recording_data_manager(dataset_key, "pre", seed, split_map, args)
    exc_dm = _make_recording_data_manager(dataset_key, "cue-excluded", seed, split_map, args)
    rand_dm = _make_recording_data_manager(dataset_key, "random", seed, split_map, args)
    return cueaware_dm, pre_dm, exc_dm, rand_dm


def _build_cuefilter(adapter, args, device: torch.device) -> CueFilter:
    return CueFilter(
        input_dim=adapter.feature_dim,
        n_blocks=args.n_blocks,
        kernel_size=args.kernel_size,
        groups=args.groups,
        alpha=args.alpha,
        gamma=args.gamma,
        cue_threshold=args.cue_threshold,
    ).to(device)


def _unpack_batch(batch, device: torch.device):
    if isinstance(batch, dict):
        audios = batch["audio"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        sample_ids = [str(item) for item in batch["sample_id"]]
        cue_spans = batch.get("cue_spans_sec")
        durations = batch.get("duration_sec")
        return audios, labels, sample_ids, cue_spans, durations
    audios, labels, sample_ids = batch
    return audios.to(device, non_blocking=True), labels.to(device, non_blocking=True).squeeze(-1), [str(s) for s in sample_ids], None, None


def _randomize_gate(gate: torch.Tensor) -> torch.Tensor:
    pieces = []
    for row in gate:
        pieces.append(row[torch.randperm(row.shape[0], device=row.device)])
    return torch.stack(pieces, dim=0)


def _shuffled_gate(gate: torch.Tensor) -> torch.Tensor:
    if gate.shape[0] > 1:
        return torch.roll(gate, shifts=1, dims=0)
    return _randomize_gate(gate)


def _agent_gate(
    cuefilter: CueFilter,
    sequence_features: torch.Tensor,
    cue_spans,
    durations,
    device: torch.device,
    args,
) -> torch.Tensor:
    if cue_spans is None or durations is None:
        return torch.ones(sequence_features.shape[:2], dtype=sequence_features.dtype, device=device)
    cue_labels, _coverage = build_cue_supervision_batch(
        cue_spans_batch=cue_spans,
        num_frames=sequence_features.shape[1],
        durations_sec=durations,
        device=device,
        expansion_frames=args.boundary_expand_frames,
        expansion_sec=getattr(args, "boundary_tolerance_sec", None),
    )
    return cuefilter.compute_gate(cue_labels, mode="soft")


def _apply_custom_gate(
    method: str,
    sequence_features: torch.Tensor,
    cuefilter: CueFilter,
    device: torch.device,
    args,
    cue_spans=None,
    durations=None,
    alpha: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if method == "cuefilter":
        outputs = cuefilter(sequence_features, mode="soft", alpha=alpha, renorm=True)
        return outputs["renormed_features"], outputs["gate"]

    base_outputs = cuefilter(sequence_features, mode="soft", alpha=alpha, renorm=False)
    base_gate = base_outputs["gate"]
    if method == "agent":
        gate = _agent_gate(cuefilter, sequence_features, cue_spans, durations, device, args)
    elif method == "random":
        gate = _randomize_gate(base_gate)
    elif method == "uniform":
        gate = base_gate.mean(dim=1, keepdim=True).expand_as(base_gate)
    elif method == "shuffled":
        gate = _shuffled_gate(base_gate)
    else:
        raise ValueError(f"Unsupported gate method: {method}")

    filtered = gate.unsqueeze(-1) * sequence_features
    return cuefilter.renormalize(filtered), gate


def _evaluate_gated_loader(
    adapter,
    loader,
    inverse_transform_fn,
    device: torch.device,
    eval_level: str,
    gate_method: str,
    cuefilter: Optional[CueFilter] = None,
    args=None,
    alpha: Optional[float] = None,
) -> Dict[str, float]:
    adapter.eval()
    if cuefilter is not None:
        cuefilter.eval()

    preds_scaled: List[float] = []
    labels_scaled: List[float] = []
    sample_ids: List[str] = []

    with torch.no_grad():
        for batch in loader:
            audios, labels, batch_sample_ids, cue_spans, durations = _unpack_batch(batch, device)
            sequence_features, _aux = adapter.encode(audios)
            pred_features = sequence_features
            if gate_method != "none":
                if cuefilter is None:
                    raise ValueError(f"gate_method={gate_method} requires cuefilter")
                pred_features, _gate = _apply_custom_gate(
                    gate_method,
                    sequence_features,
                    cuefilter,
                    device=device,
                    args=args,
                    cue_spans=cue_spans,
                    durations=durations,
                    alpha=alpha,
                )
            y_hat = adapter.predict_from_sequence(pred_features)
            if y_hat.ndim > 1 and y_hat.shape[-1] == 1:
                y_hat = y_hat.squeeze(-1)
            preds_scaled.extend(y_hat.detach().cpu().numpy().reshape(-1).tolist())
            labels_scaled.extend(labels.detach().cpu().numpy().reshape(-1).tolist())
            sample_ids.extend(batch_sample_ids)

    preds_raw = inverse_transform_fn(np.asarray(preds_scaled, dtype=np.float32))
    labels_raw = inverse_transform_fn(np.asarray(labels_scaled, dtype=np.float32))
    preds_eval, labels_eval = finalize_predictions(preds_raw, labels_raw, sample_ids, eval_level=eval_level)
    return _compute_regression_metrics(preds_eval, labels_eval)


def _method_metrics(
    dataset_key: str,
    method: str,
    pre_metrics: Dict[str, float],
    exc_metrics: Dict[str, float],
    rand_metrics: Dict[str, float],
    backbone_pre_norm: float,
) -> Dict[str, float]:
    original = _normal(pre_metrics["MAE"], dataset_key)
    cue_rm = _normal(exc_metrics["MAE"] - pre_metrics["MAE"], dataset_key)
    rand_rm = _normal(rand_metrics["MAE"] - pre_metrics["MAE"], dataset_key)
    cue_extra = _normal(exc_metrics["MAE"] - rand_metrics["MAE"], dataset_key)
    return {
        "Method": method,
        "Original": original,
        "Cost": original - backbone_pre_norm,
        "Cue rm.": cue_rm,
        "Rand rm.": rand_rm,
        "Cue extra": cue_extra,
        "Raw Original MAE": pre_metrics["MAE"],
        "Raw RMSE": pre_metrics["RMSE"],
        "R2": pre_metrics["R2"],
    }


def _evaluate_functional_methods(
    dataset_key: str,
    baseline_adapter,
    mitigation_adapter,
    cuefilter: CueFilter,
    cueaware_dm: CueAwarePreDataManager,
    exc_dm: RecordingSplitDataManager,
    rand_dm: RecordingSplitDataManager,
    device: torch.device,
    args,
) -> List[Dict[str, float]]:
    pre_loader = cueaware_dm.get_loader("test", shuffle=False)
    exc_loader = exc_dm.get_loader("test", shuffle=False)
    rand_loader = rand_dm.get_loader("test", shuffle=False)

    base_pre = _evaluate_gated_loader(baseline_adapter, pre_loader, cueaware_dm.inverse_transform, device, args.eval_level, "none")
    base_exc = _evaluate_gated_loader(baseline_adapter, exc_loader, exc_dm.inverse_transform, device, args.eval_level, "none")
    base_rand = _evaluate_gated_loader(baseline_adapter, rand_loader, rand_dm.inverse_transform, device, args.eval_level, "none")
    backbone_pre_norm = _normal(base_pre["MAE"], dataset_key)

    rows = [_method_metrics(dataset_key, "Backbone", base_pre, base_exc, base_rand, backbone_pre_norm)]
    for display, gate_method in [
        ("CueFilter", "cuefilter"),
        ("Agent mask", "agent"),
        ("Random gate", "random"),
        ("Uniform gate", "uniform"),
        ("Shuffled gate", "shuffled"),
    ]:
        pre = _evaluate_gated_loader(
            mitigation_adapter, pre_loader, cueaware_dm.inverse_transform, device, args.eval_level, gate_method, cuefilter, args
        )
        exc = _evaluate_gated_loader(
            mitigation_adapter, exc_loader, exc_dm.inverse_transform, device, args.eval_level, gate_method, cuefilter, args
        )
        rand = _evaluate_gated_loader(
            mitigation_adapter, rand_loader, rand_dm.inverse_transform, device, args.eval_level, gate_method, cuefilter, args
        )
        rows.append(_method_metrics(dataset_key, display, pre, exc, rand, backbone_pre_norm))
    return rows


def _evaluate_functional_methods_sequential(
    dataset_key: str,
    seed: int,
    baseline_adapter,
    mitigation_adapter,
    cuefilter: CueFilter,
    cueaware_dm: CueAwarePreDataManager,
    device: torch.device,
    args,
) -> List[Dict[str, float]]:
    method_specs = [
        ("CueFilter", "cuefilter"),
        ("Agent mask", "agent"),
        ("Random gate", "random"),
        ("Uniform gate", "uniform"),
        ("Shuffled gate", "shuffled"),
    ]

    metrics_by_method: Dict[str, Dict[str, Dict[str, float]]] = {"Backbone": {}}
    for display, _gate_method in method_specs:
        metrics_by_method[display] = {}

    for variant_name, slot in [("pre", "pre"), ("cue-excluded", "exc"), ("random", "rand")]:
        dm = _make_recording_data_manager(dataset_key, variant_name, seed, cueaware_dm.split_map, args)
        loader = dm.get_loader("test", shuffle=False)
        metrics_by_method["Backbone"][slot] = _evaluate_gated_loader(
            baseline_adapter, loader, dm.inverse_transform, device, args.eval_level, "none"
        )
        for display, gate_method in method_specs:
            metrics_by_method[display][slot] = _evaluate_gated_loader(
                mitigation_adapter, loader, dm.inverse_transform, device, args.eval_level, gate_method, cuefilter, args
            )
        del loader
        del dm

    backbone_pre_norm = _normal(metrics_by_method["Backbone"]["pre"]["MAE"], dataset_key)
    rows = []
    for method in ["Backbone"] + [display for display, _gate_method in method_specs]:
        rows.append(
            _method_metrics(
                dataset_key,
                method,
                metrics_by_method[method]["pre"],
                metrics_by_method[method]["exc"],
                metrics_by_method[method]["rand"],
                backbone_pre_norm,
            )
        )
    return rows


def _evaluate_strength_methods(
    dataset_key: str,
    mitigation_adapter,
    cuefilter: CueFilter,
    cueaware_dm: CueAwarePreDataManager,
    exc_dm: RecordingSplitDataManager,
    rand_dm: RecordingSplitDataManager,
    backbone_pre_norm: float,
    device: torch.device,
    args,
) -> List[Dict[str, float]]:
    pre_loader = cueaware_dm.get_loader("test", shuffle=False)
    exc_loader = exc_dm.get_loader("test", shuffle=False)
    rand_loader = rand_dm.get_loader("test", shuffle=False)
    rows = []
    alpha0_extra = None

    for alpha in args.strengths:
        pre = _evaluate_gated_loader(
            mitigation_adapter, pre_loader, cueaware_dm.inverse_transform, device, args.eval_level, "cuefilter", cuefilter, args, alpha
        )
        exc = _evaluate_gated_loader(
            mitigation_adapter, exc_loader, exc_dm.inverse_transform, device, args.eval_level, "cuefilter", cuefilter, args, alpha
        )
        rand = _evaluate_gated_loader(
            mitigation_adapter, rand_loader, rand_dm.inverse_transform, device, args.eval_level, "cuefilter", cuefilter, args, alpha
        )
        row = _method_metrics(dataset_key, "CueFilter", pre, exc, rand, backbone_pre_norm)
        row["Strength"] = float(alpha)
        if float(alpha) == 0.0:
            alpha0_extra = row["Cue extra"]
        rows.append(row)

    if alpha0_extra is None and rows:
        alpha0_extra = rows[0]["Cue extra"]

    for display, gate_method in [("Random gate", "random"), ("Uniform gate", "uniform")]:
        pre = _evaluate_gated_loader(
            mitigation_adapter, pre_loader, cueaware_dm.inverse_transform, device, args.eval_level, gate_method, cuefilter, args, args.alpha
        )
        exc = _evaluate_gated_loader(
            mitigation_adapter, exc_loader, exc_dm.inverse_transform, device, args.eval_level, gate_method, cuefilter, args, args.alpha
        )
        rand = _evaluate_gated_loader(
            mitigation_adapter, rand_loader, rand_dm.inverse_transform, device, args.eval_level, gate_method, cuefilter, args, args.alpha
        )
        row = _method_metrics(dataset_key, display, pre, exc, rand, backbone_pre_norm)
        row["Strength"] = float(args.alpha)
        rows.append(row)

    for row in rows:
        denom = alpha0_extra if alpha0_extra and abs(alpha0_extra) > 1e-6 else np.nan
        row["Extra red."] = float((alpha0_extra - row["Cue extra"]) / denom * 100.0) if np.isfinite(denom) else np.nan
    return rows


def _pool_features(sequence_features: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            sequence_features.mean(dim=1),
            sequence_features.std(dim=1, unbiased=False),
        ],
        dim=1,
    )


def _gate_stats(gate: torch.Tensor, cue_threshold: float, gamma: float) -> torch.Tensor:
    strong = (gate <= gamma + 1e-6).float()
    attenuated = (gate < 1.0 - 1e-6).float()
    return torch.stack(
        [
            gate.mean(dim=1),
            gate.std(dim=1, unbiased=False),
            gate.min(dim=1).values,
            gate.max(dim=1).values,
            attenuated.mean(dim=1),
            strong.mean(dim=1),
        ],
        dim=1,
    )


def _aggregate_feature_rows(features: np.ndarray, labels: np.ndarray, sample_ids: Sequence[str]):
    grouped = defaultdict(lambda: {"x": [], "y": []})
    for x, y, sample_id in zip(features, labels, sample_ids):
        grouped[str(sample_id)]["x"].append(np.asarray(x, dtype=np.float32))
        grouped[str(sample_id)]["y"].append(float(y))
    out_ids = sorted(grouped)
    x_out = np.stack([np.mean(grouped[sid]["x"], axis=0) for sid in out_ids], axis=0)
    y_out = np.asarray([np.mean(grouped[sid]["y"]) for sid in out_ids], dtype=np.float32)
    return x_out, y_out, out_ids


def _extract_feature_matrix(
    feature_method: str,
    split: str,
    variant_loader,
    inverse_transform_fn,
    baseline_adapter,
    mitigation_adapter,
    cuefilter: CueFilter,
    device: torch.device,
    args,
):
    baseline_adapter.eval()
    mitigation_adapter.eval()
    cuefilter.eval()
    all_features: List[np.ndarray] = []
    all_labels_scaled: List[float] = []
    all_sample_ids: List[str] = []

    with torch.no_grad():
        for batch in variant_loader:
            audios, labels, sample_ids, cue_spans, durations = _unpack_batch(batch, device)
            if feature_method == "Backbone feature":
                sequence_features, _aux = baseline_adapter.encode(audios)
                features = _pool_features(sequence_features)
            elif feature_method == "Backbone+gate stats":
                baseline_seq, _aux = baseline_adapter.encode(audios)
                mitigation_seq, _aux = mitigation_adapter.encode(audios)
                cue_outputs = cuefilter(mitigation_seq, mode="soft", renorm=False)
                features = torch.cat(
                    [_pool_features(baseline_seq), _gate_stats(cue_outputs["gate"], args.cue_threshold, args.gamma)],
                    dim=1,
                )
            else:
                sequence_features, _aux = mitigation_adapter.encode(audios)
                if feature_method == "CueFilter feature":
                    filtered, _gate = _apply_custom_gate("cuefilter", sequence_features, cuefilter, device, args)
                elif feature_method == "Random-gated feature":
                    filtered, _gate = _apply_custom_gate("random", sequence_features, cuefilter, device, args)
                elif feature_method == "Agent-mask feature":
                    filtered, _gate = _apply_custom_gate(
                        "agent", sequence_features, cuefilter, device, args, cue_spans=cue_spans, durations=durations
                    )
                else:
                    raise ValueError(f"Unsupported feature method: {feature_method}")
                features = _pool_features(filtered)
            all_features.append(features.detach().cpu().numpy())
            all_labels_scaled.extend(labels.detach().cpu().numpy().reshape(-1).tolist())
            all_sample_ids.extend(sample_ids)

    x_seg = np.concatenate(all_features, axis=0)
    y_raw = inverse_transform_fn(np.asarray(all_labels_scaled, dtype=np.float32))
    return _aggregate_feature_rows(x_seg, y_raw, all_sample_ids)


def _build_feature_regressors(seed: int):
    return {
        "Ridge": RidgeCV(alphas=RIDGE_ALPHAS),
        "SVR": SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale"),
        "RF": RandomForestRegressor(n_estimators=200, min_samples_leaf=1, random_state=seed, n_jobs=-1),
    }


def _fit_eval_regressor(regressor, x_train, y_train, x_test, y_test) -> Dict[str, float]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    regressor.fit(x_train_scaled, y_train)
    preds = np.asarray(regressor.predict(x_test_scaled), dtype=np.float32)
    preds = np.clip(preds, float(np.min(y_train)), float(np.max(y_train)))
    return _compute_regression_metrics(preds, y_test)


def _evaluate_frozen_features(
    dataset_key: str,
    baseline_adapter,
    mitigation_adapter,
    cuefilter: CueFilter,
    cueaware_dm: CueAwarePreDataManager,
    exc_dm: RecordingSplitDataManager,
    rand_dm: RecordingSplitDataManager,
    device: torch.device,
    args,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    train_loader = cueaware_dm.get_loader("train", shuffle=False)
    pre_loader = cueaware_dm.get_loader("test", shuffle=False)
    exc_loader = exc_dm.get_loader("test", shuffle=False)
    rand_loader = rand_dm.get_loader("test", shuffle=False)

    for feature_method in FEATURE_METHODS:
        x_train, y_train, _train_ids = _extract_feature_matrix(
            feature_method,
            "train",
            train_loader,
            cueaware_dm.inverse_transform,
            baseline_adapter,
            mitigation_adapter,
            cuefilter,
            device,
            args,
        )
        x_pre, y_pre, _pre_ids = _extract_feature_matrix(
            feature_method,
            "test",
            pre_loader,
            cueaware_dm.inverse_transform,
            baseline_adapter,
            mitigation_adapter,
            cuefilter,
            device,
            args,
        )
        x_exc, y_exc, _exc_ids = _extract_feature_matrix(
            feature_method,
            "test",
            exc_loader,
            exc_dm.inverse_transform,
            baseline_adapter,
            mitigation_adapter,
            cuefilter,
            device,
            args,
        )
        x_rand, y_rand, _rand_ids = _extract_feature_matrix(
            feature_method,
            "test",
            rand_loader,
            rand_dm.inverse_transform,
            baseline_adapter,
            mitigation_adapter,
            cuefilter,
            device,
            args,
        )

        for reg_name, regressor in _build_feature_regressors(args.seed_for_regressors).items():
            pre = _fit_eval_regressor(copy.deepcopy(regressor), x_train, y_train, x_pre, y_pre)
            exc = _fit_eval_regressor(copy.deepcopy(regressor), x_train, y_train, x_exc, y_exc)
            rand = _fit_eval_regressor(copy.deepcopy(regressor), x_train, y_train, x_rand, y_rand)
            rows.append(
                {
                    "Feature used": feature_method,
                    "Regressor": reg_name,
                    "Original": _normal(pre["MAE"], dataset_key),
                    "Cue rm.": _normal(exc["MAE"] - pre["MAE"], dataset_key),
                    "Rand rm.": _normal(rand["MAE"] - pre["MAE"], dataset_key),
                    "Cue extra": _normal(exc["MAE"] - rand["MAE"], dataset_key),
                }
            )
    return rows


def _span_iou(span_a: Tuple[float, float], span_b: Tuple[float, float]) -> float:
    inter = max(0.0, min(span_a[1], span_b[1]) - max(span_a[0], span_b[0]))
    union = max(0.0, span_a[1] - span_a[0]) + max(0.0, span_b[1] - span_b[0]) - inter
    return inter / union if union > 0 else 0.0


def _cue_category_summary(dataset_key: str, cue_role: str) -> Dict[str, List[str]]:
    summary: Dict[str, List[str]] = defaultdict(list)
    for sample in iter_output_samples(dataset_key):
        sample_id = str(sample["sample_id"])
        transcript = load_json(Path(sample["transcript_path"]))
        cues = load_json(Path(sample["cues_path"]))
        ann_path = annotation_path(dataset_key, sample_id)
        original = cues.get("cues", [])
        selected_spans: List[Tuple[float, float]] = []
        if ann_path.exists():
            annotation = load_json(ann_path)
            for cue in annotation.get("cues", []):
                if cue.get("status") == "deleted":
                    continue
                span = cue.get("corrected_span") or cue.get("original_span") or {}
                start = span.get("start")
                end = span.get("end")
                if start is None or end is None:
                    continue
                start = float(start)
                end = float(end)
                if end > start and role_matches_filter(cue_role_from_overlap(start, end, transcript), cue_role):
                    selected_spans.append((start, end))
        else:
            for cue in original:
                role = cue.get("speaker_role") or cue.get("speaker")
                if not role_matches_filter(role, cue_role):
                    continue
                start = cue.get("start")
                end = cue.get("end")
                if start is not None and end is not None and float(end) > float(start):
                    selected_spans.append((float(start), float(end)))

        for span in selected_spans:
            best = None
            best_iou = 0.0
            for cue in original:
                start = cue.get("start")
                end = cue.get("end")
                if start is None or end is None:
                    continue
                iou = _span_iou(span, (float(start), float(end)))
                if iou > best_iou:
                    best_iou = iou
                    best = cue
            category = str((best or {}).get("category", "unknown"))
            if category and category != "unknown":
                summary[sample_id].append(category)
    return {key: sorted(set(values)) for key, values in summary.items()}


def _cue_numeric_summary(cueaware_dm: CueAwarePreDataManager) -> Dict[str, Dict[str, float]]:
    grouped = defaultdict(lambda: {"duration": 0.0, "cue_duration": 0.0, "count": 0})
    for item in cueaware_dm.items:
        sample_id = str(item["sample_id"])
        duration = float(item.get("duration_sec", cueaware_dm.segment_length))
        spans = item.get("cue_spans_sec", []) or []
        grouped[sample_id]["duration"] += duration
        grouped[sample_id]["cue_duration"] += sum(max(0.0, float(e) - float(s)) for s, e in spans)
        grouped[sample_id]["count"] += len(spans)
    return {
        sample_id: {
            "coverage": values["cue_duration"] / values["duration"] if values["duration"] > 0 else 0.0,
            "count": float(values["count"]),
        }
        for sample_id, values in grouped.items()
    }


def _safe_auc(y_true, y_score) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return np.nan


def _safe_average_precision(y_true, y_score) -> float:
    try:
        valid_cols = [idx for idx in range(y_true.shape[1]) if len(np.unique(y_true[:, idx])) > 1]
        if not valid_cols:
            return np.nan
        return float(average_precision_score(y_true[:, valid_cols], y_score[:, valid_cols], average="macro"))
    except Exception:
        return np.nan


def _evaluate_probe_features(
    dataset_key: str,
    baseline_adapter,
    mitigation_adapter,
    cuefilter: CueFilter,
    cueaware_dm: CueAwarePreDataManager,
    device: torch.device,
    args,
) -> List[Dict[str, object]]:
    numeric_summary = _cue_numeric_summary(cueaware_dm)
    category_summary = _cue_category_summary(dataset_key, args.cue_role)
    train_loader = cueaware_dm.get_loader("train", shuffle=False)
    test_loader = cueaware_dm.get_loader("test", shuffle=False)
    rows: List[Dict[str, object]] = []

    for feature_method in PROBE_FEATURE_METHODS:
        x_train, y_train_dep, train_ids = _extract_feature_matrix(
            feature_method,
            "train",
            train_loader,
            cueaware_dm.inverse_transform,
            baseline_adapter,
            mitigation_adapter,
            cuefilter,
            device,
            args,
        )
        x_test, y_test_dep, test_ids = _extract_feature_matrix(
            feature_method,
            "test",
            test_loader,
            cueaware_dm.inverse_transform,
            baseline_adapter,
            mitigation_adapter,
            cuefilter,
            device,
            args,
        )

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        train_cov = np.asarray([numeric_summary.get(sid, {}).get("coverage", 0.0) for sid in train_ids], dtype=np.float32)
        test_cov = np.asarray([numeric_summary.get(sid, {}).get("coverage", 0.0) for sid in test_ids], dtype=np.float32)
        cov_threshold = float(np.median(train_cov))
        y_cov_train = (train_cov > cov_threshold).astype(int)
        y_cov_test = (test_cov > cov_threshold).astype(int)
        try:
            cov_probe = LogisticRegression(max_iter=1000, class_weight="balanced")
            cov_probe.fit(x_train_scaled, y_cov_train)
            cov_score = cov_probe.predict_proba(x_test_scaled)[:, 1]
            coverage_auc = _safe_auc(y_cov_test, cov_score)
        except Exception:
            coverage_auc = np.nan

        train_categories = [category_summary.get(sid, []) for sid in train_ids]
        test_categories = [category_summary.get(sid, []) for sid in test_ids]
        try:
            mlb = MultiLabelBinarizer()
            y_cat_train = mlb.fit_transform(train_categories)
            y_cat_test = mlb.transform(test_categories)
            if y_cat_train.shape[1] == 0:
                category_map = np.nan
            else:
                cat_probe = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))
                cat_probe.fit(x_train_scaled, y_cat_train)
                cat_score = cat_probe.predict_proba(x_test_scaled)
                category_map = _safe_average_precision(y_cat_test, cat_score)
        except Exception:
            category_map = np.nan

        train_count = np.asarray([numeric_summary.get(sid, {}).get("count", 0.0) for sid in train_ids], dtype=np.float32)
        test_count = np.asarray([numeric_summary.get(sid, {}).get("count", 0.0) for sid in test_ids], dtype=np.float32)
        q1, q2 = np.quantile(train_count, [1.0 / 3.0, 2.0 / 3.0])
        y_count_train = np.digitize(train_count, [q1, q2], right=True)
        y_count_test = np.digitize(test_count, [q1, q2], right=True)
        try:
            count_probe = LogisticRegression(max_iter=1000, class_weight="balanced")
            count_probe.fit(x_train_scaled, y_count_train)
            count_pred = count_probe.predict(x_test_scaled)
            count_acc = float(accuracy_score(y_count_test, count_pred))
        except Exception:
            count_acc = np.nan

        dep_metrics = _fit_eval_regressor(RidgeCV(alphas=RIDGE_ALPHAS), x_train, y_train_dep, x_test, y_test_dep)
        rows.append(
            {
                "Feature used": feature_method,
                "Coverage": coverage_auc,
                "Category": category_map,
                "Count": count_acc,
                "Dep. error": _normal(dep_metrics["MAE"], dataset_key),
            }
        )
    return rows


def _summarize(df: pd.DataFrame, group_cols: Sequence[str], value_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    for keys, group in df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        for value_col in value_cols:
            row[value_col] = format_mean_std(group[value_col].astype(float).tolist())
        row["N"] = int(len(group))
        rows.append(row)
    return pd.DataFrame(rows)


def _write_pair(output_dir: Path, stem: str, raw_rows: List[Dict[str, object]], group_cols: Sequence[str], value_cols: Sequence[str]) -> None:
    raw_df = pd.DataFrame(raw_rows)
    raw_path = output_dir / f"{stem}_raw.csv"
    summary_path = output_dir / f"{stem}.csv"
    raw_df.to_csv(raw_path, index=False)
    summary_df = _summarize(raw_df, group_cols, value_cols)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved {raw_path}")
    print(f"Saved {summary_path}")
    if not summary_df.empty:
        print("\n" + summary_df.to_markdown(index=False))


def run_single(dataset_key: str, model_name: str, seed: int, args):
    effective_args = _with_model_defaults(args, model_name)
    effective_args.seed_for_regressors = seed
    set_seed(seed)
    device = torch.device(effective_args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if effective_args.sequential_variant_load:
        if not effective_args.skip_frozen or not effective_args.skip_probe:
            raise ValueError("--sequential-variant-load requires --skip-frozen and --skip-probe")
        cueaware_dm = _make_cueaware_data_manager(dataset_key, seed, effective_args)
        pre_dm = exc_dm = rand_dm = None
    else:
        cueaware_dm, pre_dm, exc_dm, rand_dm = _make_data_managers(dataset_key, seed, effective_args)
    if effective_args.info_only:
        print(f"[{DATASET_CONFIGS[dataset_key].display_name} | {model_name} | seed={seed}] {cueaware_dm.get_info()}")
        return [], [], [], []

    baseline_adapter = build_adapter(model_name).to(device)
    baseline_adapter = _train_baseline_adapter(
        adapter=baseline_adapter,
        train_loader=cueaware_dm.get_loader("train", shuffle=True),
        val_loader=cueaware_dm.get_loader("val", shuffle=False),
        inverse_transform_fn=cueaware_dm.inverse_transform,
        device=device,
        args=effective_args,
    )

    mitigation_adapter = copy.deepcopy(baseline_adapter).to(device)
    cuefilter = _build_cuefilter(mitigation_adapter, effective_args, device)
    shared_stage1_metrics = _initialize_shared_cuefilter_if_needed(
        cuefilter=cuefilter,
        adapter=mitigation_adapter,
        model_name=model_name,
        seed=seed,
        device=device,
        args=effective_args,
    )
    if shared_stage1_metrics is None:
        cuefilter, _stage1_metrics = _train_stage1_cuefilter(
            cuefilter=cuefilter,
            adapter=mitigation_adapter,
            train_loader=cueaware_dm.get_loader("train", shuffle=True),
            val_loader=cueaware_dm.get_loader("val", shuffle=False),
            device=device,
            args=effective_args,
        )

    cuefilter, mitigation_adapter, _stage2_metrics = _train_stage2_joint(
        cuefilter=cuefilter,
        adapter=mitigation_adapter,
        train_loader=cueaware_dm.get_loader("train", shuffle=True),
        val_loader=cueaware_dm.get_loader("val", shuffle=False),
        inverse_transform_fn=cueaware_dm.inverse_transform,
        device=device,
        args=effective_args,
    )

    if effective_args.sequential_variant_load:
        functional_rows = _evaluate_functional_methods_sequential(
            dataset_key,
            seed,
            baseline_adapter,
            mitigation_adapter,
            cuefilter,
            cueaware_dm,
            device,
            effective_args,
        )
    else:
        functional_rows = _evaluate_functional_methods(
            dataset_key,
            baseline_adapter,
            mitigation_adapter,
            cuefilter,
            cueaware_dm,
            exc_dm,
            rand_dm,
            device,
            effective_args,
        )
    backbone_pre_norm = next(row["Original"] for row in functional_rows if row["Method"] == "Backbone")
    strength_rows: List[Dict[str, object]] = []
    if not effective_args.skip_strength:
        if effective_args.sequential_variant_load:
            raise ValueError("--sequential-variant-load requires --skip-strength")
        strength_rows = _evaluate_strength_methods(
            dataset_key,
            mitigation_adapter,
            cuefilter,
            cueaware_dm,
            exc_dm,
            rand_dm,
            backbone_pre_norm,
            device,
            effective_args,
        )

    frozen_rows: List[Dict[str, object]] = []
    if not effective_args.skip_frozen:
        frozen_rows = _evaluate_frozen_features(
            dataset_key,
            baseline_adapter,
            mitigation_adapter,
            cuefilter,
            cueaware_dm,
            exc_dm,
            rand_dm,
            device,
            effective_args,
        )

    probe_rows: List[Dict[str, object]] = []
    if not effective_args.skip_probe:
        probe_rows = _evaluate_probe_features(
            dataset_key,
            baseline_adapter,
            mitigation_adapter,
            cuefilter,
            cueaware_dm,
            device,
            effective_args,
        )

    dataset_name = DATASET_CONFIGS[dataset_key].display_name
    common = {
        "Dataset": dataset_name,
        "DatasetKey": dataset_key,
        "Backbone": model_name,
        "Seed": seed,
        "CueRole": effective_args.cue_role,
        "SpeechScope": effective_args.speech_scope,
        "EvalLevel": effective_args.eval_level,
        "SegmentFirst": effective_args.segment_first,
    }
    for rows in (functional_rows, frozen_rows, strength_rows, probe_rows):
        for row in rows:
            row.update(common)
    return functional_rows, frozen_rows, strength_rows, probe_rows


def main() -> None:
    args = create_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    functional_rows: List[Dict[str, object]] = []
    frozen_rows: List[Dict[str, object]] = []
    strength_rows: List[Dict[str, object]] = []
    probe_rows: List[Dict[str, object]] = []

    for dataset_key in args.datasets:
        dataset_name = DATASET_CONFIGS[dataset_key].display_name
        for model_name in args.models:
            for seed in args.seeds:
                print(f"Running functional CueFilter | {dataset_name} | {model_name} | seed={seed}", flush=True)
                try:
                    f_rows, fr_rows, s_rows, p_rows = run_single(dataset_key, model_name, seed, args)
                except Exception as exc:
                    print(f"  failed: {exc}", flush=True)
                    continue
                functional_rows.extend(f_rows)
                frozen_rows.extend(fr_rows)
                strength_rows.extend(s_rows)
                probe_rows.extend(p_rows)

    if functional_rows:
        _write_pair(
            output_dir,
            "table_cuefilter_functional",
            functional_rows,
            ["Method"],
            ["Original", "Cost", "Cue rm.", "Rand rm.", "Cue extra"],
        )
    if frozen_rows:
        _write_pair(
            output_dir,
            "table_frozen_feature_reuse",
            frozen_rows,
            ["Feature used"],
            ["Original", "Cue rm.", "Rand rm.", "Cue extra"],
        )
    if strength_rows:
        _write_pair(
            output_dir,
            "table_suppression_strength",
            strength_rows,
            ["Method", "Strength"],
            ["Original", "Cost", "Cue extra", "Extra red."],
        )
    if probe_rows:
        _write_pair(
            output_dir,
            "table_semantic_probe",
            probe_rows,
            ["Feature used"],
            ["Coverage", "Category", "Count", "Dep. error"],
        )


if __name__ == "__main__":
    main()
