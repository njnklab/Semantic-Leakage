from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .Baseline.audio_views import DATASET_CONFIGS
    from .Baseline.experiment_utils import RecordingSplitDataManager, finalize_predictions, format_mean_std, safe_r2_score, set_seed
    from .integration.common import build_cue_supervision_batch, estimate_sequence_feature_stats
    from .integration.data import CueAwarePreDataManager
    from .integration.registry import build_adapter, list_supported_adapters
    from .models import CueFilter
    from .run_mitigation_experiments import (
        _evaluate_stage1,
        _evaluate_stage2_joint,
        _evaluate_variant,
        _initialize_shared_cuefilter_if_needed,
        _train_baseline_adapter,
        _train_stage1_cuefilter,
        _train_stage2_joint,
        _with_model_defaults,
    )
except ImportError:
    from CueFilter.Baseline.audio_views import DATASET_CONFIGS
    from CueFilter.Baseline.experiment_utils import RecordingSplitDataManager, finalize_predictions, format_mean_std, safe_r2_score, set_seed
    from CueFilter.integration.common import build_cue_supervision_batch, estimate_sequence_feature_stats
    from CueFilter.integration.data import CueAwarePreDataManager
    from CueFilter.integration.registry import build_adapter, list_supported_adapters
    from CueFilter.models import CueFilter
    from CueFilter.run_mitigation_experiments import (
        _evaluate_stage1,
        _evaluate_stage2_joint,
        _evaluate_variant,
        _initialize_shared_cuefilter_if_needed,
        _train_baseline_adapter,
        _train_stage1_cuefilter,
        _train_stage2_joint,
        _with_model_defaults,
    )


VARIANT_DISPLAY = [
    "Backbone",
    "Full CueFilter",
    "No cue pretrain",
    "No cue loss",
    "No budget loss",
    "No feature norm",
    "Hard mask",
    "Random gate",
    "Shuffled gate",
]

LABEL_RANGES = {
    "edaic": 24.0,
    "cmdc": 27.0,
    "pdch": 52.0,
    "mandic": 52.0,
}


@dataclass
class AblationVariantMetrics:
    error: float
    cost: float
    cue_rm: float
    cue_extra: float
    gate_density: float
    density_err: float
    raw_original_mae: float
    raw_cue_removed_mae: float
    raw_random_removed_mae: float
    prediction_rows: Optional[List[Dict[str, object]]] = None


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CueFilter ablation experiments.")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_CONFIGS.keys()), default=["mandic"])
    parser.add_argument("--models", nargs="+", choices=list_supported_adapters(), default=["DepAudioNet"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--cue-role", choices=["patient", "doctor", "all"], default="patient")
    parser.add_argument("--speech-scope", choices=["participant", "interviewer", "dialogue"], default="participant")
    parser.add_argument(
        "--shared-pretrain-datasets",
        nargs="+",
        choices=list(DATASET_CONFIGS.keys()),
        default=None,
    )
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
    parser.add_argument(
        "--boundary-tolerance-sec",
        type=float,
        default=1.0,
        help="Boundary tolerance converted to each backbone frame grid; uses the largest whole-frame expansion within this value before normal grid quantization.",
    )
    parser.add_argument("--max-groups-per-split", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval-level", choices=["recording", "sample"], default="sample")
    parser.add_argument("--segment-first", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--detail-output-dir", type=str, default=None)
    parser.add_argument("--info-only", action="store_true")
    return parser


def _compute_regression_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(labels - preds)))
    rmse = float(np.sqrt(np.mean((labels - preds) ** 2)))
    r2 = float(safe_r2_score(labels, preds))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def _score_range(dataset_key: str) -> float:
    return LABEL_RANGES.get(dataset_key, 1.0)


def _normal(value: float, dataset_key: str) -> float:
    return float(value) / _score_range(dataset_key)


def _clone_args(args, **updates):
    cloned = argparse.Namespace(**vars(args))
    for key, value in updates.items():
        setattr(cloned, key, value)
    return cloned


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


def _make_data_managers(dataset_key: str, seed: int, args):
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
        max_groups_per_split=args.max_groups_per_split,
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
        max_groups_per_split=args.max_groups_per_split,
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
        max_groups_per_split=args.max_groups_per_split,
        eval_level=args.eval_level,
        segment_first=args.segment_first,
    )
    return pre_dm, exc_dm, rand_dm, cueaware_dm


def _estimate_sequence_stats_on_segment_loader(adapter, loader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    adapter.eval()
    total_sum = None
    total_sq_sum = None
    total_frames = 0

    with torch.no_grad():
        for audios, _labels, _sample_ids in loader:
            audios = audios.to(device, non_blocking=True)
            sequence_features, _aux = adapter.encode(audios)
            batch_sum = sequence_features.sum(dim=(0, 1)).double()
            batch_sq_sum = sequence_features.pow(2).sum(dim=(0, 1)).double()
            if total_sum is None:
                total_sum = batch_sum
                total_sq_sum = batch_sq_sum
            else:
                total_sum += batch_sum
                total_sq_sum += batch_sq_sum
            total_frames += sequence_features.shape[0] * sequence_features.shape[1]

    mean = total_sum / total_frames
    var = total_sq_sum / total_frames - mean.pow(2)
    std = var.clamp_min(1e-6).sqrt()
    return mean.float(), std.float()


def _renormalize_with_stats(features: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    sample_mean = features.mean(dim=1, keepdim=True)
    sample_std = features.std(dim=1, unbiased=False, keepdim=True).clamp_min(eps)
    target_mean = mean.to(features).view(1, 1, -1)
    target_std = std.to(features).view(1, 1, -1).clamp_min(eps)
    return (features - sample_mean) / sample_std * target_std + target_mean


def _evaluate_adapter_on_loader(
    adapter,
    loader,
    inverse_transform_fn,
    device,
    eval_level: str,
) -> Dict[str, float]:
    adapter.eval()
    preds_scaled = []
    labels_scaled = []
    sample_ids = []
    with torch.no_grad():
        for audios, labels, batch_sample_ids in loader:
            audios = audios.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).squeeze(-1)
            sequence_features, _aux = adapter.encode(audios)
            y_hat = adapter.predict_from_sequence(sequence_features)
            if y_hat.ndim > 1 and y_hat.shape[-1] == 1:
                y_hat = y_hat.squeeze(-1)
            preds_scaled.extend(y_hat.detach().cpu().numpy().tolist())
            labels_scaled.extend(labels.detach().cpu().numpy().tolist())
            sample_ids.extend(list(batch_sample_ids))

    preds_raw = inverse_transform_fn(np.asarray(preds_scaled, dtype=np.float32))
    labels_raw = inverse_transform_fn(np.asarray(labels_scaled, dtype=np.float32))
    preds_eval, labels_eval = finalize_predictions(preds_raw, labels_raw, sample_ids, eval_level=eval_level)
    return _compute_regression_metrics(preds_eval, labels_eval)


def _unpack_eval_batch(batch, device: torch.device):
    if isinstance(batch, dict):
        audios = batch["audio"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).squeeze(-1)
        sample_ids = [str(item) for item in batch["sample_id"]]
        return audios, labels, sample_ids
    audios, labels, sample_ids = batch
    return audios.to(device, non_blocking=True), labels.to(device, non_blocking=True).squeeze(-1), [str(item) for item in sample_ids]


def _randomize_gate(gate: torch.Tensor) -> torch.Tensor:
    pieces = [row[torch.randperm(row.shape[0], device=row.device)] for row in gate]
    return torch.stack(pieces, dim=0)


def _shuffled_gate(gate: torch.Tensor) -> torch.Tensor:
    if gate.shape[0] > 1:
        return torch.roll(gate, shifts=1, dims=0)
    return _randomize_gate(gate)


def _apply_gate_strategy(
    sequence_features: torch.Tensor,
    cuefilter: Optional[CueFilter],
    gate_strategy: str,
    gate_mode: str,
    use_renorm: bool,
) -> torch.Tensor:
    if cuefilter is None or gate_strategy == "none":
        return sequence_features
    if gate_strategy == "cuefilter":
        outputs = cuefilter(sequence_features, mode=gate_mode, renorm=use_renorm)
        return outputs["renormed_features"]

    base_outputs = cuefilter(sequence_features, mode=gate_mode, renorm=False)
    if gate_strategy == "random":
        gate = _randomize_gate(base_outputs["gate"])
    elif gate_strategy == "shuffled":
        gate = _shuffled_gate(base_outputs["gate"])
    else:
        raise ValueError(f"Unsupported gate strategy: {gate_strategy}")
    filtered = gate.unsqueeze(-1) * sequence_features
    return cuefilter.renormalize(filtered) if use_renorm else filtered


def _evaluate_gate_variant(
    adapter,
    data_manager,
    device: torch.device,
    cuefilter: Optional[CueFilter] = None,
    gate_strategy: str = "none",
    gate_mode: str = "soft",
    use_renorm: bool = True,
) -> Dict[str, float]:
    loader = data_manager.get_loader("test", shuffle=False)
    adapter.eval()
    if cuefilter is not None:
        cuefilter.eval()

    preds_scaled: List[float] = []
    labels_scaled: List[float] = []
    sample_ids: List[str] = []

    with torch.no_grad():
        for batch in loader:
            audios, labels, batch_sample_ids = _unpack_eval_batch(batch, device)
            sequence_features, _aux = adapter.encode(audios)
            pred_features = _apply_gate_strategy(sequence_features, cuefilter, gate_strategy, gate_mode, use_renorm)
            y_hat = adapter.predict_from_sequence(pred_features)
            if y_hat.ndim > 1 and y_hat.shape[-1] == 1:
                y_hat = y_hat.squeeze(-1)
            preds_scaled.extend(y_hat.detach().cpu().numpy().reshape(-1).tolist())
            labels_scaled.extend(labels.detach().cpu().numpy().reshape(-1).tolist())
            sample_ids.extend(batch_sample_ids)

    preds_raw = data_manager.inverse_transform(np.asarray(preds_scaled, dtype=np.float32))
    labels_raw = data_manager.inverse_transform(np.asarray(labels_scaled, dtype=np.float32))
    preds_eval, labels_eval = finalize_predictions(preds_raw, labels_raw, sample_ids, eval_level=data_manager.eval_level)
    return _compute_regression_metrics(preds_eval, labels_eval)


def _evaluate_gate_density(
    adapter,
    cuefilter: Optional[CueFilter],
    cueaware_dm: CueAwarePreDataManager,
    device: torch.device,
    args,
    gate_strategy: str = "cuefilter",
    gate_mode: str = "soft",
) -> Tuple[float, float]:
    if cuefilter is None or gate_strategy == "none":
        return float("nan"), float("nan")

    adapter.eval()
    cuefilter.eval()
    densities: List[float] = []
    coverages: List[float] = []

    with torch.no_grad():
        for batch in cueaware_dm.get_loader("test", shuffle=False):
            audios = batch["audio"].to(device, non_blocking=True)
            sequence_features, _aux = adapter.encode(audios)
            cue_labels, cue_coverage = build_cue_supervision_batch(
                cue_spans_batch=batch["cue_spans_sec"],
                num_frames=sequence_features.shape[1],
                durations_sec=batch["duration_sec"],
                device=device,
                expansion_frames=args.boundary_expand_frames,
                expansion_sec=getattr(args, "boundary_tolerance_sec", None),
            )
            outputs = cuefilter(sequence_features, mode=gate_mode, renorm=False)
            if gate_mode == "binary":
                gate_density = (outputs["p_cue"] >= args.cue_threshold).float().mean(dim=1)
            else:
                gate_density = outputs["p_cue"].mean(dim=1)
            densities.extend(gate_density.detach().cpu().numpy().reshape(-1).tolist())
            coverages.extend(cue_coverage.detach().cpu().numpy().reshape(-1).tolist())

    gate_density = float(np.mean(densities)) if densities else float("nan")
    if gate_strategy in {"random", "shuffled"}:
        return gate_density, float("nan")
    density_err = float(np.mean(np.abs(np.asarray(densities, dtype=np.float32) - np.asarray(coverages, dtype=np.float32)))) if densities else float("nan")
    return gate_density, density_err


def _build_ablation_metrics(
    dataset_key: str,
    baseline_error_norm: float,
    adapter,
    cuefilter: Optional[CueFilter],
    pre_dm: RecordingSplitDataManager,
    exc_dm: RecordingSplitDataManager,
    rand_dm: RecordingSplitDataManager,
    cueaware_dm: CueAwarePreDataManager,
    device: torch.device,
    args,
    gate_strategy: str = "cuefilter",
    gate_mode: str = "soft",
    use_renorm: bool = True,
) -> AblationVariantMetrics:
    pre = _evaluate_gate_variant(adapter, pre_dm, device, cuefilter, gate_strategy, gate_mode, use_renorm)
    exc = _evaluate_gate_variant(adapter, exc_dm, device, cuefilter, gate_strategy, gate_mode, use_renorm)
    rand = _evaluate_gate_variant(adapter, rand_dm, device, cuefilter, gate_strategy, gate_mode, use_renorm)
    error = _normal(pre["MAE"], dataset_key)
    gate_density, density_err = _evaluate_gate_density(adapter, cuefilter, cueaware_dm, device, args, gate_strategy, gate_mode)
    return AblationVariantMetrics(
        error=error,
        cost=error - baseline_error_norm,
        cue_rm=_normal(exc["MAE"] - pre["MAE"], dataset_key),
        cue_extra=_normal(exc["MAE"] - rand["MAE"], dataset_key),
        gate_density=gate_density,
        density_err=density_err,
        raw_original_mae=pre["MAE"],
        raw_cue_removed_mae=exc["MAE"],
        raw_random_removed_mae=rand["MAE"],
    )


def _mean_cue_coverage(cueaware_dm: CueAwarePreDataManager) -> float:
    records = cueaware_dm.get_items("train")
    coverages = []
    for record in records:
        cue_spans = record.get("cue_spans_sec", []) or []
        duration_sec = float(record.get("duration_sec", cueaware_dm.segment_length))
        cue_duration = sum(max(0.0, float(end) - float(start)) for start, end in cue_spans)
        if duration_sec > 0:
            coverages.append(cue_duration / duration_sec)
    if not coverages:
        return 0.05
    return float(np.mean(coverages))


def _random_gate(batch_size: int, num_frames: int, mask_ratio: float, gamma: float, device: torch.device) -> torch.Tensor:
    masked = (torch.rand(batch_size, num_frames, device=device) < mask_ratio).float()
    return torch.where(masked > 0, torch.full_like(masked, gamma), torch.ones_like(masked))


def _train_random_mask_adapter(adapter, pre_dm, device, args, mask_ratio: float) -> nn.Module:
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=max(2, args.stage2_patience // 3))
    best_mae = float("inf")
    best_state = None
    patience_counter = 0

    train_loader = pre_dm.get_loader("train", shuffle=True)
    val_loader = pre_dm.get_loader("val", shuffle=False)
    feature_mean, feature_std = _estimate_sequence_stats_on_segment_loader(adapter, pre_dm.get_loader("train", shuffle=False), device)

    for _epoch in range(args.stage2_epochs):
        adapter.train()
        for audios, labels, _sample_ids in train_loader:
            audios = audios.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).squeeze(-1)
            optimizer.zero_grad()
            sequence_features, aux = adapter.encode(audios)
            gate = _random_gate(
                batch_size=sequence_features.shape[0],
                num_frames=sequence_features.shape[1],
                mask_ratio=mask_ratio,
                gamma=args.gamma,
                device=device,
            )
            masked_features = gate.unsqueeze(-1) * sequence_features
            masked_features = _renormalize_with_stats(masked_features, feature_mean, feature_std)
            y_hat = adapter.predict_from_sequence(masked_features)
            if y_hat.ndim > 1 and y_hat.shape[-1] == 1:
                y_hat = y_hat.squeeze(-1)
            loss = criterion(y_hat, labels)
            if "hcpc_loss" in aux:
                loss = loss + args.hcpc_weight * aux["hcpc_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=args.grad_clip)
            optimizer.step()

        val_metrics = _evaluate_adapter_on_loader(
            adapter=adapter,
            loader=val_loader,
            inverse_transform_fn=pre_dm.inverse_transform,
            device=device,
            eval_level=args.eval_level,
        )
        scheduler.step(val_metrics["MAE"])
        if val_metrics["MAE"] < best_mae:
            best_mae = val_metrics["MAE"]
            best_state = copy.deepcopy(adapter.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.stage2_patience:
                break

    if best_state is not None:
        adapter.load_state_dict(best_state)
    return adapter


def _ensure_cuefilter_stats(cuefilter: CueFilter, adapter, train_loader, device: torch.device) -> None:
    feature_mean, feature_std = estimate_sequence_feature_stats(adapter, train_loader, device)
    cuefilter.set_feature_stats(feature_mean, feature_std)


def run_single_ablation(dataset_key: str, model_name: str, seed: int, args) -> Dict[str, AblationVariantMetrics]:
    args = _with_model_defaults(args, model_name)
    set_seed(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    pre_dm, exc_dm, rand_dm, cueaware_dm = _make_data_managers(dataset_key, seed, args)
    if args.info_only:
        print(f"[{DATASET_CONFIGS[dataset_key].display_name} | {model_name}] {pre_dm.get_info()}")
        return {}

    baseline_adapter = build_adapter(model_name).to(device)
    baseline_adapter = _train_baseline_adapter(
        adapter=baseline_adapter,
        train_loader=cueaware_dm.get_loader("train", shuffle=True),
        val_loader=cueaware_dm.get_loader("val", shuffle=False),
        inverse_transform_fn=cueaware_dm.inverse_transform,
        device=device,
        args=args,
    )

    train_loader = cueaware_dm.get_loader("train", shuffle=True)
    val_loader = cueaware_dm.get_loader("val", shuffle=False)
    fixed_train_loader = cueaware_dm.get_loader("train", shuffle=False)

    full_args = _clone_args(args)
    full_adapter_seed = copy.deepcopy(baseline_adapter).to(device)
    full_cuefilter_seed = _build_cuefilter(full_adapter_seed, full_args, device)
    shared_stage1 = _initialize_shared_cuefilter_if_needed(
        cuefilter=full_cuefilter_seed,
        adapter=full_adapter_seed,
        model_name=model_name,
        seed=seed,
        device=device,
        args=full_args,
    )
    if shared_stage1 is None:
        full_cuefilter_seed, _ = _train_stage1_cuefilter(
            cuefilter=full_cuefilter_seed,
            adapter=full_adapter_seed,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            args=full_args,
        )
    else:
        _ensure_cuefilter_stats(full_cuefilter_seed, full_adapter_seed, fixed_train_loader, device)

    baseline_pre = _evaluate_gate_variant(baseline_adapter, pre_dm, device, cuefilter=None, gate_strategy="none")
    baseline_error_norm = _normal(baseline_pre["MAE"], dataset_key)
    results: Dict[str, AblationVariantMetrics] = {
        "Backbone": _build_ablation_metrics(
            dataset_key,
            baseline_error_norm,
            baseline_adapter,
            None,
            pre_dm,
            exc_dm,
            rand_dm,
            cueaware_dm,
            device,
            args,
            gate_strategy="none",
        )
    }

    full_adapter = copy.deepcopy(baseline_adapter).to(device)
    full_cuefilter = copy.deepcopy(full_cuefilter_seed).to(device)
    full_cuefilter, full_adapter, _full_stage2 = _train_stage2_joint(
        cuefilter=full_cuefilter,
        adapter=full_adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        inverse_transform_fn=pre_dm.inverse_transform,
        device=device,
        args=full_args,
        gate_mode="soft",
        use_renorm=True,
    )
    results["Full CueFilter"] = _build_ablation_metrics(
        dataset_key,
        baseline_error_norm,
        full_adapter,
        full_cuefilter,
        pre_dm,
        exc_dm,
        rand_dm,
        cueaware_dm,
        device,
        args,
        gate_strategy="cuefilter",
        gate_mode="soft",
        use_renorm=True,
    )

    no_pretrain_args = _clone_args(args, load_shared_cuefilter=None, shared_pretrain_datasets=None, save_shared_cuefilter=None)
    no_pretrain_adapter = copy.deepcopy(baseline_adapter).to(device)
    no_pretrain_cuefilter = _build_cuefilter(no_pretrain_adapter, no_pretrain_args, device)
    _ensure_cuefilter_stats(no_pretrain_cuefilter, no_pretrain_adapter, fixed_train_loader, device)
    no_pretrain_cuefilter, no_pretrain_adapter, no_pretrain_stage2 = _train_stage2_joint(
        cuefilter=no_pretrain_cuefilter,
        adapter=no_pretrain_adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        inverse_transform_fn=pre_dm.inverse_transform,
        device=device,
        args=no_pretrain_args,
        gate_mode="soft",
        use_renorm=True,
    )
    results["No cue pretrain"] = _build_ablation_metrics(
        dataset_key,
        baseline_error_norm,
        no_pretrain_adapter,
        no_pretrain_cuefilter,
        pre_dm,
        exc_dm,
        rand_dm,
        cueaware_dm,
        device,
        no_pretrain_args,
        gate_strategy="cuefilter",
        gate_mode="soft",
        use_renorm=True,
    )

    no_cue_loss_args = _clone_args(args, lambda_c=0.0)
    no_cue_loss_adapter = copy.deepcopy(baseline_adapter).to(device)
    no_cue_loss_cuefilter = copy.deepcopy(full_cuefilter_seed).to(device)
    no_cue_loss_cuefilter, no_cue_loss_adapter, _no_cue_loss_stage2 = _train_stage2_joint(
        cuefilter=no_cue_loss_cuefilter,
        adapter=no_cue_loss_adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        inverse_transform_fn=pre_dm.inverse_transform,
        device=device,
        args=no_cue_loss_args,
        gate_mode="soft",
        use_renorm=True,
    )
    results["No cue loss"] = _build_ablation_metrics(
        dataset_key,
        baseline_error_norm,
        no_cue_loss_adapter,
        no_cue_loss_cuefilter,
        pre_dm,
        exc_dm,
        rand_dm,
        cueaware_dm,
        device,
        no_cue_loss_args,
        gate_strategy="cuefilter",
        gate_mode="soft",
        use_renorm=True,
    )

    no_budget_args = _clone_args(
        args,
        lambda_b=0.0,
        load_shared_cuefilter=None,
        shared_pretrain_datasets=None,
        save_shared_cuefilter=None,
    )
    no_budget_adapter_seed = copy.deepcopy(baseline_adapter).to(device)
    no_budget_cuefilter_seed = _build_cuefilter(no_budget_adapter_seed, no_budget_args, device)
    no_budget_cuefilter_seed, _ = _train_stage1_cuefilter(
        cuefilter=no_budget_cuefilter_seed,
        adapter=no_budget_adapter_seed,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        args=no_budget_args,
    )
    no_budget_adapter = copy.deepcopy(baseline_adapter).to(device)
    no_budget_cuefilter = copy.deepcopy(no_budget_cuefilter_seed).to(device)
    no_budget_cuefilter, no_budget_adapter, no_budget_stage2 = _train_stage2_joint(
        cuefilter=no_budget_cuefilter,
        adapter=no_budget_adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        inverse_transform_fn=pre_dm.inverse_transform,
        device=device,
        args=no_budget_args,
        gate_mode="soft",
        use_renorm=True,
    )
    results["No budget loss"] = _build_ablation_metrics(
        dataset_key,
        baseline_error_norm,
        no_budget_adapter,
        no_budget_cuefilter,
        pre_dm,
        exc_dm,
        rand_dm,
        cueaware_dm,
        device,
        no_budget_args,
        gate_strategy="cuefilter",
        gate_mode="soft",
        use_renorm=True,
    )

    no_renorm_adapter = copy.deepcopy(baseline_adapter).to(device)
    no_renorm_cuefilter = copy.deepcopy(full_cuefilter_seed).to(device)
    no_renorm_cuefilter, no_renorm_adapter, no_renorm_stage2 = _train_stage2_joint(
        cuefilter=no_renorm_cuefilter,
        adapter=no_renorm_adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        inverse_transform_fn=pre_dm.inverse_transform,
        device=device,
        args=args,
        gate_mode="soft",
        use_renorm=False,
    )
    results["No feature norm"] = _build_ablation_metrics(
        dataset_key,
        baseline_error_norm,
        no_renorm_adapter,
        no_renorm_cuefilter,
        pre_dm,
        exc_dm,
        rand_dm,
        cueaware_dm,
        device,
        args,
        gate_strategy="cuefilter",
        gate_mode="soft",
        use_renorm=False,
    )

    hard_adapter = copy.deepcopy(baseline_adapter).to(device)
    hard_cuefilter = copy.deepcopy(full_cuefilter_seed).to(device)
    hard_cuefilter, hard_adapter, hard_stage2 = _train_stage2_joint(
        cuefilter=hard_cuefilter,
        adapter=hard_adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        inverse_transform_fn=pre_dm.inverse_transform,
        device=device,
        args=args,
        gate_mode="binary",
        use_renorm=True,
    )
    results["Hard mask"] = _build_ablation_metrics(
        dataset_key,
        baseline_error_norm,
        hard_adapter,
        hard_cuefilter,
        pre_dm,
        exc_dm,
        rand_dm,
        cueaware_dm,
        device,
        args,
        gate_strategy="cuefilter",
        gate_mode="binary",
        use_renorm=True,
    )

    results["Random gate"] = _build_ablation_metrics(
        dataset_key,
        baseline_error_norm,
        full_adapter,
        full_cuefilter,
        pre_dm,
        exc_dm,
        rand_dm,
        cueaware_dm,
        device,
        args,
        gate_strategy="random",
        gate_mode="soft",
        use_renorm=True,
    )
    results["Shuffled gate"] = _build_ablation_metrics(
        dataset_key,
        baseline_error_norm,
        full_adapter,
        full_cuefilter,
        pre_dm,
        exc_dm,
        rand_dm,
        cueaware_dm,
        device,
        args,
        gate_strategy="shuffled",
        gate_mode="soft",
        use_renorm=True,
    )

    return results


def _write_detail_rows(detail_output_dir: Optional[str], relative_path: str, rows: List[Dict[str, object]]) -> None:
    if not detail_output_dir or not rows:
        return
    output_path = Path(detail_output_dir) / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> None:
    args = create_parser().parse_args()
    summary_rows: List[Dict[str, object]] = []

    for dataset_key in args.datasets:
        dataset_name = DATASET_CONFIGS[dataset_key].display_name
        for model_name in args.models:
            per_variant: Dict[str, List[AblationVariantMetrics]] = {name: [] for name in VARIANT_DISPLAY}
            for seed in args.seeds:
                print(f"Running ablation | {dataset_name} | {model_name} | seed={seed}", flush=True)
                try:
                    run_result = run_single_ablation(dataset_key, model_name, seed, args)
                except Exception as exc:
                    print(f"  failed: {exc}")
                    continue
                for name, metrics in run_result.items():
                    per_variant[name].append(metrics)
                    detail_rows = []
                    for row in metrics.prediction_rows or []:
                        detail_rows.append(
                            {
                                "Dataset": dataset_name,
                                "Backbone": model_name,
                                "Seed": seed,
                                "Variant": name,
                                "CueRole": args.cue_role,
                                "SpeechScope": args.speech_scope,
                                **row,
                            }
                        )
                    _write_detail_rows(
                        args.detail_output_dir,
                        f"ablation_predictions/{dataset_name.replace('-', '').lower()}__{model_name}__seed{seed}__{name.replace(' ', '_').lower()}.csv",
                        detail_rows,
                    )

            for variant_name in VARIANT_DISPLAY:
                metrics_list = per_variant[variant_name]
                if not metrics_list:
                    continue
                summary_rows.append(
                    {
                        "Dataset": dataset_name,
                        "Backbone": model_name,
                        "Variant": variant_name,
                        "Error": format_mean_std([m.error for m in metrics_list]),
                        "Cost": format_mean_std([m.cost for m in metrics_list]),
                        "Cue rm.": format_mean_std([m.cue_rm for m in metrics_list]),
                        "Cue extra": format_mean_std([m.cue_extra for m in metrics_list]),
                        "Gate dens.": format_mean_std([m.gate_density for m in metrics_list]),
                        "Density err.": format_mean_std([m.density_err for m in metrics_list]),
                        "Raw Original MAE": format_mean_std([m.raw_original_mae for m in metrics_list]),
                        "Raw Cue-removed MAE": format_mean_std([m.raw_cue_removed_mae for m in metrics_list]),
                        "Raw Random-removed MAE": format_mean_std([m.raw_random_removed_mae for m in metrics_list]),
                        "Seeds": len(metrics_list),
                    }
                )
                _write_detail_rows(
                    args.detail_output_dir,
                    f"ablation_gate_density/{dataset_name.replace('-', '').lower()}__{model_name}__{variant_name.replace(' ', '_').lower()}.csv",
                    [
                        {
                            "Dataset": dataset_name,
                            "Backbone": model_name,
                            "Variant": variant_name,
                            "Gate dens.": format_mean_std([m.gate_density for m in metrics_list]),
                            "Density err.": format_mean_std([m.density_err for m in metrics_list]),
                            "Seeds": len(metrics_list),
                        }
                    ],
                )

    if not summary_rows:
        print("No ablation results were produced.")
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
