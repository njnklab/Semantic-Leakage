"""
Generalization experiments for CueFilter under:
1. cross-dataset transfer,
2. cross-gender validation, and
3. cross-age validation.

The script reuses the shared multi-dataset CueFilter prior when provided and
reports baseline-vs-CueFilter target-domain performance together with cue
removal sensitivity before and after mitigation.
"""

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
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .Baseline.audio_views import DATASET_CONFIGS, load_segment_records
    from .Baseline.experiment_utils import (
        AudioSegmentDataset,
        build_effective_split_map,
        finalize_predictions,
        format_mean_std,
        random_split_map,
        safe_r2_score,
        set_seed,
    )
    from .integration.data import CueAwareAudioDataset, CueAwarePreDataManager, cueaware_collate_fn, load_cueaware_items
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
    from CueFilter.Baseline.audio_views import DATASET_CONFIGS, load_segment_records
    from CueFilter.Baseline.experiment_utils import (
        AudioSegmentDataset,
        build_effective_split_map,
        finalize_predictions,
        format_mean_std,
        random_split_map,
        safe_r2_score,
        set_seed,
    )
    from CueFilter.integration.data import CueAwareAudioDataset, CueAwarePreDataManager, cueaware_collate_fn, load_cueaware_items
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


AGE_BINS: Dict[str, Tuple[float, float]] = {
    "adolescent": (-np.inf, 18.0),
    "young-adult": (18.0, 25.0),
    "adult": (25.0, np.inf),
}

LABEL_RANGES = {
    "edaic": 24.0,
    "cmdc": 27.0,
    "pdch": 52.0,
    "mandic": 52.0,
}


def _normal(value: float, dataset_key: str) -> float:
    return float(value) / LABEL_RANGES.get(dataset_key, 1.0)


@dataclass
class LabelScaler:
    mean: float
    std: float

    def transform(self, values: Sequence[float]) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        return (arr - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) * self.std + self.mean


class InMemoryRecordingEvalManager:
    def __init__(
        self,
        records: Sequence[Dict[str, object]],
        scaler: LabelScaler,
        batch_size: int,
        num_workers: int,
    ):
        self.records = list(records)
        self.scaler = scaler
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_loader(self, split: str = "test", shuffle: bool = False):
        del split
        segments = [record["audio"] for record in self.records]
        labels_scaled = self.scaler.transform([float(record["label"]) for record in self.records])
        sample_ids = [str(record["sample_id"]) for record in self.records]
        dataset = AudioSegmentDataset(segments, labels_scaled, sample_ids, is_training=False)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(values)


def _make_cueaware_loader(items: Sequence[Dict], batch_size: int, num_workers: int, shuffle: bool, is_training: bool):
    dataset = CueAwareAudioDataset(items, is_training=is_training)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=cueaware_collate_fn,
    )


def _fit_source_scaler(train_items: Sequence[Dict]) -> LabelScaler:
    train_sample_labels: Dict[str, float] = {}
    for item in train_items:
        key = f"{item.get('dataset_key', 'unknown')}::{item['sample_id']}"
        train_sample_labels[key] = float(item["label_raw"])
    values = np.asarray(list(train_sample_labels.values()), dtype=np.float32)
    mean = float(values.mean())
    std = float(max(values.std(ddof=0), 1e-6))
    return LabelScaler(mean=mean, std=std)


def _attach_scaled_labels(items: Sequence[Dict], scaler: LabelScaler) -> List[Dict]:
    scaled_items: List[Dict] = []
    for item in items:
        updated = dict(item)
        updated["label_scaled"] = float(scaler.transform([float(item["label_raw"])])[0])
        scaled_items.append(updated)
    return scaled_items


def _select_records_for_test_split(
    dataset_key: str,
    variant: str,
    seed: int,
    args,
) -> List[Dict[str, object]]:
    records = load_segment_records(
        dataset_key=dataset_key,
        variant=variant,
        segment_length=args.segment_length,
        sample_rate=args.sample_rate,
        seed=seed,
        cue_role=args.cue_role,
        speech_scope=args.speech_scope,
        segment_first=args.segment_first,
    )
    split_map = build_effective_split_map(
        dataset_key=dataset_key,
        group_ids=[str(record["group_id"]) for record in records],
        seed=seed,
        max_groups_per_split=args.max_groups_per_split,
    )
    return [record for record in records if split_map.get(str(record["group_id"])) == "test"]


def _subset_by_gender(items: Sequence[Dict], gender: str) -> List[Dict]:
    return [item for item in items if item.get("gender") == gender]


def _subset_by_age_bin(items: Sequence[Dict], age_bin: str) -> List[Dict]:
    lower, upper = AGE_BINS[age_bin]
    selected = []
    for item in items:
        age = item.get("age")
        if age is None:
            continue
        age = float(age)
        if lower <= age < upper:
            selected.append(item)
    return selected


def _build_train_val_split(group_ids: Sequence[str], seed: int) -> Dict[str, str]:
    unique_ids = sorted(set(group_ids))
    if len(unique_ids) < 3:
        raise ValueError(f"Need at least 3 groups for source train/val split, got {len(unique_ids)}")
    split_map = random_split_map(unique_ids, seed)
    return {
        gid: ("train" if split != "val" else "val")
        for gid, split in split_map.items()
        if split in {"train", "val", "test"}
    }


def _partition_items(items: Sequence[Dict], split_map: Dict[str, str], split: str) -> List[Dict]:
    return [item for item in items if split_map.get(str(item["group_id"])) == split]


def _prepare_cross_dataset_source(seed: int, args) -> Tuple[List[Dict], List[Dict]]:
    train_items: List[Dict] = []
    val_items: List[Dict] = []
    for dataset_key in args.train_datasets:
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
        train_items.extend(dm.get_items("train"))
        val_items.extend(dm.get_items("val"))
    return train_items, val_items


def _prepare_subgroup_source_and_target(seed: int, args) -> Tuple[List[Dict], List[Dict], Dict[str, List[Dict]]]:
    cue_items = load_cueaware_items(
        dataset_key=args.dataset,
        segment_length=args.segment_length,
        sample_rate=args.sample_rate,
        cue_role=args.cue_role,
        speech_scope=args.speech_scope,
    )

    if args.mode == "cross-gender":
        source_items = _subset_by_gender(cue_items, args.train_gender)
        target_filter = lambda items: _subset_by_gender(items, args.test_gender)
    else:
        source_items = _subset_by_age_bin(cue_items, args.train_age_bin)
        target_filter = lambda items: _subset_by_age_bin(items, args.test_age_bin)

    split_map = _build_train_val_split([str(item["group_id"]) for item in source_items], seed)
    train_items = _partition_items(source_items, split_map, "train")
    val_items = _partition_items(source_items, split_map, "val")

    target_records: Dict[str, List[Dict]] = {}
    for variant in ("pre", "cue-excluded", "random"):
        records = load_segment_records(
            dataset_key=args.dataset,
            variant=variant,
            segment_length=args.segment_length,
            sample_rate=args.sample_rate,
            seed=seed,
            cue_role=args.cue_role,
            speech_scope=args.speech_scope,
            segment_first=args.segment_first,
        )
        target_records[variant] = target_filter(records)

    return train_items, val_items, target_records


def _evaluate_on_records(
    adapter,
    cuefilter,
    records: Sequence[Dict],
    scaler: LabelScaler,
    device,
    args,
    return_predictions: bool = False,
    condition: Optional[str] = None,
):
    if not records:
        metrics = {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}
        return (metrics, []) if return_predictions else metrics

    manager = InMemoryRecordingEvalManager(
        records=records,
        scaler=scaler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    loader = manager.get_loader(shuffle=False)
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
            sequence_features, aux = adapter.encode(audios)
            pred_features = sequence_features
            if cuefilter is not None:
                cue_outputs = cuefilter(sequence_features, mode="soft")
                pred_features = cue_outputs["renormed_features"]
            y_hat = adapter.predict_from_sequence(pred_features)
            if y_hat.ndim > 1 and y_hat.shape[-1] == 1:
                y_hat = y_hat.squeeze(-1)
            if "hcpc_loss" in aux:
                _ = aux["hcpc_loss"]

            batch_preds = y_hat.detach().cpu().numpy().tolist()
            batch_labels = labels.detach().cpu().numpy().tolist()
            preds_scaled.extend(batch_preds)
            labels_scaled.extend(batch_labels)
            sample_ids.extend(batch_sample_ids)
            if return_predictions:
                batch_pred_raw = scaler.inverse_transform(np.asarray(batch_preds, dtype=np.float32))
                batch_label_raw = scaler.inverse_transform(np.asarray(batch_labels, dtype=np.float32))
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

    preds_raw = scaler.inverse_transform(np.asarray(preds_scaled, dtype=np.float32))
    labels_raw = scaler.inverse_transform(np.asarray(labels_scaled, dtype=np.float32))
    preds_eval, labels_eval = finalize_predictions(preds_raw, labels_raw, sample_ids, eval_level=args.eval_level)
    metrics = _compute_regression_metrics(preds_eval, labels_eval)
    return (metrics, rows) if return_predictions else metrics


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CueFilter generalization experiments.")
    parser.add_argument("--mode", choices=["cross-dataset", "cross-gender", "cross-age"], required=True)
    parser.add_argument("--train-datasets", nargs="+", choices=list(DATASET_CONFIGS.keys()), default=None)
    parser.add_argument("--test-datasets", nargs="+", choices=list(DATASET_CONFIGS.keys()), default=None)
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()), default=None)
    parser.add_argument("--train-gender", choices=["male", "female"], default=None)
    parser.add_argument("--test-gender", choices=["male", "female"], default=None)
    parser.add_argument("--train-age-bin", choices=list(AGE_BINS.keys()), default=None)
    parser.add_argument("--test-age-bin", choices=list(AGE_BINS.keys()), default=None)
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
    parser.add_argument("--max-groups-per-split", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval-level", choices=["recording", "sample"], default="recording")
    parser.add_argument("--segment-first", action="store_true", help="Segment pre audio first, then apply cue removal within each 30-s sample.")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--detail-output-dir", type=str, default=None)
    parser.add_argument("--info-only", action="store_true")

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
    return parser


def validate_args(args) -> None:
    if args.mode == "cross-dataset":
        if not args.train_datasets or not args.test_datasets:
            raise ValueError("cross-dataset mode requires both --train-datasets and --test-datasets")
    elif args.mode == "cross-gender":
        if not args.dataset or not args.train_gender or not args.test_gender:
            raise ValueError("cross-gender mode requires --dataset, --train-gender, and --test-gender")
    elif args.mode == "cross-age":
        if not args.dataset or not args.train_age_bin or not args.test_age_bin:
            raise ValueError("cross-age mode requires --dataset, --train-age-bin, and --test-age-bin")


def _write_prediction_details(detail_output_dir: Optional[str], relative_path: str, rows: List[Dict[str, object]]) -> None:
    if not detail_output_dir or not rows:
        return
    output_path = Path(detail_output_dir) / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main():
    args = create_parser().parse_args()
    validate_args(args)
    summary_rows: List[Dict[str, object]] = []

    for model_name in args.models:
        effective_args = _with_model_defaults(args, model_name)
        for seed in args.seeds:
            set_seed(seed)
            device = torch.device(effective_args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

            if args.mode == "cross-dataset":
                source_train_items, source_val_items = _prepare_cross_dataset_source(seed, effective_args)
                target_variants_by_dataset = {
                    dataset_key: {
                        variant: _select_records_for_test_split(dataset_key, variant, seed, effective_args)
                        for variant in ("pre", "cue-excluded", "random")
                    }
                    for dataset_key in args.test_datasets
                }
            else:
                source_train_items, source_val_items, target_records = _prepare_subgroup_source_and_target(seed, effective_args)
                target_variants_by_dataset = {args.dataset: target_records}

            if not source_train_items or not source_val_items:
                print(f"[{model_name} | seed={seed}] skipped: no source items available")
                continue

            scaler = _fit_source_scaler(source_train_items)
            source_train_items = _attach_scaled_labels(source_train_items, scaler)
            source_val_items = _attach_scaled_labels(source_val_items, scaler)

            if effective_args.info_only:
                print(
                    f"[{model_name} | seed={seed}] "
                    f"source train={len(source_train_items)} val={len(source_val_items)}"
                )
                continue

            baseline_adapter = build_adapter(model_name).to(device)
            baseline_adapter = _train_baseline_adapter(
                adapter=baseline_adapter,
                train_loader=_make_cueaware_loader(source_train_items, effective_args.batch_size, effective_args.num_workers, True, True),
                val_loader=_make_cueaware_loader(source_val_items, effective_args.batch_size, effective_args.num_workers, False, False),
                inverse_transform_fn=scaler.inverse_transform,
                device=device,
                args=effective_args,
            )

            mitigation_adapter = copy.deepcopy(baseline_adapter).to(device)
            cuefilter = CueFilter(
                input_dim=mitigation_adapter.feature_dim,
                n_blocks=effective_args.n_blocks,
                kernel_size=effective_args.kernel_size,
                groups=effective_args.groups,
                alpha=effective_args.alpha,
                gamma=effective_args.gamma,
                cue_threshold=effective_args.cue_threshold,
            ).to(device)

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
                    train_loader=_make_cueaware_loader(source_train_items, effective_args.batch_size, effective_args.num_workers, True, True),
                    val_loader=_make_cueaware_loader(source_val_items, effective_args.batch_size, effective_args.num_workers, False, False),
                    device=device,
                    args=effective_args,
                )

            cuefilter, mitigation_adapter, _stage2_metrics = _train_stage2_joint(
                cuefilter=cuefilter,
                adapter=mitigation_adapter,
                train_loader=_make_cueaware_loader(source_train_items, effective_args.batch_size, effective_args.num_workers, True, True),
                val_loader=_make_cueaware_loader(source_val_items, effective_args.batch_size, effective_args.num_workers, False, False),
                inverse_transform_fn=scaler.inverse_transform,
                device=device,
                args=effective_args,
            )

            for target_dataset, variant_records in target_variants_by_dataset.items():
                if not variant_records["pre"]:
                    print(f"[{model_name} | seed={seed} | target={target_dataset}] skipped: empty target test set")
                    continue

                base_pre, base_pre_rows = _evaluate_on_records(
                    baseline_adapter, None, variant_records["pre"], scaler, device, effective_args, True, "baseline_original"
                )
                base_exc, base_exc_rows = _evaluate_on_records(
                    baseline_adapter, None, variant_records["cue-excluded"], scaler, device, effective_args, True, "baseline_cue_removed"
                )
                base_rand, base_rand_rows = _evaluate_on_records(
                    baseline_adapter, None, variant_records["random"], scaler, device, effective_args, True, "baseline_random"
                )

                cf_pre, cf_pre_rows = _evaluate_on_records(
                    mitigation_adapter, cuefilter, variant_records["pre"], scaler, device, effective_args, True, "cuefilter_original"
                )
                cf_exc, cf_exc_rows = _evaluate_on_records(
                    mitigation_adapter, cuefilter, variant_records["cue-excluded"], scaler, device, effective_args, True, "cuefilter_cue_removed"
                )
                cf_rand, cf_rand_rows = _evaluate_on_records(
                    mitigation_adapter, cuefilter, variant_records["random"], scaler, device, effective_args, True, "cuefilter_random"
                )

                source_name = ",".join(args.train_datasets) if args.mode == "cross-dataset" else args.dataset
                detail_rows: List[Dict[str, object]] = []
                for row in base_pre_rows + base_exc_rows + base_rand_rows + cf_pre_rows + cf_exc_rows + cf_rand_rows:
                    detail_rows.append(
                        {
                            "Mode": args.mode,
                            "Source": source_name,
                            "Target": target_dataset,
                            "Model": model_name,
                            "Seed": seed,
                            "CueRole": effective_args.cue_role,
                            "SpeechScope": effective_args.speech_scope,
                            **row,
                        }
                    )
                safe_source = str(source_name).replace(",", "_")
                _write_prediction_details(
                    effective_args.detail_output_dir,
                    f"generalization_predictions/{args.mode}__{safe_source}_to_{target_dataset}__{model_name}__seed{seed}.csv",
                    detail_rows,
                )

                summary_rows.append(
                    {
                        "Mode": args.mode,
                        "Source": source_name,
                        "Target": target_dataset,
                        "Model": model_name,
                        "Seed": seed,
                        "CueRole": effective_args.cue_role,
                        "SpeechScope": effective_args.speech_scope,
                        "EvalLevel": effective_args.eval_level,
                        "SegmentFirst": effective_args.segment_first,
                        "Baseline MAE": base_pre["MAE"],
                        "Baseline normalized Error": _normal(base_pre["MAE"], target_dataset),
                        "Baseline RMSE": base_pre["RMSE"],
                        "Baseline normalized RMSE": _normal(base_pre["RMSE"], target_dataset),
                        "Baseline R2": base_pre["R2"],
                        "CueFilter MAE": cf_pre["MAE"],
                        "CueFilter normalized Error": _normal(cf_pre["MAE"], target_dataset),
                        "CueFilter RMSE": cf_pre["RMSE"],
                        "CueFilter normalized RMSE": _normal(cf_pre["RMSE"], target_dataset),
                        "CueFilter R2": cf_pre["R2"],
                        "Baseline cue-removal increase": base_exc["MAE"] - base_pre["MAE"],
                        "Baseline normalized cue-removal increase": _normal(base_exc["MAE"] - base_pre["MAE"], target_dataset),
                        "CueFilter cue-removal increase": cf_exc["MAE"] - cf_pre["MAE"],
                        "CueFilter normalized cue-removal increase": _normal(cf_exc["MAE"] - cf_pre["MAE"], target_dataset),
                        "Baseline extra-over-random": base_exc["MAE"] - base_rand["MAE"],
                        "Baseline normalized extra-over-random": _normal(base_exc["MAE"] - base_rand["MAE"], target_dataset),
                        "CueFilter extra-over-random": cf_exc["MAE"] - cf_rand["MAE"],
                        "CueFilter normalized extra-over-random": _normal(cf_exc["MAE"] - cf_rand["MAE"], target_dataset),
                    }
                )

    if not summary_rows:
        print("No generalization results were produced.")
        return

    raw_df = pd.DataFrame(summary_rows)
    group_cols = ["Mode", "Source", "Target", "Model", "CueRole", "SpeechScope", "EvalLevel", "SegmentFirst"]
    summary_df = (
        raw_df.groupby(group_cols, dropna=False)
        .agg(
            BaselineMAE=("Baseline MAE", format_mean_std),
            BaselineError=("Baseline normalized Error", format_mean_std),
            BaselineRMSE=("Baseline normalized RMSE", format_mean_std),
            BaselineR2=("Baseline R2", format_mean_std),
            CueFilterMAE=("CueFilter MAE", format_mean_std),
            CueFilterError=("CueFilter normalized Error", format_mean_std),
            CueFilterRMSE=("CueFilter normalized RMSE", format_mean_std),
            CueFilterR2=("CueFilter R2", format_mean_std),
            BaselineCueRemoval=("Baseline cue-removal increase", format_mean_std),
            BaselineNormCueRemoval=("Baseline normalized cue-removal increase", format_mean_std),
            CueFilterCueRemoval=("CueFilter cue-removal increase", format_mean_std),
            CueFilterNormCueRemoval=("CueFilter normalized cue-removal increase", format_mean_std),
            BaselineExtraRandom=("Baseline extra-over-random", format_mean_std),
            BaselineNormExtraRandom=("Baseline normalized extra-over-random", format_mean_std),
            CueFilterExtraRandom=("CueFilter extra-over-random", format_mean_std),
            CueFilterNormExtraRandom=("CueFilter normalized extra-over-random", format_mean_std),
            Seeds=("Seed", "count"),
        )
        .reset_index()
    )

    print("\n" + summary_df.to_markdown(index=False))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".csv":
            summary_df.to_csv(output_path, index=False)
        else:
            output_path.write_text(summary_df.to_markdown(index=False), encoding="utf-8")
        print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
