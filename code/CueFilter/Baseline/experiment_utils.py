from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

try:
    from .audio_views import load_all_segments, split_group_ids
except ImportError:
    from audio_views import load_all_segments, split_group_ids


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) <= 1:
            return float("nan")
        return float(r2_score(y_true, y_pred))
    except Exception:
        return float("nan")


def format_mean_std(values: Sequence[float]) -> str:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return "-"
    return f"{arr.mean():.3f}±{arr.std(ddof=0):.3f}"


def random_split_map(group_ids: Sequence[str], seed: int) -> Dict[str, str]:
    unique_ids = sorted(set(group_ids))
    if len(unique_ids) < 3:
        raise ValueError(f"Need at least 3 groups to build train/val/test splits, got {len(unique_ids)}")

    if len(unique_ids) < 10:
        n_test = 1
        n_val = 1
    else:
        n_test = max(1, int(round(len(unique_ids) * 0.1)))
        n_val = max(1, int(round(len(unique_ids) * 0.1)))
        if n_test + n_val >= len(unique_ids):
            n_test = 1
            n_val = 1

    train_ids, test_ids = train_test_split(unique_ids, test_size=n_test, random_state=seed)
    train_ids, val_ids = train_test_split(train_ids, test_size=n_val, random_state=seed)

    split_map = {gid: "train" for gid in train_ids}
    split_map.update({gid: "val" for gid in val_ids})
    split_map.update({gid: "test" for gid in test_ids})
    return split_map


def has_complete_split(split_map: Dict[str, str], group_ids: Sequence[str]) -> bool:
    split_values = [split_map.get(gid) for gid in set(group_ids) if split_map.get(gid) is not None]
    return all(name in split_values for name in ("train", "val", "test"))


def limit_split_map(
    split_map: Dict[str, str],
    group_ids: Sequence[str],
    max_groups_per_split: Optional[int],
) -> Dict[str, str]:
    if max_groups_per_split is None:
        return split_map

    available_groups = set(group_ids)
    kept_groups = set()
    for split_name in ("train", "val", "test"):
        groups = sorted(gid for gid, split in split_map.items() if split == split_name and gid in available_groups)
        kept_groups.update(groups[:max_groups_per_split])
    return {gid: split for gid, split in split_map.items() if gid in kept_groups}


def build_effective_split_map(
    dataset_key: str,
    group_ids: Sequence[str],
    seed: int,
    max_groups_per_split: Optional[int] = None,
) -> Dict[str, str]:
    split_map = split_group_ids(dataset_key, group_ids, seed)
    if not has_complete_split(split_map, group_ids):
        split_map = random_split_map(group_ids, seed)
    split_map = limit_split_map(split_map, group_ids, max_groups_per_split)
    return split_map


def filter_split_map_to_available_groups(
    split_map: Dict[str, str],
    group_ids: Sequence[str],
) -> Dict[str, str]:
    available = set(group_ids)
    return {gid: split for gid, split in split_map.items() if gid in available}


def aggregate_recording_predictions(
    preds_raw: Sequence[float],
    labels_raw: Sequence[float],
    sample_ids: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for pred, label, sample_id in zip(preds_raw, labels_raw, sample_ids):
        slot = grouped.setdefault(sample_id, {"preds": [], "labels": []})
        slot["preds"].append(float(pred))
        slot["labels"].append(float(label))

    agg_preds = []
    agg_labels = []
    for sample_id in sorted(grouped):
        agg_preds.append(float(np.mean(grouped[sample_id]["preds"])))
        agg_labels.append(float(np.mean(grouped[sample_id]["labels"])))
    return np.asarray(agg_preds, dtype=np.float32), np.asarray(agg_labels, dtype=np.float32)


def finalize_predictions(
    preds_raw: Sequence[float],
    labels_raw: Sequence[float],
    sample_ids: Sequence[str],
    eval_level: str = "recording",
) -> Tuple[np.ndarray, np.ndarray]:
    eval_level = str(eval_level).strip().lower()
    preds = np.asarray(preds_raw, dtype=np.float32)
    labels = np.asarray(labels_raw, dtype=np.float32)
    if eval_level == "sample":
        return preds, labels
    if eval_level == "recording":
        return aggregate_recording_predictions(preds, labels, sample_ids)
    raise ValueError(f"Unsupported eval_level: {eval_level}")


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(labels, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(labels, preds))),
        "R2": safe_r2_score(labels, preds),
    }


class AudioSegmentDataset(Dataset):
    def __init__(
        self,
        segments: Sequence[np.ndarray],
        labels: Sequence[float],
        sample_ids: Sequence[str],
        is_training: bool,
    ):
        self.segments = list(segments)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.sample_ids = list(sample_ids)
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int):
        audio = torch.tensor(self.segments[idx], dtype=torch.float32)
        if audio.abs().max() > 0:
            audio = audio / audio.abs().max()

        if self.is_training:
            gain = torch.empty(1).uniform_(0.8, 1.2).item()
            audio = audio * gain
            audio = audio + torch.randn_like(audio) * 0.003

        audio = audio.unsqueeze(0)
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        sample_id = self.sample_ids[idx]
        return audio, label, sample_id


class RecordingSplitDataManager:
    def __init__(
        self,
        dataset_key: str,
        variant: str,
        segment_length: float,
        sample_rate: int,
        batch_size: int,
        num_workers: int,
        random_state: int,
        max_groups_per_split: Optional[int] = None,
        split_map: Optional[Dict[str, str]] = None,
        cue_role: str = "patient",
        speech_scope: str = "participant",
        eval_level: str = "recording",
        segment_first: bool = False,
    ):
        self.dataset_key = dataset_key
        self.variant = variant
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.eval_level = eval_level
        self.segment_first = segment_first

        segments, labels_raw, sample_ids, group_ids = load_all_segments(
            dataset_key=dataset_key,
            variant=variant,
            segment_length=segment_length,
            sample_rate=sample_rate,
            seed=random_state,
            cue_role=cue_role,
            speech_scope=speech_scope,
            segment_first=segment_first,
        )
        if len(segments) == 0:
            raise ValueError(f"No segments loaded for dataset={dataset_key}, variant={variant}")

        self.segments = list(segments)
        self.labels_raw = np.asarray(labels_raw, dtype=np.float32)
        self.sample_ids = list(sample_ids)
        self.group_ids = list(group_ids)

        if split_map is None:
            self.split_map = build_effective_split_map(
                dataset_key=dataset_key,
                group_ids=self.group_ids,
                seed=random_state,
                max_groups_per_split=max_groups_per_split,
            )
        else:
            self.split_map = filter_split_map_to_available_groups(split_map, self.group_ids)

        self.train_indices = [i for i, gid in enumerate(self.group_ids) if self.split_map.get(gid) == "train"]
        self.val_indices = [i for i, gid in enumerate(self.group_ids) if self.split_map.get(gid) == "val"]
        self.test_indices = [i for i, gid in enumerate(self.group_ids) if self.split_map.get(gid) == "test"]

        if not self.train_indices or not self.val_indices or not self.test_indices:
            raise ValueError(
                f"Insufficient split coverage for dataset={dataset_key}, variant={variant}: "
                f"train={len(self.train_indices)}, val={len(self.val_indices)}, test={len(self.test_indices)}"
            )

        train_sample_labels: Dict[str, float] = {}
        for idx in self.train_indices:
            train_sample_labels[self.sample_ids[idx]] = float(self.labels_raw[idx])

        self.scaler = StandardScaler()
        self.scaler.fit(np.asarray(list(train_sample_labels.values()), dtype=np.float32).reshape(-1, 1))
        self.labels_scaled = self.scaler.transform(self.labels_raw.reshape(-1, 1)).flatten()

    def _subset(self, indices: Sequence[int]) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
        segments = [self.segments[i] for i in indices]
        labels = self.labels_scaled[list(indices)]
        sample_ids = [self.sample_ids[i] for i in indices]
        return segments, labels, sample_ids

    def get_loader(self, split: str, shuffle: bool) -> DataLoader:
        indices = {
            "train": self.train_indices,
            "val": self.val_indices,
            "test": self.test_indices,
        }[split]
        segments, labels, sample_ids = self._subset(indices)
        dataset = AudioSegmentDataset(segments, labels, sample_ids, is_training=(split == "train"))
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).flatten()

    def get_info(self) -> Dict[str, int]:
        def uniq(indices: Sequence[int]) -> int:
            return len({self.sample_ids[i] for i in indices})

        return {
            "total_segments": len(self.segments),
            "train_segments": len(self.train_indices),
            "val_segments": len(self.val_indices),
            "test_segments": len(self.test_indices),
            "total_recordings": len(set(self.sample_ids)),
            "train_recordings": uniq(self.train_indices),
            "val_recordings": uniq(self.val_indices),
            "test_recordings": uniq(self.test_indices),
        }
