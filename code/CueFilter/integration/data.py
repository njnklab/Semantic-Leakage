from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..Baseline.audio_views import (
    audio_path_for_sample,
    build_preprocessed_audio_bundle,
    group_id_for_sample,
    iter_output_samples,
    label_key_for_sample,
    load_json,
    load_metadata_mapping,
    lookup_sample_metadata,
    segment_audio_fixed_length,
)
from ..Baseline.experiment_utils import build_effective_split_map, filter_split_map_to_available_groups


def _intersect_relative_spans(
    spans: Sequence[Tuple[float, float]],
    start_sec: float,
    end_sec: float,
) -> List[Tuple[float, float]]:
    clipped: List[Tuple[float, float]] = []
    for span_start, span_end in spans:
        overlap_start = max(float(span_start), float(start_sec))
        overlap_end = min(float(span_end), float(end_sec))
        if overlap_end <= overlap_start:
            continue
        clipped.append((overlap_start - start_sec, overlap_end - start_sec))
    return clipped


class CueAwareAudioDataset(Dataset):
    def __init__(self, items: Sequence[Dict], is_training: bool):
        self.items = list(items)
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        item = self.items[idx]
        audio = torch.tensor(item["audio"], dtype=torch.float32)
        if audio.abs().max() > 0:
            audio = audio / audio.abs().max()

        if self.is_training:
            gain = torch.empty(1).uniform_(0.85, 1.15).item()
            audio = audio * gain
            audio = audio + torch.randn_like(audio) * 0.002

        return {
            "audio": audio.unsqueeze(0),
            "label": torch.tensor(item["label_scaled"], dtype=torch.float32),
            "label_raw": torch.tensor(item["label_raw"], dtype=torch.float32),
            "sample_id": item["sample_id"],
            "group_id": item["group_id"],
            "cue_spans_sec": list(item["cue_spans_sec"]),
            "duration_sec": float(item["duration_sec"]),
        }


def cueaware_collate_fn(batch: Sequence[Dict]) -> Dict:
    return {
        "audio": torch.stack([item["audio"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0),
        "label_raw": torch.stack([item["label_raw"] for item in batch], dim=0),
        "sample_id": [item["sample_id"] for item in batch],
        "group_id": [item["group_id"] for item in batch],
        "cue_spans_sec": [item["cue_spans_sec"] for item in batch],
        "duration_sec": [item["duration_sec"] for item in batch],
    }


def load_cueaware_items(
    dataset_key: str,
    segment_length: float = 30.0,
    sample_rate: int = 16000,
    cue_role: str = "patient",
    speech_scope: str = "participant",
) -> List[Dict]:
    segment_samples = int(round(segment_length * sample_rate))
    metadata_mapping = load_metadata_mapping(dataset_key)
    items: List[Dict] = []

    for sample in iter_output_samples(dataset_key):
        transcript_path = sample["transcript_path"]
        cues_path = sample["cues_path"]
        transcript = load_json(transcript_path)
        cues = load_json(cues_path)
        sample_id = str(sample["sample_id"])
        transcript["sample_id"] = sample_id
        metadata = lookup_sample_metadata(
            dataset_key=dataset_key,
            sample_id=sample_id,
            metadata_mapping=metadata_mapping,
        )
        if metadata is None:
            continue

        audio_path = audio_path_for_sample(dataset_key, sample_id)
        if not audio_path.exists():
            continue

        try:
            audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        except Exception:
            continue

        pre_audio, _retained_intervals, cue_spans_pre = build_preprocessed_audio_bundle(
            dataset_key=dataset_key,
            transcript=transcript,
            cues=cues,
            audio=audio,
            sr=sr,
            cue_role=cue_role,
            speech_scope=speech_scope,
        )
        if len(pre_audio) == 0:
            continue

        split_segments = segment_audio_fixed_length(pre_audio, segment_samples)
        group_id = group_id_for_sample(dataset_key, sample_id)
        label_raw = float(metadata["label"])

        for seg_idx, segment in enumerate(split_segments):
            seg_start = seg_idx * segment_length
            seg_end = seg_start + segment_length
            cue_spans_segment = _intersect_relative_spans(cue_spans_pre, seg_start, seg_end)
            items.append(
                {
                    "audio": segment.astype(np.float32),
                    "label_raw": label_raw,
                    "sample_id": sample_id,
                    "group_id": group_id,
                    "dataset_key": dataset_key,
                    "gender": metadata.get("gender"),
                    "age": metadata.get("age"),
                    "cue_spans_sec": cue_spans_segment,
                    "duration_sec": segment_length,
                }
            )

    return items


class CueAwarePreDataManager:
    def __init__(
        self,
        dataset_key: str,
        segment_length: float = 30.0,
        sample_rate: int = 16000,
        batch_size: int = 8,
        num_workers: int = 0,
        random_state: int = 42,
        max_groups_per_split: Optional[int] = None,
        split_map: Optional[Dict[str, str]] = None,
        cue_role: str = "patient",
        speech_scope: str = "participant",
    ):
        self.dataset_key = dataset_key
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.segment_samples = int(round(segment_length * sample_rate))
        items = load_cueaware_items(
            dataset_key=dataset_key,
            segment_length=segment_length,
            sample_rate=sample_rate,
            cue_role=cue_role,
            speech_scope=speech_scope,
        )
        group_ids = [item["group_id"] for item in items]

        if not items:
            raise ValueError(f"No cue-aware preprocessed items available for dataset={dataset_key}")

        self.items = items
        self.group_ids = group_ids
        self.split_map = (
            build_effective_split_map(dataset_key, self.group_ids, random_state, max_groups_per_split)
            if split_map is None
            else filter_split_map_to_available_groups(split_map, self.group_ids)
        )

        self.train_indices = [i for i, item in enumerate(self.items) if self.split_map.get(item["group_id"]) == "train"]
        self.val_indices = [i for i, item in enumerate(self.items) if self.split_map.get(item["group_id"]) == "val"]
        self.test_indices = [i for i, item in enumerate(self.items) if self.split_map.get(item["group_id"]) == "test"]

        if not self.train_indices or not self.val_indices or not self.test_indices:
            raise ValueError(
                f"Insufficient cue-aware split coverage for dataset={dataset_key}: "
                f"train={len(self.train_indices)}, val={len(self.val_indices)}, test={len(self.test_indices)}"
            )

        train_sample_labels: Dict[str, float] = {}
        for idx in self.train_indices:
            train_sample_labels[self.items[idx]["sample_id"]] = float(self.items[idx]["label_raw"])
        train_labels = np.asarray(list(train_sample_labels.values()), dtype=np.float32)
        self.label_mean = float(train_labels.mean())
        self.label_std = float(max(train_labels.std(ddof=0), 1e-6))
        for item in self.items:
            item["label_scaled"] = (item["label_raw"] - self.label_mean) / self.label_std

    def _subset(self, indices: Sequence[int]) -> List[Dict]:
        return [self.items[i] for i in indices]

    def get_items(self, split: str) -> List[Dict]:
        indices = {
            "train": self.train_indices,
            "val": self.val_indices,
            "test": self.test_indices,
        }[split]
        return self._subset(indices)

    def get_loader(self, split: str, shuffle: bool) -> DataLoader:
        indices = {
            "train": self.train_indices,
            "val": self.val_indices,
            "test": self.test_indices,
        }[split]
        dataset = CueAwareAudioDataset(self._subset(indices), is_training=(split == "train"))
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=cueaware_collate_fn,
        )

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) * self.label_std + self.label_mean

    def get_info(self) -> Dict[str, int]:
        def uniq(indices: Sequence[int]) -> int:
            return len({self.items[i]["sample_id"] for i in indices})

        return {
            "total_segments": len(self.items),
            "train_segments": len(self.train_indices),
            "val_segments": len(self.val_indices),
            "test_segments": len(self.test_indices),
            "total_recordings": len({item["sample_id"] for item in self.items}),
            "train_recordings": uniq(self.train_indices),
            "val_recordings": uniq(self.val_indices),
            "test_recordings": uniq(self.test_indices),
        }
