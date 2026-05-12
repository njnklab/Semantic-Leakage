"""
Export cue coverage statistics for 30-second sample-level auditing.

For each dataset, this script counts:
- total number of 30-second samples after segmentation on the retained-speech timeline,
- number of samples that contain at least one cue interval,
- mean cue duration per 30-second sample,
- mean cue coverage ratio per 30-second sample.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CueFilter.Baseline.audio_views import (  # noqa: E402
    DATASET_CONFIGS,
    cue_intervals_from_outputs,
    intersect_intervals,
    iter_output_samples,
    load_json,
    map_intervals_to_concatenated_timeline,
    speech_intervals_from_transcript,
)


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


def _duration(intervals: Sequence[Tuple[float, float]]) -> float:
    return float(sum(max(0.0, end - start) for start, end in intervals))


def summarize_dataset(
    dataset_key: str,
    segment_length: float,
    cue_role: str,
    speech_scope: str,
) -> Dict[str, object]:
    total_segments = 0
    cue_segments = 0
    cue_durations: List[float] = []
    cue_coverages: List[float] = []

    for sample in iter_output_samples(dataset_key):
        transcript_path = Path(sample["transcript_path"])
        cues_path = Path(sample["cues_path"])
        transcript = load_json(transcript_path)
        cues = load_json(cues_path)
        sample_id = str(sample["sample_id"])
        transcript["sample_id"] = sample_id

        retained_intervals = speech_intervals_from_transcript(transcript, speech_scope=speech_scope)
        if not retained_intervals:
            continue

        cue_intervals = cue_intervals_from_outputs(
            dataset_key=dataset_key,
            sample_id=sample_id,
            transcript=transcript,
            cues=cues,
            cue_role=cue_role,
        )
        cue_intervals = intersect_intervals(retained_intervals, cue_intervals)
        mapped_cues = map_intervals_to_concatenated_timeline(retained_intervals, cue_intervals)
        pre_duration = _duration(retained_intervals)
        if pre_duration <= 0:
            continue

        segment_count = int(math.ceil(pre_duration / segment_length))
        total_segments += segment_count
        for seg_idx in range(segment_count):
            seg_start = seg_idx * segment_length
            seg_end = min(seg_start + segment_length, pre_duration)
            effective_duration = max(seg_end - seg_start, 1e-8)
            local_cues = _intersect_relative_spans(mapped_cues, seg_start, seg_end)
            cue_duration = _duration(local_cues)
            cue_coverage = cue_duration / effective_duration
            if cue_duration > 0:
                cue_segments += 1
            cue_durations.append(cue_duration)
            cue_coverages.append(cue_coverage)

    return {
        "Dataset": DATASET_CONFIGS[dataset_key].display_name,
        "# 30-s samples": int(total_segments),
        "# cue-containing samples": int(cue_segments),
        "Mean cue duration (s)": float(np.mean(cue_durations)) if cue_durations else float("nan"),
        "Mean cue coverage": float(np.mean(cue_coverages)) if cue_coverages else float("nan"),
    }


def format_markdown(rows: Sequence[Dict[str, object]]) -> str:
    lines = [
        "| Dataset | # 30-s samples | # cue-containing samples | Mean cue duration (s) | Mean cue coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['Dataset']} | {row['# 30-s samples']} | {row['# cue-containing samples']} | "
            f"{row['Mean cue duration (s)']:.3f} | {row['Mean cue coverage']:.3f} |"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cue coverage statistics for 30-second sample-level auditing.")
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        choices=[*DATASET_CONFIGS.keys(), "all"],
    )
    parser.add_argument("--segment-length", type=float, default=30.0)
    parser.add_argument("--cue-role", choices=["patient", "doctor", "all"], default="patient")
    parser.add_argument("--speech-scope", choices=["participant", "interviewer", "dialogue"], default="participant")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_keys = list(DATASET_CONFIGS.keys()) if args.dataset == "all" else [args.dataset]
    rows = [
        summarize_dataset(
            dataset_key=dataset_key,
            segment_length=args.segment_length,
            cue_role=args.cue_role,
            speech_scope=args.speech_scope,
        )
        for dataset_key in dataset_keys
    ]

    table = format_markdown(rows)
    print(table)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() == ".csv":
            import pandas as pd

            pd.DataFrame(rows).to_csv(args.output, index=False)
        else:
            args.output.write_text(table + "\n", encoding="utf-8")
        print(f"\nSaved table to {args.output}")


if __name__ == "__main__":
    main()
