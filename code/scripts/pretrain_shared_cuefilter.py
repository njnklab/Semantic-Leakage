#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CueFilter.Baseline.audio_views import DATASET_CONFIGS
from CueFilter.Baseline.experiment_utils import set_seed
from CueFilter.integration.registry import build_adapter, list_supported_adapters
from CueFilter.models import CueFilter
from CueFilter.run_mitigation_experiments import MITIGATION_DEFAULTS, _initialize_shared_cuefilter_if_needed, _with_model_defaults


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain shared stage-1 CueFilter checkpoints.")
    parser.add_argument("--models", nargs="+", choices=list_supported_adapters(), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--shared-pretrain-datasets",
        nargs="+",
        choices=list(DATASET_CONFIGS.keys()),
        required=True,
    )
    parser.add_argument("--cue-role", choices=["patient", "doctor", "all"], default="patient")
    parser.add_argument("--speech-scope", choices=["participant", "interviewer", "dialogue"], default="participant")
    parser.add_argument("--segment-length", type=float, default=30.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--stage1-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--stage1-patience", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=1.0)
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
    parser.add_argument("--load-shared-cuefilter", type=str, default=None)
    parser.add_argument("--save-shared-cuefilter", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    return parser


def main() -> None:
    args = create_parser().parse_args()
    summary_rows: List[Dict[str, object]] = []

    for model_name in args.models:
        effective_args = _with_model_defaults(args, model_name)
        for seed in args.seeds:
            set_seed(seed)
            device = torch.device(effective_args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
            adapter = build_adapter(model_name).to(device)
            cuefilter = CueFilter(
                input_dim=adapter.feature_dim,
                n_blocks=effective_args.n_blocks,
                kernel_size=effective_args.kernel_size,
                groups=effective_args.groups,
                alpha=effective_args.alpha,
                gamma=effective_args.gamma,
                cue_threshold=effective_args.cue_threshold,
            ).to(device)
            metrics = _initialize_shared_cuefilter_if_needed(
                cuefilter=cuefilter,
                adapter=adapter,
                model_name=model_name,
                seed=seed,
                device=device,
                args=effective_args,
            )
            ckpt_path = Path(effective_args.save_shared_cuefilter.format(model=model_name, seed=seed))
            row = {
                "Model": model_name,
                "Seed": seed,
                "Checkpoint": str(ckpt_path),
                "Exists": ckpt_path.exists(),
            }
            if metrics is not None:
                row.update(
                    {
                        "Precision": metrics.get("frame_precision"),
                        "Recall": metrics.get("frame_recall"),
                        "FrameF1": metrics.get("frame_f1"),
                        "AUCPR": metrics.get("auc_pr"),
                        "SpanF1": metrics.get("span_f1"),
                        "BudgetMAE": metrics.get("budget_mae"),
                    }
                )
            summary_rows.append(row)
            print(f"Prepared shared CueFilter | model={model_name} | seed={seed} | ckpt={ckpt_path}")

    if not summary_rows:
        print("No shared CueFilter checkpoints were prepared.")
        return

    df = pd.DataFrame(summary_rows)
    print("\n" + df.to_markdown(index=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".csv":
            df.to_csv(output_path, index=False)
        else:
            output_path.write_text(df.to_markdown(index=False), encoding="utf-8")
        print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
