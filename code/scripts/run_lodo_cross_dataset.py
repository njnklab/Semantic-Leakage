#!/usr/bin/env python3
"""
Standalone LODO (Leave-One-Dataset-Out) cross-dataset experiment runner.
Runs the paper's Table tab:cross_dataset_cf experiment:
    for each target dataset, train on the remaining 3, test on target,
    with full CueFilter pipeline (pretrain + joint train), then audit.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FORMAL_DATASETS = ["edaic", "pdch", "cmdc", "mandic"]
CF_MODELS = ["DepAudioNet", "DisNet", "DMPF", "DALF", "STFN"]

CHECKPOINT_TEMPLATE = str(
    (PROJECT_ROOT / "CueFilter" / "results" / "formal_sample_parallel" /
     "shared_stage1_ckpts" / "{model}_seed{seed}.pt").resolve()
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LODO cross-dataset generalization experiments.")
    parser.add_argument("--targets", nargs="+", choices=FORMAL_DATASETS, default=FORMAL_DATASETS)
    parser.add_argument("--models", nargs="+", choices=CF_MODELS, default=CF_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--output-dir", type=str, default="CueFilter/results/lodo_cross_dataset")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-shared-ckpt", action="store_true",
                        help="Skip shared CueFilter checkpoint loading (train stage1 from scratch)")
    parser.add_argument("--info-only", action="store_true")
    return parser


def run_task(cmd: List[str], log_path: Path, dry_run: bool) -> int:
    print(f"  {'[DRY RUN]' if dry_run else 'Running'}: {' '.join(cmd)}")
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_file:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=log_file, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"  FAILED (returncode={proc.returncode}), see {log_path}")
    else:
        print(f"  Done, log: {log_path}")
    return proc.returncode


def main() -> None:
    args = create_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    shard_dir = output_dir / "shards"
    raw_dir = output_dir / "raw"
    for d in (log_dir, shard_dir, raw_dir):
        d.mkdir(parents=True, exist_ok=True)

    failures: List[str] = []

    for test_dataset in args.targets:
        train_datasets = [d for d in FORMAL_DATASETS if d != test_dataset]
        for model in args.models:
            shard_path = shard_dir / f"cross_lodo_{test_dataset}__{model}.csv"
            log_path = log_dir / f"cross_lodo_{test_dataset}__{model}.log"

            cmd = [
                args.python, "-u",
                str(PROJECT_ROOT / "CueFilter" / "run_generalization_experiments.py"),
                "--mode", "cross-dataset",
                "--train-datasets", *train_datasets,
                "--test-datasets", test_dataset,
                "--models", model,
                "--seeds", *[str(s) for s in args.seeds],
                "--cue-role", "patient",
                "--speech-scope", "participant",
                "--boundary-tolerance-sec", "1.0",
                "--eval-level", "sample",
                "--segment-first",
                "--device", args.device,
                "--detail-output-dir", str(output_dir / "details"),
                "--output", str(shard_path),
            ]

            if not args.no_shared_ckpt:
                cmd.extend(["--load-shared-cuefilter", CHECKPOINT_TEMPLATE])

            name = f"LODO target={test_dataset} model={model}"
            print(f"\n[{name}]")
            rc = run_task(cmd, log_path, args.dry_run or args.info_only)
            if rc != 0:
                failures.append(name)

    # Merge shards per target dataset
    print("\n=== Merging shards ===")
    import pandas as pd
    for test_dataset in args.targets:
        pattern = f"cross_lodo_{test_dataset}__*.csv"
        shard_files = sorted(shard_dir.glob(pattern))
        if not shard_files:
            print(f"  No shards found for {test_dataset}")
            continue
        dfs = []
        for f in shard_files:
            try:
                dfs.append(pd.read_csv(f))
            except Exception as e:
                print(f"  Error reading {f}: {e}")
        if dfs:
            merged = pd.concat(dfs, ignore_index=True)
            merged_path = raw_dir / f"table7_generalization_cross_lodo_{test_dataset}.csv"
            merged.to_csv(merged_path, index=False)
            print(f"  Merged {len(dfs)} shards → {merged_path} ({len(merged)} rows)")

    # Assemble final table
    print("\n=== Assembling paper table ===")
    assemble_cmd = [
        args.python, "-u",
        str(PROJECT_ROOT / "scripts" / "assemble_formal_tables.py"),
        "--run-root", str(output_dir),
    ]
    run_task(assemble_cmd, log_dir / "assemble_table.log", args.dry_run)

    if failures:
        print(f"\n{len(failures)} tasks failed:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("\nAll tasks completed successfully.")

    # Show the assembled table
    table_path = output_dir / "tables" / "Table_cross_dataset_cf.md"
    if table_path.exists():
        print("\n=== Cross-Dataset LODO Table ===")
        print(table_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
