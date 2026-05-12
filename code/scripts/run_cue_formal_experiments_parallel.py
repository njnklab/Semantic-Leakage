#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence

import pandas as pd


FORMAL_DATASETS = ["edaic", "pdch", "cmdc", "mandic"]
CPU_BASELINE_MODELS = ["SVM", "RF"]
GPU_BASELINE_MODELS = ["DepAudioNet", "DisNet", "DMPF", "DALF", "STFN"]
BASELINE_MODELS = CPU_BASELINE_MODELS + GPU_BASELINE_MODELS
CF_MODELS = ["DepAudioNet", "DisNet", "DMPF", "DALF", "STFN"]
DEFAULT_SEEDS = ["42"]
GENDER_DATASETS = ["cmdc", "mandic"]
AGE_SETTINGS = [
    ("mandic", "adolescent", "young-adult"),
    ("mandic", "young-adult", "adolescent"),
    ("cmdc", "young-adult", "adult"),
    ("cmdc", "adult", "young-adult"),
]


@dataclass
class Task:
    name: str
    cmd: List[str]
    kind: str
    log_path: Path
    shard_path: Optional[Path] = None
    merge_key: Optional[str] = None


@dataclass
class RunningTask:
    task: Task
    process: subprocess.Popen
    log_handle: object
    gpu_id: Optional[int]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parallel formal sample-level CueFilter queue.")
    parser.add_argument("--run-root", type=Path, default=Path("CueFilter/results/formal_sample_parallel"))
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--gpu-jobs-per-gpu", type=int, default=1)
    parser.add_argument("--cpu-jobs", type=int, default=4)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--seeds", nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--skip-table1", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument(
        "--start-phase",
        choices=[
            "03_main_patient_audit",
            "04_role_specific_audit",
            "05_shared_pretrain",
            "06_functional_cuefilter",
            "07_generalization",
            "08_ablation",
        ],
        default="03_main_patient_audit",
        help="Start from this parallel phase. Earlier phases are not run.",
    )
    parser.add_argument(
        "--end-phase",
        choices=[
            "03_main_patient_audit",
            "04_role_specific_audit",
            "05_shared_pretrain",
            "06_functional_cuefilter",
            "07_generalization",
            "08_ablation",
        ],
        default=None,
        help="Stop after this parallel phase.",
    )
    parser.add_argument("--skip-serial", action="store_true", help="Skip serial cue coverage/validity steps.")
    parser.add_argument("--include-demographic-generalization", action="store_true")
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def ordered_pairs(items: Sequence[str]) -> List[tuple[str, str]]:
    return [(a, b) for a, b in itertools.product(items, items) if a != b]


def _env_for_task(gpu_id: Optional[int]) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if gpu_id is None:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def _launch_task(task: Task, cwd: Path, gpu_id: Optional[int]) -> RunningTask:
    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(task.log_path, "w", encoding="utf-8")
    env = _env_for_task(gpu_id)
    process = subprocess.Popen(
        task.cmd,
        cwd=str(cwd),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    return RunningTask(task=task, process=process, log_handle=log_handle, gpu_id=gpu_id)


def run_task_list(
    tasks: Sequence[Task],
    cwd: Path,
    gpu_ids: Sequence[int],
    gpu_jobs_per_gpu: int,
    cpu_jobs: int,
    rerun_existing: bool,
) -> None:
    runnable_tasks: List[Task] = []
    for task in tasks:
        if task.shard_path is not None and task.shard_path.exists() and not rerun_existing:
            print(f"[skip][exists] {task.name} -> {task.shard_path}")
            continue
        runnable_tasks.append(task)

    gpu_pending: Deque[Task] = deque([task for task in runnable_tasks if task.kind == "gpu"])
    cpu_pending: Deque[Task] = deque([task for task in runnable_tasks if task.kind == "cpu"])
    gpu_slots: Deque[int] = deque(gpu_id for gpu_id in gpu_ids for _ in range(gpu_jobs_per_gpu))
    running: List[RunningTask] = []
    failures: List[str] = []

    while gpu_pending or cpu_pending or running:
        while cpu_pending and sum(task.gpu_id is None for task in running) < cpu_jobs:
            task = cpu_pending.popleft()
            print(f"[launch][cpu] {task.name}")
            running.append(_launch_task(task, cwd=cwd, gpu_id=None))

        while gpu_pending and gpu_slots:
            task = gpu_pending.popleft()
            gpu_id = gpu_slots.popleft()
            print(f"[launch][gpu:{gpu_id}] {task.name}")
            running.append(_launch_task(task, cwd=cwd, gpu_id=gpu_id))

        if not running:
            continue

        time.sleep(2)
        next_running: List[RunningTask] = []
        for item in running:
            status = item.process.poll()
            if status is None:
                next_running.append(item)
                continue

            item.log_handle.close()
            if item.gpu_id is not None:
                gpu_slots.append(item.gpu_id)
            if status != 0:
                failures.append(item.task.name)
                print(f"[failed] {item.task.name} -> {status}")
            else:
                print(f"[done] {item.task.name}")
        running = next_running

    if failures:
        raise RuntimeError(f"Task failures: {', '.join(failures)}")


def run_serial_step(name: str, cmd: List[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[serial] {name}")
    with open(log_path, "w", encoding="utf-8") as log_handle:
        status = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=_env_for_task(None),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    if status != 0:
        raise RuntimeError(f"Serial step failed: {name}")


def merge_csv_shards(shard_paths: Iterable[Path], output_path: Path, sort_cols: Sequence[str]) -> None:
    frames = [pd.read_csv(path) for path in shard_paths if path.exists()]
    if not frames:
        raise FileNotFoundError(f"No shard CSVs found for {output_path.name}")
    merged = pd.concat(frames, ignore_index=True)
    existing_sort_cols = [col for col in sort_cols if col in merged.columns]
    if existing_sort_cols:
        merged = merged.sort_values(existing_sort_cols).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)


def build_table3_tasks(python_bin: str, run_root: Path, log_dir: Path, seeds: Sequence[str]) -> List[Task]:
    shard_dir = run_root / "shards" / "table3"
    tasks: List[Task] = []
    for dataset in FORMAL_DATASETS:
        for model in BASELINE_MODELS:
            shard_path = shard_dir / f"{dataset}__{model}.csv"
            kind = "cpu" if model in CPU_BASELINE_MODELS else "gpu"
            cmd = [
                python_bin, "-u", "CueFilter/Baseline/run_segment_experiments.py",
                "--datasets", dataset,
                "--experiment", "patient-baseline",
                "--train-variant", "pre",
                "--variants", "pre", "cue-only", "cue-removed", "random-removed",
                "--models", model,
                "--seeds", *seeds,
                "--eval-level", "sample",
                "--segment-first",
                "--output", str(shard_path),
            ]
            if kind == "cpu":
                cmd.extend(["--device", "cpu"])
            else:
                cmd.extend(["--device", "cuda:0"])
            tasks.append(
                Task(
                    name=f"table3__{dataset}__{model}",
                    cmd=cmd,
                    kind=kind,
                    log_path=log_dir / f"table3__{dataset}__{model}.log",
                    shard_path=shard_path,
                    merge_key="table3",
                )
            )
    return tasks


def build_table4_tasks(python_bin: str, run_root: Path, log_dir: Path, seeds: Sequence[str]) -> List[Task]:
    shard_dir = run_root / "shards" / "table4"
    tasks: List[Task] = []
    for dataset in FORMAL_DATASETS:
        for model in BASELINE_MODELS:
            shard_path = shard_dir / f"{dataset}__{model}.csv"
            kind = "cpu" if model in CPU_BASELINE_MODELS else "gpu"
            cmd = [
                python_bin, "-u", "CueFilter/Baseline/run_segment_experiments.py",
                "--datasets", dataset,
                "--experiment", "role-effect",
                "--train-variant", "pre",
                "--variants", "pre", "cue-removed", "random-removed",
                "--models", model,
                "--seeds", *seeds,
                "--eval-level", "sample",
                "--segment-first",
                "--output", str(shard_path),
            ]
            if kind == "cpu":
                cmd.extend(["--device", "cpu"])
            else:
                cmd.extend(["--device", "cuda:0"])
            tasks.append(
                Task(
                    name=f"table4__{dataset}__{model}",
                    cmd=cmd,
                    kind=kind,
                    log_path=log_dir / f"table4__{dataset}__{model}.log",
                    shard_path=shard_path,
                    merge_key="table4",
                )
            )
    return tasks


def build_shared_pretrain_tasks(
    python_bin: str,
    run_root: Path,
    log_dir: Path,
    ckpt_template: str,
    seeds: Sequence[str],
) -> List[Task]:
    shard_dir = run_root / "shards" / "shared_pretrain"
    tasks: List[Task] = []
    for model in CF_MODELS:
        shard_path = shard_dir / f"{model}.csv"
        cmd = [
            python_bin, "-u", "scripts/pretrain_shared_cuefilter.py",
            "--models", model,
            "--seeds", *seeds,
            "--shared-pretrain-datasets", *FORMAL_DATASETS,
            "--cue-role", "patient",
            "--speech-scope", "participant",
            "--load-shared-cuefilter", ckpt_template,
            "--save-shared-cuefilter", ckpt_template,
            "--boundary-tolerance-sec", "1.0",
            "--device", "cuda:0",
            "--output", str(shard_path),
        ]
        tasks.append(
            Task(
                name=f"shared_pretrain__{model}",
                cmd=cmd,
                kind="gpu",
                log_path=log_dir / f"shared_pretrain__{model}.log",
                shard_path=shard_path,
                merge_key="shared_pretrain",
            )
        )
    return tasks


def build_functional_tasks(
    python_bin: str,
    run_root: Path,
    log_dir: Path,
    ckpt_template: str,
    seeds: Sequence[str],
) -> List[Task]:
    shard_dir = run_root / "shards" / "functional"
    tasks: List[Task] = []
    for dataset in FORMAL_DATASETS:
        for model in CF_MODELS:
            output_dir = shard_dir / f"{dataset}__{model}"
            shard_path = output_dir / "table_cuefilter_functional.csv"
            cmd = [
                python_bin, "-u", "CueFilter/run_functional_experiments.py",
                "--datasets", dataset,
                "--models", model,
                "--seeds", *seeds,
                "--cue-role", "patient",
                "--speech-scope", "participant",
                "--shared-pretrain-datasets", *FORMAL_DATASETS,
                "--load-shared-cuefilter", ckpt_template,
                "--save-shared-cuefilter", ckpt_template,
                "--boundary-tolerance-sec", "1.0",
                "--eval-level", "sample",
                "--segment-first",
                "--device", "cuda:0",
                "--output-dir", str(output_dir),
            ]
            tasks.append(
                Task(
                    name=f"functional__{dataset}__{model}",
                    cmd=cmd,
                    kind="gpu",
                    log_path=log_dir / f"functional__{dataset}__{model}.log",
                    shard_path=shard_path,
                    merge_key="functional",
                )
            )
    return tasks


def build_table8_tasks(
    python_bin: str,
    run_root: Path,
    log_dir: Path,
    ckpt_template: str,
    seeds: Sequence[str],
) -> List[Task]:
    shard_dir = run_root / "shards" / "table8"
    tasks: List[Task] = []
    for dataset in FORMAL_DATASETS:
        for model in CF_MODELS:
            shard_path = shard_dir / f"{dataset}__{model}.csv"
            cmd = [
                python_bin, "-u", "CueFilter/run_ablation_experiments.py",
                "--datasets", dataset,
                "--models", model,
                "--seeds", *seeds,
                "--cue-role", "patient",
                "--speech-scope", "participant",
                "--load-shared-cuefilter", ckpt_template,
                "--boundary-tolerance-sec", "1.0",
                "--eval-level", "sample",
                "--segment-first",
                "--device", "cuda:0",
                "--detail-output-dir", str(run_root / "details" / "table8"),
                "--output", str(shard_path),
            ]
            tasks.append(
                Task(
                    name=f"table8__{dataset}__{model}",
                    cmd=cmd,
                    kind="gpu",
                    log_path=log_dir / f"table8__{dataset}__{model}.log",
                    shard_path=shard_path,
                    merge_key="table8",
                )
            )
    return tasks


def build_table7_tasks(
    python_bin: str,
    run_root: Path,
    log_dir: Path,
    ckpt_template: str,
    seeds: Sequence[str],
    include_demographic: bool,
) -> List[Task]:
    shard_dir = run_root / "shards" / "table7"
    tasks: List[Task] = []

    # LODO (Leave-One-Dataset-Out): for each target dataset, train on the union
    # of the remaining three corpora and evaluate on the held-out target.
    for test_dataset in FORMAL_DATASETS:
        train_datasets = [d for d in FORMAL_DATASETS if d != test_dataset]
        for model in CF_MODELS:
            shard_path = shard_dir / f"cross_dataset__lodo_{test_dataset}__{model}.csv"
            cmd = [
                python_bin, "-u", "CueFilter/run_generalization_experiments.py",
                "--mode", "cross-dataset",
                "--train-datasets", *train_datasets,
                "--test-datasets", test_dataset,
                "--models", model,
                "--seeds", *seeds,
                "--cue-role", "patient",
                "--speech-scope", "participant",
                "--load-shared-cuefilter", ckpt_template,
                "--boundary-tolerance-sec", "1.0",
                "--eval-level", "sample",
                "--segment-first",
                "--device", "cuda:0",
                "--detail-output-dir", str(run_root / "details" / "table7"),
                "--output", str(shard_path),
            ]
            tasks.append(
                Task(
                    name=f"table7__cross_lodo__{test_dataset}__{model}",
                    cmd=cmd,
                    kind="gpu",
                    log_path=log_dir / f"table7__cross_lodo__{test_dataset}__{model}.log",
                    shard_path=shard_path,
                    merge_key=f"table7_cross_lodo_{test_dataset}",
                )
            )

    if include_demographic:
        for dataset in GENDER_DATASETS:
            for train_gender, test_gender in [("male", "female"), ("female", "male")]:
                for model in CF_MODELS:
                    shard_path = shard_dir / f"cross_gender__{dataset}__{train_gender}_to_{test_gender}__{model}.csv"
                    cmd = [
                        python_bin, "-u", "CueFilter/run_generalization_experiments.py",
                        "--mode", "cross-gender",
                        "--dataset", dataset,
                        "--train-gender", train_gender,
                        "--test-gender", test_gender,
                        "--models", model,
                        "--seeds", *seeds,
                        "--cue-role", "patient",
                        "--speech-scope", "participant",
                        "--load-shared-cuefilter", ckpt_template,
                        "--boundary-tolerance-sec", "1.0",
                        "--eval-level", "sample",
                        "--segment-first",
                        "--device", "cuda:0",
                        "--detail-output-dir", str(run_root / "details" / "table7"),
                        "--output", str(shard_path),
                    ]
                    tasks.append(
                        Task(
                            name=f"table7__cross_gender__{dataset}__{train_gender}_to_{test_gender}__{model}",
                            cmd=cmd,
                            kind="gpu",
                            log_path=log_dir / f"table7__cross_gender__{dataset}__{train_gender}_to_{test_gender}__{model}.log",
                            shard_path=shard_path,
                            merge_key=f"table7_cross_gender__{dataset}__{train_gender}_to_{test_gender}",
                        )
                    )

        for dataset, train_age, test_age in AGE_SETTINGS:
            for model in CF_MODELS:
                shard_path = shard_dir / f"cross_age__{dataset}__{train_age}_to_{test_age}__{model}.csv"
                cmd = [
                    python_bin, "-u", "CueFilter/run_generalization_experiments.py",
                    "--mode", "cross-age",
                    "--dataset", dataset,
                    "--train-age-bin", train_age,
                    "--test-age-bin", test_age,
                    "--models", model,
                    "--seeds", *seeds,
                    "--cue-role", "patient",
                    "--speech-scope", "participant",
                    "--load-shared-cuefilter", ckpt_template,
                    "--boundary-tolerance-sec", "1.0",
                    "--eval-level", "sample",
                    "--segment-first",
                    "--device", "cuda:0",
                    "--detail-output-dir", str(run_root / "details" / "table7"),
                    "--output", str(shard_path),
                ]
                tasks.append(
                    Task(
                        name=f"table7__cross_age__{dataset}__{train_age}_to_{test_age}__{model}",
                        cmd=cmd,
                        kind="gpu",
                        log_path=log_dir / f"table7__cross_age__{dataset}__{train_age}_to_{test_age}__{model}.log",
                        shard_path=shard_path,
                        merge_key=f"table7_cross_age__{dataset}__{train_age}_to_{test_age}",
                    )
                )

    return tasks


def merge_grouped_shards(tasks: Sequence[Task], dest_dir: Path, prefix_to_sort_cols: Dict[str, Sequence[str]]) -> None:
    grouped: Dict[str, List[Path]] = {}
    for task in tasks:
        if task.merge_key and task.shard_path is not None:
            grouped.setdefault(task.merge_key, []).append(task.shard_path)

    for merge_key, shard_paths in grouped.items():
        if merge_key == "functional":
            merge_functional_shards(shard_paths, dest_dir.parent / "cuefilter_functional")
            continue
        if merge_key == "table3":
            dest = dest_dir / "table3_main_patient_audit_raw.csv"
        elif merge_key == "table4":
            dest = dest_dir / "table4_role_specific_raw.csv"
        elif merge_key == "table56":
            dest = dest_dir / "table5_6_localization_mitigation_raw.csv"
        elif merge_key == "shared_pretrain":
            dest = dest_dir / "table5_shared_pretrain_summary.csv"
        elif merge_key == "table8":
            dest = dest_dir / "table8_ablation_raw.csv"
        elif merge_key.startswith("table7_"):
            dest = dest_dir / f"{merge_key.replace('table7_', 'table7_generalization_')}.csv"
        else:
            raise KeyError(merge_key)

        merge_prefix = merge_key.split("__", 1)[0]
        if merge_key.startswith("table7_"):
            merge_prefix = "table7"
        sort_cols = prefix_to_sort_cols.get(merge_prefix, [])
        merge_csv_shards(shard_paths, dest, sort_cols=sort_cols)
        print(f"[merged] {dest}")


def _format_mean_std(values: Sequence[float]) -> str:
    series = pd.Series(values, dtype="float64").dropna()
    if series.empty:
        return "-"
    return f"{series.mean():.3f}±{series.std(ddof=0):.3f}"


def _summarize_functional(raw_df: pd.DataFrame, group_cols: Sequence[str], value_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for keys, group in raw_df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        for value_col in value_cols:
            row[value_col] = _format_mean_std(group[value_col].astype(float).tolist())
        row["N"] = len(group)
        rows.append(row)
    return pd.DataFrame(rows)


def merge_functional_shards(shard_paths: Iterable[Path], output_dir: Path) -> None:
    specs = {
        "table_cuefilter_functional": (["Method"], ["Original", "Cost", "Cue rm.", "Rand rm.", "Cue extra"]),
        "table_frozen_feature_reuse": (["Feature used"], ["Original", "Cue rm.", "Rand rm.", "Cue extra"]),
        "table_suppression_strength": (["Method", "Strength"], ["Original", "Cost", "Cue extra", "Extra red."]),
        "table_semantic_probe": (["Feature used"], ["Coverage", "Category", "Count", "Dep. error"]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dirs = [path.parent for path in shard_paths]
    for stem, (group_cols, value_cols) in specs.items():
        raw_files = [path / f"{stem}_raw.csv" for path in shard_dirs if (path / f"{stem}_raw.csv").exists()]
        if not raw_files:
            continue
        raw = pd.concat([pd.read_csv(path) for path in raw_files], ignore_index=True)
        raw_path = output_dir / f"{stem}_raw.csv"
        summary_path = output_dir / f"{stem}.csv"
        raw.to_csv(raw_path, index=False)
        summary = _summarize_functional(raw, group_cols, value_cols)
        summary.to_csv(summary_path, index=False)
        print(f"[merged] {raw_path}")
        print(f"[merged] {summary_path}")


def merge_table56_curve_points(run_root: Path) -> None:
    curve_dir = run_root / "curves" / "table56"
    if not curve_dir.exists():
        return
    for suffix in ("roc_points", "pr_points"):
        files = sorted(curve_dir.glob(f"*__{suffix}.csv"))
        if not files:
            continue
        merged = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
        merged = merged.sort_values(["Dataset", "Model", "Seed", "point_idx"]).reset_index(drop=True)
        output_path = curve_dir / f"all_{suffix}.csv"
        merged.to_csv(output_path, index=False)
        print(f"[merged] {output_path}")


def build_serial_steps(args, raw_dir: Path) -> List[tuple[str, List[str]]]:
    steps: List[tuple[str, List[str]]] = []
    if not args.skip_table1:
        steps.append(
            (
                "01_table1_cue_validity",
                [
                    args.python, "-u", "Experiments/Comparison/emotion_classification.py",
                    "-d", "all",
                    "--output", str(raw_dir / "table1_cue_validity.csv"),
                ],
            )
        )
    steps.extend(
        [
            (
                "02_table2_cue_coverage",
                [
                    args.python, "-u", "Experiments/CueAnaly/cue_coverage_stats.py",
                    "-d", "all",
                    "--output", str(raw_dir / "table2_cue_coverage.csv"),
                ],
            ),
            (
                "02b_table4_role_coverage_patient",
                [
                    args.python, "-u", "Experiments/CueAnaly/cue_coverage_stats.py",
                    "-d", "all",
                    "--cue-role", "patient",
                    "--speech-scope", "dialogue",
                    "--output", str(raw_dir / "table4_role_coverage_patient.csv"),
                ],
            ),
            (
                "02c_table4_role_coverage_doctor",
                [
                    args.python, "-u", "Experiments/CueAnaly/cue_coverage_stats.py",
                    "-d", "all",
                    "--cue-role", "doctor",
                    "--speech-scope", "dialogue",
                    "--output", str(raw_dir / "table4_role_coverage_doctor.csv"),
                ],
            ),
            (
                "02d_table4_role_coverage_all",
                [
                    args.python, "-u", "Experiments/CueAnaly/cue_coverage_stats.py",
                    "-d", "all",
                    "--cue-role", "all",
                    "--speech-scope", "dialogue",
                    "--output", str(raw_dir / "table4_role_coverage_all.csv"),
                ],
            ),
        ]
    )
    return steps


def print_execution_plan(
    serial_steps: Sequence[tuple[str, List[str]]],
    phase_specs: Sequence[tuple[str, Sequence[Task]]],
    args,
) -> None:
    print("Execution plan")
    print(f"  skip_table1={args.skip_table1}")
    print(f"  skip_serial={args.skip_serial}")
    print(f"  start_phase={args.start_phase}")
    print(f"  end_phase={args.end_phase}")
    print(f"  seeds={args.seeds}")
    print(f"  include_demographic_generalization={args.include_demographic_generalization}")
    print(f"  rerun_existing={args.rerun_existing}")
    print(f"  gpus={args.gpus} gpu_jobs_per_gpu={args.gpu_jobs_per_gpu} cpu_jobs={args.cpu_jobs}")
    print("Serial steps:")
    for step_name, cmd in serial_steps:
        print(f"  - {step_name}: {' '.join(cmd)}")
    print("Parallel phases:")
    for phase_name, tasks in phase_specs:
        gpu_count = sum(task.kind == 'gpu' for task in tasks)
        cpu_count = sum(task.kind == 'cpu' for task in tasks)
        print(f"  - {phase_name}: {len(tasks)} tasks ({gpu_count} gpu, {cpu_count} cpu)")
        preview = [task.name for task in list(tasks)[:5]]
        for task_name in preview:
            print(f"      {task_name}")
        if len(tasks) > 5:
            print("      ...")


def main() -> None:
    args = create_parser().parse_args()
    root = Path.cwd()
    run_root = args.run_root
    raw_dir = run_root / "raw"
    table_dir = run_root / "tables"
    figure_dir = run_root / "figures"
    log_dir = run_root / "logs"
    shard_dir = run_root / "shards"
    ckpt_dir = run_root / "shared_stage1_ckpts"
    for path in (raw_dir, table_dir, figure_dir, log_dir, shard_dir, ckpt_dir):
        path.mkdir(parents=True, exist_ok=True)

    ckpt_template = str((ckpt_dir / "{model}_seed{seed}.pt").resolve())
    sort_cols_map = {
        "table3": ["Dataset", "Model", "Variant"],
        "table4": ["Dataset", "CueRole", "Model", "Variant"],
        "functional": ["Dataset", "Backbone", "Method"],
        "table56": ["Dataset", "Model"],
        "shared_pretrain": ["Model", "Seed"],
        "table8": ["Dataset", "Backbone", "Variant"],
        "table7": ["Mode", "Source", "Target", "Model"],
    }

    print(f"Run root: {run_root.resolve()}")
    print(f"GPUs: {args.gpus} | gpu_jobs_per_gpu={args.gpu_jobs_per_gpu} | cpu_jobs={args.cpu_jobs}")
    print(f"Seeds: {args.seeds} | rerun_existing={args.rerun_existing}")

    serial_steps = build_serial_steps(args, raw_dir)

    phase_specs = [
        ("03_main_patient_audit", build_table3_tasks(args.python, run_root, log_dir, args.seeds)),
        ("04_role_specific_audit", build_table4_tasks(args.python, run_root, log_dir, args.seeds)),
        ("05_shared_pretrain", build_shared_pretrain_tasks(args.python, run_root, log_dir, ckpt_template, args.seeds)),
        ("06_functional_cuefilter", build_functional_tasks(args.python, run_root, log_dir, ckpt_template, args.seeds)),
        (
            "07_generalization",
            build_table7_tasks(
                args.python,
                run_root,
                log_dir,
                ckpt_template,
                args.seeds,
                args.include_demographic_generalization,
            ),
        ),
        ("08_ablation", build_table8_tasks(args.python, run_root, log_dir, ckpt_template, args.seeds)),
    ]
    if args.start_phase:
        phase_names = [name for name, _ in phase_specs]
        phase_specs = phase_specs[phase_names.index(args.start_phase):]
    if args.end_phase:
        phase_names = [name for name, _ in phase_specs]
        phase_specs = phase_specs[: phase_names.index(args.end_phase) + 1]
    if args.dry_run:
        print_execution_plan(serial_steps, phase_specs, args)
        return

    if not args.skip_serial:
        for step_name, cmd in serial_steps:
            run_serial_step(step_name, cmd, cwd=root, log_path=log_dir / f"{step_name}.log")

    for phase_name, tasks in phase_specs:
        print(f"\n===== {phase_name} =====")
        run_task_list(
            tasks=tasks,
            cwd=root,
            gpu_ids=args.gpus,
            gpu_jobs_per_gpu=args.gpu_jobs_per_gpu,
            cpu_jobs=args.cpu_jobs,
            rerun_existing=args.rerun_existing,
        )
        merge_grouped_shards(tasks, raw_dir, prefix_to_sort_cols=sort_cols_map)

    run_serial_step(
        "09_assemble_tables",
        [args.python, "-u", "scripts/assemble_formal_tables.py", "--run-root", str(run_root)],
        cwd=root,
        log_path=log_dir / "09_assemble_tables.log",
    )
    if not args.skip_figures:
        run_serial_step(
            "10_generate_figures",
            [args.python, "-u", "scripts/generate_formal_figures.py", "--run-root", str(run_root)],
            cwd=root,
            log_path=log_dir / "10_generate_figures.log",
        )

    print("\nFormal parallel queue finished.")


if __name__ == "__main__":
    main()
