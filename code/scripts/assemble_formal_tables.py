#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

TABLE8_VARIANT_ORDER = [
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


ROLE_DISPLAY = {
    "patient": "Patient cue",
    "doctor": "Interviewer cue",
    "all": "All cues",
}

SETTING_DISPLAY = {
    "cross-dataset": "Cross-dataset",
    "cross-gender": "Cross-gender",
    "cross-age": "Cross-age",
}


def parse_mean(cell) -> float:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return float("nan")
    text = str(cell).strip()
    if not text or text == "-":
        return float("nan")
    for sep in ("±", "+/-"):
        if sep in text:
            text = text.split(sep, 1)[0].strip()
            break
    try:
        return float(text)
    except ValueError:
        return float("nan")


def fmt_delta(value: float) -> str:
    if np.isnan(value):
        return "-"
    return f"{value:.3f}"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def write_outputs(df: pd.DataFrame, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_stem.with_suffix(".csv"), index=False)
    output_stem.with_suffix(".md").write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")


def assemble_table1(raw_dir: Path, table_dir: Path) -> None:
    path = raw_dir / "table1_cue_validity.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    keep = df[[
        "Dataset",
        "Language",
        "Sentiment model",
        "Cue negative score",
        "Non-cue negative score",
        "Difference",
    ]].copy()
    write_outputs(keep, table_dir / "Table1_cue_validity")


def assemble_table2(raw_dir: Path, table_dir: Path) -> None:
    if not (raw_dir / "table2_cue_coverage.csv").exists():
        return
    df = load_csv(raw_dir / "table2_cue_coverage.csv")
    keep = df[[
        "Dataset",
        "# 30-s samples",
        "# cue-containing samples",
        "Mean cue duration (s)",
        "Mean cue coverage",
    ]].copy()
    write_outputs(keep, table_dir / "Table2_cue_coverage")


def assemble_table3(raw_dir: Path, table_dir: Path) -> None:
    if not (raw_dir / "table3_main_patient_audit_raw.csv").exists():
        return
    df = load_csv(raw_dir / "table3_main_patient_audit_raw.csv")
    pivot = (
        df.pivot_table(
            index=["Dataset", "Model"],
            columns="Variant",
            values="MAE",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    rows: List[Dict[str, object]] = []
    for _, row in pivot.iterrows():
        pre = row.get("pre", "-")
        cue_only = row.get("cue-only", "-")
        cue_removed = row.get("cue-removed", "-")
        random_removed = row.get("random-removed", "-")
        rows.append(
            {
                "Dataset": row["Dataset"],
                "Model": row["Model"],
                "Original MAE": pre,
                "Cue-only MAE": cue_only,
                "Cue-removed MAE": cue_removed,
                "Random-removed MAE": random_removed,
                "MAE increase after cue removal": fmt_delta(parse_mean(cue_removed) - parse_mean(pre)),
                "Extra increase over random removal": fmt_delta(parse_mean(cue_removed) - parse_mean(random_removed)),
            }
        )
    write_outputs(pd.DataFrame(rows), table_dir / "Table3_main_patient_cue_auditing")


def _load_role_coverages(raw_dir: Path) -> pd.DataFrame:
    parts = []
    for role in ("patient", "doctor", "all"):
        path = raw_dir / f"table4_role_coverage_{role}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[["Dataset", "Mean cue coverage"]].copy()
        df["CueRole"] = role
        parts.append(df)
    if not parts:
        return pd.DataFrame(columns=["Dataset", "CueRole", "Mean cue coverage"])
    return pd.concat(parts, ignore_index=True)


def assemble_table4(raw_dir: Path, table_dir: Path) -> None:
    if not (raw_dir / "table4_role_specific_raw.csv").exists():
        return
    df = load_csv(raw_dir / "table4_role_specific_raw.csv")
    coverage_df = _load_role_coverages(raw_dir)
    pivot = (
        df.pivot_table(
            index=["Dataset", "CueRole"],
            columns="Variant",
            values="MAE",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    merged = pivot.merge(coverage_df, on=["Dataset", "CueRole"], how="left")
    rows: List[Dict[str, object]] = []
    for _, row in merged.iterrows():
        pre = row.get("pre", "-")
        cue_removed = row.get("cue-removed", "-")
        random_removed = row.get("random-removed", "-")
        rows.append(
            {
                "Dataset": row["Dataset"],
                "Cue role": ROLE_DISPLAY.get(str(row["CueRole"]), str(row["CueRole"])),
                "Cue coverage": "-" if pd.isna(row.get("Mean cue coverage")) else f"{float(row['Mean cue coverage']):.3f}",
                "Original MAE": pre,
                "Cue-removed MAE": cue_removed,
                "Random-removed MAE": random_removed,
                "MAE increase after removal": fmt_delta(parse_mean(cue_removed) - parse_mean(pre)),
            }
        )
    write_outputs(pd.DataFrame(rows), table_dir / "Table4_role_specific_cue_impact")


def assemble_table5_and_6(raw_dir: Path, table_dir: Path) -> None:
    functional_dir = raw_dir.parent / "cuefilter_functional"
    functional_path = functional_dir / "table_cuefilter_functional.csv"
    frozen_path = functional_dir / "table_frozen_feature_reuse.csv"
    strength_path = functional_dir / "table_suppression_strength.csv"
    probe_path = functional_dir / "table_semantic_probe.csv"
    if functional_path.exists():
        functional = pd.read_csv(functional_path)
        keep = functional[["Method", "Original", "Cost", "Cue rm.", "Rand rm.", "Cue extra"]].copy()
        write_outputs(keep, table_dir / "Table5_cuefilter_functional")
        if frozen_path.exists():
            frozen = pd.read_csv(frozen_path)
            keep = frozen[["Feature used", "Original", "Cue rm.", "Rand rm.", "Cue extra"]].copy()
            write_outputs(keep, table_dir / "Table6_frozen_feature_reuse")
        if strength_path.exists():
            strength = pd.read_csv(strength_path)
            keep = strength[["Method", "Strength", "Original", "Cost", "Cue extra", "Extra red."]].copy()
            write_outputs(keep, table_dir / "Table7_suppression_strength")
        if probe_path.exists():
            probe = pd.read_csv(probe_path)
            keep = probe[["Feature used", "Coverage", "Category", "Count", "Dep. error"]].copy()
            write_outputs(keep, table_dir / "Table8_semantic_probe")
        return

    if not (raw_dir / "table5_6_localization_mitigation_raw.csv").exists():
        return
    df = load_csv(raw_dir / "table5_6_localization_mitigation_raw.csv")
    table5 = df[[
        "Dataset",
        "Model",
        "Precision",
        "Recall",
        "Stage1 Frame F1",
        "AUC-ROC",
        "AUC-PR",
        "Span F1",
        "BErr",
    ]].rename(
        columns={
            "Model": "Backbone",
            "Stage1 Frame F1": "Frame F1",
        }
    )
    write_outputs(table5, table_dir / "Table5_cuefilter_localization")

    table6 = df[[
        "Dataset",
        "Model",
        "Baseline Original MAE",
        "CueFilter Original MAE",
        "Baseline cue-removal increase",
        "CueFilter cue-removal increase",
        "Reduction",
    ]].rename(columns={"Model": "Backbone"})
    write_outputs(table6, table_dir / "Table6_plug_and_play_mitigation")


def _compute_model_average(values: pd.Series) -> float:
    """Parse mean±std or plain float values and return the average across models."""
    means = []
    for val in values.dropna():
        v = parse_mean(val)
        if not np.isnan(v):
            means.append(v)
    if not means:
        return float("nan")
    return float(np.mean(means))


def _compute_model_mean_std(values: pd.Series) -> str:
    """Parse mean±std or plain float values and return mean±std across models."""
    means = []
    for val in values.dropna():
        v = parse_mean(val)
        if not np.isnan(v):
            means.append(v)
    if not means:
        return "-"
    m = float(np.mean(means))
    s = float(np.std(means, ddof=1)) if len(means) > 1 else 0.0
    return f"{m:.3f}±{s:.3f}"


def assemble_table7(raw_dir: Path, table_dir: Path) -> None:
    files = sorted(raw_dir.glob("table7_generalization_*.csv"))
    if not files:
        return

    # Check if this is a LODO (cross-lodo) experiment — paper Table "cross_dataset_cf"
    is_lodo = any("cross_lodo" in f.name for f in files)
    output_name = "Table9_generalization" if (
        raw_dir.parent / "cuefilter_functional" / "table_cuefilter_functional.csv"
    ).exists() else "Table7_generalization"

    if is_lodo and {"BaselineError", "CueFilterError", "BaselineNormExtraRandom", "CueFilterNormExtraRandom"}.issubset(
        pd.read_csv(files[0], nrows=1).columns
    ):
        # LODO cross-dataset: aggregate across models per target dataset
        lodo_rows: List[Dict[str, object]] = []
        for path in files:
            df = pd.read_csv(path)
            if df.empty:
                continue
            target = str(df["Target"].iloc[0]) if "Target" in df.columns else path.stem
            bb_error = _compute_model_average(df["BaselineError"])
            cf_error = _compute_model_average(df["CueFilterError"])
            bb_delta_cue = _compute_model_average(df["BaselineNormExtraRandom"])
            cf_delta_cue = _compute_model_average(df["CueFilterNormExtraRandom"])

            delta_nmae = cf_error - bb_error
            delta_percent = (cf_delta_cue - bb_delta_cue) / bb_delta_cue * 100.0 if abs(bb_delta_cue) > 1e-6 else float("nan")

            lodo_rows.append({
                "Target dataset": target,
                "nMAE (BB)": fmt_delta(bb_error),
                "nMAE (CF)": f"{cf_error:.3f} ({delta_nmae:+.3f})",
                "\u0394cue (BB)": fmt_delta(bb_delta_cue),
                "\u0394cue (CF)": f"{cf_delta_cue:.3f} ({delta_percent:+.0f}\\%)",
            })

        if lodo_rows:
            # Add Mean row
            bb_errors = [parse_mean(r["nMAE (BB)"]) for r in lodo_rows]
            cf_errors = [parse_mean(r["nMAE (CF)"]) for r in lodo_rows]
            bb_deltas = [parse_mean(r["\u0394cue (BB)"]) for r in lodo_rows]
            cf_deltas = [parse_mean(r["\u0394cue (CF)"]) for r in lodo_rows]
            valid_bb = [v for v in bb_errors if not np.isnan(v)]
            valid_cf = [v for v in cf_errors if not np.isnan(v)]
            valid_bbd = [v for v in bb_deltas if not np.isnan(v)]
            valid_cfd = [v for v in cf_deltas if not np.isnan(v)]
            mean_bb = np.mean(valid_bb) if valid_bb else float("nan")
            mean_cf = np.mean(valid_cf) if valid_cf else float("nan")
            mean_bbd = np.mean(valid_bbd) if valid_bbd else float("nan")
            mean_cfd = np.mean(valid_cfd) if valid_cfd else float("nan")
            delta_mean = mean_cf - mean_bb
            delta_pct = (mean_cfd - mean_bbd) / mean_bbd * 100.0 if abs(mean_bbd) > 1e-6 else float("nan")

            lodo_rows.append({
                "Target dataset": "Mean",
                "nMAE (BB)": fmt_delta(mean_bb),
                "nMAE (CF)": f"{mean_cf:.3f} ({delta_mean:+.3f})",
                "\u0394cue (BB)": fmt_delta(mean_bbd),
                "\u0394cue (CF)": f"{mean_cfd:.3f} ({delta_pct:+.0f}\\%)",
            })

        write_outputs(pd.DataFrame(lodo_rows), table_dir / "Table_cross_dataset_cf")
        return

    df = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    if {"BaselineError", "CueFilterError", "BaselineNormCueRemoval", "CueFilterNormCueRemoval"}.issubset(df.columns):
        rows: List[Dict[str, object]] = []
        for _, row in df.iterrows():
            rows.append(
                {
                    "Target": row["Target"],
                    "Method": "Backbone",
                    "Error": row["BaselineError"],
                    "RMSE": row.get("BaselineRMSE", "-"),
                    "R2": row.get("BaselineR2", "-"),
                    "Cue rm.": row["BaselineNormCueRemoval"],
                    "Cue extra": row.get("BaselineNormExtraRandom", "-"),
                }
            )
            rows.append(
                {
                    "Target": row["Target"],
                    "Method": "CueFilter",
                    "Error": row["CueFilterError"],
                    "RMSE": row.get("CueFilterRMSE", "-"),
                    "R2": row.get("CueFilterR2", "-"),
                    "Cue rm.": row["CueFilterNormCueRemoval"],
                    "Cue extra": row.get("CueFilterNormExtraRandom", "-"),
                }
            )
        write_outputs(pd.DataFrame(rows), table_dir / output_name)
        return

    table7 = df[[
        "Mode",
        "Source",
        "Target",
        "Model",
        "BaselineMAE",
        "CueFilterMAE",
        "BaselineCueRemoval",
        "CueFilterCueRemoval",
    ]].rename(
        columns={
            "Mode": "Setting",
            "Source": "Train",
            "Target": "Test",
            "Model": "Backbone",
            "BaselineMAE": "Baseline MAE",
            "CueFilterMAE": "CueFilter MAE",
            "BaselineCueRemoval": "Baseline cue-removal increase",
            "CueFilterCueRemoval": "CueFilter cue-removal increase",
        }
    )
    table7["Setting"] = table7["Setting"].map(lambda x: SETTING_DISPLAY.get(str(x), str(x)))
    write_outputs(table7, table_dir / output_name)


def assemble_table8(table_dir: Path, raw_dir: Path) -> None:
    raw_path = raw_dir / "table8_ablation_raw.csv"
    if raw_path.exists():
        df = pd.read_csv(raw_path)
        summary_rows: List[Dict[str, object]] = []
        for variant_name in TABLE8_VARIANT_ORDER:
            variant_df = df[df["Variant"] == variant_name].copy()
            if variant_df.empty:
                continue

            summary_row: Dict[str, object] = {"Variant": variant_name}
            for column in ["Error", "Cost", "Cue rm.", "Cue extra", "Gate dens.", "Density err."]:
                values = variant_df[column].map(parse_mean).to_numpy(dtype=float)
                values = values[~np.isnan(values)]
                if len(values) == 0:
                    summary_row[column] = "-"
                elif len(values) == 1:
                    summary_row[column] = f"{values[0]:.3f}"
                else:
                    summary_row[column] = f"{np.mean(values):.3f} ± {np.std(values):.3f}"
            summary_rows.append(summary_row)

        keep = pd.DataFrame(summary_rows)
        output_name = "Table10_ablation_and_sanity_checks" if (raw_dir.parent / "cuefilter_functional" / "table_cuefilter_functional.csv").exists() else "Table8_ablation_and_sanity_checks"
        write_outputs(keep, table_dir / output_name)
        return

    table8 = pd.DataFrame(
        [
            {
                "Variant": "Backbone",
                "Error": "TBD",
                "Cost": "TBD",
                "Cue rm.": "TBD",
                "Cue extra": "TBD",
                "Gate dens.": "TBD",
                "Density err.": "TBD",
            },
            {
                "Variant": "Full CueFilter",
                "Error": "TBD",
                "Cost": "TBD",
                "Cue rm.": "TBD",
                "Cue extra": "TBD",
                "Gate dens.": "TBD",
                "Density err.": "TBD",
            },
            {
                "Variant": "No cue pretrain",
                "Error": "TBD",
                "Cost": "TBD",
                "Cue rm.": "TBD",
                "Cue extra": "TBD",
                "Gate dens.": "TBD",
                "Density err.": "TBD",
            },
            {
                "Variant": "No cue loss",
                "Error": "TBD",
                "Cost": "TBD",
                "Cue rm.": "TBD",
                "Cue extra": "TBD",
                "Gate dens.": "TBD",
                "Density err.": "TBD",
            },
            {
                "Variant": "No budget loss",
                "Error": "TBD",
                "Cost": "TBD",
                "Cue rm.": "TBD",
                "Cue extra": "TBD",
                "Gate dens.": "TBD",
                "Density err.": "TBD",
            },
            {
                "Variant": "No feature norm",
                "Error": "TBD",
                "Cost": "TBD",
                "Cue rm.": "TBD",
                "Cue extra": "TBD",
                "Gate dens.": "TBD",
                "Density err.": "TBD",
            },
            {
                "Variant": "Hard mask",
                "Error": "TBD",
                "Cost": "TBD",
                "Cue rm.": "TBD",
                "Cue extra": "TBD",
                "Gate dens.": "TBD",
                "Density err.": "TBD",
            },
            {
                "Variant": "Random gate",
                "Error": "TBD",
                "Cost": "TBD",
                "Cue rm.": "TBD",
                "Cue extra": "TBD",
                "Gate dens.": "TBD",
                "Density err.": "TBD",
            },
            {
                "Variant": "Shuffled gate",
                "Error": "TBD",
                "Cost": "TBD",
                "Cue rm.": "TBD",
                "Cue extra": "TBD",
                "Gate dens.": "TBD",
                "Density err.": "TBD",
            },
        ]
    )
    output_name = "Table10_ablation_and_sanity_checks" if (raw_dir.parent / "cuefilter_functional" / "table_cuefilter_functional.csv").exists() else "Table8_ablation_and_sanity_checks"
    write_outputs(table8, table_dir / output_name)


def assemble_manifest(table_dir: Path) -> None:
    files = sorted(p.name for p in table_dir.glob("Table*.md"))
    manifest = pd.DataFrame({"table_file": files})
    write_outputs(manifest, table_dir / "manifest")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble formal paper tables from raw experiment CSVs.")
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    raw_dir = args.run_root / "raw"
    table_dir = args.run_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    assemble_table1(raw_dir, table_dir)
    assemble_table2(raw_dir, table_dir)
    assemble_table3(raw_dir, table_dir)
    assemble_table4(raw_dir, table_dir)
    assemble_table5_and_6(raw_dir, table_dir)
    assemble_table7(raw_dir, table_dir)
    assemble_table8(table_dir, raw_dir)
    assemble_manifest(table_dir)


if __name__ == "__main__":
    main()
