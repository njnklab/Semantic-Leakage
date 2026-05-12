#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_mean(cell) -> float:
    text = str(cell).strip()
    if not text or text == "-" or text.lower() == "nan":
        return float("nan")
    for sep in ("±", "+/-"):
        if sep in text:
            text = text.split(sep, 1)[0].strip()
            break
    return float(text)


def save_figure(fig, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def generate_figure4(table_dir: Path, figure_dir: Path) -> None:
    table_path = table_dir / "Table1_cue_validity.csv"
    if not table_path.exists():
        return
    df = pd.read_csv(table_path)
    agg = (
        df.groupby("Dataset", as_index=False)[["Cue negative score", "Non-cue negative score"]]
        .mean()
        .melt(id_vars="Dataset", var_name="Segment", value_name="Negative confidence")
    )
    agg["Segment"] = agg["Segment"].map({
        "Cue negative score": "Cue",
        "Non-cue negative score": "Non-cue",
    })

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.barplot(data=agg, x="Dataset", y="Negative confidence", hue="Segment", palette=["#8c1d18", "#3b5b92"], ax=ax)
    ax.set_title("Figure 4. Cue validity analysis")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Negative confidence")
    ax.legend(title="")
    save_figure(fig, figure_dir / "Figure4_cue_validity_grouped_bar")


def generate_figure5(table_dir: Path, figure_dir: Path) -> None:
    df = pd.read_csv(table_dir / "Table3_main_patient_cue_auditing.csv")
    df["MAE increase after cue removal (mean)"] = df["MAE increase after cue removal"].map(parse_mean)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 5.2))
    sns.barplot(
        data=df,
        x="Dataset",
        y="MAE increase after cue removal (mean)",
        hue="Model",
        ax=ax,
    )
    ax.set_title("Figure 5. Main patient-cue auditing")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("MAE increase after cue removal")
    ax.legend(title="Model", ncols=2, fontsize=8)
    save_figure(fig, figure_dir / "Figure5_main_patient_audit_grouped_bar")


def generate_figure6(table_dir: Path, figure_dir: Path) -> None:
    df = pd.read_csv(table_dir / "Table4_role_specific_cue_impact.csv")
    df["MAE increase after removal (mean)"] = df["MAE increase after removal"].map(parse_mean)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.barplot(
        data=df,
        x="Dataset",
        y="MAE increase after removal (mean)",
        hue="Cue role",
        palette=["#1f77b4", "#ff7f0e", "#2ca02c"],
        ax=ax,
    )
    ax.set_title("Figure 6. Role-specific cue impact")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("MAE increase after removal")
    ax.legend(title="")
    save_figure(fig, figure_dir / "Figure6_role_specific_impact_grouped_bar")


def generate_figure8(table_dir: Path, figure_dir: Path) -> None:
    df = pd.read_csv(table_dir / "Table6_plug_and_play_mitigation.csv")
    melted = df.melt(
        id_vars=["Dataset", "Backbone"],
        value_vars=["Baseline cue-removal increase", "CueFilter cue-removal increase"],
        var_name="Condition",
        value_name="MAE increase",
    )
    melted["MAE increase"] = melted["MAE increase"].map(parse_mean)
    melted["Condition"] = melted["Condition"].map({
        "Baseline cue-removal increase": "Baseline",
        "CueFilter cue-removal increase": "CueFilter",
    })

    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=melted,
        x="Backbone",
        y="MAE increase",
        hue="Condition",
        col="Dataset",
        kind="bar",
        sharey=False,
        height=4.2,
        aspect=1.0,
    )
    g.set_axis_labels("Backbone", "Cue-removal increase")
    g.set_titles("{col_name}")
    g.figure.suptitle("Figure 8. Plug-and-play mitigation", y=1.04)
    save_figure(g.figure, figure_dir / "Figure8_plug_and_play_mitigation_bar")


def write_figure7_placeholder(figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / "Figure7_case_localization_pending.md"
    path.write_text(
        "\n".join(
            [
                "# Figure 7 Placeholder",
                "",
                "Expected content:",
                "- waveform or spectrogram",
                "- human-verified cue spans",
                "- CueFilter cue probability curve",
                "",
                "This figure requires exporting one sample-level cue-probability timeline from the localization stage.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_manifest(figure_dir: Path) -> None:
    files = sorted(p.name for p in figure_dir.iterdir() if p.is_file())
    manifest = pd.DataFrame({"figure_file": files})
    manifest.to_csv(figure_dir / "manifest.csv", index=False)
    (figure_dir / "manifest.md").write_text(manifest.to_markdown(index=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate formal paper figures from assembled tables.")
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    table_dir = args.run_root / "tables"
    figure_dir = args.run_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    generate_figure4(table_dir, figure_dir)
    generate_figure5(table_dir, figure_dir)
    generate_figure6(table_dir, figure_dir)
    generate_figure8(table_dir, figure_dir)
    write_figure7_placeholder(figure_dir)
    write_manifest(figure_dir)


if __name__ == "__main__":
    main()
