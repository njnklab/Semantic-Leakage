"""
Cue summary export.

For each dataset, export one Excel workbook that contains:
- top-25 cue phrases for interviewer
- top-25 cue phrases for interviewee
- cue count vs score table
- cue duration vs score table
- additional compact distribution indicators
- two JointGrid plots for cue count / duration vs score

Each block is written on the same sheet and separated by one blank column.

Plot reading guide / 读图指南:
- x-axis / 横轴: Scale Score, i.e. the clinical scale total score matched to each sample.
- y-axis / 纵轴: Cue Count or Cue Duration (s), depending on the exported figure.
- joint heatmap color depth / 联合热图颜色深度: darker cells mean more samples fall into that score-metric bin.
- top marginal histogram / 上侧柱状图: the distribution of Scale Score for interviewer and interviewee cues.
- right marginal histogram / 右侧柱状图: the distribution of Cue Count or Cue Duration for interviewer and interviewee cues.
- smoothed lines / 平滑趋势线: role-wise trend lines across score bins; the legend rho value is the Spearman correlation.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager, pyplot as plt
from matplotlib.patches import Patch
from openpyxl.styles import Font

BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent.parent
BASE_OUTPUT_DIR = BASE_DIR / "outputs"
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AGENT_OUTPUTS_DIR = PROJECT_ROOT / "agent" / "outputs"
DATA_DIR = PROJECT_ROOT / "data"
LOCAL_ARIAL_DIR = Path.home() / ".local" / "share" / "fonts" / "arial-ms"

ROLES = ("interviewer", "interviewee")
ROLE_DISPLAY = {
    "interviewer": "Interviewer",
    "interviewee": "Interviewee",
}
ROLE_COLORS = {
    "interviewee": "#03012D",
    "interviewer": "#BF360C",
}
DATASETS = {
    "edaic": ("E-DAIC", "PHQ-8"),
    "mandic": ("ManDIC", "HAMD-17"),
    "cmdc": ("CMDC", "PHQ"),
    "pdch": ("PDCH", "HAMD"),
}
TOP_K_PHRASES = 35
PLOT_FONT_SCALE = 2.0
AXIS_LABEL_FONT_SIZE = 24
TICK_LABEL_FONT_SIZE = 24
LEGEND_FONT_SIZE = 20
CBAR_TICK_FONT_SIZE = 16
CBAR_TITLE_FONT_SIZE = 18

if LOCAL_ARIAL_DIR.exists():
    for font_path in sorted(LOCAL_ARIAL_DIR.glob("*.TTF")) + sorted(LOCAL_ARIAL_DIR.glob("*.ttf")):
        font_manager.fontManager.addfont(str(font_path))

try:
    font_manager.findfont("Arial", fallback_to_default=False)
    AXIS_FONT_FAMILY = "Arial"
except ValueError:
    AXIS_FONT_FAMILY = "Liberation Sans"


def normalize_role(cue: Dict) -> str:
    """Normalize cue speaker role to interviewer/interviewee."""
    role = str(cue.get("speaker_role") or cue.get("speaker") or "").strip().lower()
    if any(tag in role for tag in ("interviewer", "doctor", "clinician", "therapist")):
        return "interviewer"
    if any(tag in role for tag in ("interviewee", "participant", "patient", "subject")):
        return "interviewee"
    return role


def infer_subject_id(dataset: str, cue_file: Path, dataset_dir: Path, sample_id: str) -> str:
    """Infer the score-matching subject id from the cue path when possible."""
    dataset_key = dataset.strip().lower()
    rel_parts = cue_file.relative_to(dataset_dir).parts

    if dataset_key == "e-daic":
        match = re.search(r"(\d+)", sample_id)
        return match.group(1) if match else sample_id

    if dataset_key == "mandic":
        return sample_id

    if dataset_key == "cmdc":
        for part in rel_parts:
            if re.fullmatch(r"(HC|MDD)\d+", part, flags=re.IGNORECASE):
                return part

    if dataset_key == "pdch":
        for part in rel_parts:
            if re.fullmatch(r"\d{3}[AB]", part, flags=re.IGNORECASE):
                return part

    return sample_id


def load_cue_samples(dataset: str) -> List[Dict]:
    """Load cue results from agent outputs with robust subject ids."""
    dataset_dir = AGENT_OUTPUTS_DIR / dataset
    samples: List[Dict] = []

    if not dataset_dir.exists():
        return samples

    cue_files = sorted(set(dataset_dir.rglob("cues.json")) | set(dataset_dir.rglob("*_cues.json")))
    for cue_file in cue_files:
        with open(cue_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        cues = [cue for cue in data.get("cues", []) if isinstance(cue, dict)]
        if not cues:
            continue

        sample_id = str(data.get("sample_id", cue_file.stem)).strip()
        subject_id = infer_subject_id(dataset, cue_file, dataset_dir, sample_id)
        relative_path = str(cue_file.relative_to(dataset_dir))

        samples.append(
            {
                "sample_id": sample_id,
                "subject_id": subject_id,
                "relative_path": relative_path,
                "cues": cues,
            }
        )

    return samples


def load_score_data(dataset: str) -> Dict[str, float]:
    """Load score table for a dataset."""
    scores: Dict[str, float] = {}

    if dataset == "E-DAIC":
        for score_file in ("train_split_Depression_AVEC2017.csv", "dev_split_Depression_AVEC2017.csv"):
            file_path = DATA_DIR / "E-DAIC" / score_file
            if not file_path.exists():
                continue
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                participant_id = str(int(row.get("Participant_ID", 0)))
                scores[participant_id] = float(row.get("PHQ8_Score", 0))

    elif dataset == "ManDIC":
        file_path = DATA_DIR / "ManDIC" / "info.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                sample_id = str(row.get("standard_id", "")).strip()
                score = row.get("HAMD-17_total_score", np.nan)
                if not pd.isna(score):
                    scores[sample_id] = float(score)

    elif dataset == "CMDC":
        file_path = DATA_DIR / "CMDC_EULA" / "SubjectInfo.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                sample_id = str(row.get("ID", "")).strip()
                score = row.get("PHQtotal", np.nan)
                if not pd.isna(score):
                    scores[sample_id] = float(score)

    elif dataset == "PDCH":
        file_path = DATA_DIR / "PDCH" / "HAMD_annotation_en.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                sample_id = str(row.get("Serial", "")).strip()
                score = row.get("total", np.nan)
                if not pd.isna(score):
                    scores[sample_id] = float(score)

    return scores


def match_score(sample_id: str, subject_id: str, scores: Dict[str, float]) -> Optional[float]:
    """Match a cue sample to a score value."""
    for key in (subject_id, sample_id):
        if key in scores:
            return scores[key]

    candidates = [subject_id, sample_id]
    for candidate in candidates:
        numbers = re.findall(r"\d+", candidate)
        for key, value in scores.items():
            key_clean = str(key).replace(".0", "")
            if numbers and numbers[0] == key_clean:
                return value
            if key in candidate or candidate in str(key):
                return value

    return None


def safe_duration(cue: Dict) -> float:
    """Return a non-negative cue duration."""
    try:
        start = float(cue.get("start", 0.0))
        end = float(cue.get("end", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, end - start)


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return a ratio, or NaN when the denominator is zero."""
    if denominator == 0:
        return np.nan
    return numerator / denominator


def summarize_sample(
    dataset: str,
    score_name: str,
    sample_id: str,
    subject_id: str,
    relative_path: str,
    cues: List[Dict],
    score: Optional[float],
) -> Tuple[Dict, Dict[str, Counter]]:
    """Build compact sample-level metrics and role phrase counts."""
    role_stats = {
        role: {
            "count": 0,
            "duration": 0.0,
            "phrases": Counter(),
            "categories": Counter(),
        }
        for role in ROLES
    }

    for cue in cues:
        role = normalize_role(cue)
        if role not in ROLES:
            continue

        text = str(cue.get("text", "")).strip().lower()
        category = str(cue.get("category", "unknown")).strip().lower() or "unknown"
        duration = safe_duration(cue)

        role_stats[role]["count"] += 1
        role_stats[role]["duration"] += duration
        if text:
            role_stats[role]["phrases"][text] += 1
        role_stats[role]["categories"][category] += 1

    total_cue_count = sum(role_stats[role]["count"] for role in ROLES)
    total_cue_duration = sum(role_stats[role]["duration"] for role in ROLES)

    row = {
        "dataset": dataset,
        "subject_id": subject_id,
        "sample_id": sample_id,
        "relative_path": relative_path,
        "score_name": score_name,
        "score": score,
        "total_cue_count": total_cue_count,
        "total_cue_duration": total_cue_duration,
    }

    for role in ROLES:
        count = role_stats[role]["count"]
        duration = role_stats[role]["duration"]
        row[f"{role}_cue_count"] = count
        row[f"{role}_cue_duration"] = duration
        row[f"{role}_avg_cue_duration"] = duration / count if count else 0.0
        row[f"{role}_unique_phrase_count"] = len(role_stats[role]["phrases"])
        row[f"{role}_unique_category_count"] = len(role_stats[role]["categories"])
        row[f"{role}_cue_count_share"] = count / total_cue_count if total_cue_count else 0.0
        row[f"{role}_cue_duration_share"] = duration / total_cue_duration if total_cue_duration else 0.0

    row["count_gap_interviewer_minus_interviewee"] = row["interviewer_cue_count"] - row["interviewee_cue_count"]
    row["duration_gap_interviewer_minus_interviewee"] = row["interviewer_cue_duration"] - row["interviewee_cue_duration"]
    row["count_ratio_interviewer_to_interviewee"] = safe_ratio(
        row["interviewer_cue_count"], row["interviewee_cue_count"]
    )
    row["duration_ratio_interviewer_to_interviewee"] = safe_ratio(
        row["interviewer_cue_duration"], row["interviewee_cue_duration"]
    )

    role_phrase_counts = {
        role: role_stats[role]["phrases"]
        for role in ROLES
    }
    return row, role_phrase_counts


def build_top_phrase_table(dataset: str, role: str, counter: Counter) -> pd.DataFrame:
    """Build a top-k phrase table for one role."""
    total_count = sum(counter.values())
    rows = []
    for rank, (phrase, count) in enumerate(counter.most_common(TOP_K_PHRASES), start=1):
        rows.append(
            {
                "dataset": dataset,
                "speaker_role": role,
                "rank": rank,
                "phrase": phrase,
                "count": count,
                "frequency_ratio": count / total_count if total_count else 0.0,
            }
        )
    return pd.DataFrame(rows)


def with_blank_separator(df: pd.DataFrame) -> pd.DataFrame:
    """Return a one-column blank separator."""
    return pd.DataFrame({"": [""] * max(len(df), 1)})


def write_dataset_workbook(
    dataset: str,
    dataset_output_dir: Path,
    sections: List[Tuple[str, pd.DataFrame]],
    notes: List[Tuple[str, str]],
) -> Path:
    """Write one workbook per dataset with grouped sections on one sheet."""
    output_path = dataset_output_dir / f"{dataset.lower()}_summary.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        start_col = 0
        for title, df in sections:
            df_to_write = df.copy()
            df_to_write.to_excel(writer, sheet_name="summary", startrow=1, startcol=start_col, index=False)
            ws = writer.sheets["summary"]
            cell = ws.cell(row=1, column=start_col + 1)
            cell.value = title
            cell.font = Font(bold=True)
            start_col += len(df_to_write.columns) + 1

        notes_df = pd.DataFrame(notes, columns=["section", "description"])
        notes_df.to_excel(writer, sheet_name="说明", index=False)

        for ws in writer.book.worksheets:
            for column_cells in ws.columns:
                max_len = 0
                column_letter = column_cells[0].column_letter
                for cell in column_cells:
                    value = "" if cell.value is None else str(cell.value)
                    max_len = max(max_len, len(value))
                ws.column_dimensions[column_letter].width = min(max(max_len + 2, 10), 40)

    return output_path


def build_role_metric_plot_df(matched_df: pd.DataFrame, metric_suffix: str) -> pd.DataFrame:
    """Convert wide role columns into a long table for plotting."""
    plot_frames = []
    for role in ROLES:
        metric_col = f"{role}_{metric_suffix}"
        role_df = matched_df[["subject_id", "sample_id", "relative_path", "score"]].copy()
        role_df["speaker_role"] = role
        role_df["metric_value"] = matched_df[metric_col].astype(float)
        plot_frames.append(role_df)

    return pd.concat(plot_frames, ignore_index=True)


def style_axis_text(ax) -> None:
    """Apply Arial bold styling to axis labels and tick labels."""
    if ax.get_xlabel():
        ax.set_xlabel(
            ax.get_xlabel(),
            fontname=AXIS_FONT_FAMILY,
            fontweight="bold",
            fontsize=AXIS_LABEL_FONT_SIZE,
        )
    if ax.get_ylabel():
        ax.set_ylabel(
            ax.get_ylabel(),
            fontname=AXIS_FONT_FAMILY,
            fontweight="bold",
            fontsize=AXIS_LABEL_FONT_SIZE,
        )

    for label in ax.get_xticklabels():
        label.set_fontname(AXIS_FONT_FAMILY)
        label.set_fontweight("bold")
        label.set_fontsize(TICK_LABEL_FONT_SIZE)
    for label in ax.get_yticklabels():
        label.set_fontname(AXIS_FONT_FAMILY)
        label.set_fontweight("bold")
        label.set_fontsize(TICK_LABEL_FONT_SIZE)


def save_joint_role_plot(
    dataset: str,
    score_name: str,
    plot_df: pd.DataFrame,
    y_label: str,
    output_stem: str,
    dataset_output_dir: Path,
    *,
    y_discrete: bool = False,
) -> Path:
    """Save one JointGrid heatmap-style plot as JPG."""
    sns.set_theme(style="ticks", font_scale=PLOT_FONT_SCALE)

    g = sns.JointGrid(
        data=plot_df,
        x="score",
        y="metric_value",
        height=9.6,
        ratio=5,
        space=0.08,
        marginal_ticks=True,
    )

    g.figure.subplots_adjust(top=0.985, left=0.12, right=0.95, bottom=0.11)
    cax_interviewee = g.ax_joint.inset_axes([0.80, 0.77, 0.028, 0.20], zorder=6)
    cax_interviewer = g.ax_joint.inset_axes([0.87, 0.77, 0.028, 0.20], zorder=6)
    for cax in (cax_interviewee, cax_interviewer):
        cax.set_facecolor((1, 1, 1, 0.92))

    role_stats = {}
    for role in ("interviewee", "interviewer"):
        role_df = plot_df[plot_df["speaker_role"] == role]
        color = ROLE_COLORS[role]
        cmap = sns.light_palette(color, as_cmap=True)
        cbar_ax = cax_interviewee if role == "interviewee" else cax_interviewer

        prev_collections = len(g.ax_joint.collections)
        sns.histplot(
            data=role_df,
            x="score",
            y="metric_value",
            ax=g.ax_joint,
            discrete=(True, y_discrete),
            cmap=cmap,
            pmax=0.8,
            alpha=0.78,
        )
        if len(g.ax_joint.collections) > prev_collections:
            mesh = g.ax_joint.collections[-1]
            cbar = g.figure.colorbar(mesh, cax=cbar_ax, orientation="vertical")
            cbar.outline.set_linewidth(0.6)
            cbar.ax.tick_params(labelsize=CBAR_TICK_FONT_SIZE, length=2)
            for tick in cbar.ax.get_xticklabels():
                tick.set_fontname(AXIS_FONT_FAMILY)
                tick.set_fontweight("bold")
                tick.set_fontsize(CBAR_TICK_FONT_SIZE)
            for tick in cbar.ax.get_yticklabels():
                tick.set_fontname(AXIS_FONT_FAMILY)
                tick.set_fontweight("bold")
                tick.set_fontsize(CBAR_TICK_FONT_SIZE)

        trend_df = role_df.groupby("score", as_index=False)["metric_value"].mean().sort_values("score")
        if not trend_df.empty:
            smooth_window = 5 if len(trend_df) >= 9 else 3
            trend_df["trend_value"] = (
                trend_df["metric_value"].rolling(window=smooth_window, center=True, min_periods=1).mean()
            )
            g.ax_joint.plot(
                trend_df["score"],
                trend_df["trend_value"],
                color=color,
                linewidth=2.2,
                alpha=0.95,
                zorder=5,
            )
        rho = role_df["score"].corr(role_df["metric_value"], method="spearman")
        role_stats[role] = {
            "rho": rho,
            "n": len(role_df),
        }

        sns.histplot(
            data=role_df,
            x="score",
            ax=g.ax_marg_x,
            color=color,
            element="step",
            fill=True,
            alpha=0.22,
            common_norm=False,
            discrete=True,
            linewidth=1.5,
        )
        sns.histplot(
            data=role_df,
            y="metric_value",
            ax=g.ax_marg_y,
            color=color,
            element="step",
            fill=True,
            alpha=0.22,
            common_norm=False,
            discrete=y_discrete,
            linewidth=1.5,
        )

    g.set_axis_labels("Scale Score", y_label)
    legend_handles = [
        Patch(
            facecolor=ROLE_COLORS["interviewee"],
            edgecolor="none",
            alpha=0.78,
            label=f"{ROLE_DISPLAY['interviewee']}  ρ={role_stats['interviewee']['rho']:.2f}",
        ),
        Patch(
            facecolor=ROLE_COLORS["interviewer"],
            edgecolor="none",
            alpha=0.78,
            label=f"{ROLE_DISPLAY['interviewer']}  ρ={role_stats['interviewer']['rho']:.2f}",
        ),
    ]
    legend = g.ax_joint.legend(
        handles=legend_handles,
        title=None,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.975),
        fontsize=LEGEND_FONT_SIZE,
    )
    for text in legend.get_texts():
        text.set_fontname(AXIS_FONT_FAMILY)
        text.set_fontweight("bold")
        text.set_fontsize(LEGEND_FONT_SIZE)
    g.ax_joint.grid(alpha=0.18, linewidth=0.6)

    style_axis_text(g.ax_joint)
    style_axis_text(g.ax_marg_x)
    style_axis_text(g.ax_marg_y)

    jpg_path = dataset_output_dir / f"{output_stem}.jpg"

    g.figure.savefig(
        jpg_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
    )
    plt.close(g.figure)

    return jpg_path


def export_dataset_plots(
    dataset: str,
    score_name: str,
    matched_df: pd.DataFrame,
    dataset_output_dir: Path,
) -> List[Path]:
    """Export cue-count and cue-duration JointGrid plots."""
    if matched_df.empty:
        return []

    saved_paths: List[Path] = []

    count_plot_df = build_role_metric_plot_df(matched_df, "cue_count")
    saved_paths.append(
        save_joint_role_plot(
            dataset=dataset,
            score_name=score_name,
            plot_df=count_plot_df,
            y_label="Cue Count",
            output_stem=f"{dataset.lower()}_cue_count_vs_score",
            dataset_output_dir=dataset_output_dir,
            y_discrete=True,
        )
    )

    duration_plot_df = build_role_metric_plot_df(matched_df, "cue_duration")
    saved_paths.append(
        save_joint_role_plot(
            dataset=dataset,
            score_name=score_name,
            plot_df=duration_plot_df,
            y_label="Cue Duration (s)",
            output_stem=f"{dataset.lower()}_cue_duration_vs_score",
            dataset_output_dir=dataset_output_dir,
            y_discrete=False,
        )
    )

    return saved_paths


def export_dataset_summary(dataset: str, score_name: str) -> Optional[Path]:
    """Export one compact workbook per dataset."""
    print(f"\n{'=' * 70}")
    print(f"{dataset} summary export")
    print("=" * 70)

    samples = load_cue_samples(dataset)
    if not samples:
        print(f"[{dataset}] No cue files found under {AGENT_OUTPUTS_DIR / dataset}")
        return None

    scores = load_score_data(dataset)
    role_phrase_totals = {role: Counter() for role in ROLES}
    rows: List[Dict] = []

    for sample in samples:
        score = match_score(sample["sample_id"], sample["subject_id"], scores)
        row, role_phrase_counts = summarize_sample(
            dataset=dataset,
            score_name=score_name,
            sample_id=sample["sample_id"],
            subject_id=sample["subject_id"],
            relative_path=sample["relative_path"],
            cues=sample["cues"],
            score=score,
        )
        rows.append(row)
        for role in ROLES:
            role_phrase_totals[role].update(role_phrase_counts[role])

    df = pd.DataFrame(rows).sort_values(by=["subject_id", "sample_id", "relative_path"]).reset_index(drop=True)
    matched_df = df.dropna(subset=["score"]).copy()
    print(f"Loaded samples: {len(df)}")
    print(f"Matched scores: {len(matched_df)}/{len(df)}")

    top25_interviewer_df = build_top_phrase_table(dataset, "interviewer", role_phrase_totals["interviewer"])
    top25_interviewee_df = build_top_phrase_table(dataset, "interviewee", role_phrase_totals["interviewee"])

    score_count_df = matched_df[
        [
            "subject_id",
            "sample_id",
            "relative_path",
            "score",
            "interviewer_cue_count",
            "interviewee_cue_count",
            "total_cue_count",
            "count_gap_interviewer_minus_interviewee",
            "count_ratio_interviewer_to_interviewee",
        ]
    ].copy()

    score_duration_df = matched_df[
        [
            "subject_id",
            "sample_id",
            "relative_path",
            "score",
            "interviewer_cue_duration",
            "interviewee_cue_duration",
            "total_cue_duration",
            "duration_gap_interviewer_minus_interviewee",
            "duration_ratio_interviewer_to_interviewee",
        ]
    ].copy()

    distribution_df = matched_df[
        [
            "subject_id",
            "sample_id",
            "relative_path",
            "score",
            "interviewer_unique_phrase_count",
            "interviewee_unique_phrase_count",
            "interviewer_unique_category_count",
            "interviewee_unique_category_count",
            "interviewer_avg_cue_duration",
            "interviewee_avg_cue_duration",
            "interviewer_cue_count_share",
            "interviewee_cue_count_share",
            "interviewer_cue_duration_share",
            "interviewee_cue_duration_share",
        ]
    ].copy()

    dataset_output_dir = BASE_OUTPUT_DIR / dataset.lower()
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    sections = [
        ("Top25 Interviewer Phrases", top25_interviewer_df),
        ("Top25 Interviewee Phrases", top25_interviewee_df),
        ("Cue Count vs Score", score_count_df),
        ("Cue Duration vs Score", score_duration_df),
        ("Other Distribution Metrics", distribution_df),
    ]
    notes = [
        ("Top25 Interviewer Phrases", "Top-25 interviewer cue phrases with count and frequency_ratio."),
        ("Top25 Interviewee Phrases", "Top-25 interviewee cue phrases with count and frequency_ratio."),
        ("Cue Count vs Score", "Per sample cue count table for later plotting against the clinical score."),
        ("Cue Duration vs Score", "Per sample cue duration table for later plotting against the clinical score."),
        (
            "Other Distribution Metrics",
            "Compact distribution indicators: phrase diversity, category diversity, average cue duration, and role shares.",
        ),
        (
            "subject_id / sample_id",
            "subject_id is used for score matching when the dataset keeps multiple utterances per subject, such as CMDC and PDCH.",
        ),
    ]
    workbook_path = write_dataset_workbook(dataset, dataset_output_dir, sections, notes)
    print(f"Saved: {workbook_path}")
    plot_paths = export_dataset_plots(dataset, score_name, matched_df, dataset_output_dir)
    for plot_path in plot_paths:
        print(f"Saved: {plot_path}")
    return workbook_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Cue summary export")
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS.keys(), "all"],
        default="all",
        help="Dataset to analyze",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        for _, (dataset, score_name) in DATASETS.items():
            export_dataset_summary(dataset, score_name)
    else:
        dataset, score_name = DATASETS[args.dataset]
        export_dataset_summary(dataset, score_name)

    print(f"\nResults saved to: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
