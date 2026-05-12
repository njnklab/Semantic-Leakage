"""
DSM-5 symptom-domain alignment for detected cue texts.

This analysis checks whether each detected cue can be associated with one or
more of the nine DSM-5 major depressive episode symptom domains using a
transparent bilingual keyword matcher. It exports cue-level matches, dataset
summary tables, and a compact figure combining a dataset-by-domain heatmap with
the overall any-domain hit rate.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "dsm_cue_alignment"
ANNOTATION_DIR = PROJECT_ROOT / "CueAnnotatorSys" / "annotations"
AGENT_OUTPUTS_DIR = PROJECT_ROOT / "agent" / "outputs"
LOCAL_ARIAL_DIR = Path.home() / ".local" / "share" / "fonts" / "arial-ms"

DATASET_ORDER = ["E-DAIC", "PDCH", "CMDC", "ManDIC"]
FIG_FONT_SIZE = 18

if LOCAL_ARIAL_DIR.exists():
    for font_path in sorted(LOCAL_ARIAL_DIR.glob("*.TTF")) + sorted(LOCAL_ARIAL_DIR.glob("*.ttf")):
        font_manager.fontManager.addfont(str(font_path))

try:
    font_manager.findfont("Arial", fallback_to_default=False)
    FIG_FONT_FAMILY = "Arial"
except ValueError:
    FIG_FONT_FAMILY = "Liberation Sans"


@dataclass(frozen=True)
class DSMDomain:
    key: str
    label: str
    english_patterns: Sequence[str]
    chinese_keywords: Sequence[str]


DSM_DOMAINS: Sequence[DSMDomain] = (
    DSMDomain(
        key="depressed_mood",
        label="Depressed\nmood",
        english_patterns=(
            r"\bdepress(?:ed|ion)?\b",
            r"\bdepressing\b",
            r"\bsad(?:ness)?\b",
            r"\bdown\b",
            r"\bhopeless(?:ness)?\b",
            r"\bempty\b",
            r"\bnumb\b",
            r"\bcry(?:ing)?\b",
            r"\btears?\b",
            r"\bblue\b",
            r"\bgloomy\b",
            r"\bmelanchol(?:y|ic)\b",
            r"\bunhappy\b",
            r"\bmiserable\b",
            r"\bupset\b",
            r"\blow mood\b",
            r"\bfeel(?:ing)? low\b",
        ),
        chinese_keywords=(
            "抑郁",
            "难过",
            "伤心",
            "绝望",
            "低落",
            "沮丧",
            "悲观",
            "空虚",
            "麻木",
            "哭",
            "不开心",
            "压抑",
            "崩溃",
            "情绪不好",
            "悲伤",
            "心情不好",
            "心情非常不好",
            "情绪低",
            "情绪低落",
            "郁闷",
            "丧",
        ),
    ),
    DSMDomain(
        key="anhedonia",
        label="Anhedonia",
        english_patterns=(
            r"\banhedonia\b",
            r"\binterest\b",
            r"\binterested\b",
            r"\bpleasure\b",
            r"\benjoy(?:ment|ed|ing)?\b",
            r"\bhobb(?:y|ies)\b",
            r"\bfun\b",
            r"\bmotivation\b",
            r"\bmotivated\b",
            r"\bexcited\b",
            r"\benthusiasm\b",
            r"\bbored\b",
            r"\bboring\b",
            r"\bno interest\b",
            r"\bnot interested\b",
            r"\bno pleasure\b",
            r"\bloss of pleasure\b",
        ),
        chinese_keywords=(
            "没兴趣",
            "提不起兴趣",
            "兴趣",
            "无聊",
            "不喜欢",
            "没乐趣",
            "快感缺失",
            "乐趣",
            "爱好",
            "动力",
            "没动力",
            "享受",
            "高兴不起来",
            "不能够体验到快乐",
            "没有愉快感",
            "没有快乐",
            "缺乏快感",
            "缺乏乐趣",
        ),
    ),
    DSMDomain(
        key="appetite_weight",
        label="Appetite /\nweight",
        english_patterns=(
            r"\bappetite\b",
            r"\beat(?:ing)?\b",
            r"\bfood\b",
            r"\bhunger\b",
            r"\bhungry\b",
            r"\bweight\b",
            r"\bnausea\b",
            r"\bvomit(?:ing)?\b",
            r"\bovereat(?:ing)?\b",
            r"\bstarv(?:e|ing)\b",
        ),
        chinese_keywords=(
            "食欲",
            "吃不下",
            "暴食",
            "体重",
            "食物",
            "恶心",
            "呕吐",
            "饥饿",
            "没胃口",
            "胃口",
            "吃饭",
        ),
    ),
    DSMDomain(
        key="sleep",
        label="Sleep",
        english_patterns=(
            r"\binsomnia\b",
            r"\bsleep(?:ing)?\b",
            r"\basleep\b",
            r"\bawake\b",
            r"\bwake(?:up)?\b",
            r"\bwake up\b",
            r"\bnightmare\b",
            r"\bdream\b",
            r"\bnight\b",
            r"\bbed\b",
            r"\bsleepless\b",
            r"\boversleep(?:ing)?\b",
            r"\bnap\b",
        ),
        chinese_keywords=(
            "失眠",
            "睡不着",
            "早醒",
            "睡眠",
            "困",
            "嗜睡",
            "睡不好",
            "入睡",
            "做梦",
            "噩梦",
            "半夜",
            "睡觉",
            "睡",
            "醒",
        ),
    ),
    DSMDomain(
        key="psychomotor",
        label="Psychomotor\nchange",
        english_patterns=(
            r"\bpsychomotor\b",
            r"\brestless(?:ness)?\b",
            r"\bagitat(?:ed|ion)\b",
            r"\bfidget(?:ing)?\b",
            r"\bpacing\b",
            r"\bslowed?\b",
            r"\bslowly\b",
            r"\bmove(?:ment|ing)? slowly\b",
            r"\bretardation\b",
            r"\bcan(?:not|'t) sit\b",
            r"\bsit still\b",
        ),
        chinese_keywords=(
            "精神运动",
            "迟缓",
            "动作慢",
            "行动慢",
            "反应慢",
            "语速慢",
            "坐立不安",
            "手脚不停",
            "活动减少",
            "动不了",
            "激越",
        ),
    ),
    DSMDomain(
        key="fatigue",
        label="Fatigue /\nenergy loss",
        english_patterns=(
            r"\btired\b",
            r"\bexhaust(?:ed|ion)?\b",
            r"\bfatigue\b",
            r"\bweak(?:ness)?\b",
            r"\benergy\b",
            r"\bletharg(?:ic|y)\b",
            r"\bdrained\b",
            r"\bburnout\b",
            r"\bweary\b",
            r"\bno strength\b",
        ),
        chinese_keywords=(
            "疲劳",
            "疲倦",
            "乏力",
            "没劲",
            "倦怠",
            "虚弱",
            "精力不足",
            "累",
            "疲惫",
            "无力",
            "四肢乏力",
            "没力气",
            "缺乏活力",
            "无精打采",
            "精神差",
            "疲乏",
        ),
    ),
    DSMDomain(
        key="guilt_worthlessness",
        label="Worthlessness /\nguilt",
        english_patterns=(
            r"\bguilt(?:y)?\b",
            r"\bshame\b",
            r"\bashamed\b",
            r"\bworthless(?:ness)?\b",
            r"\bfailure\b",
            r"\bfail(?:ed|ing)?\b",
            r"\bblame\b",
            r"\bregret\b",
            r"\bregrets\b",
            r"\bregretted\b",
            r"\bburden\b",
            r"\buseless\b",
            r"\blow self[- ]esteem\b",
            r"\bself[- ]esteem\b",
        ),
        chinese_keywords=(
            "自责",
            "内疚",
            "无用",
            "废物",
            "失败",
            "后悔",
            "拖累",
            "愧疚",
            "负罪",
            "没价值",
            "存在价值",
            "让家人失望",
            "低自尊",
            "干不好",
            "懊悔",
            "对不起",
            "自卑",
            "对自己失望",
        ),
    ),
    DSMDomain(
        key="concentration",
        label="Concentration /\ndecision",
        english_patterns=(
            r"\bconcentrat(?:e|ion|ing)\b",
            r"\bfocus(?:ing|ed)?\b",
            r"\battention\b",
            r"\bforget(?:ful|ting)?\b",
            r"\bconfused\b",
            r"\bthinking\b",
            r"\bmemory\b",
            r"\bfoggy\b",
            r"\bdistract(?:ed|ion)?\b",
            r"\bdecision\b",
            r"\bdecide\b",
            r"\bindecis(?:ive|ion)\b",
        ),
        chinese_keywords=(
            "注意力",
            "不集中",
            "走神",
            "忘记",
            "混乱",
            "思维",
            "记忆",
            "迟钝",
            "糊涂",
            "决定",
            "选择困难",
            "犹豫",
        ),
    ),
    DSMDomain(
        key="suicidality",
        label="Death /\nsuicidality",
        english_patterns=(
            r"\bsuicid(?:e|al)\b",
            r"\bdie\b",
            r"\bdied\b",
            r"\bdeath\b",
            r"\bdead\b",
            r"\bkill\b",
            r"\bhurt\b",
            r"\bharm\b",
            r"\bself[- ]harm\b",
            r"\bend my life\b",
            r"\bnot worth living\b",
            r"\bmeaningless\b",
            r"\balive\b",
        ),
        chinese_keywords=(
            "自杀",
            "想死",
            "轻生",
            "不想活",
            "结束生命",
            "死亡",
            "伤害自己",
            "自残",
            "了结",
            "没有意义",
            "活着没有意思",
            "活着没有意义",
            "活着没意思",
            "活着没意义",
            "生不如死",
            "濒死感",
            "割腕",
        ),
    ),
)


def normalize_dataset_name(name: str) -> str:
    dataset = str(name).strip()
    aliases = {
        "edaic": "E-DAIC",
        "e-daic": "E-DAIC",
        "pdch": "PDCH",
        "cmdc": "CMDC",
        "mandic": "ManDIC",
    }
    return aliases.get(dataset.lower(), dataset)


def normalize_role(cue: Dict) -> str:
    role = str(cue.get("speaker_role") or cue.get("speaker") or "").strip().lower()
    if any(tag in role for tag in ("interviewer", "doctor", "clinician", "therapist")):
        return "interviewer"
    if any(tag in role for tag in ("interviewee", "participant", "patient", "subject")):
        return "patient"
    return "unknown"


def cue_duration(cue: Dict) -> float:
    span = cue.get("corrected_span") or cue.get("original_span") or {}
    start = cue.get("start", span.get("start"))
    end = cue.get("end", span.get("end"))
    try:
        return max(0.0, float(end) - float(start))
    except (TypeError, ValueError):
        return 0.0


def iter_annotation_records() -> Iterable[Dict]:
    for path in sorted(ANNOTATION_DIR.glob("annotation_*.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        dataset = normalize_dataset_name(data.get("dataset") or path.stem.split("_")[1])
        sample_id = str(data.get("sample_id") or path.stem.replace(f"annotation_{dataset}_", "")).strip()
        for cue in data.get("cues", []):
            if not isinstance(cue, dict) or str(cue.get("status", "")).lower() == "deleted":
                continue
            yield {
                "source": "annotations",
                "dataset": dataset,
                "sample_id": sample_id,
                "role": "unknown",
                "text": str(cue.get("text", "")).strip(),
                "duration": cue_duration(cue),
            }


def iter_agent_records(role_filter: str) -> Iterable[Dict]:
    cue_files = sorted(
        set(AGENT_OUTPUTS_DIR.rglob("*_cues.json")) | set(AGENT_OUTPUTS_DIR.rglob("cues.json"))
    )
    for path in cue_files:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        dataset = normalize_dataset_name(data.get("dataset") or path.relative_to(AGENT_OUTPUTS_DIR).parts[0])
        sample_id = str(data.get("sample_id") or path.stem.replace("_cues", "")).strip()
        for cue in data.get("cues", []):
            if not isinstance(cue, dict):
                continue
            role = normalize_role(cue)
            if role_filter != "all" and role != role_filter:
                continue
            yield {
                "source": "agent",
                "dataset": dataset,
                "sample_id": sample_id,
                "role": role,
                "text": str(cue.get("text", "")).strip(),
                "duration": cue_duration(cue),
            }


def match_domains(text: str) -> List[str]:
    normalized = " ".join(str(text).lower().split())
    matched: List[str] = []
    if not normalized:
        return matched

    for domain in DSM_DOMAINS:
        is_match = any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in domain.english_patterns)
        if not is_match:
            is_match = any(keyword in text for keyword in domain.chinese_keywords)
        if is_match:
            matched.append(domain.key)

    return matched


def build_cue_table(source: str, role_filter: str) -> pd.DataFrame:
    if source == "annotations":
        records = list(iter_annotation_records())
    elif source == "agent":
        records = list(iter_agent_records(role_filter=role_filter))
    else:
        raise ValueError(f"Unsupported source: {source}")

    rows = []
    for idx, record in enumerate(records):
        domains = match_domains(record["text"])
        rows.append(
            {
                "cue_index": idx,
                **record,
                "matched": bool(domains),
                "matched_domains": ";".join(domains),
                "match_count": len(domains),
            }
        )
    return pd.DataFrame(rows)


def summarize_matches(cue_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows = []
    summary_rows = []

    for dataset in DATASET_ORDER:
        dataset_df = cue_df[cue_df["dataset"] == dataset].copy()
        total = int(len(dataset_df))
        matched_total = int(dataset_df["matched"].sum()) if total else 0
        summary = {
            "dataset": dataset,
            "total_cues": total,
            "matched_cues": matched_total,
            "hit_rate": matched_total / total if total else np.nan,
        }

        domain_counter: Counter[str] = Counter()
        for domains in dataset_df["matched_domains"].fillna(""):
            for domain in str(domains).split(";"):
                if domain:
                    domain_counter[domain] += 1

        for domain in DSM_DOMAINS:
            count = int(domain_counter[domain.key])
            share = count / total if total else np.nan
            summary[f"{domain.key}_count"] = count
            summary[f"{domain.key}_share"] = share
            long_rows.append(
                {
                    "dataset": dataset,
                    "domain": domain.key,
                    "domain_label": domain.label.replace("\n", " "),
                    "matched_cues": count,
                    "total_cues": total,
                    "share_of_all_cues": share,
                }
            )

        summary_rows.append(summary)

    return pd.DataFrame(summary_rows), pd.DataFrame(long_rows)


def write_unmatched_examples(cue_df: pd.DataFrame, output_dir: Path, per_dataset: int = 30) -> Path:
    rows = []
    for dataset in DATASET_ORDER:
        unmatched = cue_df[(cue_df["dataset"] == dataset) & (~cue_df["matched"])].copy()
        counts = unmatched["text"].value_counts().head(per_dataset)
        for text, count in counts.items():
            rows.append({"dataset": dataset, "cue_text": text, "count": int(count)})
    output_path = output_dir / "dsm_unmatched_examples.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def build_fractional_domain_distribution(cue_df: pd.DataFrame) -> pd.DataFrame:
    """Allocate each cue once across matched DSM domains, with unmatched as gray remainder."""
    rows = []
    domain_keys = [domain.key for domain in DSM_DOMAINS]
    domain_labels = {domain.key: domain.label.replace("\n", " ") for domain in DSM_DOMAINS}

    for dataset in DATASET_ORDER:
        dataset_df = cue_df[cue_df["dataset"] == dataset].copy()
        total = len(dataset_df)
        weights = Counter({domain: 0.0 for domain in domain_keys})
        unmatched = 0.0

        for domains_text in dataset_df["matched_domains"].fillna(""):
            domains = [domain for domain in str(domains_text).split(";") if domain]
            if not domains:
                unmatched += 1.0
                continue
            share = 1.0 / len(domains)
            for domain in domains:
                weights[domain] += share

        for domain in domain_keys:
            rows.append(
                {
                    "dataset": dataset,
                    "domain": domain,
                    "domain_label": domain_labels[domain],
                    "percentage": (weights[domain] / total * 100.0) if total else np.nan,
                }
            )
        rows.append(
            {
                "dataset": dataset,
                "domain": "unmatched",
                "domain_label": "No DSM-5 match",
                "percentage": (unmatched / total * 100.0) if total else np.nan,
            }
        )

    return pd.DataFrame(rows)


def plot_alignment(summary_df: pd.DataFrame, cue_df: pd.DataFrame, output_dir: Path, source: str) -> Path:
    sns.set_theme(style="white")
    plt.rcParams.update(
        {
            "font.family": FIG_FONT_FAMILY,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
        }
    )

    plot_df = build_fractional_domain_distribution(cue_df)
    domain_order = [domain.key for domain in DSM_DOMAINS] + ["unmatched"]
    label_lookup = {domain.key: domain.label.replace("\n", " ") for domain in DSM_DOMAINS}
    label_lookup["unmatched"] = "No DSM-5 match"
    colors = {
        "depressed_mood": "#4C78A8",
        "anhedonia": "#F58518",
        "appetite_weight": "#E45756",
        "sleep": "#72B7B2",
        "psychomotor": "#54A24B",
        "fatigue": "#EECA3B",
        "guilt_worthlessness": "#B279A2",
        "concentration": "#FF9DA6",
        "suicidality": "#9D755D",
        "unmatched": "#D8D8D8",
    }

    summary_index = summary_df.set_index("dataset").reindex(DATASET_ORDER)
    hit_rates = summary_index["hit_rate"] * 100.0

    fig, ax = plt.subplots(figsize=(10, 5))
    y_positions = np.arange(len(DATASET_ORDER))
    left = np.zeros(len(DATASET_ORDER), dtype=float)

    for domain in domain_order:
        values = (
            plot_df[plot_df["domain"] == domain]
            .set_index("dataset")
            .reindex(DATASET_ORDER)["percentage"]
            .fillna(0.0)
            .to_numpy()
        )
        ax.barh(
            y_positions,
            values,
            left=left,
            height=0.68,
            color=colors[domain],
            edgecolor="white",
            linewidth=0.8,
            label=label_lookup[domain],
        )
        left += values

    yticklabels = DATASET_ORDER
    ax.set_yticks(y_positions)
    ax.set_yticklabels(yticklabels)
    ax.invert_yaxis()
    ax.set_ylim(len(DATASET_ORDER) - 0.36, -0.64)
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("Cue proportion (%)")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=FIG_FONT_SIZE)
    ax.xaxis.label.set_size(FIG_FONT_SIZE)
    ax.xaxis.label.set_weight("bold")
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname(FIG_FONT_FAMILY)
        tick.set_fontweight("bold")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    for idx, dataset in enumerate(DATASET_ORDER):
        hit_rate = hit_rates.loc[dataset]
        if pd.isna(hit_rate):
            continue
        ax.text(
            min(float(hit_rate) + 1.0, 94.0),
            idx,
            f"{hit_rate:.1f}%",
            va="center",
            ha="left",
            fontsize=FIG_FONT_SIZE,
            color="#222222",
            weight="bold",
            fontname=FIG_FONT_FAMILY,
        )

    legend_handles = [Patch(facecolor=colors[domain], label=label_lookup[domain]) for domain in domain_order]
    ax.legend(
        handles=legend_handles,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.04, 0.5),
        frameon=False,
        prop={"family": FIG_FONT_FAMILY, "weight": "bold", "size": FIG_FONT_SIZE},
        columnspacing=1.0,
        handlelength=1.4,
        handleheight=1.1,
        labelspacing=0.55,
    )
    fig.subplots_adjust(left=0.12, right=0.60, top=0.99, bottom=0.18)

    output_path = output_dir / "dsm_cue_alignment.png"
    fig.savefig(output_path, dpi=300)
    jpg_output_path = output_dir / "dsm_cue_alignment_compact.jpg"
    fig.savefig(jpg_output_path, dpi=300, format="jpg", pil_kwargs={"quality": 95, "optimize": True})
    plt.close(fig)
    return jpg_output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Align cue texts to DSM-5 symptom-domain keywords.")
    parser.add_argument(
        "--source",
        choices=["annotations", "agent"],
        default="agent",
        help="Use manually reviewed annotation JSONs or raw Agent cue outputs.",
    )
    parser.add_argument(
        "--role",
        choices=["all", "patient", "interviewer"],
        default="all",
        help="Role filter for --source agent. Annotation JSONs do not contain role labels.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cue_df = build_cue_table(source=args.source, role_filter=args.role)
    summary_df, long_df = summarize_matches(cue_df)
    plot_distribution_df = build_fractional_domain_distribution(cue_df)

    suffix = args.source if args.source == "annotations" else f"{args.source}_{args.role}"
    cue_path = output_dir / f"dsm_cue_matches_{suffix}.csv"
    summary_path = output_dir / f"dsm_cue_alignment_summary_{suffix}.csv"
    long_path = output_dir / f"dsm_cue_alignment_long_{suffix}.csv"
    plot_distribution_path = output_dir / f"dsm_cue_alignment_plot_distribution_{suffix}.csv"

    cue_df.to_csv(cue_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    long_df.to_csv(long_path, index=False)
    plot_distribution_df.to_csv(plot_distribution_path, index=False)
    unmatched_path = write_unmatched_examples(cue_df, output_dir)
    figure_path = plot_alignment(summary_df, cue_df, output_dir, source=suffix)

    display_df = summary_df[["dataset", "total_cues", "matched_cues", "hit_rate"]].copy()
    display_df["hit_rate"] = display_df["hit_rate"].map(lambda value: f"{value:.3f}" if pd.notna(value) else "nan")
    print(display_df.to_string(index=False))
    print(f"\nSaved cue-level matches to {cue_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved long domain table to {long_path}")
    print(f"Saved plot distribution to {plot_distribution_path}")
    print(f"Saved unmatched examples to {unmatched_path}")
    print(f"Saved figure to {figure_path}")


if __name__ == "__main__":
    main()
