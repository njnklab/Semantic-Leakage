"""
Emotion classification audit for manually verified cue spans.

This script reads the unified agent outputs:
    agent/outputs/<DATASET>/<SAMPLE_ID>/transcript.json
    agent/outputs/<DATASET>/<SAMPLE_ID>/cues.json

It uses human-reviewed cue annotations when available and falls back to
agent-generated cues otherwise. Non-cue texts are sampled from participant
sentences that do not overlap any cue span.

Usage:
    python emotion_classification.py -d edaic
    python emotion_classification.py -d all --output results.md
    python emotion_classification.py -d edaic --max-samples 10
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Bypass legacy local model loading checks.
os.environ["SAFETENSORS_FAST_GPU"] = "1"
import transformers.modeling_utils as _mu
import transformers.utils.import_utils as _iu

_mu.check_torch_load_is_safe = lambda *a, **k: None
_iu.check_torch_load_is_safe = lambda *a, **k: None

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODELS_DIR = BASE_DIR / "models"
AGENT_OUTPUTS_DIR = PROJECT_ROOT / "agent" / "outputs"
ANNOTATIONS_DIR = PROJECT_ROOT / "CueAnnotatorSys" / "annotations"

from CueFilter.Baseline.audio_views import iter_output_samples  # noqa: E402

PARTICIPANT_ROLES = {"interviewee", "participant", "patient"}
NEGATIVE_LABELS = {
    "negative",
    "neg",
    "1 star",
    "2 stars",
    "1",
    "2",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "grief",
    "disappointment",
    "remorse",
    "nervousness",
    "annoyance",
    "负面",
    "消极",
    "负面情感",
    "负面情绪",
    "悲伤",
    "愤怒",
    "恐惧",
    "厌恶",
    "焦虑",
    "抑郁",
    "难过",
    "痛苦",
    "失望",
    "沮丧",
    "烦躁",
    "担心",
}

DATASET_CONFIG = {
    "edaic": {
        "name": "E-DAIC",
        "language": "en",
        "outputs_dir": AGENT_OUTPUTS_DIR / "E-DAIC",
        "annotation_prefix": "E-DAIC",
    },
    "pdch": {
        "name": "PDCH",
        "language": "zh",
        "outputs_dir": AGENT_OUTPUTS_DIR / "PDCH",
        "annotation_prefix": "PDCH",
    },
    "cmdc": {
        "name": "CMDC",
        "language": "zh",
        "outputs_dir": AGENT_OUTPUTS_DIR / "CMDC",
        "annotation_prefix": "CMDC",
    },
    "mandic": {
        "name": "ManDIC",
        "language": "zh",
        "outputs_dir": AGENT_OUTPUTS_DIR / "ManDIC",
        "annotation_prefix": "ManDIC",
    },
}

MODEL_CONFIG = {
    "english": [
        ("NLPTown BERT", MODELS_DIR / "nlptown-bert-base-multilingual-uncased-sentiment"),
        ("Lxyuan DistilBERT", MODELS_DIR / "lxyuan-distilbert-base-multilingual-cased-sentiments-student"),
    ],
    "chinese": [
        ("Lxyuan DistilBERT", MODELS_DIR / "lxyuan-distilbert-base-multilingual-cased-sentiments-student"),
        ("DavidLanz Chinese", MODELS_DIR / "DavidLanz-fine_tune_chinese_sentiment"),
        ("Xuyuan BERT", MODELS_DIR / "touch20032003-xuyuan-trial-sentiment-bert-chinese"),
    ],
}


@dataclass
class CueSpan:
    text: str
    start: float
    end: float


@dataclass
class SampleTexts:
    sample_id: str
    cue_texts: List[str]
    non_cue_texts: List[str]
    cue_source: str


def load_json(path: Path) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def clean_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def normalize_role(role: Optional[str]) -> str:
    return str(role or "").strip().lower()


def is_participant(role: Optional[str]) -> bool:
    role_norm = normalize_role(role)
    return any(token in role_norm for token in PARTICIPANT_ROLES)


def overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def annotation_path(dataset_key: str, sample_id: str) -> Path:
    prefix = DATASET_CONFIG[dataset_key]["annotation_prefix"]
    return ANNOTATIONS_DIR / f"annotation_{prefix}_{sample_id}.json"


def cue_is_participant(cue_start: float, cue_end: float, sentences: Sequence[Dict]) -> bool:
    best_overlap = 0.0
    best_role = ""
    for sentence in sentences:
        sent_start = float(sentence.get("start", 0.0))
        sent_end = float(sentence.get("end", 0.0))
        overlap = max(0.0, min(cue_end, sent_end) - max(cue_start, sent_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_role = sentence.get("speaker", "")
    return best_overlap > 0 and is_participant(best_role)


def load_verified_cues(dataset_key: str, sample_id: str, transcript: Dict, cues_path: Path) -> Tuple[List[CueSpan], str]:
    ann_path = annotation_path(dataset_key, sample_id)
    sentences = transcript.get("sentences", [])

    if ann_path.exists():
        ann = load_json(ann_path)
        spans: List[CueSpan] = []
        for cue in ann.get("cues", []):
            if cue.get("status") == "deleted":
                continue
            span = cue.get("corrected_span") or cue.get("original_span") or {}
            start = span.get("start")
            end = span.get("end")
            text = clean_text(cue.get("text", ""))
            if start is None or end is None or not text:
                continue
            start = float(start)
            end = float(end)
            if end <= start:
                continue
            if sentences and not cue_is_participant(start, end, sentences):
                continue
            spans.append(CueSpan(text=text, start=start, end=end))
        return spans, "human"

    if cues_path.exists():
        cue_data = load_json(cues_path)
        spans = []
        for cue in cue_data.get("cues", []):
            if not is_participant(cue.get("speaker_role") or cue.get("speaker")):
                continue
            text = clean_text(cue.get("text", ""))
            start = cue.get("start")
            end = cue.get("end")
            if start is None or end is None or not text:
                continue
            start = float(start)
            end = float(end)
            if end <= start:
                continue
            spans.append(CueSpan(text=text, start=start, end=end))
        return spans, "agent"

    return [], "missing"


def sample_non_cue_texts(
    transcript: Dict,
    cue_spans: Sequence[CueSpan],
    max_non_cue_per_sample: int,
    rng: random.Random,
) -> List[str]:
    collected: List[str] = []
    seen = set()

    for sentence in transcript.get("sentences", []):
        if not is_participant(sentence.get("speaker")):
            continue

        text = clean_text(sentence.get("text", ""))
        if len(text) < 2:
            continue

        sent_start = float(sentence.get("start", 0.0))
        sent_end = float(sentence.get("end", 0.0))
        if any(overlaps(sent_start, sent_end, cue.start, cue.end) for cue in cue_spans):
            continue

        if text not in seen:
            seen.add(text)
            collected.append(text)

    if len(collected) > max_non_cue_per_sample:
        collected = rng.sample(collected, max_non_cue_per_sample)

    return collected


def build_samples(
    dataset_key: str,
    max_non_cue_per_sample: int,
    max_samples: Optional[int],
    seed: int,
) -> List[SampleTexts]:
    rng = random.Random(seed)
    samples: List[SampleTexts] = []

    for sample in iter_output_samples(dataset_key):
        transcript_path = Path(sample["transcript_path"])
        cues_path = Path(sample["cues_path"])
        transcript = load_json(transcript_path)
        sample_id = str(sample["sample_id"])
        transcript["sample_id"] = sample_id

        cue_spans, cue_source = load_verified_cues(dataset_key, sample_id, transcript, cues_path)
        cue_texts = [clean_text(cue.text) for cue in cue_spans if len(clean_text(cue.text)) >= 2]
        if not cue_texts:
            continue

        non_cue_texts = sample_non_cue_texts(
            transcript=transcript,
            cue_spans=cue_spans,
            max_non_cue_per_sample=max_non_cue_per_sample,
            rng=rng,
        )
        if not non_cue_texts:
            continue

        samples.append(
            SampleTexts(
                sample_id=sample_id,
                cue_texts=cue_texts,
                non_cue_texts=non_cue_texts,
                cue_source=cue_source,
            )
        )

        if max_samples is not None and len(samples) >= max_samples:
            break

    return samples


def is_negative(label: str) -> bool:
    label_norm = str(label).strip().lower()
    return any(neg in label_norm for neg in NEGATIVE_LABELS)


def score_output(output) -> float:
    if isinstance(output, list) and output and isinstance(output[0], list):
        output = output[0]
    if not isinstance(output, list):
        return 0.0
    return float(sum(float(item.get("score", 0.0)) for item in output if is_negative(item.get("label", ""))))


def classify_texts(texts: Sequence[str], pipe, batch_size: int) -> List[float]:
    if not texts:
        return []
    outputs = pipe(list(texts), batch_size=batch_size, truncation=True, top_k=None)
    return [score_output(output) for output in outputs]


def aggregate_by_sample(scores: Sequence[float], owners: Sequence[int], sample_count: int) -> List[float]:
    buckets: Dict[int, List[float]] = defaultdict(list)
    for score, owner in zip(scores, owners):
        buckets[owner].append(score)
    means = []
    for idx in range(sample_count):
        if buckets[idx]:
            means.append(float(np.mean(buckets[idx])))
    return means


def load_models(language_group: str):
    device = 0 if torch.cuda.is_available() else -1
    pipelines = []
    for display_name, model_path in MODEL_CONFIG[language_group]:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        clf = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device,
            top_k=None,
        )
        pipelines.append((display_name, clf))
    return pipelines


def evaluate_dataset(dataset_key: str, samples: Sequence[SampleTexts], batch_size: int) -> List[Dict]:
    if not samples:
        return []

    language_group = "english" if DATASET_CONFIG[dataset_key]["language"].startswith("en") else "chinese"
    records = []

    cue_texts: List[str] = []
    cue_owners: List[int] = []
    non_cue_texts: List[str] = []
    non_cue_owners: List[int] = []

    for idx, sample in enumerate(samples):
        cue_texts.extend(sample.cue_texts)
        cue_owners.extend([idx] * len(sample.cue_texts))
        non_cue_texts.extend(sample.non_cue_texts)
        non_cue_owners.extend([idx] * len(sample.non_cue_texts))

    for model_name, clf in load_models(language_group):
        cue_scores = classify_texts(cue_texts, clf, batch_size=batch_size)
        non_cue_scores = classify_texts(non_cue_texts, clf, batch_size=batch_size)

        cue_means = aggregate_by_sample(cue_scores, cue_owners, len(samples))
        non_cue_means = aggregate_by_sample(non_cue_scores, non_cue_owners, len(samples))

        cue_mean = float(np.mean(cue_means)) if cue_means else 0.0
        non_cue_mean = float(np.mean(non_cue_means)) if non_cue_means else 0.0

        records.append(
            {
                "dataset": DATASET_CONFIG[dataset_key]["name"],
                "language": DATASET_CONFIG[dataset_key]["language"],
                "model": model_name,
                "cue_conf": cue_mean,
                "non_cue_conf": non_cue_mean,
                "delta": cue_mean - non_cue_mean,
                "samples": len(samples),
                "cue_texts": len(cue_texts),
                "non_cue_texts": len(non_cue_texts),
            }
        )

    return records


def format_markdown_table(records: Sequence[Dict]) -> str:
    lines = [
        "| Dataset | Language | Sentiment model | Cue negative score | Non-cue negative score | Difference |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            "| {dataset} | {language} | {model} | {cue_conf:.3f} | {non_cue_conf:.3f} | {delta:+.3f} |".format(
                **row
            )
        )
    return "\n".join(lines)


def dataset_summary(samples: Sequence[SampleTexts]) -> str:
    human = sum(1 for sample in samples if sample.cue_source == "human")
    agent = sum(1 for sample in samples if sample.cue_source == "agent")
    return f"samples={len(samples)}, human={human}, agent={agent}"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cue vs non-cue sentiment audit")
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        choices=["edaic", "pdch", "cmdc", "mandic", "all"],
        help="Dataset to evaluate",
    )
    parser.add_argument(
        "--max-non-cue-per-sample",
        type=int,
        default=30,
        help="Maximum number of non-cue participant sentences sampled per recording",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for non-cue sampling",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the markdown table",
    )
    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    dataset_keys = list(DATASET_CONFIG.keys()) if args.dataset == "all" else [args.dataset]
    all_records: List[Dict] = []

    for dataset_key in dataset_keys:
        samples = build_samples(
            dataset_key=dataset_key,
            max_non_cue_per_sample=args.max_non_cue_per_sample,
            max_samples=args.max_samples,
            seed=args.seed,
        )
        print(f"[{DATASET_CONFIG[dataset_key]['name']}] {dataset_summary(samples)}")
        if not samples:
            continue

        dataset_records = evaluate_dataset(
            dataset_key=dataset_key,
            samples=samples,
            batch_size=args.batch_size,
        )
        all_records.extend(dataset_records)

    table = format_markdown_table(all_records)
    print()
    print(table)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() == ".csv":
            import pandas as pd

            pd.DataFrame(all_records).rename(
                columns={
                    "dataset": "Dataset",
                    "language": "Language",
                    "model": "Sentiment model",
                    "cue_conf": "Cue negative score",
                    "non_cue_conf": "Non-cue negative score",
                    "delta": "Difference",
                    "samples": "Samples",
                    "cue_texts": "Cue texts",
                    "non_cue_texts": "Non-cue texts",
                }
            ).to_csv(args.output, index=False)
        else:
            args.output.write_text(table + "\n", encoding="utf-8")
        print(f"\nSaved table to {args.output}")


if __name__ == "__main__":
    main()
