import json
import random
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "agent" / "outputs"
ANNOTATION_ROOT = PROJECT_ROOT / "CueAnnotatorSys" / "annotations"

PARTICIPANT_ROLES = {"interviewee", "participant", "patient"}
INTERVIEWER_ROLES = {"interviewer", "doctor", "clinician", "therapist", "ellie"}
VALID_CUE_ROLES = {"patient", "doctor", "all"}
VALID_SPEECH_SCOPES = {"participant", "interviewer", "dialogue"}
VALID_VARIANTS = {"pre", "cue-only", "cue-excluded", "cue-removed", "random", "random-removed"}
VARIANT_ALIASES = {
    "cue-removed": "cue-excluded",
    "random-removed": "random",
}


@dataclass
class DatasetConfig:
    key: str
    display_name: str
    output_dir: Path
    annotation_prefix: str
    label_path: Path


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "edaic": DatasetConfig(
        key="edaic",
        display_name="E-DAIC",
        output_dir=OUTPUT_ROOT / "E-DAIC",
        annotation_prefix="E-DAIC",
        label_path=DATA_ROOT / "E-DAIC" / "Detailed_PHQ8_Labels.csv",
    ),
    "pdch": DatasetConfig(
        key="pdch",
        display_name="PDCH",
        output_dir=OUTPUT_ROOT / "PDCH",
        annotation_prefix="PDCH",
        label_path=DATA_ROOT / "PDCH" / "HAMD_annotation_en.csv",
    ),
    "cmdc": DatasetConfig(
        key="cmdc",
        display_name="CMDC",
        output_dir=OUTPUT_ROOT / "CMDC",
        annotation_prefix="CMDC",
        label_path=DATA_ROOT / "CMDC_EULA" / "SubjectInfo.csv",
    ),
    "mandic": DatasetConfig(
        key="mandic",
        display_name="ManDIC",
        output_dir=OUTPUT_ROOT / "ManDIC",
        annotation_prefix="ManDIC",
        label_path=DATA_ROOT / "ManDIC" / "info.csv",
    ),
}


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def paired_cues_path(transcript_path: Path) -> Path:
    if transcript_path.name == "transcript.json":
        return transcript_path.with_name("cues.json")
    if transcript_path.name.endswith("_transcript.json"):
        return transcript_path.with_name(transcript_path.name.replace("_transcript.json", "_cues.json"))
    raise ValueError(f"Unsupported transcript filename: {transcript_path.name}")


def infer_sample_id_from_output_path(dataset_key: str, transcript_path: Path) -> str:
    output_dir = DATASET_CONFIGS[dataset_key].output_dir
    rel_path = transcript_path.relative_to(output_dir)

    if dataset_key == "cmdc":
        if len(rel_path.parts) < 3:
            raise ValueError(f"Unrecognized CMDC output path: {transcript_path}")
        part, subject = rel_path.parts[0], rel_path.parts[1]
        question = transcript_path.name.replace("_transcript.json", "").replace(".json", "")
        return f"{part}{subject}{question}"

    if dataset_key == "pdch":
        if len(rel_path.parts) < 2:
            raise ValueError(f"Unrecognized PDCH output path: {transcript_path}")
        serial = rel_path.parts[0]
        session = transcript_path.name.replace("_transcript.json", "").replace(".json", "")
        return f"{serial}_{session}"

    if len(rel_path.parts) >= 2 and transcript_path.name == "transcript.json":
        return rel_path.parts[0]

    stem = transcript_path.name
    if stem.endswith("_transcript.json"):
        return stem[: -len("_transcript.json")]
    if stem.endswith(".json"):
        return stem[:-len(".json")]
    return stem


def iter_output_samples(dataset_key: str) -> List[Dict[str, object]]:
    output_dir = DATASET_CONFIGS[dataset_key].output_dir
    if not output_dir.exists():
        return []

    transcript_paths = sorted(set(output_dir.rglob("transcript.json")) | set(output_dir.rglob("*_transcript.json")))
    samples: List[Dict[str, object]] = []
    for transcript_path in transcript_paths:
        cues_path = paired_cues_path(transcript_path)
        if not cues_path.exists():
            continue
        samples.append(
            {
                "sample_dir": transcript_path.parent,
                "transcript_path": transcript_path,
                "cues_path": cues_path,
                "sample_id": infer_sample_id_from_output_path(dataset_key, transcript_path),
            }
        )
    return samples


def normalize_role(role: Optional[str]) -> str:
    role_norm = str(role or "").strip().lower()
    if any(token in role_norm for token in PARTICIPANT_ROLES):
        return "patient"
    if any(token in role_norm for token in INTERVIEWER_ROLES):
        return "doctor"
    return "unknown"


def is_participant(role: Optional[str]) -> bool:
    return normalize_role(role) == "patient"


def is_interviewer(role: Optional[str]) -> bool:
    return normalize_role(role) == "doctor"


def role_matches_filter(role: Optional[str], cue_role: str) -> bool:
    if cue_role not in VALID_CUE_ROLES:
        raise ValueError(f"Unsupported cue role filter: {cue_role}")
    if cue_role == "all":
        return normalize_role(role) in {"patient", "doctor"}
    return normalize_role(role) == cue_role


def merge_intervals(intervals: Sequence[Tuple[float, float]], gap: float = 0.0) -> List[Tuple[float, float]]:
    cleaned = sorted((float(s), float(e)) for s, e in intervals if e > s)
    if not cleaned:
        return []

    merged: List[Tuple[float, float]] = []
    cur_start, cur_end = cleaned[0]
    for start, end in cleaned[1:]:
        if start <= cur_end + gap:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    return merged


def subtract_intervals(
    base_intervals: Sequence[Tuple[float, float]],
    remove_intervals: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    result = list(base_intervals)
    for rem_start, rem_end in merge_intervals(remove_intervals):
        updated: List[Tuple[float, float]] = []
        for start, end in result:
            if rem_end <= start or rem_start >= end:
                updated.append((start, end))
                continue
            if rem_start > start:
                updated.append((start, rem_start))
            if rem_end < end:
                updated.append((rem_end, end))
        result = updated
    return merge_intervals(result)


def intersect_intervals(
    base_intervals: Sequence[Tuple[float, float]],
    target_intervals: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    intersections: List[Tuple[float, float]] = []
    merged_base = merge_intervals(base_intervals)
    merged_target = merge_intervals(target_intervals)

    for base_start, base_end in merged_base:
        for target_start, target_end in merged_target:
            overlap_start = max(base_start, target_start)
            overlap_end = min(base_end, target_end)
            if overlap_end > overlap_start:
                intersections.append((overlap_start, overlap_end))

    return merge_intervals(intersections)


def map_intervals_to_concatenated_timeline(
    base_intervals: Sequence[Tuple[float, float]],
    target_intervals: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """
    Map target intervals onto the timeline formed by concatenating base intervals.

    Example:
        base_intervals = [(10, 20), (30, 40)]
        target_intervals = [(12, 14), (33, 35)]
        -> [(2, 4), (13, 15)]
    """
    mapped: List[Tuple[float, float]] = []
    merged_base = merge_intervals(base_intervals)
    merged_target = merge_intervals(target_intervals)
    concat_cursor = 0.0

    for base_start, base_end in merged_base:
        for target_start, target_end in merged_target:
            overlap_start = max(base_start, target_start)
            overlap_end = min(base_end, target_end)
            if overlap_end <= overlap_start:
                continue
            mapped_start = concat_cursor + (overlap_start - base_start)
            mapped_end = concat_cursor + (overlap_end - base_start)
            if mapped_end > mapped_start:
                mapped.append((mapped_start, mapped_end))
        concat_cursor += base_end - base_start

    return merge_intervals(mapped)


def extract_intervals(audio: np.ndarray, sr: int, intervals: Sequence[Tuple[float, float]]) -> np.ndarray:
    segments: List[np.ndarray] = []
    for start, end in intervals:
        s = max(0, int(round(start * sr)))
        e = min(len(audio), int(round(end * sr)))
        if e > s:
            segments.append(audio[s:e])
    if not segments:
        return np.array([], dtype=audio.dtype)
    return np.concatenate(segments)


def segment_audio_fixed_length(audio: np.ndarray, segment_samples: int) -> List[np.ndarray]:
    if len(audio) == 0:
        return []

    segments: List[np.ndarray] = []
    for start in range(0, len(audio), segment_samples):
        chunk = audio[start:start + segment_samples]
        if len(chunk) == 0:
            continue
        if len(chunk) < segment_samples:
            chunk = np.pad(chunk, (0, segment_samples - len(chunk)))
        segments.append(chunk.astype(np.float32, copy=False))
    return segments


def intervals_total_duration(intervals: Sequence[Tuple[float, float]]) -> float:
    return float(sum(max(0.0, end - start) for start, end in intervals))


def sample_random_intervals(
    candidate_intervals: Sequence[Tuple[float, float]],
    target_duration: float,
    rng: random.Random,
) -> List[Tuple[float, float]]:
    available = list(merge_intervals(candidate_intervals))
    sampled: List[Tuple[float, float]] = []
    remaining = max(0.0, float(target_duration))

    while remaining > 1e-4 and available:
        lengths = [end - start for start, end in available]
        total = sum(lengths)
        if total <= 1e-6:
            break

        pick = rng.uniform(0.0, total)
        chosen_idx = 0
        cursor = 0.0
        for idx, length in enumerate(lengths):
            cursor += length
            if pick <= cursor:
                chosen_idx = idx
                break

        start, end = available.pop(chosen_idx)
        length = end - start
        cut_len = min(length, remaining)
        if cut_len <= 0:
            continue

        if cut_len == length:
            seg_start, seg_end = start, end
        else:
            seg_start = rng.uniform(start, end - cut_len)
            seg_end = seg_start + cut_len

        sampled.append((seg_start, seg_end))
        remaining -= cut_len

        if seg_start - start > 1e-4:
            available.append((start, seg_start))
        if end - seg_end > 1e-4:
            available.append((seg_end, end))

        available = merge_intervals(available)

    return merge_intervals(sampled)


def normalize_variant_name(variant: str) -> str:
    variant_norm = str(variant).strip().lower()
    if variant_norm not in VALID_VARIANTS:
        raise ValueError(f"Unsupported variant: {variant}")
    return VARIANT_ALIASES.get(variant_norm, variant_norm)


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


def _pad_audio(audio: np.ndarray, target_samples: int) -> np.ndarray:
    if len(audio) >= target_samples:
        return audio[:target_samples].astype(np.float32, copy=False)
    return np.pad(audio, (0, target_samples - len(audio))).astype(np.float32, copy=False)


def build_segment_view(
    pre_segment_audio: np.ndarray,
    sr: int,
    cue_intervals: Sequence[Tuple[float, float]],
    variant: str,
    seed: int,
    stable_token: str,
) -> np.ndarray:
    variant = normalize_variant_name(variant)
    segment_duration = len(pre_segment_audio) / float(sr) if sr > 0 else 0.0
    base_intervals = [(0.0, segment_duration)] if segment_duration > 0 else []
    cue_intervals = intersect_intervals(base_intervals, cue_intervals)
    non_cue_intervals = subtract_intervals(base_intervals, cue_intervals)

    if variant == "pre":
        return pre_segment_audio.astype(np.float32, copy=False)
    if variant == "cue-only":
        return extract_intervals(pre_segment_audio, sr, cue_intervals).astype(np.float32, copy=False)
    if variant == "cue-excluded":
        return extract_intervals(pre_segment_audio, sr, non_cue_intervals).astype(np.float32, copy=False)
    if variant == "random":
        stable_seed = int(hashlib.md5(stable_token.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(stable_seed + int(seed))
        target_duration = intervals_total_duration(cue_intervals)
        random_remove = sample_random_intervals(non_cue_intervals, target_duration, rng)
        kept_intervals = subtract_intervals(base_intervals, random_remove)
        return extract_intervals(pre_segment_audio, sr, kept_intervals).astype(np.float32, copy=False)

    raise ValueError(f"Unsupported variant: {variant}")


def annotation_path(dataset_key: str, sample_id: str) -> Path:
    prefix = DATASET_CONFIGS[dataset_key].annotation_prefix
    return ANNOTATION_ROOT / f"annotation_{prefix}_{sample_id}.json"


def speech_intervals_from_transcript(
    transcript: Dict,
    speech_scope: str = "participant",
) -> List[Tuple[float, float]]:
    if speech_scope not in VALID_SPEECH_SCOPES:
        raise ValueError(f"Unsupported speech scope: {speech_scope}")

    intervals = []
    for sentence in transcript.get("sentences", []):
        speaker_role = normalize_role(sentence.get("speaker"))
        if speech_scope == "participant" and speaker_role != "patient":
            continue
        if speech_scope == "interviewer" and speaker_role != "doctor":
            continue
        start = float(sentence.get("start", 0.0))
        end = float(sentence.get("end", 0.0))
        if end > start:
            intervals.append((start, end))
    return merge_intervals(intervals, gap=0.05)


def participant_intervals_from_transcript(transcript: Dict) -> List[Tuple[float, float]]:
    return speech_intervals_from_transcript(transcript, speech_scope="participant")


def interviewer_intervals_from_transcript(transcript: Dict) -> List[Tuple[float, float]]:
    return speech_intervals_from_transcript(transcript, speech_scope="interviewer")


def cue_role_from_overlap(cue_start: float, cue_end: float, transcript: Dict) -> str:
    best_overlap = 0.0
    best_role = "unknown"
    for sentence in transcript.get("sentences", []):
        start = float(sentence.get("start", 0.0))
        end = float(sentence.get("end", 0.0))
        overlap = max(0.0, min(cue_end, end) - max(cue_start, start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_role = normalize_role(sentence.get("speaker", ""))
    return best_role if best_overlap > 0 else "unknown"


def cue_is_participant(cue_start: float, cue_end: float, transcript: Dict) -> bool:
    return cue_role_from_overlap(cue_start, cue_end, transcript) == "patient"


def cue_intervals_from_outputs(
    dataset_key: str,
    sample_id: str,
    transcript: Dict,
    cues: Dict,
    cue_role: str = "patient",
) -> List[Tuple[float, float]]:
    if cue_role not in VALID_CUE_ROLES:
        raise ValueError(f"Unsupported cue role filter: {cue_role}")

    ann_path = annotation_path(dataset_key, sample_id)
    intervals: List[Tuple[float, float]] = []

    if ann_path.exists():
        annotation = load_json(ann_path)
        for cue in annotation.get("cues", []):
            if cue.get("status") == "deleted":
                continue
            span = cue.get("corrected_span") or cue.get("original_span") or {}
            start = span.get("start")
            end = span.get("end")
            if start is None or end is None:
                continue
            start = float(start)
            end = float(end)
            if end <= start:
                continue
            inferred_role = cue_role_from_overlap(start, end, transcript)
            if role_matches_filter(inferred_role, cue_role):
                intervals.append((start, end))
    else:
        for cue in cues.get("cues", []):
            role = cue.get("speaker_role") or cue.get("speaker")
            if not role_matches_filter(role, cue_role):
                continue
            start = cue.get("start")
            end = cue.get("end")
            if start is None or end is None:
                continue
            start = float(start)
            end = float(end)
            if end > start:
                intervals.append((start, end))

    return merge_intervals(intervals, gap=0.05)


def iter_sample_dirs(dataset_key: str) -> Iterable[Path]:
    return [Path(sample["sample_dir"]) for sample in iter_output_samples(dataset_key)]


def edaic_audio_path(sample_id: str) -> Path:
    participant_id = sample_id.replace("_AUDIO", "")
    return DATA_ROOT / "E-DAIC" / f"{participant_id}_P" / f"{participant_id}_AUDIO.wav"


def pdch_audio_path(sample_id: str) -> Path:
    match = re.match(r"^([0-9]{3}[AB])_(\d+)$", sample_id)
    if not match:
        raise ValueError(f"Unrecognized PDCH sample_id: {sample_id}")
    serial, session = match.groups()
    return DATA_ROOT / "PDCH" / serial / f"{session}.wav"


def cmdc_audio_path(sample_id: str) -> Path:
    match = re.match(r"^(part\d+)(HC\d+|MDD\d+)(Q\d+)$", sample_id)
    if not match:
        raise ValueError(f"Unrecognized CMDC sample_id: {sample_id}")
    part, subject, question = match.groups()
    return DATA_ROOT / "CMDC_EULA" / part / subject / f"{question}.wav"


def mandic_audio_path(sample_id: str) -> Path:
    upper = DATA_ROOT / "ManDIC" / "data" / f"{sample_id}.WAV"
    lower = DATA_ROOT / "ManDIC" / "data" / f"{sample_id}.wav"
    return upper if upper.exists() else lower


def audio_path_for_sample(dataset_key: str, sample_id: str) -> Path:
    if dataset_key == "edaic":
        return edaic_audio_path(sample_id)
    if dataset_key == "pdch":
        return pdch_audio_path(sample_id)
    if dataset_key == "cmdc":
        return cmdc_audio_path(sample_id)
    if dataset_key == "mandic":
        return mandic_audio_path(sample_id)
    raise KeyError(dataset_key)


def normalize_gender(value) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"m", "male", "1", "man"}:
        return "male"
    if text in {"f", "female", "0", "2", "woman"}:
        return "female"
    return None


def _safe_float(value) -> Optional[float]:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_metadata_mapping(dataset_key: str) -> Dict[str, Dict[str, object]]:
    config = DATASET_CONFIGS[dataset_key]
    df = pd.read_csv(config.label_path)

    if dataset_key == "edaic":
        return {
            f"{int(row['Participant_ID'])}_AUDIO": {
                "label": float(row["PHQ_8Total"]),
                "gender": None,
                "age": None,
            }
            for _, row in df.iterrows()
        }
    if dataset_key == "pdch":
        return {
            str(row["Serial"]).strip(): {
                "label": float(row["total"]),
                "gender": None,
                "age": None,
            }
            for _, row in df.iterrows()
        }
    if dataset_key == "cmdc":
        return {
            str(row["ID"]).strip(): {
                "label": float(row["PHQtotal"]),
                "gender": normalize_gender(row.get("gender")),
                "age": _safe_float(row.get("age")),
            }
            for _, row in df.iterrows()
        }
    if dataset_key == "mandic":
        df = df.dropna(subset=["HAMD-17_total_score"])
        return {
            str(row["standard_id"]).strip(): {
                "label": float(row["HAMD-17_total_score"]),
                "gender": normalize_gender(row.get("sex")),
                "age": _safe_float(row.get("age")),
            }
            for _, row in df.iterrows()
        }
    raise KeyError(dataset_key)


def load_label_mapping(dataset_key: str) -> Dict[str, float]:
    metadata = load_metadata_mapping(dataset_key)
    return {key: float(value["label"]) for key, value in metadata.items()}


def label_key_for_sample(dataset_key: str, sample_id: str) -> str:
    if dataset_key == "pdch":
        match = re.match(r"^([0-9]{3}[AB])_(\d+)$", sample_id)
        return match.group(1) if match else sample_id
    if dataset_key == "cmdc":
        match = re.match(r"^(part\d+)(HC\d+|MDD\d+)(Q\d+)$", sample_id)
        return match.group(2) if match else sample_id
    return sample_id


def group_id_for_sample(dataset_key: str, sample_id: str) -> str:
    if dataset_key == "pdch":
        return label_key_for_sample(dataset_key, sample_id)
    if dataset_key == "cmdc":
        return label_key_for_sample(dataset_key, sample_id)
    return sample_id


def lookup_sample_metadata(
    dataset_key: str,
    sample_id: str,
    metadata_mapping: Optional[Dict[str, Dict[str, object]]] = None,
) -> Optional[Dict[str, object]]:
    metadata_mapping = metadata_mapping or load_metadata_mapping(dataset_key)
    label_key = label_key_for_sample(dataset_key, sample_id)
    metadata = metadata_mapping.get(label_key)
    if metadata is None:
        return None
    return {
        "label": float(metadata["label"]),
        "gender": metadata.get("gender"),
        "age": metadata.get("age"),
    }


def official_edaic_splits() -> Dict[str, str]:
    split_map: Dict[str, str] = {}
    data_dir = DATA_ROOT / "E-DAIC"
    train_df = pd.read_csv(data_dir / "train_split_Depression_AVEC2017.csv")
    dev_df = pd.read_csv(data_dir / "dev_split_Depression_AVEC2017.csv")
    test_df = pd.read_csv(data_dir / "test_split_Depression_AVEC2017.csv")

    for pid in train_df["Participant_ID"].tolist():
        split_map[f"{int(pid)}_AUDIO"] = "train"
    for pid in dev_df["Participant_ID"].tolist():
        split_map[f"{int(pid)}_AUDIO"] = "val"
    for pid in test_df["participant_ID"].tolist():
        split_map[f"{int(pid)}_AUDIO"] = "test"
    return split_map


def split_group_ids(dataset_key: str, group_ids: Sequence[str], seed: int) -> Dict[str, str]:
    if dataset_key == "edaic":
        return official_edaic_splits()

    unique_ids = sorted(set(group_ids))
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.1, random_state=seed)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.1 / 0.9, random_state=seed)

    split_map = {gid: "train" for gid in train_ids}
    split_map.update({gid: "val" for gid in val_ids})
    split_map.update({gid: "test" for gid in test_ids})
    return split_map


def build_audio_view(
    dataset_key: str,
    transcript: Dict,
    cues: Dict,
    audio: np.ndarray,
    sr: int,
    variant: str,
    seed: int = 42,
    cue_role: str = "patient",
    speech_scope: str = "participant",
) -> np.ndarray:
    variant = normalize_variant_name(variant)
    sample_id = transcript.get("sample_id", "")
    base_intervals = speech_intervals_from_transcript(transcript, speech_scope=speech_scope)
    cue_intervals = cue_intervals_from_outputs(
        dataset_key,
        sample_id,
        transcript,
        cues,
        cue_role=cue_role,
    )
    cue_intervals = intersect_intervals(base_intervals, cue_intervals)
    non_cue_intervals = subtract_intervals(base_intervals, cue_intervals)

    if variant == "pre":
        intervals = base_intervals
    elif variant == "cue-only":
        intervals = cue_intervals
    elif variant == "cue-excluded":
        intervals = non_cue_intervals
    elif variant == "random":
        stable_seed = int(hashlib.md5(sample_id.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(stable_seed + int(seed))
        target_duration = intervals_total_duration(cue_intervals)
        random_remove = sample_random_intervals(non_cue_intervals, target_duration, rng)
        intervals = subtract_intervals(base_intervals, random_remove)
    else:
        raise ValueError(f"Unsupported variant: {variant}")

    return extract_intervals(audio, sr, intervals)


def build_preprocessed_audio_bundle(
    dataset_key: str,
    transcript: Dict,
    cues: Dict,
    audio: np.ndarray,
    sr: int,
    cue_role: str = "patient",
    speech_scope: str = "participant",
) -> Tuple[np.ndarray, List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Build the preprocessed speech view and map cue intervals onto the
    concatenated retained-speech timeline.

    Returns:
        pre_audio, retained_intervals, cue_intervals_on_pre_timeline
    """
    sample_id = transcript.get("sample_id", "")
    retained_intervals = speech_intervals_from_transcript(transcript, speech_scope=speech_scope)
    cue_intervals = cue_intervals_from_outputs(
        dataset_key,
        sample_id,
        transcript,
        cues,
        cue_role=cue_role,
    )
    cue_intervals = intersect_intervals(retained_intervals, cue_intervals)
    pre_audio = extract_intervals(audio, sr, retained_intervals)
    mapped_cues = map_intervals_to_concatenated_timeline(retained_intervals, cue_intervals)
    return pre_audio, retained_intervals, mapped_cues


def load_all_segments(
    dataset_key: str,
    variant: str,
    segment_length: float = 30.0,
    sample_rate: int = 16000,
    seed: int = 42,
    cue_role: str = "patient",
    speech_scope: str = "participant",
    segment_first: bool = False,
) -> Tuple[List[np.ndarray], np.ndarray, List[str], List[str]]:
    records = load_segment_records(
        dataset_key=dataset_key,
        variant=variant,
        segment_length=segment_length,
        sample_rate=sample_rate,
        seed=seed,
        cue_role=cue_role,
        speech_scope=speech_scope,
        segment_first=segment_first,
    )
    segments = [record["audio"] for record in records]
    labels = [float(record["label"]) for record in records]
    sample_ids = [str(record["sample_id"]) for record in records]
    group_ids = [str(record["group_id"]) for record in records]
    return segments, np.array(labels, dtype=np.float32), sample_ids, group_ids


def load_segment_records(
    dataset_key: str,
    variant: str,
    segment_length: float = 30.0,
    sample_rate: int = 16000,
    seed: int = 42,
    cue_role: str = "patient",
    speech_scope: str = "participant",
    segment_first: bool = False,
) -> List[Dict[str, object]]:
    variant = normalize_variant_name(variant)
    segment_samples = int(segment_length * sample_rate)
    metadata_mapping = load_metadata_mapping(dataset_key)
    records: List[Dict[str, object]] = []

    for sample in iter_output_samples(dataset_key):
        transcript_path = Path(sample["transcript_path"])
        cues_path = Path(sample["cues_path"])
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

        if segment_first:
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

            total_duration_sec = len(pre_audio) / float(sample_rate)
            group_id = group_id_for_sample(dataset_key, sample_id)
            segment_count = int(np.ceil(total_duration_sec / segment_length))
            for seg_idx in range(segment_count):
                seg_start_sec = seg_idx * segment_length
                seg_end_sec = min(seg_start_sec + segment_length, total_duration_sec)
                start_sample = int(round(seg_start_sec * sample_rate))
                end_sample = int(round(seg_end_sec * sample_rate))
                raw_segment = pre_audio[start_sample:end_sample]
                if len(raw_segment) == 0:
                    continue

                cue_spans_segment = _intersect_relative_spans(cue_spans_pre, seg_start_sec, seg_end_sec)
                seg_audio = build_segment_view(
                    pre_segment_audio=raw_segment,
                    sr=sample_rate,
                    cue_intervals=cue_spans_segment,
                    variant=variant,
                    seed=seed,
                    stable_token=f"{sample_id}__seg{seg_idx}",
                )
                seg_audio = _pad_audio(seg_audio, segment_samples)
                records.append(
                    {
                        "audio": seg_audio,
                        "label": metadata["label"],
                        "sample_id": sample_id,
                        "group_id": group_id,
                        "dataset_key": dataset_key,
                        "gender": metadata.get("gender"),
                        "age": metadata.get("age"),
                        "segment_index": seg_idx,
                    }
                )
        else:
            view_audio = build_audio_view(
                dataset_key=dataset_key,
                transcript=transcript,
                cues=cues,
                audio=audio,
                sr=sr,
                variant=variant,
                seed=seed,
                cue_role=cue_role,
                speech_scope=speech_scope,
            )
            split_segments = segment_audio_fixed_length(view_audio, segment_samples)

            for seg_idx, seg in enumerate(split_segments):
                records.append(
                    {
                        "audio": seg,
                        "label": metadata["label"],
                        "sample_id": sample_id,
                        "group_id": group_id_for_sample(dataset_key, sample_id),
                        "dataset_key": dataset_key,
                        "gender": metadata.get("gender"),
                        "age": metadata.get("age"),
                        "segment_index": seg_idx,
                    }
                )

    return records
