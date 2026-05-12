"""
Cue wordcloud export.

For each dataset, generate two separate token-based wordcloud images:
- interviewer
- interviewee
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Counter as CounterType
from typing import Dict, Iterable, List

import jieba
import jieba.posseg as pseg
import matplotlib.pyplot as plt
import nltk
from wordcloud import WordCloud

BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent.parent
BASE_OUTPUT_DIR = BASE_DIR / "outputs"
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AGENT_OUTPUTS_DIR = PROJECT_ROOT / "agent" / "outputs"
ROLES = ("interviewer", "interviewee")
DATASETS = {
    "edaic": "E-DAIC",
    "mandic": "ManDIC",
    "cmdc": "CMDC",
    "pdch": "PDCH",
}
TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z]+(?:'[A-Za-z]+)?")
EN_CONTENT_TAG_PREFIXES = ("JJ", "NN")
EN_BRIDGE_TOKENS = {"of", "and"}
EN_ADJ_SUFFIXES = (
    "y", "ful", "ous", "ive", "able", "ible", "al", "ish", "less", "ic",
    "ical", "ary", "ory", "ant", "ent",
)

EN_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "so", "because",
    "of", "in", "on", "at", "for", "to", "from", "with", "by", "as",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing", "have", "has", "had",
    "i", "me", "my", "mine", "you", "your", "yours", "we", "our", "ours",
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their",
    "this", "that", "these", "those", "there", "here", "what", "which", "who", "whom",
    "about", "into", "over", "under", "again", "really", "very", "just",
    "lot", "kind", "sort", "thing", "things",
}

ZH_STOPWORDS = {
    "的", "了", "呢", "吗", "啊", "吧", "呀", "哦", "哇", "嗯", "呃", "诶",
    "一个", "一种", "一些", "这个", "那个", "这些", "那些",
    "自己", "我们", "你们", "他们", "她们", "它们",
    "时候", "最近", "现在", "之前", "之后",
}
ZH_BRIDGE_TOKENS = {"很", "最", "更", "自己", "的", "地", "得"}


def get_font_path() -> str:
    """Return a font path that can render Chinese text if available."""
    font_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]
    for path in font_paths:
        if os.path.exists(path):
            return path
    return None


def normalize_role(cue: Dict) -> str:
    """Normalize cue speaker role to interviewer/interviewee."""
    role = str(cue.get("speaker_role") or cue.get("speaker") or "").strip().lower()
    if any(tag in role for tag in ("interviewer", "doctor", "clinician", "therapist")):
        return "interviewer"
    if any(tag in role for tag in ("interviewee", "participant", "patient", "subject")):
        return "interviewee"
    return role


def load_cue_samples(dataset: str) -> List[Dict]:
    """Load cue files from agent outputs for a dataset."""
    dataset_dir = AGENT_OUTPUTS_DIR / dataset
    results: List[Dict] = []

    if not dataset_dir.exists():
        return results

    cue_files = sorted(set(dataset_dir.rglob("cues.json")) | set(dataset_dir.rglob("*_cues.json")))
    for cue_file in cue_files:
        with open(cue_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        cues = [cue for cue in data.get("cues", []) if isinstance(cue, dict)]
        if not cues:
            continue

        results.append(
            {
                "sample_id": data.get("sample_id", cue_file.stem),
                "cues": cues,
            }
        )

    return results


def is_chinese_token(token: str) -> bool:
    """Check whether a token contains Chinese characters."""
    return any("\u4e00" <= ch <= "\u9fff" for ch in token)


def keep_chinese_pos(flag: str) -> bool:
    """Keep Chinese content words for symptom-like phrases."""
    return flag.startswith(("a", "n", "nr", "ns", "nt", "nz", "v", "vn"))


def keep_english_pos(word: str, tag: str) -> bool:
    """Keep English adjectives and nouns/proper nouns."""
    if tag.startswith(EN_CONTENT_TAG_PREFIXES):
        return True

    # Fallback heuristic when no tagger resource is available.
    if word[:1].isupper():
        return True
    return word.lower().endswith(EN_ADJ_SUFFIXES)


def ensure_nltk_tagger() -> bool:
    """Ensure the NLTK POS tagger is available."""
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
        return True
    except LookupError:
        try:
            return bool(nltk.download("averaged_perceptron_tagger_eng", quiet=True))
        except Exception:
            return False


def tokenize_chinese_chunk(text: str) -> List[str]:
    """Extract short Chinese phrases instead of isolated words."""
    phrases: List[str] = []
    current_phrase: List[str] = []

    def flush() -> None:
        if not current_phrase:
            return
        phrase = "".join(current_phrase).strip()
        if phrase and phrase not in ZH_STOPWORDS and len(phrase) >= 2:
            phrases.append(phrase)
        current_phrase.clear()

    for word, flag in pseg.lcut(text):
        token = word.strip().lower()
        if not token:
            continue
        if re.fullmatch(r"[\W_]+", token):
            flush()
            continue

        if token in ZH_BRIDGE_TOKENS and current_phrase:
            continue
        if token in ZH_STOPWORDS:
            flush()
            continue

        if keep_chinese_pos(flag):
            current_phrase.append(token)
        else:
            flush()

    flush()
    return phrases


def tokenize_english_chunk(text: str) -> List[str]:
    """Extract English adjective/noun phrases instead of isolated words."""
    raw_tokens = [
        token.strip()
        for token in TOKEN_PATTERN.findall(text)
        if token.strip() and not is_chinese_token(token)
    ]
    raw_tokens = [token for token in raw_tokens if len(token) > 1]
    if not raw_tokens:
        return []

    if ensure_nltk_tagger():
        tagged_tokens = nltk.pos_tag(raw_tokens)
    else:
        tagged_tokens = [(token, "") for token in raw_tokens]

    phrases: List[str] = []
    current_phrase: List[str] = []

    def flush() -> None:
        if not current_phrase:
            return
        cleaned = [token.lower() for token in current_phrase if token.lower() not in EN_BRIDGE_TOKENS]
        if len(cleaned) > 1:
            phrases.append(" ".join(current_phrase).lower())
        elif not phrases:
            phrases.append(cleaned[0] if cleaned else current_phrase[0].lower())
        current_phrase.clear()

    for token, tag in tagged_tokens:
        token_lower = token.lower()
        if token_lower in EN_BRIDGE_TOKENS and current_phrase:
            current_phrase.append(token_lower)
            continue
        if token_lower in EN_STOPWORDS:
            flush()
            continue
        if keep_english_pos(token, tag):
            current_phrase.append(token)
        elif current_phrase and token.isalpha():
            # Keep a trailing content-looking token to preserve short phrases
            # like "good night's sleep".
            current_phrase.append(token)
        else:
            flush()

    flush()
    return phrases


def tokenize_text(text: str) -> List[str]:
    """Tokenize mixed Chinese/English cue text and keep adjectives/nouns only."""
    zh_chunks: List[str] = []
    en_chunks: List[str] = []

    for chunk in TOKEN_PATTERN.findall(text):
        if is_chinese_token(chunk):
            zh_chunks.append(chunk)
        else:
            en_chunks.append(chunk)

    tokens: List[str] = []
    if zh_chunks:
        tokens.extend(tokenize_chinese_chunk(" ".join(zh_chunks)))
    if en_chunks:
        tokens.extend(tokenize_english_chunk(" ".join(en_chunks)))
    return tokens


def build_role_word_counts(results: Iterable[Dict]) -> Dict[str, CounterType[str]]:
    """Count cue tokens by speaker role."""
    role_word_counts: Dict[str, CounterType[str]] = defaultdict(Counter)

    for sample in results:
        for cue in sample.get("cues", []):
            text = str(cue.get("text", "")).strip()
            role = normalize_role(cue)
            if not text or role not in ROLES:
                continue
            for token in tokenize_text(text):
                role_word_counts[role][token] += 1

    return {role: role_word_counts.get(role, Counter()) for role in ROLES}


def generate_single_wordcloud(output_path: Path, word_counts: Dict[str, int], font_path: str) -> None:
    """Render one role-specific wordcloud image."""
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    ax.axis("off")

    if not word_counts:
        ax.text(
            0.5,
            0.5,
            "No tokens available",
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            transform=ax.transAxes,
        )
    else:
        wc_kwargs = {
            "width": 1200,
            "height": 840,
            "background_color": "white",
            "max_words": 120,
            "min_font_size": 12,
            "max_font_size": 140,
            "relative_scaling": 0.45,
            "prefer_horizontal": 1.0,
            "colormap": "viridis",
            "margin": 10,
        }
        if font_path:
            wc_kwargs["font_path"] = font_path

        wc = WordCloud(**wc_kwargs)
        wc.generate_from_frequencies(word_counts)
        ax.imshow(wc, interpolation="bilinear")

    plt.tight_layout()
    fig.savefig(output_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved wordcloud: {output_path}")


def process_dataset(dataset: str) -> None:
    """Generate two separate wordcloud images for one dataset."""
    print(f"\n{'=' * 70}")
    print(f"{dataset} wordcloud analysis")
    print("=" * 70)

    samples = load_cue_samples(dataset)
    if not samples:
        print(f"[{dataset}] No cue files found under {AGENT_OUTPUTS_DIR / dataset}")
        return

    role_word_counts = build_role_word_counts(samples)
    dataset_output_dir = BASE_OUTPUT_DIR / dataset.lower()
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    font_path = get_font_path()

    old_combined = dataset_output_dir / f"{dataset.lower()}_wordcloud_by_role.png"
    old_combined.unlink(missing_ok=True)

    for role in ROLES:
        output_path = dataset_output_dir / f"{dataset.lower()}_{role}_wordcloud.png"
        generate_single_wordcloud(output_path, dict(role_word_counts.get(role, Counter())), font_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cue wordcloud analysis by speaker role")
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS.keys(), "all"],
        default="all",
        help="Dataset to analyze",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        for dataset in DATASETS.values():
            process_dataset(dataset)
    else:
        process_dataset(DATASETS[args.dataset])

    print(f"\nResults saved to: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
