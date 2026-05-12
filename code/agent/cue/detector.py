"""
Unified Cue Detector
统一的抑郁线索检测器
"""
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from agent.cue.categories import get_categories
from agent.cue.llm_extractor import LLMCueExtractor
from agent.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CueDetector:
    """
    抑郁线索检测器

    支持中英文，检测所有说话人的内容
    集成LLM进行短语提取和QA配对
    """

    def __init__(self, ollama_config: Optional[Dict] = None, use_llm_extraction: bool = True):
        self.llm_client = LLMClient(**ollama_config) if ollama_config else None
        self.use_llm_extraction = use_llm_extraction
        self.llm_extractor = LLMCueExtractor(self.llm_client) if use_llm_extraction else None

    def detect(
        self,
        transcript: Dict[str, Any],
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        检测transcript中的抑郁线索

        优先使用 word-level 时间戳进行精确定位，
        如果没有 words，则回退到 sentence-level

        Args:
            transcript: transcript.json内容
            language: 语言代码 (en/zh)，None则自动检测

        Returns:
            cues.json格式结果
        """
        if not language:
            language = transcript.get("language", "en")

        sample_id = transcript.get("sample_id", "unknown")
        dataset = transcript.get("dataset", "unknown")
        words = transcript.get("words", [])
        sentences = transcript.get("sentences", [])

        # 优先使用 word-level 进行检测
        if words:
            logger.info(f"Detecting cues for {sample_id}: {len(words)} words, language={language}")
            cues = self._detect_in_words(words, language)
        else:
            logger.info(f"Detecting cues for {sample_id}: {len(sentences)} sentences (no words), language={language}")
            cues = self._detect_in_sentences(sentences, language)

        # 去重
        cues = self._deduplicate_cues(cues)

        # LLM后处理：短语提取和QA配对
        if self.use_llm_extraction and self.llm_extractor and cues:
            logger.info(f"Running LLM extraction for {len(cues)} cues...")
            # 步骤1: 短语提取（问题1）
            cues = self.llm_extractor.batch_extract_phrases(cues, sentences, language)
            # 步骤2: QA配对（问题2）
            cues = self.llm_extractor.batch_pair_qa(cues, sentences, language)
            logger.info(f"LLM extraction completed")

        # 构建输出
        speakers = list(set(c.get("speaker", "unknown") for c in cues)) if cues else []

        # 构建cue输出，保留LLM提取的额外信息
        cue_outputs = []
        for i, c in enumerate(cues):
            speaker = c.get("speaker", "unknown")
            speaker_lower = speaker.lower()
            # 判断角色
            if "interviewer" in speaker_lower or "doctor" in speaker_lower or "ellie" in speaker_lower:
                speaker_role = "interviewer"
            elif "interviewee" in speaker_lower or "patient" in speaker_lower or "participant" in speaker_lower:
                speaker_role = "interviewee"
            else:
                speaker_role = "unknown"

            cue_out = {
                "id": i,
                "text": c["text"],
                "sentence_id": c.get("sentence_id", 0),
                "start": round(c["start"], 4),
                "end": round(c["end"], 4),
                "speaker": speaker,
                "speaker_role": speaker_role,
                "category": c.get("category", "depression_related"),
                "confidence": round(c.get("confidence", 0.8), 2),
            }
            # 保留LLM提取的额外信息（如果有）
            if "original_keyword" in c:
                cue_out["original_keyword"] = c["original_keyword"]
            if "original_answer" in c:
                cue_out["original_answer"] = c["original_answer"]
            if "context_question" in c:
                cue_out["context_question"] = c["context_question"]
            if "inferred_topic" in c:
                cue_out["inferred_topic"] = c["inferred_topic"]
            if "extraction_confidence" in c:
                cue_out["extraction_confidence"] = round(c["extraction_confidence"], 2)
            if "pairing_confidence" in c:
                cue_out["pairing_confidence"] = round(c["pairing_confidence"], 2)
            cue_outputs.append(cue_out)

        result = {
            "sample_id": sample_id,
            "dataset": dataset,
            "language": language,
            "total_cues": len(cues),
            "speakers": speakers,
            "cues": cue_outputs
        }

        logger.info(f"Detected {len(cues)} cues")
        return result

    def _detect_in_words(
        self,
        words: List[Dict[str, Any]],
        language: str
    ) -> List[Dict[str, Any]]:
        """
        使用 word-level 时间戳精确检测线索

        策略：
        1. 将连续的 words 组合成短语进行匹配
        2. 优先匹配最长的关键词
        3. 精确记录关键词的时间戳
        """
        cues = []
        categories = get_categories(language)

        # 构建文本字符串（用于匹配）
        word_texts = [w.get("text", "") for w in words]
        full_text = "".join(word_texts) if language == "zh" else " ".join(word_texts)
        full_text_lower = full_text.lower() if language == "en" else full_text

        # 字符到 word 索引的映射（用于时间戳查找）
        char_to_word_idx = []
        for i, w in enumerate(word_texts):
            char_to_word_idx.extend([i] * len(w))
            if language != "zh":
                char_to_word_idx.append(i)  # 空格

        # 对每个类别进行匹配
        for cat_name, cat_info in categories.items():
            for keyword in cat_info["keywords"]:
                keyword_lower = keyword.lower() if language == "en" else keyword

                # 在文本中查找所有匹配
                start_pos = 0
                while True:
                    idx = full_text_lower.find(keyword_lower, start_pos)
                    if idx == -1:
                        break

                    # 找到对应的 words
                    end_idx = idx + len(keyword)
                    if end_idx >= len(char_to_word_idx):
                        end_idx = len(char_to_word_idx) - 1

                    start_word_idx = char_to_word_idx[idx] if idx < len(char_to_word_idx) else 0
                    end_word_idx = char_to_word_idx[end_idx - 1] if end_idx <= len(char_to_word_idx) else len(words) - 1

                    # 获取时间戳
                    start_word = words[start_word_idx]
                    end_word = words[end_word_idx]

                    # 构建上下文（前后各2个词）
                    context_start = max(0, start_word_idx - 2)
                    context_end = min(len(words), end_word_idx + 3)
                    context_words = [words[i].get("text", "") for i in range(context_start, context_end)]
                    context = "".join(context_words) if language == "zh" else " ".join(context_words)

                    cue = {
                        "text": keyword,
                        "context": context,
                        "sentence_id": start_word.get("sentence_id", 0),
                        "start": start_word.get("start", 0),
                        "end": end_word.get("end", 0),
                        "speaker": start_word.get("speaker", "unknown"),
                        "category": cat_name,
                        "confidence": self._estimate_confidence(context, keyword, language),
                    }
                    cues.append(cue)

                    start_pos = idx + 1

        return cues

    def _detect_in_sentences(
        self,
        sentences: List[Dict[str, Any]],
        language: str
    ) -> List[Dict[str, Any]]:
        """在句子中检测线索（回退方案）"""
        cues = []
        categories = get_categories(language)

        for sent in sentences:
            text = sent.get("text", "")
            if not text:
                continue

            text_lower = text.lower() if language == "en" else text

            # 对每个类别进行匹配
            for cat_name, cat_info in categories.items():
                for keyword in cat_info["keywords"]:
                    if keyword in text_lower:
                        # 估算关键词位置
                        start_ratio, end_ratio = self._find_keyword_position(text, keyword, language)
                        sent_start = sent.get("start", 0)
                        sent_end = sent.get("end", 0)
                        sent_duration = sent_end - sent_start

                        cue = {
                            "text": keyword,
                            "context": text,
                            "sentence_id": sent.get("id", 0),
                            "start": sent_start + sent_duration * start_ratio if start_ratio else sent_start,
                            "end": sent_start + sent_duration * end_ratio if end_ratio else sent_end,
                            "speaker": sent.get("speaker", "unknown"),
                            "category": cat_name,
                            "confidence": self._estimate_confidence(text, keyword, language) * 0.8,  # 降低置信度
                        }
                        cues.append(cue)

        return cues

    def _find_keyword_position(
        self,
        text: str,
        keyword: str,
        language: str
    ) -> tuple:
        """找到关键词在文本中的位置（用于估计时间戳）"""
        idx = text.lower().find(keyword) if language == "en" else text.find(keyword)
        if idx == -1:
            return None, None

        # 简单估计：关键词在整个文本中的位置比例
        text_len = len(text)
        keyword_len = len(keyword)

        start_ratio = idx / text_len if text_len > 0 else 0
        end_ratio = (idx + keyword_len) / text_len if text_len > 0 else 0

        # 返回相对位置（秒），调用者需要加上句子起始时间
        # 这里返回比例，由调用者计算具体时间
        return start_ratio, end_ratio

    def _estimate_confidence(
        self,
        text: str,
        keyword: str,
        language: str
    ) -> float:
        """估计检测置信度"""
        # 基础置信度
        base_conf = 0.8

        # 根据上下文调整
        text_lower = text.lower() if language == "en" else text

        # 如果有更长的匹配短语，置信度更高
        if len(keyword) > 4:
            base_conf += 0.05

        # 如果关键词前后有语境支持，置信度更高
        idx = text_lower.find(keyword)
        if idx > 0:
            # 检查前文
            prev_text = text_lower[max(0, idx-10):idx]
            context_words = ["feel", "have", "am", "very"] if language == "en" else ["感到", "很", "非常"]
            if any(w in prev_text for w in context_words):
                base_conf += 0.05

        return min(base_conf, 0.95)

    def _deduplicate_cues(self, cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去重：相同sentence_id和相似text的只保留一个"""
        if not cues:
            return []

        # 按sentence_id分组去重
        result = []
        seen_keys = set()  # (sentence_id, text_lower)

        for cue in cues:
            sid = cue.get("sentence_id", 0)
            text = cue.get("text", "").lower().strip()

            # 检查是否已存在相似文本
            is_dup = False
            for existing_sid, existing_text in list(seen_keys):
                if sid == existing_sid and self._text_similarity(text, existing_text) > 0.7:
                    is_dup = True
                    break

            if not is_dup:
                result.append(cue)
                seen_keys.add((sid, text))

        return result

    def _text_similarity(self, s1: str, s2: str) -> float:
        """计算文本相似度"""
        if s1 == s2:
            return 1.0
        if s1 in s2 or s2 in s1:
            return 0.8

        # 计算共有字符比例
        set1, set2 = set(s1), set(s2)
        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def process_directory(
        self,
        input_dir: str,
        dataset: str,
        skip_existing: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        批量处理目录中的所有transcript.json

        Args:
            input_dir: 输入目录（包含transcript.json的目录）
            dataset: 数据集名称
            skip_existing: 是否跳过已处理的文件（cues.json已存在则跳过）

        Returns:
            处理结果列表
        """
        input_path = Path(input_dir)
        results = []

        # 查找所有 transcript 文件，兼容 CMDC 的 *_transcript.json 命名
        transcript_files = sorted(
            set(input_path.rglob("transcript.json")) | set(input_path.rglob("*_transcript.json")),
            key=lambda p: str(p),
        )
        logger.info(f"Found {len(transcript_files)} transcript files")

        # 过滤已处理的文件
        if skip_existing:
            transcript_files = [
                tf for tf in transcript_files
                if not self._get_cue_path(tf).exists()
            ]
            logger.info(f"Skipping processed files, {len(transcript_files)} remaining")

        for i, tf in enumerate(transcript_files, 1):
            sample_id = self._get_sample_id(tf)
            cue_file = self._get_cue_path(tf)

            logger.info(f"[{i}/{len(transcript_files)}] Processing {sample_id}")

            try:
                # 加载transcript
                with open(tf, "r", encoding="utf-8") as f:
                    transcript = json.load(f)

                # 检测cue
                cues = self.detect(transcript)

                # 保存结果
                with open(cue_file, "w", encoding="utf-8") as f:
                    json.dump(cues, f, indent=2, ensure_ascii=False)

                logger.info(f"  Saved {cues['total_cues']} cues to {cue_file}")
                results.append({"sample": sample_id, "status": "success", "cues": cues["total_cues"]})

            except Exception as e:
                logger.error(f"  Failed: {e}")
                results.append({"sample": sample_id, "status": "error", "error": str(e)})

        return results

    @staticmethod
    def _get_sample_id(transcript_path: Path) -> str:
        if transcript_path.name == "transcript.json":
            return transcript_path.parent.name
        if transcript_path.name.endswith("_transcript.json"):
            return transcript_path.name[:-len("_transcript.json")]
        return transcript_path.stem

    @staticmethod
    def _get_cue_path(transcript_path: Path) -> Path:
        if transcript_path.name == "transcript.json":
            return transcript_path.parent / "cues.json"
        if transcript_path.name.endswith("_transcript.json"):
            return transcript_path.with_name(
                transcript_path.name.replace("_transcript.json", "_cues.json")
            )
        return transcript_path.with_name("cues.json")
