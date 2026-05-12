"""
Multi-ASR Reconcile Engine
使用本地LLM融合多个ASR结果
"""
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from pydantic import BaseModel, Field

from agent.asr.base import ASRResult, ASRSegment

logger = logging.getLogger(__name__)


class ASRSelectionResult(BaseModel):
    """ASR选择结果Schema"""
    selected_model: str = Field(description="选择的模型名称，如 'whisper' 或 'qwen-audio'")
    confidence: float = Field(description="选择置信度 0-1")
    reasoning: str = Field(description="简要选择理由")
    improved_text: Optional[str] = Field(description="如果有改进，提供改进后的文本；否则为null", default=None)


@dataclass
class ReconcileConfig:
    """Reconcile配置"""
    # 模型权重 (Whisper, SenseVoice, Qwen-Audio)
    model_weights: Dict[str, float] = None
    # 置信度阈值：低于此值触发LLM reconcile
    confidence_threshold: float = 0.75
    # 差异阈值：两段文本相似度低于此值才需要reconcile
    diff_threshold: float = 0.6
    # LLM温度
    llm_temperature: float = 0.1

    def __post_init__(self):
        if self.model_weights is None:
            self.model_weights = {
                "whisper_stable": 0.55,
                "whisper_fast": 0.45,
            }


class TextSimilarity:
    """文本相似度计算"""

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """计算编辑距离"""
        if len(s1) < len(s2):
            return TextSimilarity.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    @classmethod
    def similarity(cls, s1: str, s2: str) -> float:
        """计算相似度 (0-1)"""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        max_len = max(len(s1), len(s2))
        distance = cls.levenshtein_distance(s1, s2)
        return 1.0 - (distance / max_len)


class SegmentAligner:
    """时间段对齐器：将不同ASR的时间段对齐"""

    @staticmethod
    def align_segments(
        results: List[ASRResult],
        time_tolerance: float = 0.3  # 时间容差（秒），减小以避免连续segment被合并
    ) -> List[Dict[str, Any]]:
        """
        对齐多个ASR结果的时间段

        Returns:
            List of aligned groups, each group contains segments from different ASRs
            [{"time": (start, end), "segments": {engine_name: segment}}]
        """
        if not results:
            return []

        # 以第一路结果为锚点进行对齐，避免跨多句的链式合并
        reference_result = results[0]
        other_results = results[1:]
        groups = []

        for ref_seg in reference_result.segments:
            group = {
                "time": (ref_seg.start, ref_seg.end),
                "segments": {reference_result.engine: ref_seg},
            }

            for result in other_results:
                aligned_seg = SegmentAligner._extract_aligned_segment(
                    result=result,
                    ref_start=ref_seg.start,
                    ref_end=ref_seg.end,
                    time_tolerance=time_tolerance,
                )
                if aligned_seg:
                    group["segments"][result.engine] = aligned_seg

            groups.append(group)

        # 保留参考结果未覆盖到的尾部/边界内容
        for result in other_results:
            for seg in result.segments:
                if not any(
                    SegmentAligner._segments_overlap(
                        seg.start, seg.end, ref_seg.start, ref_seg.end, time_tolerance
                    )
                    for ref_seg in reference_result.segments
                ):
                    groups.append({
                        "time": (seg.start, seg.end),
                        "segments": {result.engine: seg},
                    })

        groups.sort(key=lambda item: (item["time"][0], item["time"][1]))

        return groups

    @staticmethod
    def _extract_aligned_segment(
        result: ASRResult,
        ref_start: float,
        ref_end: float,
        time_tolerance: float,
    ) -> Optional[ASRSegment]:
        overlapping_segments = [
            seg
            for seg in result.segments
            if SegmentAligner._segments_overlap(
                seg.start, seg.end, ref_start, ref_end, time_tolerance
            )
        ]

        if not overlapping_segments:
            return None

        overlapping_words: List[Dict[str, Any]] = []
        for seg in overlapping_segments:
            if not seg.words:
                continue
            for word in seg.words:
                word_start = float(word.get("start", seg.start))
                word_end = float(word.get("end", seg.end))
                if SegmentAligner._segments_overlap(
                    word_start, word_end, ref_start, ref_end, time_tolerance
                ):
                    overlapping_words.append(word)

        if overlapping_words:
            overlapping_words.sort(
                key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0)))
            )
            text = "".join(str(word.get("word", "")) for word in overlapping_words).strip()
            confidences = [
                float(word.get("probability", 0.0))
                for word in overlapping_words
                if word.get("probability") is not None
            ]
            confidence = (
                sum(confidences) / len(confidences)
                if confidences
                else max(seg.confidence for seg in overlapping_segments)
            )
            return ASRSegment(
                id=overlapping_segments[0].id,
                text=text,
                start=float(overlapping_words[0].get("start", ref_start)),
                end=float(overlapping_words[-1].get("end", ref_end)),
                confidence=confidence,
                words=overlapping_words,
            )

        return max(
            overlapping_segments,
            key=lambda seg: SegmentAligner._overlap_duration(
                seg.start, seg.end, ref_start, ref_end
            ),
        )

    @staticmethod
    def _segments_overlap(
        start1: float,
        end1: float,
        start2: float,
        end2: float,
        tolerance: float = 0.0,
    ) -> bool:
        return not (end1 < start2 - tolerance or start1 > end2 + tolerance)

    @staticmethod
    def _overlap_duration(start1: float, end1: float, start2: float, end2: float) -> float:
        return max(0.0, min(end1, end2) - max(start1, start2))


class ReconcileEngine:
    """
    多ASR结果融合引擎

    算法：
    1. 时间段对齐：将多个ASR的segments按时间对齐
    2. 相似度计算：计算同一时间段内不同ASR文本的相似度
    3. 规则选择：高相似度直接选高权重模型结果
    4. LLM增强：低相似度时使用本地LLM判断最优结果
    """

    def __init__(self, llm_client=None, config: Optional[ReconcileConfig] = None):
        self.llm_client = llm_client
        self.config = config or ReconcileConfig()
        self.aligner = SegmentAligner()
        self.similarity = TextSimilarity()

    def reconcile(self, results: List[ASRResult]) -> ASRResult:
        """
        融合多个ASR结果

        Args:
            results: 多个ASR结果（通常2-3个）

        Returns:
            融合后的ASR结果
        """
        if not results:
            raise ValueError("No ASR results to reconcile")

        if len(results) == 1:
            return results[0]

        logger.info(f"Reconciling {len(results)} ASR results")

        # 1. 时间段对齐
        aligned_groups = self.aligner.align_segments(results)
        logger.info(f"Aligned into {len(aligned_groups)} time groups")

        # 2. 逐段融合
        fused_segments = []
        for i, group in enumerate(aligned_groups):
            fused_seg = self._fuse_segment_group(i, group)
            fused_segments.append(fused_seg)

        # 3. 构建最终结果
        full_text = " ".join(s.text for s in fused_segments)
        avg_confidence = sum(s.confidence for s in fused_segments) / len(fused_segments) if fused_segments else 0

        return ASRResult(
            text=full_text,
            segments=fused_segments,
            language=results[0].language,
            engine="fused",
            model=f"fused_{len(results)}",
            confidence_avg=avg_confidence,
        )

    def _fuse_segment_group(self, seg_id: int, group: Dict[str, Any]) -> ASRSegment:
        """
        融合一个时间段内的多个ASR结果
        """
        segments = group["segments"]  # {engine_name: ASRSegment}
        time_start, time_end = group["time"]

        if len(segments) == 1:
            # 只有一个ASR结果，直接返回
            seg = list(segments.values())[0]
            seg_start, seg_end = self._get_segment_time_bounds(seg, time_start, time_end)
            return ASRSegment(
                id=seg_id,
                text=seg.text,
                start=seg_start,
                end=seg_end,
                confidence=seg.confidence,
                words=seg.words,
            )

        # 计算两两相似度
        texts = {name: seg.text for name, seg in segments.items()}
        similarities = []

        names = list(texts.keys())
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                sim = self.similarity.similarity(texts[name1], texts[name2])
                similarities.append((name1, name2, sim))

        # 计算平均相似度
        avg_sim = sum(s[2] for s in similarities) / len(similarities) if similarities else 0

        logger.debug(f"Segment {seg_id}: avg_similarity={avg_sim:.2f}, texts={texts}")

        # 决策：高相似度直接选高权重，低相似度用LLM
        improved_text = None
        if avg_sim >= self.config.diff_threshold:
            # 高相似度：选择加权置信度最高的
            best_name = self._select_by_weighted_score(segments)
            best_seg = segments[best_name]
            logger.debug(f"Segment {seg_id}: rule-based selection ({best_name})")
        else:
            # 低相似度：使用LLM判断
            if self.llm_client:
                best_name, improved_text = self._llm_select_best(texts)
                best_seg = segments[best_name]
                logger.debug(f"Segment {seg_id}: LLM-based selection ({best_name})")
            else:
                # 没有LLM，回退到加权选择
                best_name = self._select_by_weighted_score(segments)
                best_seg = segments[best_name]
                logger.debug(f"Segment {seg_id}: fallback selection ({best_name})")

        # 使用改进后的文本（如果有）
        final_text = improved_text if improved_text else best_seg.text
        seg_start, seg_end = self._get_segment_time_bounds(best_seg, time_start, time_end)

        return ASRSegment(
            id=seg_id,
            text=final_text,
            start=seg_start,
            end=seg_end,
            confidence=best_seg.confidence,
            words=best_seg.words,
        )

    def _select_by_weighted_score(self, segments: Dict[str, ASRSegment]) -> str:
        """基于加权置信度选择最佳结果"""
        scores = {}
        for name, seg in segments.items():
            weight = self.config.model_weights.get(name, 0.5)
            # 加权分数 = 模型权重 * 置信度
            scores[name] = weight * seg.confidence

        return max(scores, key=scores.get)

    def _llm_select_best(self, texts: Dict[str, str]) -> Tuple[str, Optional[str]]:
        """
        使用本地LLM选择最佳文本

        Returns:
            (selected_model_name, improved_text_or_none)
        """
        if not self.llm_client:
            first_key = list(texts.keys())[0]
            return first_key, None

        # 构建prompt
        system_prompt, user_message = self._build_selection_prompt(texts)

        try:
            result = self.llm_client.chat_completion(
                system_prompt=system_prompt,
                user_message=user_message,
                output_schema=ASRSelectionResult,
                temperature=self.config.llm_temperature,
                max_tokens=800
            )

            selection = ASRSelectionResult(**result)

            # 验证返回的模型名称是否有效
            if selection.selected_model in texts:
                return selection.selected_model, selection.improved_text
            else:
                # 尝试模糊匹配
                for name in texts.keys():
                    if name.lower() in selection.selected_model.lower():
                        return name, selection.improved_text

            # 如果都没匹配到，返回第一个
            first_key = list(texts.keys())[0]
            return first_key, None

        except Exception as e:
            logger.warning(f"LLM selection failed: {e}, using fallback")
            first_key = list(texts.keys())[0]
            return first_key, None

    def _build_selection_prompt(self, texts: Dict[str, str]) -> Tuple[str, str]:
        """
        构建选择prompt

        Returns:
            (system_prompt, user_prompt)
        """
        model_names = list(texts.keys())
        lines = [f"Model '{name}': \"{text}\"" for name, text in texts.items()]

        system_prompt = """You are an expert speech recognition quality assessor.

Your task is to select the best ASR (Automatic Speech Recognition) result from multiple models.

EVALUATION CRITERIA (in priority order):
1. Semantic completeness - Does it capture the full meaning?
2. Content accuracy - Are the key words correct?
3. Context coherence - Does it make sense in context?

IMPORTANT RULES:
- Preserve the original speech exactly (including disfluencies like "um", "uh")
- Do NOT correct grammar or punctuation
- Do NOT merge or split sentences differently
- Select the version that best captures what was actually said

OUTPUT FORMAT (JSON):
{
    "selected_model": "model_name",
    "confidence": 0.95,
    "reasoning": "brief reason",
    "improved_text": null
}

Note: improved_text should be null unless you can make a clear improvement while preserving the original meaning."""

        user_message = f"""Compare these ASR results for the same speech segment:

{chr(10).join(lines)}

Which model produced the best result? Return your answer as JSON.

Example output:
{{"selected_model": "{model_names[0]}", "confidence": 0.9, "reasoning": "More accurate content words", "improved_text": null}}"""

        return system_prompt, user_message

    @staticmethod
    def _get_segment_time_bounds(
        seg: ASRSegment,
        fallback_start: float,
        fallback_end: float,
    ) -> Tuple[float, float]:
        if seg.words:
            starts = [float(word.get("start", fallback_start)) for word in seg.words if word.get("start") is not None]
            ends = [float(word.get("end", fallback_end)) for word in seg.words if word.get("end") is not None]
            if starts and ends:
                return min(starts), max(ends)
        return seg.start if seg.start is not None else fallback_start, seg.end if seg.end is not None else fallback_end
