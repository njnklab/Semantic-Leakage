"""
Speaker Assignment Module
为ASR结果分配说话人角色
"""
import logging
import re
from typing import List, Dict, Any, Optional, Literal

from pydantic import BaseModel, Field
from agent.utils.llm_client import LLMClient

from agent.asr.base import ASRResult, ASRSegment

logger = logging.getLogger(__name__)


class SpeakerRoleAssignment(BaseModel):
    id: int = Field(description="Current sentence id")
    speaker: Literal["interviewer", "interviewee"] = Field(description="Assigned role")


class SpeakerRoleBatchResult(BaseModel):
    assignments: List[SpeakerRoleAssignment]


class SpeakerAssigner:
    """
    说话人分配器

    支持两种方式：
    1. Diarization结果：使用pyannote等工具预分离的说话人
    2. 规则启发式：基于文本内容推断（问句=interviewer）
    """

    # 问句关键词（用于启发式判断）
    QUESTION_PATTERNS = {
        "en": [
            r"^(can|could|would|will|do|does|did|have|has|had|are|is|was|were)\s",
            r"^(what|when|where|who|why|how|which)\s",
            r"^(tell me|describe|explain|talk about)",
            r"\?$",  # 以问号结尾
        ],
        "zh": [
            r"^(请|能不能|可以|是否|有没有|是不是|什么|怎么|为什么|谁|哪里|多少)",
            r"(吗|呢|吧|？|\\?|请问|能否)$",  # 支持全角和半角问号
            r"^(说说|描述|谈谈|讲讲|介绍一下)",
        ]
    }

    # 短句标记（通常是interviewer的反馈）
    SHORT_PHRASES = {
        "en": ["ok", "okay", "yes", "yeah", "right", "sure", "great", "thanks", "thank you"],
        "zh": ["好的", "嗯", "对", "是的", "没错", "谢谢", "行", "可以"]
    }

    ZH_LLM_DATASETS = {"mandic", "pdch"}
    SINGLE_SPEAKER_INTERVIEWEE_DATASETS = {"cmdc"}

    def __init__(
        self,
        use_diarization: bool = True,
        llm_client: Optional[LLMClient] = None,
        force_llm_for_zh: bool = True,
        strict_llm_for_zh: bool = True,
        context_window: int = 2,
        llm_batch_size: int = 6,
        llm_window_size: int = 7,
        llm_target_size: int = 5,
        ollama_config: Optional[Dict[str, Any]] = None,
    ):
        self.use_diarization = use_diarization
        self.force_llm_for_zh = force_llm_for_zh
        self.strict_llm_for_zh = strict_llm_for_zh
        self.context_window = context_window
        self.llm_batch_size = llm_batch_size
        self.llm_window_size = max(5, llm_window_size)
        self.llm_target_size = max(1, min(llm_target_size, self.llm_window_size))
        self.llm_client = llm_client or (LLMClient(**ollama_config) if ollama_config or force_llm_for_zh else None)

    def assign(
        self,
        asr_result: ASRResult,
        diarization: Optional[List[Dict[str, Any]]] = None,
        language: str = "en",
        dataset: str = ""
    ) -> ASRResult:
        """
        为ASR结果分配说话人

        Args:
            asr_result: ASR转录结果
            diarization: 预分离的说话人时间段（可选）
            language: 语言代码

        Returns:
            分配了说话人的ASR结果
        """
        dataset_key = dataset.strip().lower()

        if dataset_key in self.SINGLE_SPEAKER_INTERVIEWEE_DATASETS:
            segments = self._assign_all_as_interviewee(asr_result.segments)
        elif self._should_use_llm(language, dataset):
            segments = self._assign_by_llm(asr_result.segments, language, dataset)
        elif diarization and self.use_diarization:
            # 使用diarization结果
            segments = self._assign_by_diarization(asr_result.segments, diarization, language, dataset)
        else:
            # 使用启发式规则
            segments = self._assign_by_heuristic(asr_result.segments, language)

        # 统一角色标签
        for seg in segments:
            seg.speaker = self._normalize_role(seg.speaker)

        # 更新结果
        asr_result.segments = segments
        return asr_result

    @staticmethod
    def _assign_all_as_interviewee(segments: List[ASRSegment]) -> List[ASRSegment]:
        """Single-speaker corpora like CMDC only contain participant responses."""
        for seg in segments:
            seg.speaker = "interviewee"
        return segments

    def _should_use_llm(self, language: str, dataset: str) -> bool:
        return (
            self.force_llm_for_zh
            and language == "zh"
            and dataset.strip().lower() in self.ZH_LLM_DATASETS
        )

    def _ensure_llm_client(self) -> LLMClient:
        if self.llm_client is None:
            self.llm_client = LLMClient()
        return self.llm_client

    def _assign_by_llm(
        self,
        segments: List[ASRSegment],
        language: str,
        dataset_name: str,
    ) -> List[ASRSegment]:
        if not segments:
            return segments

        llm_client = self._ensure_llm_client()
        fallback = self._assign_by_heuristic(list(segments), language)
        fallback_map = {seg.id: self._normalize_role(seg.speaker) for seg in fallback}
        assigned_roles: Dict[int, str] = {}

        system_prompt = (
            "You are assigning dialogue roles in a Chinese clinical interview transcript. "
            "You will receive several local transcript windows. "
            "For each window, assign roles only for the listed target sentence ids. "
            "Each role must be exactly 'interviewer' or 'interviewee'. "
            "Use the surrounding context carefully. "
            "The interviewer usually asks questions, guides topics, confirms details, "
            "or gives brief prompts. The interviewee usually answers, narrates experiences, "
            "describes symptoms, or responds to the interviewer. "
            "Return only JSON."
        )

        try:
            windows = self._build_llm_windows(segments)
            for batch_start in range(0, len(windows), self.llm_batch_size):
                batch_windows = windows[batch_start:batch_start + self.llm_batch_size]
                payload_items = []
                for window_idx, window in enumerate(batch_windows):
                    payload_items.append(
                        {
                            "window_id": batch_start + window_idx,
                            "sentences": [
                                {
                                    "id": seg.id,
                                    "text": seg.text,
                                }
                                for seg in window["sentences"]
                            ],
                            "target_ids": window["target_ids"],
                            "fallback_guess": [
                                {
                                    "id": seg_id,
                                    "speaker": fallback_map.get(seg_id, "interviewee"),
                                }
                                for seg_id in window["target_ids"]
                            ],
                        }
                    )

                user_message = (
                    f"Dataset: {dataset_name}\n"
                    "Assign dialogue roles for the following Chinese transcript windows.\n"
                    "Each window contains 5-7 consecutive sentences. "
                    "Use the full window as context, but only output assignments for the listed target_ids.\n"
                    "Output JSON with field 'assignments', where each item has fields "
                    "'id' and 'speaker'. The 'speaker' value must be exactly "
                    "'interviewer' or 'interviewee'.\n\n"
                    f"{payload_items}"
                )
                result = llm_client.chat_completion(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    output_schema=SpeakerRoleBatchResult,
                    temperature=0.0,
                    max_tokens=2000,
                )
                for item in result.get("assignments", []):
                    seg_id = int(item["id"])
                    assigned_roles[seg_id] = self._normalize_role(item["speaker"])
        except Exception as exc:
            logger.warning("LLM speaker assignment failed for %s: %s", dataset_name, exc)
            if self.strict_llm_for_zh:
                raise RuntimeError(f"LLM speaker assignment failed for {dataset_name}: {exc}") from exc
            return fallback

        for seg in segments:
            seg.speaker = assigned_roles.get(seg.id, fallback_map.get(seg.id, "interviewee"))
        return segments

    def _build_llm_windows(self, segments: List[ASRSegment]) -> List[Dict[str, Any]]:
        if not segments:
            return []

        total = len(segments)
        if total <= self.llm_window_size:
            return [
                {
                    "sentences": segments,
                    "target_ids": [seg.id for seg in segments],
                }
            ]

        windows: List[Dict[str, Any]] = []
        target_start = 0
        while target_start < total:
            target_end = min(total, target_start + self.llm_target_size)
            window_start = max(0, target_start - 1)
            window_end = min(total, window_start + self.llm_window_size)
            if window_end - window_start < self.llm_window_size and window_start > 0:
                window_start = max(0, window_end - self.llm_window_size)

            window_segments = segments[window_start:window_end]
            target_ids = [seg.id for seg in segments[target_start:target_end]]
            windows.append(
                {
                    "sentences": window_segments,
                    "target_ids": target_ids,
                }
            )
            target_start = target_end

        return windows

    def _normalize_role(self, role: Optional[str]) -> str:
        role_lower = str(role or "").strip().lower()
        if role_lower in {"interviewer", "doctor"}:
            return "interviewer"
        if role_lower in {"interviewee", "participant", "patient"}:
            return "interviewee"
        return "interviewee"

    def _assign_by_diarization(
        self,
        segments: List[ASRSegment],
        diarization: List[Dict[str, Any]],
        language: str = "en",
        dataset_name: str = ""
    ) -> List[ASRSegment]:
        """基于diarization结果分配说话人"""
        # 将diarization的speaker标签映射到interviewer/interviewee
        # 假设说话时间最长的为interviewee

        speaker_durations = {}
        for d in diarization:
            spk = d["speaker"]
            duration = d.get("end", 0) - d.get("start", 0)
            speaker_durations[spk] = speaker_durations.get(spk, 0) + duration

        # 找到说话时间最长的speaker
        if speaker_durations:
            main_speaker = max(speaker_durations, key=speaker_durations.get)
            # 对于 ManDIC 数据集，说话时间最短的是 interviewer（医生）
            # 因为医生通常问短句，患者回答较长
            if dataset_name == "ManDIC":
                main_speaker = min(speaker_durations, key=speaker_durations.get)
                speaker_map = {main_speaker: "interviewer"}
                for spk in speaker_durations:
                    if spk != main_speaker:
                        speaker_map[spk] = "interviewee"
            else:
                speaker_map = {main_speaker: "interviewee"}
                for spk in speaker_durations:
                    if spk != main_speaker:
                        speaker_map[spk] = "interviewer"
        else:
            speaker_map = {}

        # 如果只检测到1个speaker，使用启发式规则进一步区分
        if len(speaker_durations) <= 1:
            return self._assign_by_heuristic(segments, language)

        # 为每个segment分配speaker
        for seg in segments:
            # 找到时间重叠的diarization段
            overlap_speakers = {}
            for d in diarization:
                if self._time_overlap(seg.start, seg.end, d.get("start", 0), d.get("end", 0)):
                    spk = d.get("speaker", "unknown")
                    overlap_speakers[spk] = overlap_speakers.get(spk, 0) + 1

            if overlap_speakers:
                # 选择重叠最多的speaker
                best_spk = max(overlap_speakers, key=overlap_speakers.get)
                mapped_speaker = speaker_map.get(best_spk)
                if mapped_speaker:
                    seg.speaker = self._normalize_role(mapped_speaker)
                else:
                    # 映射失败，使用启发式规则
                    seg.speaker = self._heuristic_single_segment(seg, language)
            else:
                # 没有diarization重叠，使用启发式规则
                seg.speaker = self._heuristic_single_segment(seg, language)

        return segments

    def _heuristic_single_segment(self, seg: ASRSegment, language: str) -> str:
        """对单个segment使用启发式规则判断speaker"""
        patterns = self.QUESTION_PATTERNS.get(language, self.QUESTION_PATTERNS["en"])
        text_lower = seg.text.lower().strip()

        # 问句 = interviewer
        if any(re.search(p, text_lower) for p in patterns):
            return "interviewer"

        # 长句 = interviewee
        unit_count = self._count_text_units(text_lower, language)
        long_threshold = 10 if language != "zh" else 16
        if unit_count > long_threshold:
            return "interviewee"

        # 默认假设大部分是interviewee在回答
        return "interviewee"

    def _assign_by_heuristic(self, segments: List[ASRSegment], language: str) -> List[ASRSegment]:
        """基于启发式规则分配说话人（考虑对话交替模式）"""
        if not segments:
            return segments

        patterns = self.QUESTION_PATTERNS.get(language, self.QUESTION_PATTERNS["en"])
        short_phrases = self.SHORT_PHRASES.get(language, self.SHORT_PHRASES["en"])

        # 分析每句话的特征
        seg_info = []
        for i, seg in enumerate(segments):
            text_lower = seg.text.lower().strip()
            unit_count = self._count_text_units(text_lower, language)

            # 检查是否为问句
            is_question = any(re.search(p, text_lower) for p in patterns)
            # 检查是否为短句反馈
            short_threshold = 3 if language != "zh" else 4
            is_short = unit_count <= short_threshold and any(p in text_lower for p in short_phrases)

            seg_info.append({
                'idx': i,
                'unit_count': unit_count,
                'is_question': is_question,
                'is_short': is_short,
            })

        # 基于规则分配speaker（使用交替模式）
        speakers = []
        last_speaker = None

        for info in seg_info:
            i = info['idx']
            text_lower = segments[i].text.lower().strip()

            # 确定当前speaker
            if info['is_question']:
                # 问句 = interviewer
                current_speaker = "interviewer"
            elif info['is_short'] and info['unit_count'] <= (2 if language != "zh" else 3):
                # 非常短的确认词，通常是interviewer
                current_speaker = "interviewer"
            elif info['unit_count'] > (15 if language != "zh" else 24):
                # 长句回答，通常是interviewee
                current_speaker = "interviewee"
            elif last_speaker == "interviewer":
                # 前一个是interviewer，当前很可能是interviewee
                current_speaker = "interviewee"
            elif last_speaker == "interviewee":
                # 前一个是interviewee，当前可能是interviewer
                # 但如果当前较长，可能是interviewee继续说
                if info['unit_count'] > (8 if language != "zh" else 14):
                    current_speaker = "interviewee"
                else:
                    current_speaker = "interviewer"
            else:
                # 默认：第一个是interviewer
                current_speaker = "interviewer"

            speakers.append(current_speaker)
            last_speaker = current_speaker

        # 应用分配
        for i, seg in enumerate(segments):
            seg.speaker = speakers[i]

        # 后处理：修正连续多个interviewee的情况（合并长回答）
        segments = self._post_process(segments, language)

        return segments

    def _post_process(self, segments: List[ASRSegment], language: str) -> List[ASRSegment]:
        """后处理：修正明显错误的分配"""
        if len(segments) < 3:
            return segments

        # 策略1: 基于问句-回答对的验证
        # 如果 interviewee 说了问句，可能是误判
        patterns = self.QUESTION_PATTERNS.get(language, self.QUESTION_PATTERNS["en"])

        for i, seg in enumerate(segments):
            text_lower = seg.text.lower().strip()
            is_question = any(re.search(p, text_lower) for p in patterns)

            if is_question and seg.speaker == "interviewee":
                # Interviewee 通常不会问正式问题，可能是误判
                # 检查上下文来判断
                if i > 0 and segments[i-1].speaker == "interviewee":
                    # 连续interviewee，当前可能是问句，修正为interviewer
                    seg.speaker = "interviewer"

        # 策略2: 检测并修正开头的连续分配错误
        # 如果开头有连续5+个interviewee，可能是起始角色判断错误
        initial_interviewee = 0
        for seg in segments[:10]:
            if seg.speaker == "interviewee":
                initial_interviewee += 1

        # 如果开头interviewee过多，可能需要调整
        # 简单规则：前几句应该是interviewer主导
        if initial_interviewee >= 5:
            # 找到第一个明确的问句之前，标记为interviewer
            for i, seg in enumerate(segments[:10]):
                text_lower = seg.text.lower().strip()
                is_question = any(re.search(p, text_lower) for p in patterns)
                if not is_question and seg.speaker == "interviewee":
                    # 检查是否为短句确认
                    unit_count = self._count_text_units(text_lower, language)
                    if unit_count > (3 if language != "zh" else 6):
                        seg.speaker = "interviewer"

        return segments

    @staticmethod
    def _count_text_units(text: str, language: str) -> int:
        """Count coarse text units for heuristics without breaking Chinese into 1 token."""
        if not text:
            return 0
        if language == "zh":
            compact = re.sub(r"\s+", "", text)
            return len(compact)
        return len(text.split())

    @staticmethod
    def _time_overlap(start1: float, end1: float, start2: float, end2: float) -> bool:
        """检查两个时间段是否重叠"""
        return not (end1 < start2 or end2 < start1)
