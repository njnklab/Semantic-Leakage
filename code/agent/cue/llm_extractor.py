"""
LLM-based Cue Extraction
使用LLM进行智能短语提取和QA配对
"""
import json
import logging
import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from agent.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class PhraseExtractionResult(BaseModel):
    """短语提取结果"""
    original_keyword: str = Field(description="原始匹配到的关键词")
    extracted_phrase: str = Field(description="提取的完整短语（包含时态、否定、修饰语）")
    confidence: float = Field(description="提取置信度 0-1")
    reasoning: str = Field(description="提取理由，简要说明")


class QAPairingResult(BaseModel):
    """QA配对结果"""
    is_short_answer: bool = Field(description="是否是简短回答（yes/no/有一点等）")
    inferred_topic: Optional[str] = Field(description="推断的主题，如'自杀想法'、'睡眠问题'等")
    combined_text: str = Field(description="组合后的文本，格式: (主题)回答 或直接回答")
    confidence: float = Field(description="推断置信度 0-1")


class DepressionRelevanceResult(BaseModel):
    """抑郁相关性判断结果"""
    is_depression_related: bool = Field(description="是否与抑郁症状相关（是/否）")
    confidence: float = Field(description="判断置信度 0-1")
    reasoning: str = Field(description="判断理由，简要说明为什么这个短语是或不是抑郁症状")


class LLMCueExtractor:
    """
    使用LLM进行智能Cue提取

    解决两个问题：
    1. 短语边界扩展 - 提取包含关键词的完整语义短语
    2. QA配对 - 理解简短回答的指代主题
    """

    # 定义简短回答模式
    SHORT_ANSWER_PATTERNS = [
        # English
        "yes", "no", "yeah", "yep", "nope", "maybe", "not really",
        "sure", "right", "correct", "exactly", "yeah yeah",
        "i think so", "i guess", "kind of", "sort of",
        # Chinese
        "有一点", "有时候", "不太清楚", "偶尔", "很少", "经常", "总是", "从不",
        "还好", "不错", "不太好", "很差", "严重", "轻微",
    ]

    # 批量请求的默认批大小
    BATCH_SIZE = 5

    def __init__(self, llm_client: Optional[LLMClient] = None, ollama_config: Optional[Dict] = None, request_delay: float = 0.0):
        """
        Args:
            llm_client: LLM客户端
            ollama_config: Ollama配置
            request_delay: 请求间延迟（秒），本地ollama无需延迟
        """
        if llm_client:
            self.llm_client = llm_client
        elif ollama_config:
            self.llm_client = LLMClient(**ollama_config)
        else:
            self.llm_client = LLMClient()
        self.request_delay = request_delay

    def extract_phrase(
        self,
        sentence_text: str,
        keyword: str,
        language: str = "en"
    ) -> PhraseExtractionResult:
        """
        使用LLM提取包含关键词的完整短语

        Args:
            sentence_text: 完整句子文本
            keyword: 匹配到的关键词（如 "sleep"）
            language: 语言 (en/zh)

        Returns:
            PhraseExtractionResult
        """
        if language == "zh":
            system_prompt = """你是一个短语边界识别专家。你的任务是：
1. 在给定句子中找到目标关键词
2. 提取包含该关键词的完整语义短语（意群）
3. 保留否定词、时态标记、修饰语等

你必须返回以下格式的JSON：
{
    "original_keyword": "原始关键词",
    "extracted_phrase": "提取的完整短语",
    "confidence": 0.9,
    "reasoning": "提取理由"
}

示例：
- 句子: "我最近一直睡不好觉"
- 关键词: "睡眠"
- 输出: {"original_keyword": "睡眠", "extracted_phrase": "一直睡不好觉", "confidence": 0.95, "reasoning": "包含时间状语和否定"}

- 句子: "我没有自杀的想法"
- 关键词: "自杀"
- 输出: {"original_keyword": "自杀", "extracted_phrase": "没有自杀的想法", "confidence": 0.95, "reasoning": "包含否定和宾语"}

只返回JSON，不要其他解释。"""

            user_message = f"""句子: "{sentence_text}"
关键词: "{keyword}"

请提取完整短语并返回JSON格式。"""
        else:
            system_prompt = """You are a phrase boundary extraction expert. Your task is to:
1. Find the target keyword in the given sentence
2. Extract the complete semantic phrase containing that keyword
3. Preserve negations, tense markers, and modifiers

You MUST return JSON in this exact format:
{
    "original_keyword": "the original keyword",
    "extracted_phrase": "the complete extracted phrase",
    "confidence": 0.9,
    "reasoning": "brief explanation"
}

Examples:
- Sentence: "I have been having trouble sleeping lately"
- Keyword: "sleep"
- Output: {"original_keyword": "sleep", "extracted_phrase": "having trouble sleeping lately", "confidence": 0.95, "reasoning": "includes verb phrase and time modifier"}

- Sentence: "I don't have any suicidal thoughts"
- Keyword: "suicide"
- Output: {"original_keyword": "suicide", "extracted_phrase": "don't have any suicidal thoughts", "confidence": 0.95, "reasoning": "includes negation and object"}

Return ONLY valid JSON. No markdown, no explanations."""

            user_message = f"""Sentence: "{sentence_text}"
Keyword: "{keyword}"

Extract the complete phrase and return as JSON."""

        try:
            result = self.llm_client.chat_completion(
                system_prompt=system_prompt,
                user_message=user_message,
                output_schema=PhraseExtractionResult,
                temperature=0.1,
                max_tokens=200
            )
            if self.request_delay > 0:
                time.sleep(self.request_delay)
            return PhraseExtractionResult(**result)
        except Exception as e:
            logger.error(f"Phrase extraction failed: {e}")
            raise RuntimeError(f"LLM phrase extraction failed for keyword '{keyword}': {e}")

    def pair_qa(
        self,
        question_text: str,
        answer_text: str,
        language: str = "en"
    ) -> QAPairingResult:
        """
        配对问题和回答，推断简短回答的指代主题

        Args:
            question_text: 医生的问题文本
            answer_text: 患者的回答文本
            language: 语言 (en/zh)

        Returns:
            QAPairingResult
        """
        if language == "zh":
            system_prompt = """你是一个对话理解专家。分析医患对话，判断患者回答是否指代问题中的特定主题。

任务：
1. 判断患者回答是否是简短回答（如"有一点"、"是的"、"偶尔"等）
2. 如果是简短回答，从问题中提取患者所指代的主题（如"自杀想法"、"睡眠问题"、"焦虑情绪"等）
3. 生成组合文本，格式: (主题)回答

你必须返回以下格式的JSON：
{
    "is_short_answer": true/false,
    "inferred_topic": "推断的主题",
    "combined_text": "(主题)回答 或原样",
    "confidence": 0.9
}

抑郁相关主题类别：
- 自杀想法
- 睡眠问题
- 情绪低落
- 兴趣丧失
- 疲劳乏力
- 食欲变化
- 自责内疚
- 焦虑紧张
- 注意力不集中
- 社交退缩

示例1:
- 问题: "你最近有自杀的想法吗？"
- 回答: "有一点"
- 输出: {"is_short_answer": true, "inferred_topic": "自杀想法", "combined_text": "(自杀想法)有一点", "confidence": 0.95}

示例2:
- 问题: "你睡眠怎么样？"
- 回答: "不太好，经常失眠"
- 输出: {"is_short_answer": false, "inferred_topic": null, "combined_text": "不太好，经常失眠", "confidence": 0.9}

只返回JSON，不要其他解释。"""

            user_message = f"""问题: "{question_text}"
回答: "{answer_text}"

请分析并返回JSON格式。"""
        else:
            system_prompt = """You are a dialogue understanding expert. Analyze doctor-patient conversations to determine what topic a short answer refers to.

Task:
1. Determine if the patient's answer is a short/elliptical answer (e.g., "yes", "a little", "sometimes", "not really")
2. If it is a short answer, extract the topic from the question that the patient is referring to (e.g., "suicidal thoughts", "sleep problems", "depressed mood")
3. Generate combined text in format: (topic)answer

You MUST return JSON in this exact format:
{
    "is_short_answer": true/false,
    "inferred_topic": "inferred topic or null",
    "combined_text": "(topic)answer or original",
    "confidence": 0.9
}

Depression-related topic categories:
- suicidal ideation
- sleep problems
- depressed mood
- loss of interest
- fatigue
- appetite changes
- guilt/worthlessness
- anxiety
- concentration issues
- social withdrawal
- physical symptoms

Example 1:
- Question: "Have you had thoughts of hurting yourself recently?"
- Answer: "A little bit"
- Output: {"is_short_answer": true, "inferred_topic": "suicidal ideation", "combined_text": "(suicidal ideation) a little bit", "confidence": 0.95}

Example 2:
- Question: "How has your sleep been?"
- Answer: "Not great, I keep waking up at night"
- Output: {"is_short_answer": false, "inferred_topic": null, "combined_text": "Not great, I keep waking up at night", "confidence": 0.9}

Return ONLY valid JSON. No markdown, no explanations."""

            user_message = f"""Question: "{question_text}"
Answer: "{answer_text}"

Analyze and return JSON format."""

        try:
            result = self.llm_client.chat_completion(
                system_prompt=system_prompt,
                user_message=user_message,
                output_schema=QAPairingResult,
                temperature=0.1,
                max_tokens=200
            )
            if self.request_delay > 0:
                time.sleep(self.request_delay)
            return QAPairingResult(**result)
        except Exception as e:
            logger.error(f"QA pairing failed: {e}")
            raise RuntimeError(f"LLM QA pairing failed for answer '{answer_text}': {e}")

    def check_depression_relevance(
        self,
        phrase: str,
        keyword: str,
        category: str,
        language: str = "en"
    ) -> DepressionRelevanceResult:
        """
        判断提取的短语是否真正表示抑郁症状

        Args:
            phrase: 提取的完整短语
            keyword: 原始匹配的关键词
            category: 抑郁类别 (sleep/mood/suicide等)
            language: 语言 (en/zh)

        Returns:
            DepressionRelevanceResult
        """
        if language == "zh":
            system_prompt = """你是一个抑郁症状识别专家。你的任务是判断给定的短语是否真正表示抑郁症状。

判断标准：
1. 与抑郁相关：短语必须表示负面情绪、症状或困扰（如"睡不着"、"感到绝望"、"没有食欲"）
2. 与抑郁无关：短语表示正面情绪、正常状态或无关内容（如"睡得不错"、"很期待"、"梦想的工作"）

注意：
- "dream"在"your dream job"中不是睡眠问题，只是提到"梦想"
- "sleep"在"I sleep pretty good"中与睡眠问题无关
- "fun"通常与兴趣丧失无关，除非明确表示"觉得什么都没趣"

你必须返回以下格式的JSON：
{
    "is_depression_related": true/false,
    "confidence": 0.9,
    "reasoning": "判断理由"
}

示例1:
- 短语: "一直睡不好觉"
- 关键词: "睡眠"
- 类别: "sleep"
- 输出: {"is_depression_related": true, "confidence": 0.95, "reasoning": "明确表示睡眠困扰，是失眠症状"}

示例2:
- 短语: "睡得挺好的"
- 关键词: "睡眠"
- 类别: "sleep"
- 输出: {"is_depression_related": false, "confidence": 0.95, "reasoning": "表示睡眠质量好，不是睡眠问题"}

示例3:
- 短语: "你的梦想工作"
- 关键词: "梦"
- 类别: "sleep"
- 输出: {"is_depression_related": false, "confidence": 0.95, "reasoning": "这里的梦指梦想/理想，不是睡眠相关的梦或噩梦"}

只返回JSON，不要其他解释。"""

            user_message = f"""短语: "{phrase}"
关键词: "{keyword}"
类别: "{category}"

请判断是否与抑郁症状相关并返回JSON格式。"""
        else:
            system_prompt = """You are a depression symptom recognition expert. Your task is to determine if a given phrase truly represents a depression symptom.

Criteria:
1. Depression-related: The phrase must express negative emotions, symptoms, or distress (e.g., "can't sleep", "feeling hopeless", "lost appetite")
2. Not depression-related: The phrase expresses positive emotions, normal states, or irrelevant content (e.g., "sleep well", "looking forward to", "dream job")

Important Notes:
- "dream" in "your dream job" is NOT a sleep issue, it just mentions "dream/aspiration"
- "sleep" in "I sleep pretty good" is NOT a sleep problem
- "fun" is usually unrelated to loss of interest unless it explicitly says "nothing is fun anymore"
- "tired" after physical activity is different from fatigue from depression

You MUST return JSON in this exact format:
{
    "is_depression_related": true/false,
    "confidence": 0.9,
    "reasoning": "brief explanation"
}

Example 1:
- Phrase: "having trouble sleeping at night"
- Keyword: "sleep"
- Category: "sleep"
- Output: {"is_depression_related": true, "confidence": 0.95, "reasoning": "Clearly expresses sleep difficulty, an insomnia symptom"}

Example 2:
- Phrase: "I get a good night's sleep"
- Keyword: "sleep"
- Category: "sleep"
- Output: {"is_depression_related": false, "confidence": 0.95, "reasoning": "Indicates good sleep quality, not a sleep problem"}

Example 3:
- Phrase: "your dream job"
- Keyword: "dream"
- Category: "sleep"
- Output: {"is_depression_related": false, "confidence": 0.95, "reasoning": "Here 'dream' refers to aspiration/career goal, not sleep-related dreams or nightmares"}

Example 4:
- Phrase: "be fun"
- Keyword: "fun"
- Category: "interest"
- Output: {"is_depression_related": false, "confidence": 0.9, "reasoning": "Just mentions 'fun' in a general context, not indicating loss of interest in activities"}

Return ONLY valid JSON. No markdown, no explanations."""

            user_message = f"""Phrase: "{phrase}"
Keyword: "{keyword}"
Category: "{category}"

Determine if this represents a depression symptom and return JSON format."""

        try:
            result = self.llm_client.chat_completion(
                system_prompt=system_prompt,
                user_message=user_message,
                output_schema=DepressionRelevanceResult,
                temperature=0.1,
                max_tokens=200
            )
            if self.request_delay > 0:
                time.sleep(self.request_delay)
            return DepressionRelevanceResult(**result)
        except Exception as e:
            logger.error(f"Depression relevance check failed: {e}")
            # Default to keeping the cue if check fails
            return DepressionRelevanceResult(
                is_depression_related=True,
                confidence=0.5,
                reasoning="Check failed, defaulting to keep"
            )

    def _is_short_answer(self, text: str) -> bool:
        """快速判断是否是简短回答（规则回退）"""
        text_lower = text.lower().strip()
        # 检查是否是预定义的简短模式
        for pattern in self.SHORT_ANSWER_PATTERNS:
            if pattern in text_lower or text_lower in pattern:
                return True
        # 长度判断：少于5个词可能是简短回答
        words = text_lower.split()
        return len(words) <= 5

    def _batch_phrase_and_relevance(
        self,
        items: List[Dict[str, str]],
        language: str,
    ) -> List[Dict[str, Any]]:
        """
        将多个cue的短语提取+抑郁相关性检查合并为一次LLM调用。

        Args:
            items: [{"keyword": ..., "sentence": ..., "category": ...}, ...]
            language: 语言

        Returns:
            [{"extracted_phrase": ..., "confidence": ..., "reasoning": ...,
              "is_depression_related": ..., "relevance_reasoning": ...}, ...]
        """
        if not items:
            return []

        # 构建批量prompt
        if language == "zh":
            system_prompt = """你是短语提取和抑郁症状识别专家。对于每个条目，你需要：
1. 提取包含关键词的完整语义短语
2. 判断该短语是否真正表示抑郁症状（注意排除正面表述和非临床用法）

返回JSON数组，每个元素格式：
{"extracted_phrase":"完整短语","confidence":0.9,"reasoning":"理由","is_depression_related":true,"relevance_reasoning":"相关性理由"}

只返回JSON数组，不要其他内容。"""
        else:
            system_prompt = """You are an expert in phrase extraction and depression symptom recognition. For each item:
1. Extract the complete semantic phrase containing the keyword
2. Judge whether it truly represents a depression symptom (exclude positive statements and non-clinical uses)

Return a JSON array, each element:
{"extracted_phrase":"...","confidence":0.9,"reasoning":"...","is_depression_related":true,"relevance_reasoning":"..."}

Return ONLY the JSON array."""

        entries = []
        for idx, it in enumerate(items):
            if language == "zh":
                entries.append(f"{idx+1}. 句子:\"{it['sentence']}\" 关键词:\"{it['keyword']}\" 类别:{it['category']}")
            else:
                entries.append(f"{idx+1}. Sentence:\"{it['sentence']}\" Keyword:\"{it['keyword']}\" Category:{it['category']}")
        user_message = "\n".join(entries)

        try:
            result = self.llm_client.chat_completion(
                system_prompt=system_prompt,
                user_message=user_message,
                output_schema=None,
                temperature=0.1,
                max_tokens=250 * len(items),
            )
            content = result.get("content", "")
            parsed = json.loads(content)
            if isinstance(parsed, list) and len(parsed) == len(items):
                return parsed
            # 长度不匹配时逐个回退
            logger.warning(f"Batch response length mismatch: expected {len(items)}, got {len(parsed) if isinstance(parsed, list) else 'non-list'}")
        except Exception as e:
            logger.warning(f"Batch phrase+relevance failed, falling back to single: {e}")
        # 回退：逐个调用
        results = []
        for it in items:
            try:
                pr = self.extract_phrase(it["sentence"], it["keyword"], language)
                rel = self.check_depression_relevance(pr.extracted_phrase, it["keyword"], it["category"], language)
                results.append({
                    "extracted_phrase": pr.extracted_phrase,
                    "confidence": pr.confidence,
                    "reasoning": pr.reasoning,
                    "is_depression_related": rel.is_depression_related,
                    "relevance_reasoning": rel.reasoning,
                })
            except Exception:
                results.append({
                    "extracted_phrase": it["keyword"],
                    "confidence": 0.5,
                    "reasoning": "fallback",
                    "is_depression_related": True,
                    "relevance_reasoning": "fallback",
                })
        return results

    def batch_extract_phrases(
        self,
        cues: List[Dict[str, Any]],
        sentences: List[Dict[str, Any]],
        language: str = "en",
        inter_request_delay: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        批量提取短语并检查抑郁相关性，合并为批量LLM请求以加速。
        """
        # 如果没有sentences，使用context作为回退（逐个）
        if not sentences:
            logger.warning("No sentences provided for phrase extraction, using context")
            updated_cues = []
            for cue in cues:
                sentence_text = cue.get("context", "")
                keyword = cue.get("text", "")
                if not sentence_text:
                    updated_cues.append(cue)
                    continue
                result = self.extract_phrase(sentence_text, keyword, language)
                updated_cue = cue.copy()
                if result.confidence >= 0.6 and result.extracted_phrase != keyword:
                    updated_cue["text"] = result.extracted_phrase
                    updated_cue["original_keyword"] = keyword
                    updated_cue["extraction_confidence"] = result.confidence
                    updated_cue["extraction_reasoning"] = result.reasoning
                updated_cues.append(updated_cue)
            return updated_cues

        sentence_map = {s.get("id", 0): s.get("text", "") for s in sentences}

        # 收集需要LLM处理的cue和直接通过的cue
        pending = []  # (index_in_cues, cue, sentence_text, keyword, category)
        passthrough = []  # (index_in_cues, cue)
        for i, cue in enumerate(cues):
            sentence_id = cue.get("sentence_id", 0)
            sentence_text = sentence_map.get(sentence_id, "")
            keyword = cue.get("text", "")
            if not sentence_text:
                passthrough.append((i, cue))
            else:
                pending.append((i, cue, sentence_text, keyword, cue.get("category", "unknown")))

        # 按 BATCH_SIZE 分批调用
        batch_results = {}  # index -> result dict
        for batch_start in range(0, len(pending), self.BATCH_SIZE):
            batch = pending[batch_start:batch_start + self.BATCH_SIZE]
            items = [{"keyword": kw, "sentence": st, "category": cat} for _, _, st, kw, cat in batch]
            logger.info(f"  Batch phrase+relevance: {batch_start+1}-{batch_start+len(batch)}/{len(pending)}")
            results = self._batch_phrase_and_relevance(items, language)
            for j, (idx, cue, st, kw, cat) in enumerate(batch):
                batch_results[idx] = results[j] if j < len(results) else None

        # 组装结果
        updated_cues = []
        for idx, cue in passthrough:
            updated_cues.append(cue)
        for idx, cue, st, kw, cat in pending:
            r = batch_results.get(idx)
            if r is None:
                updated_cues.append(cue)
                continue
            extracted_phrase = r.get("extracted_phrase", kw)
            conf = r.get("confidence", 0.5)
            if conf < 0.6:
                extracted_phrase = kw

            if not r.get("is_depression_related", True):
                logger.info(f"Filtered out (not depression-related): '{extracted_phrase}' - {r.get('relevance_reasoning', '')}")
                continue

            updated_cue = cue.copy()
            updated_cue["text"] = extracted_phrase
            updated_cue["original_keyword"] = kw
            updated_cue["extraction_confidence"] = conf
            updated_cue["extraction_reasoning"] = r.get("reasoning", "")
            updated_cue["relevance_confidence"] = conf
            updated_cue["relevance_reasoning"] = r.get("relevance_reasoning", "")
            updated_cues.append(updated_cue)
            logger.debug(f"Extracted phrase: '{kw}' -> '{extracted_phrase}'")

        return updated_cues

    def _batch_pair_qa_llm(
        self,
        items: List[Dict[str, str]],
        language: str,
    ) -> List[Dict[str, Any]]:
        """将多个QA配对合并为一次LLM调用。"""
        if not items:
            return []

        if language == "zh":
            system_prompt = """你是对话理解专家。对于每组医患问答，判断回答是否指代问题中的特定抑郁相关主题。
返回JSON数组，每个元素：{"is_short_answer":true,"inferred_topic":"主题","combined_text":"(主题)回答","confidence":0.9}
只返回JSON数组。"""
        else:
            system_prompt = """You are a dialogue understanding expert. For each Q&A pair, determine if the answer refers to a depression-related topic from the question.
Return a JSON array, each element: {"is_short_answer":true,"inferred_topic":"topic","combined_text":"(topic)answer","confidence":0.9}
Return ONLY the JSON array."""

        entries = []
        for idx, it in enumerate(items):
            if language == "zh":
                entries.append(f"{idx+1}. 问题:\"{it['question']}\" 回答:\"{it['answer']}\"")
            else:
                entries.append(f"{idx+1}. Q:\"{it['question']}\" A:\"{it['answer']}\"")
        user_message = "\n".join(entries)

        try:
            result = self.llm_client.chat_completion(
                system_prompt=system_prompt,
                user_message=user_message,
                output_schema=None,
                temperature=0.1,
                max_tokens=200 * len(items),
            )
            content = result.get("content", "")
            parsed = json.loads(content)
            if isinstance(parsed, list) and len(parsed) == len(items):
                return parsed
            logger.warning(f"Batch QA response length mismatch")
        except Exception as e:
            logger.warning(f"Batch QA failed, falling back to single: {e}")

        # 回退：逐个调用
        results = []
        for it in items:
            try:
                r = self.pair_qa(it["question"], it["answer"], language)
                results.append(r.model_dump(mode='json'))
            except Exception:
                results.append({"is_short_answer": False, "inferred_topic": None, "combined_text": it["answer"], "confidence": 0.5})
        return results

    def batch_pair_qa(
        self,
        cues: List[Dict[str, Any]],
        sentences: List[Dict[str, Any]],
        language: str = "en",
        inter_request_delay: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """批量处理QA配对，合并为批量LLM请求以加速。"""
        if not sentences:
            logger.warning("No sentences provided for QA pairing, skipping")
            return cues

        sentence_map = {s.get("id", 0): s for s in sentences}

        # 分类：需要QA配对的 vs 直接通过的
        qa_pending = []  # (index_in_cues, cue, question_text, cue_text)
        updated_cues_map = {}  # index -> cue

        for i, cue in enumerate(cues):
            speaker = cue.get("speaker", "").lower()
            if "interviewee" not in speaker and "patient" not in speaker:
                updated_cues_map[i] = cue
                continue

            cue_text = cue.get("text", "")
            if not self._is_short_answer(cue_text):
                updated_cues_map[i] = cue
                continue

            sentence_id = cue.get("sentence_id", 0)
            if not sentence_map.get(sentence_id):
                updated_cues_map[i] = cue
                continue

            question_text = self._find_pairing_question(sentence_id, sentence_map, sentences)
            if not question_text:
                updated_cues_map[i] = cue
                continue

            qa_pending.append((i, cue, question_text, cue_text))

        # 批量LLM调用
        for batch_start in range(0, len(qa_pending), self.BATCH_SIZE):
            batch = qa_pending[batch_start:batch_start + self.BATCH_SIZE]
            items = [{"question": qt, "answer": ct} for _, _, qt, ct in batch]
            logger.info(f"  Batch QA pairing: {batch_start+1}-{batch_start+len(batch)}/{len(qa_pending)}")
            results = self._batch_pair_qa_llm(items, language)

            for j, (idx, cue, qt, ct) in enumerate(batch):
                updated_cue = cue.copy()
                r = results[j] if j < len(results) else {}
                if r.get("is_short_answer") and r.get("inferred_topic"):
                    updated_cue["text"] = r.get("combined_text", ct)
                    updated_cue["original_answer"] = ct
                    updated_cue["context_question"] = qt
                    updated_cue["inferred_topic"] = r["inferred_topic"]
                    updated_cue["pairing_confidence"] = r.get("confidence", 0.5)
                    logger.debug(f"Paired QA: '{ct}' + '{qt}' -> '{r.get('combined_text', ct)}'")
                updated_cues_map[idx] = updated_cue

        # 按原始顺序输出
        return [updated_cues_map[i] for i in sorted(updated_cues_map.keys())]

    def _find_pairing_question(
        self,
        answer_sentence_id: int,
        sentence_map: Dict[int, Dict],
        sentences: List[Dict]
    ) -> Optional[str]:
        """
        找到与回答配对的问题

        策略：
        1. 检查同sentence_id的words中是否有interviewer的发言
        2. 查找前一个sentence（如果是interviewer）
        """
        # 获取当前sentence
        current = sentence_map.get(answer_sentence_id)
        if not current:
            return None

        # 检查words中是否有interviewer的发言
        words = current.get("words", [])
        interviewer_words = [
            w for w in words
            if "interviewer" in w.get("speaker", "").lower() or "doctor" in w.get("speaker", "").lower()
        ]

        if interviewer_words:
            return "".join(w.get("text", "") for w in interviewer_words)

        # 查找前一个sentence
        prev_sent = None
        for i, sent in enumerate(sentences):
            if sent.get("id") == answer_sentence_id and i > 0:
                prev_sent = sentences[i - 1]
                break

        if prev_sent:
            prev_speaker = prev_sent.get("speaker", "").lower()
            if "interviewer" in prev_speaker or "doctor" in prev_speaker or "ellie" in prev_speaker:
                return prev_sent.get("text", "")

        return None
