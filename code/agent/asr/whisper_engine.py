"""
Whisper ASR Engine
仅保留 Whisper / Faster-Whisper 实现
"""
import gc
import math
import logging
import threading
from typing import Optional, Dict

from agent.asr.base import BaseASREngine, ASRResult, ASRSegment

logger = logging.getLogger(__name__)


class WhisperEngine(BaseASREngine):
    """Whisper/Faster-Whisper ASR引擎"""

    _shared_models = {}
    _shared_model_refs = {}
    _shared_lock = threading.Lock()

    # 支持的模型 (faster-whisper格式)
    SUPPORTED_MODELS = {
        "tiny": "tiny",
        "base": "base",
        "small": "small",
        "medium": "medium",
        "large-v3": "large-v3",
        "large-v2": "large-v2",
        "large": "large-v3",
    }

    def __init__(
        self,
        model: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        decode_config: Optional[dict] = None,
        engine_name: str = "whisper",
    ):
        super().__init__(model, device)
        self.compute_type = compute_type
        self._model = None
        self._decode_config = decode_config or {}
        self.engine_name = engine_name
        self._shared_cache_key = None

    def _cache_key(self):
        model_size = self.SUPPORTED_MODELS.get(self.model, self.model)
        return (model_size, self.device, self.compute_type)

    def load(self):
        """加载Whisper模型"""
        if self._loaded:
            return

        try:
            from faster_whisper import WhisperModel

            model_size = self.SUPPORTED_MODELS.get(self.model, self.model)
            cache_key = self._cache_key()
            with self._shared_lock:
                cached_model = self._shared_models.get(cache_key)
                if cached_model is None:
                    logger.info(f"Loading Whisper model: {model_size}")
                    cached_model = WhisperModel(
                        model_size,
                        device=self.device,
                        compute_type=self.compute_type,
                    )
                    self._shared_models[cache_key] = cached_model
                    self._shared_model_refs[cache_key] = 0
                else:
                    logger.info(f"Reusing Whisper model from cache: {model_size}")

                self._shared_model_refs[cache_key] += 1

            self._model = cached_model
            self._shared_cache_key = cache_key
            self._loaded = True
            logger.info(f"Whisper model loaded: {self.model}")

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    @staticmethod
    def _clear_device_cache():
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def unload(self):
        """释放当前引擎对共享Whisper模型的引用。"""
        if not self._loaded:
            return

        cache_key = self._shared_cache_key
        self._model = None
        self._loaded = False
        self._shared_cache_key = None

        if cache_key is None:
            self._clear_device_cache()
            gc.collect()
            return

        with self._shared_lock:
            if cache_key in self._shared_model_refs:
                self._shared_model_refs[cache_key] -= 1
                if self._shared_model_refs[cache_key] <= 0:
                    self._shared_model_refs.pop(cache_key, None)
                    self._shared_models.pop(cache_key, None)

        gc.collect()
        self._clear_device_cache()

    def reset(self):
        self.unload()

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        vad_filter: bool = False,  # 默认关闭VAD，避免时间戳错乱
        condition_on_previous_text: bool = False,
        **kwargs
    ) -> ASRResult:
        """
        转录音频

        Args:
            audio_path: 音频文件路径
            language: 语言代码 (en/zh/auto)
            vad_filter: 是否使用VAD过滤（默认False）
            condition_on_previous_text: 是否基于前文条件生成（默认False，更准确的时间戳）
        """
        if not self._loaded:
            self.load()

        try:
            logger.info(f"Transcribing: {audio_path}")

            # 自动检测语言
            if language == "auto" or not language:
                language = None

            # 转录
            # Enable word timestamps for precise cue localization
            if 'word_timestamps' not in kwargs:
                kwargs['word_timestamps'] = True

            # 应用解码配置（用于双Whisper差异化）
            # decode_config中的参数优先级高于函数默认参数
            transcribe_kwargs = {
                'vad_filter': vad_filter,
                'condition_on_previous_text': condition_on_previous_text,
                **self._decode_config,
                **kwargs
            }

            segments, info = self._model.transcribe(
                audio_path,
                language=language,
                **transcribe_kwargs
            )

            # 转换为标准格式
            asr_segments = []
            full_text_parts = []

            for i, segment in enumerate(segments):
                words = None
                if hasattr(segment, 'words') and segment.words:
                    words = [
                        {
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                            "probability": getattr(w, 'probability', 0.0)
                        }
                        for w in segment.words
                    ]

                confidence = self._segment_confidence(segment, words)
                asr_seg = ASRSegment(
                    id=i,
                    text=segment.text.strip(),
                    start=segment.start,
                    end=segment.end,
                    confidence=confidence,
                    words=words
                )
                asr_segments.append(asr_seg)
                full_text_parts.append(segment.text.strip())

            # 计算平均置信度
            avg_confidence = self._compute_confidence(asr_segments)

            result = ASRResult(
                text=" ".join(full_text_parts),
                segments=asr_segments,
                language=info.language if info else (language or "unknown"),
                engine=self.engine_name,
                model=self.model,
                confidence_avg=avg_confidence,
            )

            logger.info(
                f"Transcription complete: {len(asr_segments)} segments, "
                f"language={result.language}, confidence={avg_confidence:.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    @staticmethod
    def _segment_confidence(segment, words) -> float:
        """Normalize model confidence to a stable 0-1 range."""
        if words:
            probabilities = [
                max(0.0, min(1.0, float(w.get("probability", 0.0))))
                for w in words
                if w.get("probability") is not None
            ]
            if probabilities:
                return sum(probabilities) / len(probabilities)

        avg_logprob = getattr(segment, "avg_logprob", None)
        if avg_logprob is not None:
            return max(0.0, min(1.0, math.exp(float(avg_logprob))))

        no_speech_prob = getattr(segment, "no_speech_prob", None)
        if no_speech_prob is not None:
            return max(0.0, min(1.0, 1.0 - float(no_speech_prob)))

        return 0.0
