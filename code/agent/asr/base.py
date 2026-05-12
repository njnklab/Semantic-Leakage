"""
ASR Engine Base Classes
统一ASR引擎接口
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class ASRSegment:
    """ASR转录片段"""
    id: int
    text: str
    start: float
    end: float
    confidence: float = 0.0
    speaker: Optional[str] = None
    words: Optional[List[Dict[str, Any]]] = None


@dataclass
class ASRResult:
    """ASR转录结果"""
    text: str
    segments: List[ASRSegment]
    language: str = ""
    engine: str = ""
    model: str = ""
    confidence_avg: float = 0.0
    raw_output: Optional[Dict] = None


class BaseASREngine(ABC):
    """ASR引擎基类"""

    def __init__(self, model: str, device: str = "cuda"):
        self.model = model
        self.device = device
        self._loaded = False

    @abstractmethod
    def load(self):
        """加载模型"""
        pass

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        **kwargs
    ) -> ASRResult:
        """
        转录音频

        Args:
            audio_path: 音频文件路径
            language: 语言代码 (en/zh/auto)
            **kwargs: 额外参数

        Returns:
            ASRResult
        """
        pass

    def _compute_confidence(self, segments: List[ASRSegment]) -> float:
        """计算平均置信度"""
        if not segments:
            return 0.0
        return sum(s.confidence for s in segments) / len(segments)

    def reset(self):
        """释放运行时状态，默认无操作。"""
        self._loaded = False


class MultiASREngine:
    """多ASR引擎组合"""

    def __init__(self, engines: List[BaseASREngine], weights: Optional[List[float]] = None):
        """
        Args:
            engines: ASR引擎列表
            weights: 各引擎权重，None表示平均权重
        """
        self.engines = engines
        self.weights = weights or [1.0 / len(engines)] * len(engines)
        assert len(self.engines) == len(self.weights), "Engines and weights must match"

    def transcribe_all(
        self,
        audio_path: str,
        language: Optional[str] = None,
        **kwargs
    ) -> List[ASRResult]:
        """使用所有引擎转录"""
        results = []
        for engine in self.engines:
            if not engine._loaded:
                engine.load()
            result = engine.transcribe(audio_path, language=language, **kwargs)
            results.append(result)
        return results

    def reset(self):
        """重置所有底层引擎的运行时状态。"""
        for engine in self.engines:
            if hasattr(engine, "reset"):
                engine.reset()
