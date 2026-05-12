"""
Unified ASR Pipeline
统一的ASR处理流程
"""
import json
import gc
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import time

from agent.asr.base import BaseASREngine, ASRResult, MultiASREngine
from agent.asr.whisper_engine import WhisperEngine
from agent.core.speaker_assigner import SpeakerAssigner
from agent.core.reconcile import ReconcileEngine
from agent.utils.audio_preprocessor import AudioPreprocessor

logger = logging.getLogger(__name__)


class UnifiedPipelineConfig:
    """统一Pipeline配置"""

    def __init__(
        self,
        language: str = "auto",  # auto/en/zh
        enable_diarization: bool = True,
        device: str = "cuda",
        ollama_config: Optional[Dict[str, Any]] = None,
        force_llm_speaker_assignment_for_zh: bool = True,
        strict_llm_speaker_assignment_for_zh: bool = True,
    ):
        self.language = language
        self.enable_diarization = enable_diarization
        self.device = device
        self.ollama_config = ollama_config
        self.force_llm_speaker_assignment_for_zh = force_llm_speaker_assignment_for_zh
        self.strict_llm_speaker_assignment_for_zh = strict_llm_speaker_assignment_for_zh


class UnifiedASRPipeline:
    """
    统一ASR处理流程

    双Whisper + Reconcile (中英文通用，word-level时间戳)
    """

    ZH_DATASETS_WITHOUT_DIARIZATION = {"mandic", "pdch", "cmdc"}

    def __init__(self, config: Optional[UnifiedPipelineConfig] = None):
        self.config = config or UnifiedPipelineConfig()

        # 初始化组件
        self.audio_preprocessor = AudioPreprocessor(
            enable_diarization=self.config.enable_diarization,
            device=self.config.device,
        )

        self.speaker_assigner = SpeakerAssigner(
            use_diarization=self.config.enable_diarization,
            force_llm_for_zh=self.config.force_llm_speaker_assignment_for_zh,
            strict_llm_for_zh=self.config.strict_llm_speaker_assignment_for_zh,
            ollama_config=self.config.ollama_config,
        )

        self.reconcile_engine = ReconcileEngine()

        # ASR引擎缓存
        self._asr_engines: Dict[str, BaseASREngine] = {}

    def _resolve_language(self, dataset: str) -> str:
        language = self.config.language
        if language != "auto":
            return language
        dataset_lower = dataset.strip().lower()
        if "edaic" in dataset_lower or "e-daic" in dataset_lower:
            return "en"
        return "zh"

    def _should_use_diarization(self, language: str, dataset: str) -> bool:
        if not self.config.enable_diarization:
            return False
        if language == "zh" and dataset.strip().lower() in self.ZH_DATASETS_WITHOUT_DIARIZATION:
            return False
        return True

    def _get_asr_engine(self, language: str) -> BaseASREngine:
        """获取ASR引擎（带缓存）- 双Whisper"""
        cache_key = f"{language}_multi"

        if cache_key not in self._asr_engines:
            # 双Whisper配置 - 使用不同的beam_size和best_of参数
            # Whisper A: beam_size=5, best_of=5 (更稳定)
            # Whisper B: beam_size=3, best_of=3 (更快，可能有不同结果)
            whisper_a = WhisperEngine(
                model="large-v3",
                device=self.config.device,
                engine_name="whisper_stable",
                decode_config={
                    "beam_size": 5,
                    "best_of": 5,
                    "patience": 1.0,
                }
            )
            whisper_b = WhisperEngine(
                model="large-v3",
                device=self.config.device,
                engine_name="whisper_fast",
                decode_config={
                    "beam_size": 3,
                    "best_of": 3,
                    "patience": 1.2,
                }
            )

            engine = MultiASREngine([whisper_a, whisper_b])
            self._asr_engines[cache_key] = engine

        return self._asr_engines[cache_key]

    @staticmethod
    def _clear_device_cache():
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _is_oom_error(error: Exception) -> bool:
        message = str(error).lower()
        return "out of memory" in message or "cuda failed with error out of memory" in message

    def release_working_memory(self):
        """释放一次样本处理后遗留的临时内存。"""
        gc.collect()
        self._clear_device_cache()

    def reset_runtime_state(self):
        """释放可重建的运行时状态，供OOM后恢复。"""
        for engine in self._asr_engines.values():
            if hasattr(engine, "reset"):
                engine.reset()
        if hasattr(self.audio_preprocessor, "reset"):
            self.audio_preprocessor.reset()
        self.release_working_memory()

    def release_resources(self):
        """在阶段结束后释放ASR/diarization资源。"""
        self.reset_runtime_state()

    def process(self, audio_path: str, dataset: str = "unknown") -> Dict[str, Any]:
        """
        处理音频文件

        Args:
            audio_path: 音频文件路径
            dataset: 数据集名称（用于输出）

        Returns:
            统一格式的transcript字典
        """
        audio_path = Path(audio_path)
        dataset_key = dataset.strip().lower()
        if dataset_key in {"cmdc", "pdch"}:
            sample_id = f"{audio_path.parent.name}_{audio_path.stem}"
        else:
            sample_id = audio_path.stem
        start_time = time.time()
        preproc_result = None

        logger.info(f"Processing: {audio_path}")

        try:
            language = self._resolve_language(dataset)
            use_diarization = self._should_use_diarization(language, dataset)
            logger.info("Resolved language=%s, diarization=%s", language, use_diarization)

            # 1. 音频预处理
            logger.info("Stage 1: Audio preprocessing")
            preproc_result = self.audio_preprocessor.preprocess(
                str(audio_path),
                enable_diarization=use_diarization,
            )

            # 2. 复用预处理阶段的说话人分离结果
            diarization = None
            if use_diarization and preproc_result.diarization_segments:
                logger.info("Stage 2: Reusing diarization from preprocessing")
                diarization = [
                    {"start": s.start, "end": s.end, "speaker": s.speaker}
                    for s in preproc_result.diarization_segments
                ]

            # 3. 语言检测
            logger.info(f"Detected language: {language}")

            # 4. ASR识别 (双Whisper + Reconcile)
            logger.info("Stage 3: ASR recognition (Dual Whisper + Reconcile)")
            asr_engine = self._get_asr_engine(language)

            # 获取双ASR结果 (transcribe_all会自动加载引擎)
            asr_results = asr_engine.transcribe_all(
                preproc_result.audio_path,
                language=language,
                vad_filter=False,
            )

            if not asr_results or len(asr_results) == 0:
                raise RuntimeError("ASR failed")

            logger.info("Stage 4: Reconciling ASR results")
            if len(asr_results) > 1:
                asr_result = self.reconcile_engine.reconcile(asr_results)
            else:
                asr_result = asr_results[0]

            # 5. 说话人分配
            logger.info("Stage 5: Speaker assignment")
            asr_result = self.speaker_assigner.assign(
                asr_result,
                diarization=diarization,
                language=language,
                dataset=dataset
            )

            # 6. 格式化输出
            logger.info("Stage 6: Formatting output")
            output = self._format_output(
                sample_id=sample_id,
                dataset=dataset,
                language=language,
                asr_result=asr_result,
                preproc_result=preproc_result,
                processing_time=time.time() - start_time,
                diarization_enabled=use_diarization,
            )

            logger.info(f"Processing complete in {output['processing']['time_sec']:.1f}s")
            return output
        finally:
            if preproc_result and preproc_result.temp_file:
                try:
                    Path(preproc_result.audio_path).unlink(missing_ok=True)
                except Exception:
                    pass
            self.release_working_memory()

    def _format_output(
        self,
        sample_id: str,
        dataset: str,
        language: str,
        asr_result: ASRResult,
        preproc_result: Any,
        processing_time: float,
        diarization_enabled: bool,
    ) -> Dict[str, Any]:
        """格式化输出为统一格式，包含 sentences 和 words"""

        # 提取唯一说话人列表
        speakers = list(set(s.speaker for s in asr_result.segments if s.speaker))
        if not speakers:
            speakers = ["interviewee"]

        # 构建sentences
        sentences = []
        words = []

        for s in asr_result.segments:
            sentence_start = s.start
            sentence_end = s.end
            if s.words:
                word_starts = [float(w.get("start", s.start)) for w in s.words if w.get("start") is not None]
                word_ends = [float(w.get("end", s.end)) for w in s.words if w.get("end") is not None]
                if word_starts and word_ends:
                    sentence_start = min(word_starts)
                    sentence_end = max(word_ends)

            sentence = {
                "id": s.id,
                "text": s.text,
                "speaker": s.speaker or "interviewee",
                "start": round(sentence_start, 4),
                "end": round(sentence_end, 4),
                "confidence": round(s.confidence, 4) if s.confidence else None,
            }
            sentences.append(sentence)

            # 提取 words（如果支持 word-level）
            if s.words:
                for w in s.words:
                    word_entry = {
                        "text": w.get("word", ""),
                        "start": round(float(w.get("start", 0)), 4),
                        "end": round(float(w.get("end", 0)), 4),
                        "confidence": round(float(w.get("probability", 0)), 4) if "probability" in w else None,
                        "speaker": s.speaker or "interviewee",
                        "sentence_id": s.id,
                    }
                    words.append(word_entry)

        # 计算总时长
        duration = asr_result.segments[-1].end if asr_result.segments else 0.0

        return {
            "sample_id": sample_id,
            "dataset": dataset,
            "language": language,
            "duration_sec": round(duration, 4),
            "speaker_count": len(speakers),
            "speakers": speakers,
            "sentences": sentences,
            "words": words,  # 新增 word-level 输出
            "asr_info": {
                "engine": asr_result.engine,
                "model": asr_result.model,
                "confidence_avg": round(asr_result.confidence_avg, 4),
                "segment_count": len(asr_result.segments),
                "word_count": len(words),
            },
            "processing": {
                "time_sec": round(processing_time, 2),
                "diarization_enabled": diarization_enabled,
                "reconcile_enabled": True,
            }
        }

    def process_dataset(
        self,
        data_dir: str,
        output_dir: str,
        dataset: str,
        file_pattern: str = "*.wav",
        skip_existing: bool = True,
        max_workers: int = 1,
    ):
        """
        批量处理数据集

        Args:
            data_dir: 数据目录
            output_dir: 输出目录
            dataset: 数据集名称
            file_pattern: 音频文件匹配模式
            skip_existing: 是否跳过已处理的文件
            max_workers: 请求的并行线程数。GPU Whisper/diarization 共享状态下不安全，当前会回退为顺序执行。
        """
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 查找所有音频文件，兼容 .wav / .WAV
        expected_suffix = Path(file_pattern.replace("*", "placeholder")).suffix.lower() or ".wav"
        audio_files = sorted(
            [
                path for path in data_path.rglob("*")
                if path.is_file() and path.suffix.lower() == expected_suffix
            ],
            key=lambda path: str(path),
        )
        logger.info(f"Found {len(audio_files)} audio files in {data_dir}")

        if not audio_files:
            logger.warning("No audio files found")
            return

        # 对于 CMDC / PDCH，需要保留原始目录结构，避免同名音频相互覆盖
        preserve_structure = dataset.strip().lower() in {"cmdc", "pdch"}

        # 添加数据集层级: agent/outputs/E-DAIC/xxx_AUDIO/transcript.json
        output_path = output_path / dataset
        output_path.mkdir(parents=True, exist_ok=True)

        # 过滤已处理的文件
        if skip_existing:
            todo_files = []
            for f in audio_files:
                if preserve_structure:
                    # CMDC / PDCH: 保持原始目录结构
                    rel_path = f.relative_to(data_path)
                    out_file = output_path / rel_path.parent / f"{f.stem}_transcript.json"
                else:
                    # E-DAIC/ManDIC: 使用 sample_id 结构 (如 300_AUDIO)
                    sample_id = f.stem
                    out_file = output_path / sample_id / "transcript.json"
                if not out_file.exists():
                    todo_files.append(f)
            logger.info(f"{len(todo_files)}/{len(audio_files)} files need processing")
            audio_files = todo_files

        if max_workers != 1:
            logger.warning(
                "Requested max_workers=%s, but the pipeline uses shared GPU models and diarization state. "
                "Falling back to sequential processing for correctness.",
                max_workers,
            )

        success = 0
        failed = 0

        def process_one(audio_file, idx, total):
            if preserve_structure:
                sample_id = f"{audio_file.parent.name}_{audio_file.stem}"
            else:
                sample_id = audio_file.stem

            if preserve_structure:
                rel_path = audio_file.relative_to(data_path)
                sample_output_dir = output_path / rel_path.parent
                output_file = sample_output_dir / f"{audio_file.stem}_transcript.json"
            else:
                sample_output_dir = output_path / sample_id
                output_file = sample_output_dir / "transcript.json"

            logger.info(f"[{idx}/{total}] Processing {sample_id}")

            try:
                result = self.process(str(audio_file), dataset=dataset)
                sample_output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                logger.info(f"  Saved: {output_file}")
                return True
            except Exception as e:
                if self._is_oom_error(e):
                    logger.warning("  OOM detected for %s, resetting runtime state and retrying once", sample_id)
                    self.reset_runtime_state()
                    try:
                        result = self.process(str(audio_file), dataset=dataset)
                        sample_output_dir.mkdir(parents=True, exist_ok=True)
                        with open(output_file, "w", encoding="utf-8") as f:
                            json.dump(result, f, indent=2, ensure_ascii=False)
                        logger.info(f"  Saved after retry: {output_file}")
                        return True
                    except Exception as retry_error:
                        logger.error(f"  Retry failed: {retry_error}")
                        self.reset_runtime_state()
                        return False
                logger.error(f"  Failed: {e}")
                return False

        for i, audio_file in enumerate(audio_files, 1):
            if process_one(audio_file, i, len(audio_files)):
                success += 1
            else:
                failed += 1

        logger.info(f"\nComplete: {success} succeeded, {failed} failed")
