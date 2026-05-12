import logging
import os
import tempfile
import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import librosa
import numpy as np
import soundfile as sf


logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """Speaker diarization segment"""
    start: float
    end: float
    speaker: str  # "SPEAKER_00", "SPEAKER_01", etc.
    confidence: float = 0.0


@dataclass
class PreprocessResult:
    audio_path: str
    original_duration: float
    processed_duration: float
    energy_threshold: float
    removed_intervals: List[Tuple[float, float]]
    kept_intervals: List[Tuple[float, float]]
    temp_file: bool = False
    gain_applied: float = 1.0
    diarization_segments: List[SpeakerSegment] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        def _float(v):
            try:
                return float(v)
            except Exception:
                return v

        def _intervals(arr):
            return [(_float(s), _float(e)) for s, e in arr]

        return {
            "audio_path": self.audio_path,
            "original_duration": _float(self.original_duration),
            "processed_duration": _float(self.processed_duration),
            "energy_threshold": _float(self.energy_threshold),
            "removed_intervals": _intervals(self.removed_intervals),
            "kept_intervals": _intervals(self.kept_intervals),
            "temp_file": bool(self.temp_file),
            "gain_applied": _float(self.gain_applied),
            "diarization_segments": [
                {
                    "start": _float(seg.start),
                    "end": _float(seg.end),
                    "speaker": seg.speaker,
                    "confidence": _float(seg.confidence),
                }
                for seg in self.diarization_segments
            ],
        }


class SpeakerDiarization:
    """Speaker diarization using pyannote.audio"""

    def __init__(self, hf_token: Optional[str] = None, device: str = "cuda"):
        self.hf_token = hf_token
        self.device = device
        self._pipeline = None
        self._loaded = False

    def _load_pipeline(self):
        """Lazy load pyannote pipeline"""
        if self._loaded:
            return
        try:
            from pyannote.audio import Pipeline
            import torch
            # Use ModelScope mirror to avoid HF gated repo restrictions
            self._pipeline = Pipeline.from_pretrained(
                "jaman21/pyannote-speaker-diarization-community-1"
            )
            if self._pipeline:
                self._pipeline.to(torch.device(self.device))
            self._loaded = True
            logger.info("Speaker diarization pipeline loaded (community-1)")
        except Exception as e:
            logger.warning(f"Failed to load diarization pipeline: {e}")
            self._loaded = True  # Mark as loaded to avoid retry

    def diarize(self, audio_path: str, min_speakers: int = 1, max_speakers: int = 2) -> List[SpeakerSegment]:
        """
        Run speaker diarization on audio file

        Args:
            audio_path: Path to audio file
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers

        Returns:
            List of speaker segments
        """
        self._load_pipeline()
        if not self._pipeline:
            logger.warning("Diarization pipeline not available, returning single speaker")
            # Return single segment covering full audio
            duration = self._get_duration(audio_path)
            return [SpeakerSegment(start=0.0, end=duration, speaker="SPEAKER_00", confidence=1.0)]

        try:
            logger.info(f"Running speaker diarization on {audio_path}")
            diarization_output = self._pipeline(audio_path, min_speakers=min_speakers, max_speakers=max_speakers)

            # Handle community-1 format (DiarizeOutput) vs old format (Annotation)
            if hasattr(diarization_output, 'speaker_diarization'):
                diarization = diarization_output.speaker_diarization
            else:
                diarization = diarization_output

            segments = []
            # Support both old itertracks API and new formats
            if hasattr(diarization, 'itertracks'):
                # Standard pyannote API
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    segments.append(SpeakerSegment(
                        start=turn.start,
                        end=turn.end,
                        speaker=speaker,
                        confidence=1.0
                    ))
            elif hasattr(diarization, 'iterrows'):
                # DataFrame-like API
                for _, row in diarization.iterrows():
                    segments.append(SpeakerSegment(
                        start=row['start'],
                        end=row['end'],
                        speaker=row['speaker'],
                        confidence=1.0
                    ))
            else:
                # Fallback: try to iterate directly
                for seg in diarization:
                    segments.append(SpeakerSegment(
                        start=getattr(seg, 'start', 0),
                        end=getattr(seg, 'end', 0),
                        speaker=getattr(seg, 'speaker', 'SPEAKER_00'),
                        confidence=1.0
                    ))

            logger.info(f"Diarization complete: {len(segments)} segments, speakers: {set(s.speaker for s in segments)}")
            return segments
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            duration = self._get_duration(audio_path)
            return [SpeakerSegment(start=0.0, end=duration, speaker="SPEAKER_00", confidence=1.0)]

    def _get_duration(self, audio_path: str) -> float:
        """Get audio duration"""
        try:
            info = sf.info(audio_path)
            return info.duration
        except Exception:
            return 0.0

    @staticmethod
    def _clear_device_cache():
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def reset(self):
        """释放 diarization 运行时资源，供 OOM 恢复或阶段切换使用。"""
        self._pipeline = None
        self._loaded = False
        gc.collect()
        self._clear_device_cache()


class AudioPreprocessor:
    """Annotate low-energy intervals and apply light enhancement for ASR (no persistent output)."""

    def __init__(
        self,
        frame_length: int = 2048,
        hop_length: int = 512,
        reference_percentile: float = 92.0,
        energy_ratio: float = 0.25,
        min_silence_sec: float = 0.5,
        max_silence_sec: float = 4.0,
        tail_preserve_sec: float = 0.7,
        min_keep_sec: float = 0.5,
        merge_gap_sec: float = 0.3,
        target_sr: int = 16000,
        target_rms_db: float = -20.0,
        preemphasis_coef: float = 0.97,
        enable_diarization: bool = True,
        hf_token: Optional[str] = None,
        device: str = "cuda",
    ):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.reference_percentile = reference_percentile
        self.energy_ratio = energy_ratio
        self.min_silence_sec = min_silence_sec
        self.max_silence_sec = max_silence_sec
        self.tail_preserve_sec = tail_preserve_sec
        self.min_keep_sec = min_keep_sec
        self.merge_gap_sec = merge_gap_sec
        self.target_sr = target_sr
        self.target_rms_db = target_rms_db
        self.preemphasis_coef = preemphasis_coef
        self.hf_token = hf_token
        self.device = device
        self.enable_diarization = enable_diarization
        self.diarization = SpeakerDiarization(hf_token=hf_token, device=device) if enable_diarization else None

    def preprocess(self, audio_path: str, enable_diarization: Optional[bool] = None) -> PreprocessResult:
        """Detect low-energy regions, apply light enhancement, return temp file path for ASR."""
        logger.info("AudioPreprocessor (enhance+annotate): loading %s", audio_path)
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        original_duration = len(y) / sr if len(y) else 0.0
        effective_enable_diarization = self.enable_diarization if enable_diarization is None else enable_diarization

        if not len(y):
            logger.warning("AudioPreprocessor: empty audio detected, annotate-only skip.")
            return PreprocessResult(
                audio_path=audio_path,
                original_duration=0.0,
                processed_duration=0.0,
                energy_threshold=0.0,
                removed_intervals=[],
                kept_intervals=[],
                temp_file=False,
                gain_applied=1.0,
                diarization_segments=[],
            )

        # Low-energy detection for metadata (on original sample rate)
        rms = librosa.feature.rms(y=y, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        times = librosa.frames_to_time(
            np.arange(len(rms)), sr=sr, hop_length=self.hop_length, n_fft=self.frame_length
        )

        ref_value = np.percentile(rms, self.reference_percentile)
        if ref_value <= 1e-8:
            ref_value = np.max(rms)
        if ref_value <= 1e-8:
            logger.warning("AudioPreprocessor: RMS reference extremely low, skip annotations.")
            removed_intervals: List[Tuple[float, float]] = []
            kept_intervals: List[Tuple[float, float]] = [(0.0, original_duration)]
            threshold = 0.0
        else:
            threshold = float(ref_value * self.energy_ratio)
            low_segments: List[Tuple[float, float]] = []
            in_low = False
            start_time = 0.0
            for time_point, energy in zip(times, rms):
                if energy < threshold:
                    if not in_low:
                        start_time = time_point
                        in_low = True
                else:
                    if in_low:
                        low_segments.append((start_time, time_point))
                        in_low = False
            if in_low:
                low_segments.append((start_time, times[-1] if len(times) else original_duration))

            removal_segments: List[Tuple[float, float]] = []
            for seg_start, seg_end in low_segments:
                seg_end = min(seg_end, original_duration)
                duration = max(0.0, seg_end - seg_start)
                if duration < self.min_silence_sec:
                    continue
                if duration > self.max_silence_sec and duration > self.tail_preserve_sec:
                    adjusted_end = seg_end - self.tail_preserve_sec
                else:
                    adjusted_end = seg_end
                if adjusted_end - seg_start > 1e-3:
                    removal_segments.append((seg_start, adjusted_end))

            keep_segments: List[Tuple[float, float]] = []
            cursor = 0.0
            for seg_start, seg_end in removal_segments:
                seg_start = max(seg_start, 0.0)
                seg_end = min(seg_end, original_duration)
                if seg_start - cursor > self.min_keep_sec:
                    keep_segments.append((cursor, seg_start))
                cursor = max(cursor, seg_end)
            if original_duration - cursor > self.min_keep_sec:
                keep_segments.append((cursor, original_duration))

            kept_intervals = [
                (max(0.0, start), min(original_duration, end))
                for start, end in keep_segments
                if end - start > self.min_keep_sec
            ]
            # Merge intervals that are close together to avoid fragmentation
            kept_intervals = self._merge_close_intervals(kept_intervals, self.merge_gap_sec)
            if not kept_intervals:
                kept_intervals = [(0.0, original_duration)]
                removal_segments = []
            removed_intervals = [(float(s), float(e)) for s, e in removal_segments]

        # Enhancement pipeline (mono) -> resample -> per-speaker loudness normalize -> preemphasis
        if sr != self.target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr

        # Speaker diarization and per-speaker normalization
        speaker_segments: List[SpeakerSegment] = []
        gain_applied = 1.0
        if effective_enable_diarization:
            if self.diarization is None:
                self.diarization = SpeakerDiarization(hf_token=self.hf_token, device=self.device)
            speaker_segments = self.diarization.diarize(audio_path, min_speakers=1, max_speakers=2)
            y = self._normalize_by_speaker(y, sr, speaker_segments)
        else:
            # Global normalization fallback
            y, gain_applied = self._normalize_audio(y, self.target_rms_db)

        # clip to [-1,1]
        y = np.clip(y, -1.0, 1.0)
        # pre-emphasis to sharpen
        y = librosa.effects.preemphasis(y, coef=self.preemphasis_coef)

        # write to temp file for ASR, not persisted in outputs
        fd, tmp_path = tempfile.mkstemp(suffix="_enhanced.wav", prefix="sd_pre_")
        os.close(fd)
        Path(tmp_path).unlink(missing_ok=True)  # ensure clean write via soundfile
        sf.write(tmp_path, y, sr)

        processed_duration = len(y) / sr if len(y) else 0.0
        logger.info(
            "AudioPreprocessor: enhanced audio (gain %.2fx, sr %s) written to temp for ASR", gain_applied, sr
        )

        return PreprocessResult(
            audio_path=str(tmp_path),
            original_duration=original_duration,
            processed_duration=processed_duration,
            energy_threshold=threshold,
            removed_intervals=removed_intervals if 'removed_intervals' in locals() else [],
            kept_intervals=kept_intervals if 'kept_intervals' in locals() else [(0.0, original_duration)],
            temp_file=True,
            gain_applied=gain_applied,
            diarization_segments=speaker_segments,
        )

    def _normalize_by_speaker(self, y: np.ndarray, sr: int, speaker_segments: List[SpeakerSegment]) -> np.ndarray:
        """Apply per-speaker loudness normalization."""
        if not speaker_segments:
            y_norm, _ = self._normalize_audio(y, self.target_rms_db)
            return y_norm

        y_normalized = y.copy()
        speaker_gains = {}

        # Calculate gain per speaker
        for seg in speaker_segments:
            start_sample = int(seg.start * sr)
            end_sample = int(seg.end * sr)
            if start_sample >= len(y) or end_sample <= start_sample:
                continue
            start_sample = max(0, start_sample)
            end_sample = min(len(y), end_sample)

            segment_audio = y[start_sample:end_sample]
            rms = np.sqrt(np.mean(segment_audio ** 2)) if len(segment_audio) > 0 else 0.0
            if rms > 1e-8:
                target_rms = 10 ** (self.target_rms_db / 20.0)
                gain = target_rms / rms
                gain = min(gain, 15.0)  # limit max gain per speaker (15x for better ASR)
                speaker_gains[seg.speaker] = gain
                logger.debug(f"Speaker {seg.speaker}: RMS={rms:.4f}, gain={gain:.2f}")

        if not speaker_gains:
            y_norm, _ = self._normalize_audio(y, self.target_rms_db)
            return y_norm

        # Apply gain per segment
        for seg in speaker_segments:
            if seg.speaker not in speaker_gains:
                continue
            start_sample = int(seg.start * sr)
            end_sample = int(seg.end * sr)
            start_sample = max(0, start_sample)
            end_sample = min(len(y), end_sample)
            y_normalized[start_sample:end_sample] *= speaker_gains[seg.speaker]

        logger.info(f"Per-speaker normalization applied: {len(speaker_gains)} speakers, gains={list(speaker_gains.values())}")
        return y_normalized

    def _normalize_audio(self, y: np.ndarray, target_rms_db: float) -> tuple:
        """Apply global loudness normalization.

        Returns:
            (normalized_audio, gain_applied)
        """
        rms = np.sqrt(np.mean(y ** 2)) if len(y) else 0.0
        if rms <= 1e-8:
            return y, 1.0
        target_rms = 10 ** (target_rms_db / 20.0)
        gain = target_rms / rms
        gain = min(gain, 20.0)  # prevent extreme boost
        logger.info(f"Global normalization: RMS={rms:.4f}, gain={gain:.2f}")
        return y * gain, gain

    @staticmethod
    def _merge_close_intervals(
        intervals: List[Tuple[float, float]],
        gap_threshold: float
    ) -> List[Tuple[float, float]]:
        """Merge intervals that are closer than gap_threshold to reduce fragmentation."""
        if not intervals:
            return intervals
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        merged = [sorted_intervals[0]]
        for start, end in sorted_intervals[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end <= gap_threshold:
                # Merge with previous interval
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        return merged

    def reset(self):
        """释放预处理阶段占用的运行时资源。"""
        if self.diarization:
            self.diarization.reset()


__all__ = ["AudioPreprocessor", "PreprocessResult", "SpeakerDiarization", "SpeakerSegment"]
