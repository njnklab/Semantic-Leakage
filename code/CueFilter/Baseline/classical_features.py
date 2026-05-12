from __future__ import annotations

from typing import Iterable, List

import librosa
import numpy as np


FRAME_LENGTH = 400
HOP_LENGTH = 160
N_FFT = 512
N_MFCC = 13


def _safe_2d(feature: np.ndarray) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32)
    if feature.ndim == 1:
        feature = feature[None, :]
    if feature.size == 0:
        feature = np.zeros((1, 1), dtype=np.float32)
    return np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)


def _summary_stats(feature: np.ndarray) -> np.ndarray:
    feature = _safe_2d(feature)
    mean = feature.mean(axis=1)
    std = feature.std(axis=1)
    return np.concatenate([mean, std], axis=0).astype(np.float32, copy=False)


def extract_handcrafted_features(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return np.zeros(104, dtype=np.float32)

    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak

    try:
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        rms = librosa.feature.rms(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH)
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH)
        flatness = librosa.feature.spectral_flatness(y=audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    except Exception:
        return np.zeros(104, dtype=np.float32)

    rms_vec = _safe_2d(rms).reshape(-1)
    if rms_vec.size == 0:
        silence_ratio = 1.0
    else:
        silence_threshold = 0.05 * float(np.max(rms_vec))
        silence_ratio = float(np.mean(rms_vec <= silence_threshold))

    waveform_stats = np.asarray(
        [
            float(np.mean(np.abs(audio))),
            float(np.std(audio)),
            float(np.max(audio)),
            float(np.min(audio)),
        ],
        dtype=np.float32,
    )

    pieces = [
        waveform_stats,
        _summary_stats(zcr),
        _summary_stats(rms),
        _summary_stats(centroid),
        _summary_stats(bandwidth),
        _summary_stats(rolloff),
        _summary_stats(flatness),
        _summary_stats(chroma),
        _summary_stats(contrast),
        _summary_stats(mfcc),
        _summary_stats(mfcc_delta),
        _summary_stats(mfcc_delta2),
        np.asarray([silence_ratio], dtype=np.float32),
    ]
    features = np.concatenate(pieces, axis=0)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def extract_handcrafted_features_batch(segments: Iterable[np.ndarray], sample_rate: int) -> np.ndarray:
    features: List[np.ndarray] = [extract_handcrafted_features(seg, sample_rate) for seg in segments]
    if not features:
        return np.zeros((0, 104), dtype=np.float32)
    return np.stack(features, axis=0).astype(np.float32, copy=False)

