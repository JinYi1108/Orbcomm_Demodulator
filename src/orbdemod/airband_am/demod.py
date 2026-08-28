"""Envelope demodulation and audio conditioning for civil-airband AM voice.

Functions and outputs
---------------------
``estimate_airband_carrier(iq, sample_rate_hz, search_hz)``
    Measures the strongest carrier near baseband zero; returns
    ``(offset_hz, frequency_axis_hz, linear_psd)``.
``normalize_airband_audio(audio, percentile, target)``
    Applies percentile-based listening normalization; returns
    ``(normalized_float32_audio, original_scale)``.
``demodulate_airband_am(iq, iq_sample_rate_hz, ...)``
    Performs envelope detection, voice-band filtering, resampling, and
    optional normalization; returns ``AirbandAMDemodResult`` containing the
    envelope, float32 audio, audio sample rate, and normalization scale.

Normalization only controls WAV listening level.  It does not repair clipped
ADC samples or reconstruct information lost before this decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Tuple

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class AirbandAMDemodResult:
    """Envelope and conditioned voice audio produced from airband IQ."""

    envelope: np.ndarray
    audio: np.ndarray
    audio_sample_rate_hz: float
    normalization_scale: float


def estimate_airband_carrier(
    iq: np.ndarray,
    sample_rate_hz: float,
    search_hz: float = 5e3,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return carrier offset, sorted frequency bins, and linear IQ PSD."""

    iq = np.asarray(iq)
    if iq.ndim != 1 or len(iq) < 64:
        raise ValueError("iq must be one-dimensional with at least 64 samples.")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")
    if not 0 < search_hz <= sample_rate_hz / 2.0:
        raise ValueError("search_hz must lie in (0, sample_rate_hz / 2].")

    nperseg = min(131_072, len(iq))
    frequencies, psd = signal.welch(
        iq,
        fs=sample_rate_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend=False,
        return_onesided=False,
        scaling="density",
    )
    order = np.argsort(frequencies)
    frequencies = frequencies[order]
    psd = np.asarray(np.real(psd[order]), dtype=np.float64)
    maximum_raw_psd = float(np.max(psd))
    psd = np.maximum(psd, np.finfo(np.float64).tiny)
    search = np.abs(frequencies) <= search_hz
    indices = np.flatnonzero(search)
    if maximum_raw_psd <= np.finfo(np.float64).tiny:
        offset_hz = 0.0
    else:
        offset_hz = float(frequencies[indices[np.argmax(psd[search])]])
    return offset_hz, frequencies, psd


def normalize_airband_audio(
    audio: np.ndarray,
    percentile: float = 99.5,
    target: float = 0.8,
) -> Tuple[np.ndarray, float]:
    """Return listening-normalized float32 audio and its input scale."""

    audio = np.asarray(audio)
    if audio.ndim != 1 or len(audio) == 0:
        raise ValueError("audio must be a non-empty one-dimensional array.")
    if not 0 < percentile <= 100:
        raise ValueError("percentile must lie in (0, 100].")
    if not 0 < target <= 1:
        raise ValueError("target must lie in (0, 1].")
    scale = float(np.percentile(np.abs(audio), percentile))
    if not np.isfinite(scale) or scale <= np.finfo(float).tiny:
        return np.zeros(len(audio), dtype=np.float32), 0.0
    normalized = np.clip(target * audio / scale, -1.0, 1.0)
    return np.asarray(normalized, dtype=np.float32), scale


def demodulate_airband_am(
    iq: np.ndarray,
    iq_sample_rate_hz: float,
    *,
    audio_sample_rate_hz: float = 48e3,
    audio_low_hz: float = 300.0,
    audio_high_hz: float = 4e3,
    normalize: bool = True,
    normalization_percentile: float = 99.5,
    normalization_target: float = 0.8,
) -> AirbandAMDemodResult:
    """Return the AM envelope and voice-band audio recovered from complex IQ."""

    iq = np.asarray(iq)
    if iq.ndim != 1 or len(iq) < 64:
        raise ValueError("iq must be one-dimensional with at least 64 samples.")
    if iq_sample_rate_hz <= 0 or audio_sample_rate_hz <= 0:
        raise ValueError("IQ and audio sample rates must be positive.")
    if not 0 < audio_low_hz < audio_high_hz < iq_sample_rate_hz / 2.0:
        raise ValueError(
            "Expected 0 < audio_low_hz < audio_high_hz < IQ Nyquist."
        )

    envelope = np.asarray(np.abs(iq), dtype=np.float32)
    audio_sos = signal.butter(
        6,
        [audio_low_hz, audio_high_hz],
        btype="bandpass",
        fs=iq_sample_rate_hz,
        output="sos",
    )
    filtered = signal.sosfiltfilt(audio_sos, envelope)

    ratio = Fraction(audio_sample_rate_hz / iq_sample_rate_hz).limit_denominator(
        100_000
    )
    audio = signal.resample_poly(filtered, ratio.numerator, ratio.denominator)
    expected_count = int(round(len(iq) * audio_sample_rate_hz / iq_sample_rate_hz))
    if len(audio) > expected_count:
        audio = audio[:expected_count]
    elif len(audio) < expected_count:
        audio = np.pad(audio, (0, expected_count - len(audio)))

    scale = 1.0
    if normalize:
        audio, scale = normalize_airband_audio(
            audio,
            percentile=normalization_percentile,
            target=normalization_target,
        )
    else:
        audio = np.asarray(audio, dtype=np.float32)
    return AirbandAMDemodResult(
        envelope=envelope,
        audio=audio,
        audio_sample_rate_hz=float(audio_sample_rate_hz),
        normalization_scale=float(scale),
    )
