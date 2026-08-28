"""Basic signal and audio metrics for an already selected airband channel.

Functions and outputs
---------------------
``summarize_airband_am(...)``
    Combines carrier, IQ, envelope, and audio measurements; returns a
    JSON-serializable dictionary.  The carrier contrast is a diagnostic
    spectral contrast, not a calibrated receiver SNR or signal classifier.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def summarize_airband_am(
    *,
    iq: np.ndarray,
    envelope: np.ndarray,
    audio: np.ndarray,
    iq_sample_rate_hz: float,
    audio_sample_rate_hz: float,
    tuned_frequency_hz: float,
    carrier_offset_hz: float,
    carrier_frequencies_hz: np.ndarray,
    carrier_psd: np.ndarray,
    normalization_scale: float,
) -> Dict[str, object]:
    """Return JSON-ready carrier, sample-count, level, and duration metrics."""

    iq = np.asarray(iq)
    envelope = np.asarray(envelope)
    audio = np.asarray(audio)
    carrier_frequencies_hz = np.asarray(carrier_frequencies_hz)
    carrier_psd = np.asarray(carrier_psd)
    if any(array.ndim != 1 or len(array) == 0 for array in (iq, envelope, audio)):
        raise ValueError("iq, envelope, and audio must be non-empty 1-D arrays.")
    if carrier_frequencies_hz.shape != carrier_psd.shape:
        raise ValueError("carrier frequency and PSD arrays must have equal shapes.")

    carrier_index = int(np.argmin(np.abs(carrier_frequencies_hz - carrier_offset_hz)))
    reference = (
        (np.abs(carrier_frequencies_hz) <= 10e3)
        & (np.abs(carrier_frequencies_hz - carrier_offset_hz) >= 500.0)
    )
    if np.any(reference):
        reference_power = float(np.median(carrier_psd[reference]))
        contrast_db = float(
            10.0
            * np.log10(
                float(carrier_psd[carrier_index])
                / max(reference_power, np.finfo(float).tiny)
            )
        )
    else:
        contrast_db = float("nan")

    return {
        "tuned_frequency_hz": float(tuned_frequency_hz),
        "estimated_carrier_offset_hz": float(carrier_offset_hz),
        "estimated_carrier_frequency_hz": float(
            tuned_frequency_hz + carrier_offset_hz
        ),
        "carrier_local_contrast_db": contrast_db,
        "iq_sample_rate_hz": float(iq_sample_rate_hz),
        "iq_sample_count": int(len(iq)),
        "iq_duration_seconds": float(len(iq) / iq_sample_rate_hz),
        "iq_rms": float(np.sqrt(np.mean(np.abs(iq) ** 2))),
        "iq_peak": float(np.max(np.abs(iq))),
        "envelope_mean": float(np.mean(envelope)),
        "envelope_rms": float(np.sqrt(np.mean(envelope**2))),
        "audio_sample_rate_hz": float(audio_sample_rate_hz),
        "audio_sample_count": int(len(audio)),
        "audio_duration_seconds": float(len(audio) / audio_sample_rate_hz),
        "audio_rms": float(np.sqrt(np.mean(audio**2))),
        "audio_peak": float(np.max(np.abs(audio))),
        "audio_normalization_scale": float(normalization_scale),
        "metric_caution": (
            "Carrier contrast is diagnostic only; it is not calibrated SNR or "
            "proof that the channel is an aircraft transmission."
        ),
    }
