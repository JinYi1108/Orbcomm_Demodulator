"""Shallow broadcast-FM demodulation and MPX spectrum estimation."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import signal


def quadrature_discriminator(iq: np.ndarray, fs: float) -> Tuple[np.ndarray, float]:
    """Recover the FM composite (MPX) waveform from complex channel IQ.

    Args:
        iq: One-dimensional complex FM-channel samples.
        fs: IQ sample rate in samples/s; normally ``FMDDCConfig.fs_out``.

    Returns:
        ``(mpx_hz, carrier_offset_hz)``. ``mpx_hz`` contains one fewer sample
        than ``iq`` and expresses instantaneous frequency deviation in Hz after
        subtracting its mean. ``carrier_offset_hz`` is that removed mean.

    Notes:
        This is FM discrimination, not complete stereo/audio decoding. The MPX
        waveform can still contain 0--15 kHz audio, the 19 kHz pilot,
        23--53 kHz stereo difference, and 57 kHz RDS components.
    """

    iq = np.asarray(iq)
    if iq.ndim != 1 or len(iq) < 2:
        raise ValueError("iq must be one-dimensional and contain at least two samples.")
    if fs <= 0:
        raise ValueError("fs must be positive.")

    phase_difference = np.angle(iq[1:] * np.conj(iq[:-1]))
    instantaneous_frequency_hz = phase_difference * float(fs) / (2.0 * np.pi)
    # Broadcast-FM modulation is AC coupled, so its time average is zero.
    # The average instantaneous frequency therefore estimates tuning offset.
    # A median is not appropriate here: a valid multi-tone MPX waveform can
    # have a non-zero median even when its average frequency is exactly zero.
    carrier_offset_hz = float(np.mean(instantaneous_frequency_hz, dtype=np.float64))
    mpx_hz = instantaneous_frequency_hz - carrier_offset_hz
    return np.asarray(mpx_hz, dtype=np.float32), carrier_offset_hz


def compute_mpx_psd(
    mpx_hz: np.ndarray,
    fs: float,
    segment_seconds: float = 0.050,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate a one-sided Welch PSD of the real FM composite baseband.

    Args:
        mpx_hz: Real MPX waveform returned by
            :func:`quadrature_discriminator`.
        fs: MPX sample rate in samples/s.
        segment_seconds: Desired Welch segment duration. At 240 kS/s the
            default 0.050 s corresponds to 12,000 samples and about 20 Hz
            frequency-bin spacing.

    Returns:
        ``(frequencies_hz, psd)`` for the non-negative frequency axis. ``psd``
        uses SciPy's density scaling and is floored above zero for safe dB
        conversion.

    Notes:
        ``segment_seconds`` controls Welch PSD resolution and averaging. It is
        unrelated to the 50 ms time-domain segment shown in the third plot.
    """

    mpx_hz = np.asarray(mpx_hz)
    if mpx_hz.ndim != 1 or len(mpx_hz) < 256:
        raise ValueError("At least 256 MPX samples are required.")
    if fs <= 0 or segment_seconds <= 0:
        raise ValueError("fs and segment_seconds must be positive.")

    target = max(2048, int(round(segment_seconds * fs)))
    nperseg = min(len(mpx_hz), target)
    frequencies, psd = signal.welch(
        mpx_hz,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )
    return frequencies, np.maximum(psd, np.finfo(np.float64).tiny)


def pilot_peak_summary(
    frequencies: np.ndarray,
    psd: np.ndarray,
) -> Tuple[float, float]:
    """Measure the strongest candidate near the 19 kHz stereo-pilot frequency.

    Args:
        frequencies: Frequency axis in Hz, normally returned by
            :func:`compute_mpx_psd`.
        psd: Linear power spectral density corresponding to ``frequencies``.

    Returns:
        ``(peak_hz, contrast_db)``. The peak is searched from 18.8 to 19.2 kHz
        and compared with the neighboring 18.0--18.7 and 19.3--20.0 kHz
        regions. Both values are NaN if those bands are not covered.

    Notes:
        A strong 19 kHz candidate supports stereo broadcast FM, but this metric
        alone is not an automatic or definitive signal classification.
    """

    frequencies = np.asarray(frequencies)
    psd = np.asarray(psd)
    pilot_mask = (frequencies >= 18.8e3) & (frequencies <= 19.2e3)
    noise_mask = (
        ((frequencies >= 18.0e3) & (frequencies <= 18.7e3))
        | ((frequencies >= 19.3e3) & (frequencies <= 20.0e3))
    )
    if not np.any(pilot_mask) or not np.any(noise_mask):
        return float("nan"), float("nan")

    pilot_indices = np.flatnonzero(pilot_mask)
    peak_index = int(pilot_indices[np.argmax(psd[pilot_mask])])
    peak_hz = float(frequencies[peak_index])
    noise_median = float(np.median(psd[noise_mask]))
    contrast_db = float(
        10.0
        * np.log10(
            max(float(psd[peak_index]), np.finfo(float).tiny)
            / max(noise_median, np.finfo(float).tiny)
        )
    )
    return peak_hz, contrast_db
