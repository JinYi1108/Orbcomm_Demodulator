"""Diagnostic plotting for one decoded civil-airband AM voice window.

Functions and outputs
---------------------
``save_airband_am_diagnostic_plot(...)``
    Writes one PNG containing channel IQ spectrum, envelope level, audio
    waveform, and audio spectrogram; returns ``None``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_airband_am_diagnostic_plot(
    *,
    output_path: str | Path,
    label: str,
    tuned_frequency_hz: float,
    carrier_offset_hz: float,
    carrier_frequencies_hz: np.ndarray,
    carrier_psd: np.ndarray,
    envelope: np.ndarray,
    iq_sample_rate_hz: float,
    audio: np.ndarray,
    audio_sample_rate_hz: float,
) -> None:
    """Write the four-panel airband channel and audio diagnostic PNG."""

    output_path = Path(output_path)
    envelope = np.asarray(envelope)
    audio = np.asarray(audio)
    frequencies = np.asarray(carrier_frequencies_hz)
    psd = np.asarray(carrier_psd)
    if any(array.ndim != 1 or len(array) == 0 for array in (envelope, audio)):
        raise ValueError("envelope and audio must be non-empty 1-D arrays.")
    if frequencies.shape != psd.shape:
        raise ValueError("carrier frequency and PSD arrays must have equal shapes.")

    fig, axes = plt.subplots(4, 1, figsize=(13, 13))
    spectrum = np.abs(frequencies) <= min(15e3, iq_sample_rate_hz / 2.0)
    axes[0].plot(
        frequencies[spectrum] / 1e3,
        10.0
        * np.log10(
            np.maximum(
                np.asarray(psd[spectrum], dtype=np.float64),
                np.finfo(np.float64).tiny,
            )
        ),
        linewidth=0.8,
    )
    axes[0].axvline(carrier_offset_hz / 1e3, color="tab:red", linestyle="--")
    axes[0].set_xlabel("Offset from tuned frequency (kHz)")
    axes[0].set_ylabel("IQ PSD (dB/Hz)")
    axes[0].set_title(
        f"{label}: {tuned_frequency_hz / 1e6:.6f} MHz channel, "
        f"carrier offset {carrier_offset_hz:+.1f} Hz"
    )
    axes[0].grid(alpha=0.25)

    block_count = max(1, int(round(0.005 * iq_sample_rate_hz)))
    usable = len(envelope) // block_count * block_count
    if usable:
        envelope_blocks = envelope[:usable].reshape(-1, block_count)
        envelope_power = np.mean(envelope_blocks**2, axis=1)
        envelope_times = (
            np.arange(len(envelope_power), dtype=np.float64) + 0.5
        ) * block_count / iq_sample_rate_hz
    else:
        envelope_times = np.arange(len(envelope)) / iq_sample_rate_hz
        envelope_power = envelope**2
    axes[1].plot(
        envelope_times,
        10.0 * np.log10(np.maximum(envelope_power, np.finfo(float).tiny)),
        linewidth=0.8,
    )
    axes[1].set_xlabel("Seconds from requested window start")
    axes[1].set_ylabel("Envelope power (dB)")
    axes[1].set_title("AM channel envelope, 5 ms averages")
    axes[1].grid(alpha=0.25)

    audio_times = np.arange(len(audio), dtype=np.float64) / audio_sample_rate_hz
    axes[2].plot(audio_times, audio, linewidth=0.6)
    axes[2].set_xlabel("Seconds from requested window start")
    axes[2].set_ylabel("Audio amplitude")
    axes[2].set_title("Voice-band AM envelope audio")
    axes[2].grid(alpha=0.25)

    maximum_nfft = min(2048, len(audio))
    nfft = max(32, 2 ** int(np.floor(np.log2(maximum_nfft))))
    axes[3].specgram(
        audio,
        NFFT=nfft,
        Fs=audio_sample_rate_hz,
        noverlap=3 * nfft // 4,
        cmap="magma",
    )
    axes[3].set_ylim(0, min(5e3, audio_sample_rate_hz / 2.0))
    axes[3].set_xlabel("Seconds from requested window start")
    axes[3].set_ylabel("Audio frequency (Hz)")
    axes[3].set_title("Audio spectrogram")

    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
