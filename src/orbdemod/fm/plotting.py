"""Diagnostic spectrum plotting for shallow FM demodulation."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

_cache_root = Path(tempfile.gettempdir()) / "orbdemod-fm-cache"
(_cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
(_cache_root / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def save_fm_psd_plot(
    output_path: str | Path,
    iq: np.ndarray,
    mpx_hz: np.ndarray,
    fs: float,
    mpx_frequencies: np.ndarray,
    mpx_psd: np.ndarray,
    rf_frequency_hz: float,
    label: str,
    waveform_start_seconds: float,
    waveform_duration_seconds: float,
) -> None:
    """Save the three-panel FM diagnostic figure.

    Args:
        output_path: Destination image filename, normally ``fm_psd.png``.
        iq: Complete selected complex FM-channel IQ sequence.
        mpx_hz: Complete selected discriminator output in Hz.
        fs: IQ and MPX sample rate in samples/s.
        mpx_frequencies: Non-negative MPX PSD frequency axis in Hz.
        mpx_psd: Linear MPX power spectral density.
        rf_frequency_hz: Absolute tuned station frequency used in the title.
        label: Human-readable experiment label used in the title.
        waveform_start_seconds: Third-panel start relative to the selected
            analysis window, not relative to the raw file.
        waveform_duration_seconds: Duration displayed only in the third panel.

    Notes:
        Panel 1 computes an IQ PSD from the full selected window. Panel 2 plots
        the supplied full-window MPX PSD and marks the 0--15, 19, 23--53, and
        57 kHz broadcast-FM regions. Panel 3 plots only the requested local MPX
        time segment. The function writes the image and returns ``None``.
    """

    nperseg = min(len(iq), max(2048, int(round(0.020 * fs))))
    rf_frequencies, rf_psd = signal.welch(
        iq,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend=False,
        return_onesided=False,
        scaling="density",
    )
    order = np.argsort(rf_frequencies)
    rf_frequencies = rf_frequencies[order]
    rf_psd = np.maximum(np.real(rf_psd[order]), np.finfo(float).tiny)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11))

    axes[0].plot(rf_frequencies / 1e3, 10.0 * np.log10(rf_psd), linewidth=0.8)
    axes[0].set_xlim(-fs / 2e3, fs / 2e3)
    axes[0].set_xlabel("Frequency offset from tuned station (kHz)")
    axes[0].set_ylabel("PSD (dB/Hz)")
    axes[0].set_title(
        f"{label}: channel IQ centred at {rf_frequency_hz / 1e6:.6f} MHz"
    )
    axes[0].grid(alpha=0.25)

    display_limit_hz = min(90e3, fs / 2.0)
    display_mask = mpx_frequencies <= display_limit_hz
    axes[1].plot(
        mpx_frequencies[display_mask] / 1e3,
        10.0 * np.log10(
            np.maximum(mpx_psd[display_mask], np.finfo(float).tiny)
        ),
        linewidth=0.9,
        color="black",
    )
    axes[1].axvspan(0.1, 15.0, color="tab:blue", alpha=0.10, label="0-15 kHz audio (L+R)")
    axes[1].axvspan(18.8, 19.2, color="tab:red", alpha=0.18, label="19 kHz stereo pilot")
    axes[1].axvspan(23.0, 53.0, color="tab:orange", alpha=0.10, label="23-53 kHz stereo (L-R)")
    axes[1].axvline(38.0, color="tab:orange", linestyle="--", linewidth=1.0, label="38 kHz suppressed centre")
    axes[1].axvspan(56.5, 57.5, color="tab:green", alpha=0.18, label="57 kHz RDS")
    axes[1].set_xlim(0.0, display_limit_hz / 1e3)
    axes[1].set_xlabel("FM composite (MPX) frequency (kHz)")
    axes[1].set_ylabel("PSD (dB/Hz)")
    axes[1].set_title("FM discriminator output: four broadcast-FM regions")
    axes[1].legend(loc="best", fontsize=8, ncol=2)
    axes[1].grid(alpha=0.25)

    waveform_start_sample = int(round(waveform_start_seconds * fs))
    waveform_start_sample = min(waveform_start_sample, len(mpx_hz) - 1)
    waveform_sample_count = max(1, int(round(waveform_duration_seconds * fs)))
    waveform_stop_sample = min(
        waveform_start_sample + waveform_sample_count,
        len(mpx_hz),
    )
    waveform_time_ms = (
        np.arange(waveform_start_sample, waveform_stop_sample, dtype=np.float64)
        / fs
        * 1e3
    )
    axes[2].plot(
        waveform_time_ms,
        mpx_hz[waveform_start_sample:waveform_stop_sample] / 1e3,
        linewidth=0.7,
    )
    axes[2].set_xlabel("Time within selected analysis window (ms)")
    axes[2].set_ylabel("Instantaneous frequency deviation (kHz)")
    axes[2].set_title(
        "FM composite waveform: {:.1f}--{:.1f} ms".format(
            waveform_start_sample / fs * 1e3,
            waveform_stop_sample / fs * 1e3,
        )
    )
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
