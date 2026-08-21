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
) -> None:
    """Save RF-channel and FM-composite PSDs with four FM regions marked."""

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

    time_limit = min(len(mpx_hz), int(round(0.050 * fs)))
    time_ms = np.arange(time_limit, dtype=np.float64) / fs * 1e3
    axes[2].plot(time_ms, mpx_hz[:time_limit] / 1e3, linewidth=0.7)
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_ylabel("Instantaneous frequency deviation (kHz)")
    axes[2].set_title("First 50 ms of the FM composite waveform")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
