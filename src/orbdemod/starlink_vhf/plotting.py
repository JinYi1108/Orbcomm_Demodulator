"""Diagnostic plotting for one Starlink VHF decode window.

Functions and outputs
---------------------
``save_starlink_vhf_diagnostic_plot(output_path, iq, sample_rate_hz, packets, ...)``
    Writes a four-panel PNG containing the channel spectrogram, IQ envelope,
    selected-packet symbols, and dechirp concentration; returns the displayed
    ``(minimum_db, maximum_db)`` magnitude limits.

The figure describes signal and decoder behavior only.  It does not plot a
TLE orbit or claim a specific spacecraft identity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from .lora import LoRaPhysicalPacket


def save_starlink_vhf_diagnostic_plot(
    output_path: str | Path,
    iq: np.ndarray,
    sample_rate_hz: float,
    packets: Sequence[LoRaPhysicalPacket],
    *,
    rf_frequency_hz: float,
    window_start_seconds: float,
    requested_start_seconds: float | None = None,
    requested_stop_seconds: float | None = None,
    packet_index: int = 0,
    magnitude_limits_db: Tuple[float, float] | None = None,
    label: str,
) -> Tuple[float, float]:
    """Write the standard diagnostic PNG and return its magnitude limits."""

    iq = np.asarray(iq)
    if iq.ndim != 1 or iq.size == 0 or not np.iscomplexobj(iq):
        raise ValueError("iq must be a non-empty one-dimensional complex array.")
    if not np.all(np.isfinite(iq)):
        raise ValueError("iq must contain only finite samples.")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive and finite.")
    if not np.isfinite(rf_frequency_hz):
        raise ValueError("rf_frequency_hz must be finite.")
    if not np.isfinite(window_start_seconds):
        raise ValueError("window_start_seconds must be finite.")
    if packet_index < 0:
        raise ValueError("packet_index must be non-negative.")
    if not label.strip():
        raise ValueError("label must not be empty.")
    nperseg = min(512, iq.size)
    noverlap = max(0, nperseg - max(1, nperseg // 8))
    frequencies, times, spectrum = signal.spectrogram(
        iq,
        fs=sample_rate_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        mode="magnitude",
        return_onesided=False,
    )
    frequencies = np.fft.fftshift(frequencies)
    magnitude = np.asarray(np.fft.fftshift(spectrum, axes=0), dtype=np.float64)
    magnitude_db = 20.0 * np.log10(
        np.maximum(magnitude, np.finfo(np.float64).tiny)
    )
    if magnitude_limits_db is None:
        display_floor, display_ceiling = np.percentile(
            magnitude_db, [10.0, 99.7]
        )
    else:
        display_floor, display_ceiling = map(float, magnitude_limits_db)
        if not (
            np.isfinite(display_floor)
            and np.isfinite(display_ceiling)
            and display_floor < display_ceiling
        ):
            raise ValueError(
                "magnitude_limits_db must contain two finite increasing values."
            )
    absolute_stft_times = times + window_start_seconds
    absolute_times = np.arange(iq.size) / sample_rate_hz + window_start_seconds

    fig, axes = plt.subplots(4, 1, figsize=(13, 12), constrained_layout=True)
    mesh = axes[0].pcolormesh(
        absolute_stft_times,
        frequencies / 1e3,
        magnitude_db,
        shading="auto",
        cmap="viridis",
        vmin=display_floor,
        vmax=display_ceiling,
    )
    axes[0].set(
        ylabel="Offset (kHz)",
        title=f"{label}: channel spectrogram near {rf_frequency_hz / 1e6:.6f} MHz",
    )
    fig.colorbar(mesh, ax=axes[0], label="Magnitude (dB, arbitrary reference)")

    envelope_db = 20.0 * np.log10(np.maximum(np.abs(iq), np.finfo(float).tiny))
    axes[1].plot(absolute_times, envelope_db, lw=0.7)
    axes[1].set(ylabel="|IQ| (dB)", title="Channel envelope and decoded packet bounds")
    for bounds_index, packet in enumerate(packets):
        axes[0].axvspan(
            window_start_seconds + packet.preamble_start_seconds,
            window_start_seconds + packet.packet_stop_seconds,
            color="tab:red",
            alpha=0.18,
        )
        axes[1].axvspan(
            window_start_seconds + packet.preamble_start_seconds,
            window_start_seconds + packet.packet_stop_seconds,
            color="tab:red",
            alpha=0.18,
            label="decoded packet" if bounds_index == 0 else None,
        )
    if packets:
        axes[1].legend(loc="upper right")

    if requested_start_seconds is not None and requested_stop_seconds is not None:
        if not (
            np.isfinite(requested_start_seconds)
            and np.isfinite(requested_stop_seconds)
            and requested_start_seconds < requested_stop_seconds
        ):
            raise ValueError("requested plot bounds must be finite and increasing.")
        for axis in axes:
            axis.axvline(requested_start_seconds, color="tab:orange", ls="--", lw=0.8)
            axis.axvline(requested_stop_seconds, color="tab:orange", ls="--", lw=0.8)

    if packet_index < len(packets):
        packet = packets[packet_index]
        symbol_duration = (
            packet.packet_stop_seconds - packet.phy_data_start_seconds
        ) / max(1, packet.symbols.size)
        symbol_times = (
            window_start_seconds
            + packet.phy_data_start_seconds
            + np.arange(packet.symbols.size) * symbol_duration
        )
        step_times = np.append(symbol_times, symbol_times[-1] + symbol_duration)
        step_symbols = np.append(packet.symbols, packet.symbols[-1])
        axes[2].step(step_times, step_symbols, where="post", lw=0.8)
        axes[2].set(
            ylabel="Symbol", title=f"Decoded packet {packet_index}: SF symbols"
        )
        axes[3].plot(symbol_times, packet.concentration_db, lw=0.8)
        axes[3].set(
            ylabel="Peak / median (dB)",
            title=(
                f"Decoded packet {packet_index}: per-symbol dechirp concentration "
                "(not SNR)"
            ),
        )
    else:
        message = (
            "No decodable packet"
            if not packets
            else f"Decoded packet index {packet_index} is unavailable"
        )
        axes[2].text(0.5, 0.5, message, ha="center", va="center", transform=axes[2].transAxes)
        axes[3].text(0.5, 0.5, message, ha="center", va="center", transform=axes[3].transAxes)
        axes[2].set(title="Decoded symbols")
        axes[3].set(title="Dechirp concentration")

    axes[3].set_xlabel(
        "Nominal seconds from raw-file start"
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(Path(output_path), dpi=180)
    plt.close(fig)
    return float(display_floor), float(display_ceiling)
