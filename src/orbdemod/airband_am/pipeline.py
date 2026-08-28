"""File pipeline from a known civil-airband RF window to WAV and diagnostics.

Functions and outputs
---------------------
``_compact_path_number(value)``
    Formats one number for safe automatic directory names; returns ``str``.
``_prepare_output_dir(input_path, config, output_dir, output_root, overwrite)``
    Selects and creates a non-destructive result directory; returns ``Path``.
``_write_audio_wav(path, audio, sample_rate_hz, normalized)``
    Writes normalized int16 or unnormalized float32 WAV; returns its dtype name.
``demodulate_airband_am_file(input_path, output_dir, config, ...)``
    Reads exactly one requested raw-file window, performs DDC and AM voice
    decoding, writes ``audio.wav``, ``diagnostic.png``, and ``summary.json``;
    returns the same JSON-serializable summary dictionary.

The requested duration controls the WAV duration.  Padding is used only to
reduce filter-edge effects and is removed from all reported output arrays.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.io import wavfile

from .config import AirbandAMFileConfig
from .ddc import downconvert_airband_am_voltage
from .demod import (
    demodulate_airband_am,
    estimate_airband_carrier,
    normalize_airband_audio,
)
from .metrics import summarize_airband_am
from .plotting import save_airband_am_diagnostic_plot


def _compact_path_number(value: float) -> str:
    """Return a compact decimal string safe for use in an output path."""

    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def _prepare_output_dir(
    input_path: Path,
    config: AirbandAMFileConfig,
    output_dir: str | Path | None,
    output_root: str | Path,
    overwrite: bool,
) -> Path:
    """Create and return a result directory without silently overwriting runs."""

    if output_dir is None:
        frequency = _compact_path_number(config.rf_frequency_hz / 1e6)
        start = _compact_path_number(config.start_seconds)
        duration = _compact_path_number(config.duration_seconds)
        base = Path(output_root).expanduser().resolve() / (
            f"{input_path.stem}_airband_am_{frequency}MHz_"
            f"start{start}s_dur{duration}s"
        )
    else:
        base = Path(output_dir).expanduser().resolve()
    selected = base
    if selected.exists() and not overwrite:
        run_number = 2
        while selected.with_name(f"{base.name}_run{run_number:02d}").exists():
            run_number += 1
        selected = selected.with_name(f"{base.name}_run{run_number:02d}")
    selected.mkdir(parents=True, exist_ok=True)
    return selected


def _write_audio_wav(
    path: Path,
    audio: np.ndarray,
    sample_rate_hz: float,
    normalized: bool,
) -> str:
    """Write a WAV and return ``"int16"`` or ``"float32"`` for the summary."""

    rounded_rate = int(round(sample_rate_hz))
    if not np.isclose(sample_rate_hz, rounded_rate, rtol=0.0, atol=1e-9):
        raise ValueError("WAV output sample rate must be an integer.")
    if normalized:
        wav_data = np.asarray(
            np.round(np.clip(audio, -1.0, 1.0) * 32767.0),
            dtype=np.int16,
        )
        dtype_name = "int16"
    else:
        wav_data = np.asarray(audio, dtype=np.float32)
        dtype_name = "float32"
    wavfile.write(path, rounded_rate, wav_data)
    return dtype_name


def demodulate_airband_am_file(
    input_path: str | Path,
    output_dir: str | Path | None,
    config: AirbandAMFileConfig,
    *,
    output_root: str | Path = "results",
    overwrite: bool = False,
) -> Dict[str, object]:
    """Decode one known airband file window and return its saved-run summary."""

    config.validate()
    input_path = Path(input_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    dtype = np.dtype(config.dtype)
    file_bytes = input_path.stat().st_size
    if file_bytes % dtype.itemsize:
        raise ValueError("Input file size is not a whole number of dtype samples.")
    sample_count = file_bytes // dtype.itemsize
    file_duration = sample_count / config.ddc.fs_in
    requested_stop = config.start_seconds + config.duration_seconds
    if requested_stop > file_duration + 1e-12:
        raise ValueError(
            f"Requested window ends at {requested_stop:.6f} s, but the file "
            f"contains only {file_duration:.6f} s."
        )

    padded_start = max(0.0, config.start_seconds - config.padding_seconds)
    padded_stop = min(file_duration, requested_stop + config.padding_seconds)
    start_sample = int(round(padded_start * config.ddc.fs_in))
    stop_sample = int(round(padded_stop * config.ddc.fs_in))
    raw = np.memmap(input_path, dtype=dtype, mode="r")
    initial_phase = float(
        np.remainder(
            2.0
            * np.pi
            * config.rf_frequency_hz
            * start_sample
            / config.ddc.fs_in,
            2.0 * np.pi,
        )
    )
    padded_iq, _ = downconvert_airband_am_voltage(
        raw[start_sample:stop_sample],
        config.rf_frequency_hz,
        config.ddc,
        initial_phase,
    )
    padded_demod = demodulate_airband_am(
        padded_iq,
        config.ddc.fs_out,
        audio_sample_rate_hz=config.audio_sample_rate_hz,
        audio_low_hz=config.audio_low_hz,
        audio_high_hz=config.audio_high_hz,
        normalize=False,
    )

    trim_seconds = config.start_seconds - padded_start
    iq_start = int(round(trim_seconds * config.ddc.fs_out))
    iq_count = int(round(config.duration_seconds * config.ddc.fs_out))
    audio_start = int(round(trim_seconds * config.audio_sample_rate_hz))
    audio_count = int(round(config.duration_seconds * config.audio_sample_rate_hz))
    iq = np.asarray(padded_iq[iq_start : iq_start + iq_count], dtype=np.complex64)
    envelope = np.asarray(
        padded_demod.envelope[iq_start : iq_start + iq_count], dtype=np.float32
    )
    audio = np.asarray(
        padded_demod.audio[audio_start : audio_start + audio_count],
        dtype=np.float32,
    )
    if len(iq) != iq_count or len(audio) != audio_count:
        raise ValueError("DDC output was shorter than the requested trimmed window.")

    normalization_scale = 1.0
    if config.normalize_audio:
        audio, normalization_scale = normalize_airband_audio(
            audio,
            percentile=config.normalization_percentile,
            target=config.normalization_target,
        )
    carrier_offset, carrier_frequencies, carrier_psd = estimate_airband_carrier(
        iq,
        config.ddc.fs_out,
        search_hz=min(5e3, config.ddc.channel_passband_hz),
    )
    metrics = summarize_airband_am(
        iq=iq,
        envelope=envelope,
        audio=audio,
        iq_sample_rate_hz=config.ddc.fs_out,
        audio_sample_rate_hz=config.audio_sample_rate_hz,
        tuned_frequency_hz=config.rf_frequency_hz,
        carrier_offset_hz=carrier_offset,
        carrier_frequencies_hz=carrier_frequencies,
        carrier_psd=carrier_psd,
        normalization_scale=normalization_scale,
    )

    selected_output = _prepare_output_dir(
        input_path,
        config,
        output_dir,
        output_root,
        overwrite,
    )
    wav_path = selected_output / "audio.wav"
    plot_path = selected_output / "diagnostic.png"
    summary_path = selected_output / "summary.json"
    wav_dtype = _write_audio_wav(
        wav_path,
        audio,
        config.audio_sample_rate_hz,
        config.normalize_audio,
    )
    save_airband_am_diagnostic_plot(
        output_path=plot_path,
        label=config.label,
        tuned_frequency_hz=config.rf_frequency_hz,
        carrier_offset_hz=carrier_offset,
        carrier_frequencies_hz=carrier_frequencies,
        carrier_psd=carrier_psd,
        envelope=envelope,
        iq_sample_rate_hz=config.ddc.fs_out,
        audio=audio,
        audio_sample_rate_hz=config.audio_sample_rate_hz,
    )

    summary: Dict[str, object] = {
        "input_path": str(input_path),
        "output_dir": str(selected_output),
        "label": config.label,
        "requested_start_seconds": float(config.start_seconds),
        "requested_duration_seconds": float(config.duration_seconds),
        "padding_seconds": float(config.padding_seconds),
        "config": asdict(config),
        "metrics": metrics,
        "output_files": {
            "audio_wav": str(wav_path),
            "audio_wav_dtype": wav_dtype,
            "diagnostic_png": str(plot_path),
            "summary_json": str(summary_path),
        },
        "scope": (
            "Known-frequency airband AM voice decoding only; no clipping "
            "diagnosis, carrier search, transmitter identification, or "
            "medium-wave broadcast support."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
