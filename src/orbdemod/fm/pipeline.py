"""File-oriented pipeline: raw voltage -> FM-channel IQ -> MPX PSD plot."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .fm_ddc import FMDDCConfig, downconvert_fm_voltage
from .demod import compute_mpx_psd, pilot_peak_summary, quadrature_discriminator


@dataclass(frozen=True)
class FMPSDConfig:
    """Configuration for one offline FM spectrum window."""

    rf_frequency_hz: float
    start_seconds: float | None = None
    duration_seconds: float | None = None
    start_fraction: float | None = None
    stop_fraction: float | None = None
    dtype: str = "<i2"
    padding_seconds: float = 0.020
    label: str = "fm_psd"
    ddc: FMDDCConfig = field(default_factory=FMDDCConfig)

    def validate(self) -> None:
        self.ddc.validate()
        if not 0 < self.rf_frequency_hz < self.ddc.fs_in / 2.0:
            raise ValueError("rf_frequency_hz must lie between 0 and fs_in / 2.")
        seconds_values = (self.start_seconds, self.duration_seconds)
        fraction_values = (self.start_fraction, self.stop_fraction)
        seconds_selected = any(value is not None for value in seconds_values)
        fraction_selected = any(value is not None for value in fraction_values)
        if seconds_selected == fraction_selected:
            raise ValueError(
                "Select exactly one window mode: start_seconds with "
                "duration_seconds, or start_fraction with stop_fraction."
            )
        if seconds_selected:
            if any(value is None for value in seconds_values):
                raise ValueError(
                    "start_seconds and duration_seconds must be provided together."
                )
            if self.start_seconds < 0:
                raise ValueError("start_seconds must be non-negative.")
            if self.duration_seconds <= 0:
                raise ValueError("duration_seconds must be positive.")
        else:
            if any(value is None for value in fraction_values):
                raise ValueError(
                    "start_fraction and stop_fraction must be provided together."
                )
            if not 0.0 <= self.start_fraction < self.stop_fraction <= 1.0:
                raise ValueError(
                    "Expected 0 <= start_fraction < stop_fraction <= 1."
                )
        if self.padding_seconds < 0:
            raise ValueError("padding_seconds must be non-negative.")
        np.dtype(self.dtype)

    def resolve_window(self, file_duration_seconds: float) -> Tuple[str, float, float]:
        """Return selection mode, start time, and duration in seconds."""

        self.validate()
        if file_duration_seconds <= 0:
            raise ValueError("The input file contains no samples.")
        if self.start_seconds is not None:
            requested_start = self.start_seconds
            requested_duration = self.duration_seconds
            selection_mode = "seconds"
        else:
            requested_start = self.start_fraction * file_duration_seconds
            requested_stop = self.stop_fraction * file_duration_seconds
            requested_duration = requested_stop - requested_start
            selection_mode = "fraction"
        requested_stop = requested_start + requested_duration
        if requested_stop > file_duration_seconds + 1e-12:
            raise ValueError(
                f"Requested window ends at {requested_stop:.6f} s, "
                f"but the file contains only {file_duration_seconds:.6f} s."
            )
        return selection_mode, requested_start, requested_duration


def _json_value(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def analyze_fm_psd_file(
    input_path: str | Path,
    output_dir: str | Path,
    config: FMPSDConfig,
) -> Dict[str, object]:
    """Analyze one time-frequency window without making an FM classification."""

    config.validate()
    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = np.dtype(config.dtype)
    raw = np.memmap(input_path, dtype=dtype, mode="r")
    file_duration_seconds = len(raw) / config.ddc.fs_in
    selection_mode, requested_start, requested_duration = config.resolve_window(
        file_duration_seconds
    )
    requested_stop = requested_start + requested_duration

    padded_start = max(0.0, requested_start - config.padding_seconds)
    padded_stop = min(
        file_duration_seconds,
        requested_stop + config.padding_seconds,
    )
    start_sample = int(round(padded_start * config.ddc.fs_in))
    stop_sample = int(round(padded_stop * config.ddc.fs_in))
    raw_window = raw[start_sample:stop_sample]

    initial_phase = np.remainder(
        2.0
        * np.pi
        * config.rf_frequency_hz
        * start_sample
        / config.ddc.fs_in,
        2.0 * np.pi,
    )
    iq_padded, _ = downconvert_fm_voltage(
        raw_window,
        rf_frequency_hz=config.rf_frequency_hz,
        config=config.ddc,
        initial_phase=float(initial_phase),
    )

    trim_start = int(round((requested_start - padded_start) * config.ddc.fs_out))
    requested_samples = int(round(requested_duration * config.ddc.fs_out))
    trim_stop = min(trim_start + requested_samples, len(iq_padded))
    iq = iq_padded[trim_start:trim_stop]
    if len(iq) < 256:
        raise ValueError("The requested window is too short after down-conversion.")

    mpx_hz, carrier_offset_hz = quadrature_discriminator(iq, config.ddc.fs_out)
    frequencies, psd = compute_mpx_psd(mpx_hz, config.ddc.fs_out)
    pilot_peak_hz, pilot_local_contrast_db = pilot_peak_summary(frequencies, psd)

    summary: Dict[str, object] = {
        "input_path": str(input_path),
        "label": config.label,
        "rf_frequency_hz": config.rf_frequency_hz,
        "selection_mode": selection_mode,
        "start_seconds": requested_start,
        "duration_seconds": requested_duration,
        "start_fraction": config.start_fraction,
        "stop_fraction": config.stop_fraction,
        "input_file_duration_seconds": file_duration_seconds,
        "input_dtype": dtype.str,
        "iq_sample_rate_hz": config.ddc.fs_out,
        "carrier_offset_hz": carrier_offset_hz,
        "pilot_peak_hz": pilot_peak_hz,
        "pilot_local_contrast_db": pilot_local_contrast_db,
        "classification": "not_performed",
    }

    json_summary = {key: _json_value(value) for key, value in summary.items()}
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, ensure_ascii=False, indent=2)
    np.savez_compressed(
        output_dir / "fm_arrays.npz",
        iq=iq,
        mpx_hz=mpx_hz,
        mpx_frequencies_hz=frequencies,
        mpx_psd=psd,
        sample_rate_hz=np.array(config.ddc.fs_out),
    )

    from .plotting import save_fm_psd_plot

    save_fm_psd_plot(
        output_dir / "fm_psd.png",
        iq=iq,
        mpx_hz=mpx_hz,
        fs=config.ddc.fs_out,
        mpx_frequencies=frequencies,
        mpx_psd=psd,
        rf_frequency_hz=config.rf_frequency_hz,
        label=config.label,
    )
    return json_summary
