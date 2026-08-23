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
    """Configuration for one offline FM analysis window.

    Attributes:
        rf_frequency_hz: Target station frequency in Hz, for example
            ``98.3e6``.
        start_seconds: Analysis start relative to the beginning of the raw
            file. Use together with ``duration_seconds``.
        duration_seconds: Analysis duration when selecting by seconds.
        start_fraction: Start position as a fraction of total file duration.
            Use together with ``stop_fraction`` instead of the seconds mode.
        stop_fraction: Stop position as a fraction of total file duration.
        dtype: NumPy dtype used to interpret the raw file. ``"<i2"`` means
            little-endian signed 16-bit integers.
        padding_seconds: Extra data read before and after the requested window
            to reduce filter-edge effects; it is removed from saved IQ.
        waveform_start_seconds: Start of the third plot relative to the chosen
            analysis window. ``None`` centers the displayed segment.
        waveform_duration_seconds: Duration displayed by the third plot. This
            does not change the data used for either PSD.
        label: Human-readable label placed in plot titles and ``summary.json``.
        ddc: FM DDC sample-rate, filter, and chunk settings.

    Notes:
        Select exactly one file-window mode: either ``start_seconds`` with
        ``duration_seconds``, or ``start_fraction`` with ``stop_fraction``.
    """

    rf_frequency_hz: float
    start_seconds: float | None = None
    duration_seconds: float | None = None
    start_fraction: float | None = None
    stop_fraction: float | None = None
    dtype: str = "<i2"
    padding_seconds: float = 0.020
    waveform_start_seconds: float | None = None
    waveform_duration_seconds: float = 0.050
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
        if (
            self.waveform_start_seconds is not None
            and self.waveform_start_seconds < 0
        ):
            raise ValueError("waveform_start_seconds must be non-negative.")
        if self.waveform_duration_seconds <= 0:
            raise ValueError("waveform_duration_seconds must be positive.")
        np.dtype(self.dtype)

    def resolve_window(self, file_duration_seconds: float) -> Tuple[str, float, float]:
        """Resolve either selection mode to actual seconds in the raw file.

        Args:
            file_duration_seconds: Total duration of the input file.

        Returns:
            ``(selection_mode, start_seconds, duration_seconds)`` where
            ``selection_mode`` is ``"seconds"`` or ``"fraction"``.
        """

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

    def resolve_waveform_window(
        self,
        available_duration_seconds: float,
    ) -> Tuple[float, float]:
        """Resolve the third plot's start and duration within the analysis window.

        Args:
            available_duration_seconds: Duration covered by the available MPX
                samples after discrimination.

        Returns:
            ``(start_seconds, duration_seconds)`` relative to the beginning of
            the selected analysis window. When the configured start is
            ``None``, the requested display duration is centered.
        """

        if available_duration_seconds <= 0:
            raise ValueError("No FM composite samples are available for plotting.")

        display_duration = min(
            self.waveform_duration_seconds,
            available_duration_seconds,
        )
        if self.waveform_start_seconds is None:
            display_start = 0.5 * (
                available_duration_seconds - display_duration
            )
        else:
            display_start = self.waveform_start_seconds
            if display_start >= available_duration_seconds:
                raise ValueError(
                    "waveform_start_seconds must lie inside the selected "
                    "analysis window."
                )
            display_duration = min(
                display_duration,
                available_duration_seconds - display_start,
            )
        return display_start, display_duration


def _json_value(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _compact_path_number(value: float) -> str:
    """Format a path number with up to six decimals and no trailing zeros."""

    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def _prepare_output_dir(
    input_path: Path,
    output_dir: str | Path | None,
    output_root: str | Path,
    rf_frequency_hz: float,
    start_seconds: float,
    duration_seconds: float,
    overwrite: bool,
) -> Path:
    """Create an explicit or automatically named, non-destructive output path.

    Automatic paths use::

        output_root / input_stem / frequencyMHz / start_duration

    For example, 98.33 MHz from 30 s for 3 s becomes
    ``results/file/98.33MHz/30_3``. Existing paths receive ``_run02``,
    ``_run03``, and so on unless ``overwrite`` is true. An explicit
    ``output_dir`` replaces the automatic base path but follows the same
    collision rule.
    """

    if output_dir is None:
        frequency_name = (
            f"{_compact_path_number(rf_frequency_hz / 1e6)}MHz"
        )
        window_name = "{}_{}".format(
            _compact_path_number(start_seconds),
            _compact_path_number(duration_seconds),
        )
        base_dir = (
            Path(output_root).expanduser()
            / input_path.stem
            / frequency_name
            / window_name
        ).resolve()
    else:
        base_dir = Path(output_dir).expanduser().resolve()

    selected_dir = base_dir
    if selected_dir.exists() and not overwrite:
        run_number = 2
        while True:
            candidate = base_dir.with_name(
                f"{base_dir.name}_run{run_number:02d}"
            )
            if not candidate.exists():
                selected_dir = candidate
                break
            run_number += 1

    selected_dir.mkdir(parents=True, exist_ok=overwrite)
    return selected_dir


def analyze_fm_psd_file(
    input_path: str | Path,
    output_dir: str | Path | None,
    config: FMPSDConfig,
    *,
    output_root: str | Path = "results",
    overwrite: bool = False,
) -> Dict[str, object]:
    """Run the complete offline raw-voltage-to-FM-diagnostics pipeline.

    Args:
        input_path: Raw, real-valued voltage file interpreted with
            ``config.dtype``.
        output_dir: Explicit output-directory base. Pass ``None`` to build the
            directory automatically from input filename, frequency, resolved
            start time, and duration.
        config: RF frequency, file-window, waveform-display, and DDC settings.
        output_root: Root used only when ``output_dir`` is ``None``. The default
            is ``results`` relative to the current working directory.
        overwrite: If false, an existing base directory becomes ``_run02``,
            then ``_run03``, etc. If true, fixed result files in the base
            directory are replaced.

    Returns:
        JSON-compatible summary dictionary, including the actual output path,
        resolved file window, carrier offset, and 19 kHz pilot-candidate
        metrics.

    Saves:
        ``summary.json`` with resolved values and metrics; ``run_config.json``
        with requested configuration; ``fm_arrays.npz`` with the complete
        selected IQ, MPX, and MPX PSD arrays; and ``fm_psd.png`` with three
        diagnostic panels.

    Notes:
        The saved IQ and MPX cover the complete selected analysis window. Only
        the third panel is shortened by the waveform display parameters. This
        pipeline does not perform stereo audio decoding or automatic FM
        classification.
    """

    config.validate()
    input_path = Path(input_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

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
    waveform_start, waveform_duration = config.resolve_waveform_window(
        len(mpx_hz) / config.ddc.fs_out
    )
    output_dir = _prepare_output_dir(
        input_path=input_path,
        output_dir=output_dir,
        output_root=output_root,
        rf_frequency_hz=config.rf_frequency_hz,
        start_seconds=requested_start,
        duration_seconds=requested_duration,
        overwrite=overwrite,
    )

    summary: Dict[str, object] = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
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
        "waveform_start_seconds": waveform_start,
        "waveform_duration_seconds": waveform_duration,
        "carrier_offset_hz": carrier_offset_hz,
        "pilot_peak_hz": pilot_peak_hz,
        "pilot_local_contrast_db": pilot_local_contrast_db,
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
        waveform_start_seconds=waveform_start,
        waveform_duration_seconds=waveform_duration,
    )
    return json_summary
