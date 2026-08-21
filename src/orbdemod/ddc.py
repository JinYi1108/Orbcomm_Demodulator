import numpy as np
import scipy.signal as sp_signal
from dataclasses import dataclass
from typing import Literal, Sequence, Tuple


@dataclass(frozen=True)
class DecimationStageConfig:
    """Explicit filter and rate-change requirements for one DDC stage.

    ``passband_hz`` and ``stopband_hz`` are measured from complex baseband
    zero.  Keeping them explicit avoids relying on a fixed cutoff ratio whose
    actual alias rejection changes with every input/output sample-rate pair.
    """

    fs_in: float
    fs_out: float
    passband_hz: float
    stopband_hz: float
    passband_ripple_db: float = 0.25
    stopband_attenuation_db: float = 60.0
    filter_type: Literal["butterworth", "kaiser_fir"] = "kaiser_fir"

    def validate(self) -> None:
        if self.fs_in <= 0 or self.fs_out <= 0:
            raise ValueError("fs_in and fs_out must be positive.")
        if self.fs_in <= self.fs_out:
            raise ValueError("A decimation stage requires fs_in > fs_out.")
        integer_decimation_factor(self.fs_in, self.fs_out)
        if not 0 < self.passband_hz < self.stopband_hz:
            raise ValueError("Expected 0 < passband_hz < stopband_hz.")
        if self.stopband_hz > self.fs_out / 2.0:
            raise ValueError(
                "stopband_hz must not exceed the output Nyquist frequency."
            )
        if self.passband_ripple_db <= 0:
            raise ValueError("passband_ripple_db must be positive.")
        if self.stopband_attenuation_db <= 0:
            raise ValueError("stopband_attenuation_db must be positive.")
        if self.filter_type not in ("butterworth", "kaiser_fir"):
            raise ValueError(f"Unsupported filter_type: {self.filter_type!r}.")

    @property
    def decimation_factor(self) -> int:
        self.validate()
        return integer_decimation_factor(self.fs_in, self.fs_out)


@dataclass(frozen=True)
class FilterResponseReport:
    """Measured effective response of one generated anti-alias filter."""

    filter_type: str
    processing_passes: int
    filter_order: int
    coefficient_count: int
    passband_min_db: float
    passband_max_db: float
    passband_ripple_db: float
    worst_stopband_db: float
    stopband_attenuation_db: float
    coefficients_finite: bool
    stable: bool
    meets_specification: bool
    failure_reasons: Tuple[str, ...]

# ===========================
# 1. Basic Functions
# ===========================



def integer_decimation_factor(fs_in: float, fs_out: float) -> int:
    """Return an exact integer rate ratio instead of silently truncating it."""

    if fs_in <= 0 or fs_out <= 0:
        raise ValueError("Sample rates must be positive.")
    ratio = float(fs_in) / float(fs_out)
    factor = int(round(ratio))
    if factor < 1 or not np.isclose(ratio, factor, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"fs_in/fs_out must be an integer; got {ratio:.12g}."
        )
    return factor


def _validate_processing_passes(processing_passes: int) -> int:
    if isinstance(processing_passes, bool) or not isinstance(
        processing_passes,
        (int, np.integer),
    ):
        raise ValueError("processing_passes must be a positive integer.")
    processing_passes = int(processing_passes)
    if processing_passes < 1:
        raise ValueError("processing_passes must be a positive integer.")
    return processing_passes


def measure_filter_response(
    coefficients: np.ndarray,
    config: DecimationStageConfig,
    *,
    processing_passes: int = 1,
    passband_points: int = 4_097,
    stopband_points: int = 32_769,
    tolerance_db: float = 0.05,
) -> FilterResponseReport:
    """Measure the effective passband, stopband, and numerical validity.

    Passband and stopband grids are sampled separately. This matters when a
    passband is only a few kilohertz wide while ``fs_in`` is hundreds of
    megahertz; a uniform whole-Nyquist grid can otherwise miss it.

    ``processing_passes=2`` models ``filtfilt``/``sosfiltfilt`` by squaring
    the magnitude response, so the report describes what is applied to data.
    """

    config.validate()
    processing_passes = _validate_processing_passes(processing_passes)
    if passband_points < 2 or stopband_points < 2:
        raise ValueError("Response grids must contain at least two points.")
    if tolerance_db < 0:
        raise ValueError("tolerance_db must be non-negative.")

    coefficients = np.asarray(coefficients)
    coefficients_finite = bool(np.all(np.isfinite(coefficients)))
    passband_frequencies = np.linspace(
        0.0,
        config.passband_hz,
        passband_points,
    )
    stopband_frequencies = np.linspace(
        config.stopband_hz,
        config.fs_in / 2.0,
        stopband_points,
    )
    frequencies = np.concatenate((passband_frequencies, stopband_frequencies))

    stable = True
    filter_order = 0
    if config.filter_type == "butterworth":
        if coefficients.ndim != 2 or coefficients.shape[1] != 6:
            raise ValueError("Butterworth coefficients must be an SOS array.")
        if coefficients_finite:
            poles = []
            filter_order = 0
            for section in coefficients:
                denominator = section[3:] / section[3]
                section_tolerance = np.finfo(float).eps * max(
                    1.0,
                    float(np.max(np.abs(denominator))),
                )
                if abs(denominator[2]) > section_tolerance:
                    section_order = 2
                elif abs(denominator[1]) > section_tolerance:
                    section_order = 1
                else:
                    section_order = 0
                filter_order += section_order
                if section_order:
                    poles.extend(np.roots(denominator[: section_order + 1]))
            stable = bool(np.all(np.abs(poles) < 1.0))
            _, response = sp_signal.sosfreqz(
                coefficients,
                worN=frequencies,
                fs=config.fs_in,
            )
        else:
            response = np.full(len(frequencies), np.nan, dtype=np.complex128)
    else:
        if coefficients.ndim != 1 or len(coefficients) == 0:
            raise ValueError("Kaiser FIR coefficients must be one-dimensional.")
        filter_order = max(0, len(coefficients) - 1)
        if coefficients_finite:
            _, response = sp_signal.freqz(
                coefficients,
                worN=frequencies,
                fs=config.fs_in,
            )
        else:
            response = np.full(len(frequencies), np.nan, dtype=np.complex128)

    magnitude = np.abs(response)
    response_db = (
        20.0
        * processing_passes
        * np.log10(np.maximum(magnitude, np.finfo(float).tiny))
    )
    passband_db = response_db[:passband_points]
    stopband_db = response_db[passband_points:]

    if coefficients_finite and np.all(np.isfinite(response_db)):
        passband_min_db = float(np.min(passband_db))
        passband_max_db = float(np.max(passband_db))
        passband_ripple_db = passband_max_db - passband_min_db
        worst_stopband_db = float(np.max(stopband_db))
        stopband_attenuation_db = -worst_stopband_db
    else:
        passband_min_db = float("nan")
        passband_max_db = float("nan")
        passband_ripple_db = float("nan")
        worst_stopband_db = float("nan")
        stopband_attenuation_db = float("nan")

    failures = []
    if not coefficients_finite:
        failures.append("filter coefficients contain NaN or infinity")
    if not stable:
        failures.append("IIR filter is unstable")
    if not np.isfinite(passband_min_db) or (
        passband_min_db < -config.passband_ripple_db - tolerance_db
    ):
        failures.append(
            "passband loss exceeds "
            f"{config.passband_ripple_db:.3f} dB"
        )
    if not np.isfinite(passband_ripple_db) or (
        passband_ripple_db > config.passband_ripple_db + tolerance_db
    ):
        failures.append(
            "passband ripple exceeds "
            f"{config.passband_ripple_db:.3f} dB"
        )
    if not np.isfinite(worst_stopband_db) or (
        worst_stopband_db > -config.stopband_attenuation_db + tolerance_db
    ):
        failures.append(
            "stopband attenuation is below "
            f"{config.stopband_attenuation_db:.3f} dB"
        )

    return FilterResponseReport(
        filter_type=config.filter_type,
        processing_passes=processing_passes,
        filter_order=filter_order,
        coefficient_count=int(coefficients.size),
        passband_min_db=passband_min_db,
        passband_max_db=passband_max_db,
        passband_ripple_db=passband_ripple_db,
        worst_stopband_db=worst_stopband_db,
        stopband_attenuation_db=stopband_attenuation_db,
        coefficients_finite=coefficients_finite,
        stable=stable,
        meets_specification=not failures,
        failure_reasons=tuple(failures),
    )


def format_filter_report(report: FilterResponseReport) -> str:
    """Return a concise readable filter verification summary."""

    status = "PASS" if report.meets_specification else "FAIL"
    details = (
        f"{report.filter_type}, passes={report.processing_passes}, "
        f"order={report.filter_order}, coefficients={report.coefficient_count}, "
        f"passband_min={report.passband_min_db:.3f} dB, "
        f"passband_ripple={report.passband_ripple_db:.3f} dB, "
        f"worst_stopband={report.worst_stopband_db:.3f} dB, "
        f"stable={report.stable}, result={status}"
    )
    if report.failure_reasons:
        details += ", reasons=" + "; ".join(report.failure_reasons)
    return details


def validate_filter_response(
    coefficients: np.ndarray,
    config: DecimationStageConfig,
    *,
    processing_passes: int = 1,
    strict: bool = True,
) -> FilterResponseReport:
    """Measure a filter and optionally reject it when a requirement fails."""

    report = measure_filter_response(
        coefficients,
        config,
        processing_passes=processing_passes,
    )
    if strict and not report.meets_specification:
        raise ValueError(
            "Generated filter does not meet its specification: "
            + format_filter_report(report)
        )
    return report


def design_butterworth_lowpass(
    config: DecimationStageConfig,
    *,
    processing_passes: int = 1,
    verify: bool = False,
) -> np.ndarray:
    """Design a Butterworth SOS anti-alias filter.

    Set ``verify=True`` to measure the generated effective response and raise
    ``ValueError`` when it does not meet ``config``.
    """

    config.validate()
    processing_passes = _validate_processing_passes(processing_passes)
    if config.filter_type != "butterworth":
        raise ValueError("Butterworth design requires filter_type='butterworth'.")
    order, critical_frequency_hz = sp_signal.buttord(
        wp=config.passband_hz,
        ws=config.stopband_hz,
        gpass=config.passband_ripple_db / processing_passes,
        gstop=config.stopband_attenuation_db / processing_passes,
        fs=config.fs_in,
    )
    coefficients = sp_signal.butter(
        order,
        critical_frequency_hz,
        btype="lowpass",
        fs=config.fs_in,
        output="sos",
    )
    if verify:
        validate_filter_response(
            coefficients,
            config,
            processing_passes=processing_passes,
        )
    return coefficients


def design_kaiser_lowpass(
    config: DecimationStageConfig,
    *,
    verify: bool = False,
) -> np.ndarray:
    """Design a linear-phase Kaiser FIR anti-alias filter.

    Set ``verify=True`` to measure the generated response and raise
    ``ValueError`` when it does not meet ``config``.
    """

    config.validate()
    if config.filter_type != "kaiser_fir":
        raise ValueError("Kaiser design requires filter_type='kaiser_fir'.")
    normalized_width = (
        config.stopband_hz - config.passband_hz
    ) / (config.fs_in / 2.0)
    # kaiserord is approximate.  A small margin keeps the measured response at
    # or beyond the requested attenuation after the tap count is rounded.
    num_taps, beta = sp_signal.kaiserord(
        config.stopband_attenuation_db + 3.0,
        normalized_width,
    )
    num_taps = max(3, int(num_taps))
    if num_taps % 2 == 0:
        num_taps += 1
    cutoff_hz = 0.5 * (config.passband_hz + config.stopband_hz)
    coefficients = sp_signal.firwin(
        num_taps,
        cutoff_hz,
        window=("kaiser", beta),
        fs=config.fs_in,
    )
    if verify:
        validate_filter_response(coefficients, config)
    return coefficients


def filter_and_decimate(
    data: np.ndarray,
    config: DecimationStageConfig,
    *,
    zero_phase_iir: bool = False,
) -> np.ndarray:
    """Filter a complete array and apply the configured integer decimation.

    This whole-array helper is intended for offline experiments.  A later
    stateful processor can reuse the same filter designers for continuous
    block processing without changing the filter specifications.
    """

    config.validate()
    data = np.asarray(data)
    if data.ndim != 1 or len(data) == 0:
        raise ValueError("data must be a non-empty one-dimensional array.")

    if config.filter_type == "butterworth":
        processing_passes = 2 if zero_phase_iir else 1
        sos = design_butterworth_lowpass(
            config,
            processing_passes=processing_passes,
        )
        if zero_phase_iir:
            filtered = sp_signal.sosfiltfilt(sos, data)
        else:
            filtered = sp_signal.sosfilt(sos, data)
        return filtered[::config.decimation_factor]

    taps = design_kaiser_lowpass(config)
    return sp_signal.resample_poly(
        data,
        up=1,
        down=config.decimation_factor,
        window=taps,
    )


def validate_decimation_stages(
    stages: Sequence[DecimationStageConfig],
) -> None:
    """Validate that a sequence of stages forms one continuous rate chain."""

    if not stages:
        raise ValueError("At least one decimation stage is required.")
    for index, stage in enumerate(stages):
        stage.validate()
        if index == 0:
            continue
        previous = stages[index - 1]
        if not np.isclose(
            previous.fs_out,
            stage.fs_in,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(
                "Decimation stages are not continuous: "
                f"stage {index} ends at {previous.fs_out:.12g} Hz, "
                f"but stage {index + 1} starts at {stage.fs_in:.12g} Hz."
            )


def filter_and_decimate_stages(
    data: np.ndarray,
    stages: Sequence[DecimationStageConfig],
    *,
    zero_phase_iir: bool = False,
) -> np.ndarray:
    """Apply a validated sequence of anti-alias filters and rate changes."""

    validate_decimation_stages(stages)
    output = np.asarray(data)
    if output.ndim != 1 or len(output) == 0:
        raise ValueError("data must be a non-empty one-dimensional array.")
    for stage in stages:
        output = filter_and_decimate(
            output,
            stage,
            zero_phase_iir=zero_phase_iir,
        )
    return output


def normalize_signal(
    data: np.ndarray
) -> np.ndarray:
    """
    Normalize the signal.
    """
    data = data.astype(np.float32)
    data_normalized = data / np.median(np.abs(data))

    return data_normalized


def frequency_mixing(
    data: np.ndarray,
    freq_lo: float,
    fs: float,
    initial_phase: float = 0.0
) -> Tuple[np.ndarray, float]:
    """
    Mix the input voltage signal with a local oscillator to shift its frequency.

    Args:
        data: input voltage signal.
        freq_lo: local oscillator frequency (the frequency to shift by).
        fs: sampling frequency of the input signal.
        initial_phase: initial phase of the local oscillator in radians.

    Returns:
        mixed_signal: frequency-shifted IQ signal.
        final_phase: final phase of the local oscillator after processing the input signal, which can be used for next segment processing.
    """
    num_samples = len(data)
    
    dphi = 2 * np.pi * freq_lo / fs
    phase = initial_phase + dphi * np.arange(num_samples)
    lo_signal = np.exp(-1j * phase)
    mixed_signal = data * lo_signal

    final_phase = initial_phase + dphi * num_samples
    final_phase = np.angle(np.exp(1j * final_phase))

    return mixed_signal, final_phase





# ===========================
# 2. Deprecated Functions
# ===========================

def decimate_iir(
    data: np.ndarray,
    fs: float,
    fs_target: float,
    cutoff_ratio: float = 2.2,
    order: int = 6,
) -> np.ndarray:
    """Deprecated fixed-order IIR filtering and decimation helper.

    This remains executable for old callers. New code should use
    :func:`filter_and_decimate` with an explicit
    :class:`DecimationStageConfig`.
    """

    decimation_factor = integer_decimation_factor(fs, fs_target)
    cutoff_freq = fs_target / cutoff_ratio
    sos = sp_signal.butter(
        order,
        cutoff_freq,
        btype="low",
        fs=fs,
        output="sos",
    )
    filtered_data = sp_signal.sosfiltfilt(sos, data)
    return filtered_data[::decimation_factor]


def decimate_fir(
    data: np.ndarray,
    fs: float,
    fs_target: float,
    cutoff_ratio: float = 2.4,
    num_taps: int = 201,
) -> np.ndarray:
    """Deprecated fixed-tap FIR filtering and decimation helper.

    This remains executable for old callers. New code should use
    :func:`filter_and_decimate` with an explicit
    :class:`DecimationStageConfig`.
    """

    decimation_factor = integer_decimation_factor(fs, fs_target)
    cutoff_freq = fs_target / cutoff_ratio
    fir_filter = sp_signal.firwin(
        num_taps,
        cutoff=cutoff_freq,
        fs=fs,
    )
    filtered_data = sp_signal.filtfilt(fir_filter, 1.0, data)
    return filtered_data[::decimation_factor]
