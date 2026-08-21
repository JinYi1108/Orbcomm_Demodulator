"""Digital down conversion for ORBCOMM signals in raw voltage data."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from ..ddc import (
    DecimationStageConfig,
    filter_and_decimate,
    frequency_mixing,
    integer_decimation_factor,
    normalize_signal,
    validate_decimation_stages,
)
from ..logging_config import get_module_logger

logger = get_module_logger(__name__)


def make_orbcomm_decimation_stages(
    fs_in: float = 480e6,
    fs_mid: float = 2.4e6,
    fs_out: float = 9_600.0,
    *,
    passband_hz: float | None = None,
    stopband_attenuation_db: float = 60.0,
    final_decimation_factor: int = 10,
) -> Tuple[DecimationStageConfig, ...]:
    """Build the default ORBCOMM anti-alias and decimation plan.

    The plan inserts a 96 kS/s stage before the final 9.6 kS/s output, so the
    narrow 4.0--4.8 kHz transition is designed at a lower input rate than the 
    direct 2.4 MS/s to 9.6 kS/s reduction.

    ``fs_out / 2.4``. It remains overrideable because measured Doppler and
    receiver-frequency error may require a wider acquisition passband.
    """

    integer_decimation_factor(fs_in, fs_mid)
    integer_decimation_factor(fs_mid, fs_out)
    if final_decimation_factor < 1:
        raise ValueError("final_decimation_factor must be positive.")
    if passband_hz is None:
        passband_hz = fs_out / 2.4
    final_stopband_hz = fs_out / 2.0
    if not 0 < passband_hz < final_stopband_hz:
        raise ValueError(
            "ORBCOMM passband_hz must lie between 0 and fs_out / 2."
        )

    stages = [
        DecimationStageConfig(
            fs_in=fs_in,
            fs_out=fs_mid,
            passband_hz=passband_hz,
            stopband_hz=fs_mid / 2.0,
            stopband_attenuation_db=stopband_attenuation_db,
            filter_type="butterworth",
        )
    ]

    overall_final_factor = integer_decimation_factor(fs_mid, fs_out)
    preferred_final_factor = min(final_decimation_factor, overall_final_factor)
    while (
        preferred_final_factor > 1
        and overall_final_factor % preferred_final_factor != 0
    ):
        preferred_final_factor -= 1

    if preferred_final_factor < overall_final_factor and preferred_final_factor > 1:
        fs_prefinal = fs_out * preferred_final_factor
        stages.append(
            DecimationStageConfig(
                fs_in=fs_mid,
                fs_out=fs_prefinal,
                passband_hz=passband_hz,
                stopband_hz=fs_prefinal / 2.0,
                stopband_attenuation_db=stopband_attenuation_db,
                filter_type="kaiser_fir",
            )
        )
        final_stage_input = fs_prefinal
    else:
        final_stage_input = fs_mid

    stages.append(
        DecimationStageConfig(
            fs_in=final_stage_input,
            fs_out=fs_out,
            passband_hz=passband_hz,
            stopband_hz=final_stopband_hz,
            stopband_attenuation_db=stopband_attenuation_db,
            filter_type="kaiser_fir",
        )
    )
    result = tuple(stages)
    validate_decimation_stages(result)
    return result


def downconvert_orbcomm_voltage(
    data: np.ndarray,
    freq_lo: float,
    fs_in: float,
    fs_mid: float,
    fs_out: float,
    initial_phase: float = 0.0,
    decimation_stages: Sequence[DecimationStageConfig] | None = None,
) -> Tuple[np.ndarray, float]:
    """Convert raw voltage samples to baseband ORBCOMM complex IQ.

    Args:
        data: Input real-valued voltage samples.
        freq_lo: ORBCOMM frequency shifted to baseband.
        fs_in: Input sample rate.
        fs_mid: First intermediate sample rate.
        fs_out: Final sample rate.
        initial_phase: Initial local-oscillator phase in radians.
        decimation_stages: Optional explicit ORBCOMM rate-change plan.

    Returns:
        Final complex IQ and the local-oscillator phase for the next segment.
    """

    if decimation_stages is None:
        decimation_stages = make_orbcomm_decimation_stages(
            fs_in=fs_in,
            fs_mid=fs_mid,
            fs_out=fs_out,
        )
    else:
        decimation_stages = tuple(decimation_stages)
        validate_decimation_stages(decimation_stages)
        if not np.isclose(
            decimation_stages[0].fs_in,
            fs_in,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError("The first decimation stage must start at fs_in.")
        if not np.isclose(
            decimation_stages[-1].fs_out,
            fs_out,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError("The final decimation stage must end at fs_out.")

    rate_path = " -> ".join(
        [f"{decimation_stages[0].fs_in:.2f}"]
        + [f"{stage.fs_out:.2f}" for stage in decimation_stages]
    )
    logger.info("Starting ORBCOMM DDC rate plan (Hz): %s.", rate_path)
    logger.info(
        "Local Oscillator (LO) frequency: %.2f Hz. Initial Phase: %.2f rad.",
        freq_lo,
        initial_phase,
    )

    normalized_data = normalize_signal(data)
    mixed_data, next_phase = frequency_mixing(
        normalized_data,
        freq_lo,
        fs_in,
        initial_phase,
    )

    logger.info("DC Offset Removal.")
    mixed_data_center = mixed_data - np.mean(mixed_data)

    data_final = mixed_data_center
    total_stages = len(decimation_stages)
    for index, stage in enumerate(decimation_stages, start=1):
        logger.info(
            "ORBCOMM DDC stage %d/%d: %s filtering and decimation "
            "(%.2f Hz -> %.2f Hz).",
            index,
            total_stages,
            stage.filter_type,
            stage.fs_in,
            stage.fs_out,
        )
        data_final = filter_and_decimate(
            data_final,
            stage,
            zero_phase_iir=True,
        )

    logger.info(
        "ORBCOMM DDC pipeline has completed. Final Phase: %.2f rad.",
        next_phase,
    )
    return data_final, next_phase
