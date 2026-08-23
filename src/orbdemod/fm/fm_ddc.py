"""Channel-selecting DDC for broadcast FM in real-valued voltage data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import signal

from ..ddc import (
    DecimationStageConfig,
    design_butterworth_lowpass,
    filter_and_decimate,
    frequency_mixing,
    integer_decimation_factor,
    validate_decimation_stages,
)


@dataclass(frozen=True)
class FMDDCConfig:
    """Parameters for converting 21CMA voltage data to complex FM-channel IQ.

    Attributes:
        fs_in: Raw real-voltage sample rate in samples/s. The 21CMA default is
            480 MS/s.
        fs_mid: Sample rate after the first decimation stage. The default is
            2.4 MS/s.
        fs_out: Final complex-IQ sample rate. The default 240 kS/s supports an
            FM channel extending to +/-120 kHz.
        passband_hz: One-sided channel passband retained around the tuned
            station. The default is 90 kHz.
        stopband_hz: One-sided frequency at which the final filter must already
            meet its stopband requirement. The default is 120 kHz.
        stopband_attenuation_db: Required alias rejection in the stopband.
        first_stage_passband_ripple_db: Allowed passband ripple for the first
            Butterworth stage.
        chunk_samples: Number of high-rate input samples processed per block.
            It changes memory use and speed, not the selected file duration.
    """

    fs_in: float = 480e6
    fs_mid: float = 2.4e6
    fs_out: float = 240e3
    passband_hz: float = 90e3
    stopband_hz: float = 120e3
    stopband_attenuation_db: float = 60.0
    first_stage_passband_ripple_db: float = 0.25
    chunk_samples: int = 10_000_000

    def validate(self) -> None:
        if min(self.fs_in, self.fs_mid, self.fs_out) <= 0:
            raise ValueError("All sample rates must be positive.")
        if not self.fs_in > self.fs_mid > self.fs_out:
            raise ValueError("Expected fs_in > fs_mid > fs_out.")
        if not 0 < self.passband_hz < self.stopband_hz <= self.fs_out / 2:
            raise ValueError(
                "Expected 0 < passband_hz < stopband_hz <= fs_out / 2."
            )
        if self.stopband_attenuation_db <= 0:
            raise ValueError("stopband_attenuation_db must be positive.")
        if self.first_stage_passband_ripple_db <= 0:
            raise ValueError("first_stage_passband_ripple_db must be positive.")
        if self.chunk_samples <= 0:
            raise ValueError("chunk_samples must be positive.")
        integer_decimation_factor(self.fs_in, self.fs_mid)
        integer_decimation_factor(self.fs_mid, self.fs_out)


def make_fm_decimation_stages(
    config: FMDDCConfig = FMDDCConfig(),
) -> Tuple[DecimationStageConfig, DecimationStageConfig]:
    """Build the FM-specific two-stage anti-alias and decimation plan.

    Args:
        config: FM input, intermediate, and output sample-rate/filter settings.

    Returns:
        A two-element tuple. Stage 1 is a Butterworth IIR from ``fs_in`` to
        ``fs_mid``; stage 2 is a Kaiser FIR/polyphase stage from ``fs_mid`` to
        ``fs_out``.

    Notes:
        This function designs the processing plan only. Use
        :func:`downconvert_fm_voltage` to process voltage samples.
    """

    config.validate()
    stages = (
        DecimationStageConfig(
            fs_in=config.fs_in,
            fs_out=config.fs_mid,
            passband_hz=config.passband_hz,
            stopband_hz=config.fs_mid / 2.0,
            passband_ripple_db=config.first_stage_passband_ripple_db,
            stopband_attenuation_db=config.stopband_attenuation_db,
            filter_type="butterworth",
        ),
        DecimationStageConfig(
            fs_in=config.fs_mid,
            fs_out=config.fs_out,
            passband_hz=config.passband_hz,
            stopband_hz=config.stopband_hz,
            stopband_attenuation_db=config.stopband_attenuation_db,
            filter_type="kaiser_fir",
        ),
    )
    validate_decimation_stages(stages)
    return stages


def downconvert_fm_voltage(
    raw_data: np.ndarray,
    rf_frequency_hz: float,
    config: FMDDCConfig = FMDDCConfig(),
    initial_phase: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """Convert raw voltage samples to one FM channel as complex IQ.

    Args:
        raw_data: One-dimensional real ADC-voltage samples. Integer samples are
            converted to floating point without normalizing their ADC scale.
        rf_frequency_hz: Absolute RF frequency to translate to zero frequency,
            in Hz (for example ``98.3e6``).
        config: DDC sample-rate, filter, and chunk settings.
        initial_phase: Starting phase of the digital local oscillator, in
            radians. File-oriented processing normally calculates this from
            the absolute raw-file start sample; direct array use can leave it
            at zero.

    Returns:
        ``(iq, final_phase)`` where ``iq`` is complex64 at ``config.fs_out``
        and ``final_phase`` can continue the local oscillator in a later block.

    The high-rate input is mixed and IIR-filtered in bounded chunks. Filter
    state, NCO phase, and the decimation grid remain continuous between
    chunks. A second, linear-phase polyphase stage selects the 90 kHz FM
    passband before reducing the rate to 240 kS/s.
    """

    first_stage, second_stage = make_fm_decimation_stages(config)
    raw_data = np.asanyarray(raw_data)
    if raw_data.ndim != 1 or len(raw_data) == 0:
        raise ValueError("raw_data must be a non-empty one-dimensional array.")
    if np.iscomplexobj(raw_data):
        raise ValueError("raw_data must contain real-valued voltage samples.")
    if not 0 < rf_frequency_hz < config.fs_in / 2.0:
        raise ValueError("rf_frequency_hz must lie between 0 and fs_in / 2.")

    decimation_1 = first_stage.decimation_factor
    first_stage_sos = design_butterworth_lowpass(first_stage)
    first_stage_state = np.zeros(
        (first_stage_sos.shape[0], 2),
        dtype=np.complex128,
    )

    phase = float(initial_phase)
    input_count = 0
    stage_1_blocks = []

    for start in range(0, len(raw_data), config.chunk_samples):
        stop = min(start + config.chunk_samples, len(raw_data))
        # Preserve the original ADC-count scale while converting integer
        # samples to a type suitable for mixing and filtering.
        voltage = np.asarray(raw_data[start:stop], dtype=np.float32)
        mixed, phase = frequency_mixing(
            voltage,
            rf_frequency_hz,
            config.fs_in,
            phase,
        )
        filtered, first_stage_state = signal.sosfilt(
            first_stage_sos,
            mixed,
            zi=first_stage_state,
        )

        first_output_index = (-input_count) % decimation_1
        stage_1_blocks.append(filtered[first_output_index::decimation_1])
        input_count += len(voltage)

    stage_1 = np.concatenate(stage_1_blocks)
    baseband = filter_and_decimate(stage_1, second_stage)
    return np.asarray(baseband, dtype=np.complex64), phase
