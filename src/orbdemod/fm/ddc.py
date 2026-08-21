"""Channel-selecting DDC for broadcast FM in real-valued voltage data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import signal

from ..ddc import (
    DecimationStageConfig,
    design_butterworth_lowpass,
    design_kaiser_lowpass,
    integer_decimation_factor,
)


@dataclass(frozen=True)
class FMDDCConfig:
    """Parameters for converting 21CMA voltage data to complex FM-channel IQ."""

    fs_in: float = 480e6
    fs_mid: float = 2.4e6
    fs_out: float = 240e3
    passband_hz: float = 90e3
    stopband_hz: float = 120e3
    stopband_attenuation_db: float = 60.0
    first_stage_passband_ripple_db: float = 0.25
    chunk_samples: int = 2_000_000

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
        _integer_decimation(self.fs_in, self.fs_mid, "fs_in/fs_mid")
        _integer_decimation(self.fs_mid, self.fs_out, "fs_mid/fs_out")


def _integer_decimation(fs_from: float, fs_to: float, name: str) -> int:
    try:
        return integer_decimation_factor(fs_from, fs_to)
    except ValueError as error:
        raise ValueError(f"Invalid {name}: {error}") from error


def _adc_scale(dtype: np.dtype) -> float:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return float(max(abs(info.min), abs(info.max)))
    return 1.0


def _frequency_mix(
    data: np.ndarray,
    frequency_hz: float,
    fs: float,
    initial_phase: float,
) -> Tuple[np.ndarray, float]:
    """Shift one block to baseband while preserving NCO phase continuity."""

    phase_step = 2.0 * np.pi * float(frequency_hz) / float(fs)
    sample_index = np.arange(len(data), dtype=np.float64)
    phase = initial_phase + phase_step * sample_index
    mixed = np.asarray(data) * np.exp(-1j * phase)
    next_phase = np.remainder(
        initial_phase + phase_step * len(data) + np.pi,
        2.0 * np.pi,
    ) - np.pi
    return mixed, float(next_phase)


def _design_first_stage(config: FMDDCConfig) -> np.ndarray:
    """Design a Butterworth anti-alias filter for fs_in -> fs_mid."""

    stage = DecimationStageConfig(
        fs_in=config.fs_in,
        fs_out=config.fs_mid,
        passband_hz=config.passband_hz,
        stopband_hz=config.fs_mid / 2.0,
        passband_ripple_db=config.first_stage_passband_ripple_db,
        stopband_attenuation_db=config.stopband_attenuation_db,
        filter_type="butterworth",
    )
    return design_butterworth_lowpass(stage)


def _design_second_stage(config: FMDDCConfig) -> np.ndarray:
    """Design a Kaiser FIR with an explicit 90--120 kHz transition band."""

    stage = DecimationStageConfig(
        fs_in=config.fs_mid,
        fs_out=config.fs_out,
        passband_hz=config.passband_hz,
        stopband_hz=config.stopband_hz,
        stopband_attenuation_db=config.stopband_attenuation_db,
        filter_type="kaiser_fir",
    )
    return design_kaiser_lowpass(stage)


def downconvert_real_voltage(
    raw_data: np.ndarray,
    rf_frequency_hz: float,
    config: FMDDCConfig = FMDDCConfig(),
    initial_phase: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """Extract one FM channel as complex IQ at ``config.fs_out``.

    The high-rate input is mixed and IIR-filtered in bounded chunks. Filter
    state, NCO phase, and the decimation grid remain continuous between
    chunks. A second, linear-phase polyphase stage selects the 90 kHz FM
    passband before reducing the rate to 240 kS/s.
    """

    config.validate()
    raw_data = np.asanyarray(raw_data)
    if raw_data.ndim != 1 or len(raw_data) == 0:
        raise ValueError("raw_data must be a non-empty one-dimensional array.")
    if np.iscomplexobj(raw_data):
        raise ValueError("raw_data must contain real-valued voltage samples.")
    if not 0 < rf_frequency_hz < config.fs_in / 2.0:
        raise ValueError("rf_frequency_hz must lie between 0 and fs_in / 2.")

    decimation_1 = _integer_decimation(config.fs_in, config.fs_mid, "fs_in/fs_mid")
    decimation_2 = _integer_decimation(config.fs_mid, config.fs_out, "fs_mid/fs_out")
    first_stage_sos = _design_first_stage(config)
    first_stage_state = np.zeros(
        (first_stage_sos.shape[0], 2),
        dtype=np.complex128,
    )

    scale = _adc_scale(raw_data.dtype)
    phase = float(initial_phase)
    input_count = 0
    stage_1_blocks = []

    for start in range(0, len(raw_data), config.chunk_samples):
        stop = min(start + config.chunk_samples, len(raw_data))
        voltage = np.asarray(raw_data[start:stop], dtype=np.float32) / scale
        mixed, phase = _frequency_mix(
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
    second_stage_fir = _design_second_stage(config)
    baseband = signal.resample_poly(
        stage_1,
        up=1,
        down=decimation_2,
        window=second_stage_fir,
    )
    return np.asarray(baseband, dtype=np.complex64), phase
