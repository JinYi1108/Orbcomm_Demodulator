"""Airband-specific channel DDC built from LFdemod's shared DDC primitives.

Functions and outputs
---------------------
``make_airband_am_decimation_stages(config)``
    Builds and validates the airband two-stage rate plan; returns a tuple of
    two ``DecimationStageConfig`` objects.
``downconvert_airband_am_voltage(raw_data, rf_frequency_hz, config, initial_phase)``
    Mixes a real voltage array to one airband channel in bounded chunks;
    returns ``(complex64_iq_at_fs_out, final_nco_phase_radians)``.

This file selects civil-airband voice bandwidths.  It does not detect ADC
clipping, search for carriers, or decide whether a candidate is a nonlinear
copy of another frequency.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import signal

from ..ddc import (
    DecimationStageConfig,
    design_butterworth_lowpass,
    filter_and_decimate,
    frequency_mixing,
    validate_decimation_stages,
)
from .config import AirbandAMDDCConfig


def make_airband_am_decimation_stages(
    config: AirbandAMDDCConfig = AirbandAMDDCConfig(),
) -> Tuple[DecimationStageConfig, DecimationStageConfig]:
    """Return the validated high-rate and channel-selecting DDC stages."""

    config.validate()
    stages = (
        DecimationStageConfig(
            fs_in=config.fs_in,
            fs_out=config.fs_mid,
            passband_hz=config.first_stage_passband_hz,
            stopband_hz=config.first_stage_stopband_hz,
            passband_ripple_db=config.passband_ripple_db,
            stopband_attenuation_db=config.stopband_attenuation_db,
            filter_type="butterworth",
        ),
        DecimationStageConfig(
            fs_in=config.fs_mid,
            fs_out=config.fs_out,
            passband_hz=config.channel_passband_hz,
            stopband_hz=config.channel_stopband_hz,
            passband_ripple_db=config.passband_ripple_db,
            stopband_attenuation_db=config.stopband_attenuation_db,
            filter_type="kaiser_fir",
        ),
    )
    validate_decimation_stages(stages)
    return stages


def downconvert_airband_am_voltage(
    raw_data: np.ndarray,
    rf_frequency_hz: float,
    config: AirbandAMDDCConfig = AirbandAMDDCConfig(),
    initial_phase: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """Return one airband channel as complex IQ plus the final NCO phase.

    The first 480 MS/s-class stage is stateful and chunked so memory use stays
    bounded.  Filter state, oscillator phase, and the decimation grid remain
    continuous across chunks.  The second FIR/polyphase stage selects the
    narrow voice channel and returns IQ at ``config.fs_out``.
    """

    first_stage, channel_stage = make_airband_am_decimation_stages(config)
    raw_data = np.asanyarray(raw_data)
    if raw_data.ndim != 1 or len(raw_data) == 0:
        raise ValueError("raw_data must be a non-empty one-dimensional array.")
    if np.iscomplexobj(raw_data):
        raise ValueError("raw_data must contain real-valued voltage samples.")
    if not 0 < rf_frequency_hz < config.fs_in / 2.0:
        raise ValueError("rf_frequency_hz must lie between 0 and fs_in / 2.")

    decimation = first_stage.decimation_factor
    sos = design_butterworth_lowpass(first_stage)
    state = np.zeros((sos.shape[0], 2), dtype=np.complex128)
    phase = float(initial_phase)
    input_count = 0
    blocks = []

    for start in range(0, len(raw_data), config.chunk_samples):
        stop = min(start + config.chunk_samples, len(raw_data))
        voltage = np.asarray(raw_data[start:stop], dtype=np.float32)
        mixed, phase = frequency_mixing(
            voltage,
            rf_frequency_hz,
            config.fs_in,
            phase,
        )
        filtered, state = signal.sosfilt(sos, mixed, zi=state)
        first_index = (-input_count) % decimation
        # Copy only the decimated samples. Keeping a strided view here would
        # retain the complete high-rate filtered block until concatenation.
        blocks.append(filtered[first_index::decimation].copy())
        input_count += len(voltage)

    intermediate_iq = np.concatenate(blocks)
    channel_iq = filter_and_decimate(intermediate_iq, channel_stage)
    return np.asarray(channel_iq, dtype=np.complex64), phase
