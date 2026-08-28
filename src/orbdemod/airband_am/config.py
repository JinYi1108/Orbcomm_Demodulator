"""Configuration objects for narrowband civil-airband AM voice decoding.

Functions and outputs
---------------------
``AirbandAMDDCConfig.validate()``
    Validates the two-stage DDC rate plan and filter limits; returns ``None``
    or raises ``ValueError``.
``AirbandAMFileConfig.validate()``
    Validates one file window, audio settings, and its DDC configuration;
    returns ``None`` or raises ``ValueError``.

The defaults describe the currently tested 21CMA 480 MS/s input, but neither
class contains 21CMA event detection, clipping diagnosis, or frequency search.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..ddc import integer_decimation_factor


@dataclass(frozen=True)
class AirbandAMDDCConfig:
    """Rate and filter requirements for one civil-airband AM voice channel.

    Frequencies named ``passband`` or ``stopband`` are one-sided distances
    from the tuned carrier in complex baseband.
    """

    fs_in: float = 480e6
    fs_mid: float = 2.4e6
    fs_out: float = 60e3
    first_stage_passband_hz: float = 100e3
    first_stage_stopband_hz: float = 1.0e6
    channel_passband_hz: float = 5e3
    channel_stopband_hz: float = 10e3
    passband_ripple_db: float = 0.5
    stopband_attenuation_db: float = 70.0
    chunk_samples: int = 5_000_000

    def validate(self) -> None:
        """Validate rates and filter limits; return ``None`` on success."""

        if min(self.fs_in, self.fs_mid, self.fs_out) <= 0:
            raise ValueError("All DDC sample rates must be positive.")
        if not self.fs_in > self.fs_mid > self.fs_out:
            raise ValueError("Expected fs_in > fs_mid > fs_out.")
        integer_decimation_factor(self.fs_in, self.fs_mid)
        integer_decimation_factor(self.fs_mid, self.fs_out)
        if not (
            0
            < self.channel_passband_hz
            < self.channel_stopband_hz
            <= self.fs_out / 2.0
        ):
            raise ValueError(
                "Expected 0 < channel_passband_hz < channel_stopband_hz "
                "<= fs_out / 2."
            )
        if not (
            self.channel_stopband_hz
            < self.first_stage_passband_hz
            < self.first_stage_stopband_hz
            <= self.fs_mid / 2.0
        ):
            raise ValueError(
                "Expected channel_stopband_hz < first_stage_passband_hz < "
                "first_stage_stopband_hz <= fs_mid / 2."
            )
        if self.passband_ripple_db <= 0:
            raise ValueError("passband_ripple_db must be positive.")
        if self.stopband_attenuation_db <= 0:
            raise ValueError("stopband_attenuation_db must be positive.")
        if self.chunk_samples <= 0:
            raise ValueError("chunk_samples must be positive.")


@dataclass(frozen=True)
class AirbandAMFileConfig:
    """Configuration for decoding one known airband channel and time window."""

    rf_frequency_hz: float
    start_seconds: float
    duration_seconds: float
    dtype: str = "<i2"
    padding_seconds: float = 0.050
    audio_sample_rate_hz: float = 48e3
    audio_low_hz: float = 300.0
    audio_high_hz: float = 4e3
    normalize_audio: bool = True
    normalization_percentile: float = 99.5
    normalization_target: float = 0.8
    label: str = "airband_am"
    ddc: AirbandAMDDCConfig = field(default_factory=AirbandAMDDCConfig)

    def validate(self) -> None:
        """Validate the requested file window and audio settings."""

        self.ddc.validate()
        if not 0 < self.rf_frequency_hz < self.ddc.fs_in / 2.0:
            raise ValueError("rf_frequency_hz must lie between 0 and fs_in / 2.")
        if self.start_seconds < 0:
            raise ValueError("start_seconds must be non-negative.")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive.")
        if self.padding_seconds < 0:
            raise ValueError("padding_seconds must be non-negative.")
        if self.audio_sample_rate_hz <= 0:
            raise ValueError("audio_sample_rate_hz must be positive.")
        if not (
            0
            < self.audio_low_hz
            < self.audio_high_hz
            < min(self.ddc.channel_passband_hz, self.audio_sample_rate_hz / 2.0)
        ):
            raise ValueError(
                "Expected 0 < audio_low_hz < audio_high_hz below both the "
                "DDC channel passband and audio Nyquist frequency."
            )
        if not 0 < self.normalization_percentile <= 100:
            raise ValueError("normalization_percentile must lie in (0, 100].")
        if not 0 < self.normalization_target <= 1:
            raise ValueError("normalization_target must lie in (0, 1].")
        if not self.label.strip():
            raise ValueError("label must not be empty.")
        np.dtype(self.dtype)
