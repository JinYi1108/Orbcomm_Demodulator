"""Configuration for the Starlink-profile VHF LoRa decoder.

Functions and outputs
---------------------
``KNOWN_STARLINK_VHF_PAYLOAD_BYTES``
    Tuple of publicly reported payload lengths accepted by the conservative
    PHY/output layer; it does not assert field interpretations.
``StarlinkVHFDDCConfig.validate()``
    Validates the voltage-to-IQ rate plan and filter limits; returns ``None``
    or raises ``ValueError``.
``StarlinkVHFLoRaConfig.validate()``
    Validates LoRa physical-layer and observed-profile settings; returns
    ``None`` or raises ``ValueError``.
``StarlinkVHFFileConfig.validate()``
    Validates a raw-file time window and its nested configurations; returns
    ``None`` or raises ``ValueError``.

No spacecraft identity, TLE, event time, or 100-second cadence is encoded in
these reusable configurations.  The defaults describe the publicly observed
137.055 MHz Starlink VHF profile, while the RF center remains a required file
pipeline input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..ddc import integer_decimation_factor


# Lengths reported for publicly observed Starlink VHF LoRa payload profiles.
# Only the 81-byte profile has been decoded and structurally checked against
# the two 21CMA events available to this project.  The other lengths are
# accepted at the PHY/output layer without inventing field interpretations.
KNOWN_STARLINK_VHF_PAYLOAD_BYTES = (73, 81, 87, 104, 227)


@dataclass(frozen=True)
class StarlinkVHFDDCConfig:
    """Rate and anti-alias requirements for one Starlink VHF LoRa channel."""

    fs_in: float = 480e6
    fs_mid: float = 2.4e6
    fs_out: float = 240e3
    first_stage_passband_hz: float = 150e3
    first_stage_stopband_hz: float = 1.0e6
    channel_passband_hz: float = 50e3
    channel_stopband_hz: float = 100e3
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
        if self.passband_ripple_db <= 0 or self.stopband_attenuation_db <= 0:
            raise ValueError("DDC ripple and attenuation must be positive.")
        if self.chunk_samples <= 0:
            raise ValueError("chunk_samples must be positive.")


@dataclass(frozen=True)
class StarlinkVHFLoRaConfig:
    """Physical-layer settings for the observed Starlink VHF LoRa profile."""

    bandwidth_hz: float = 125_000.0 / 3.0
    spreading_factor: int = 8
    # Fifteen ordinary preamble upchirps are supported by the measured
    # 19.25-symbol preamble-to-PHY-data interval (15 + 2 sync + 2.25 SFD),
    # receiver boundary tests on both 21CMA events, and public receiver
    # profiles.  Keep this configurable for other observed profiles.
    preamble_symbols: int = 15
    sync_word: int = 0x12
    # The validated 21CMA frames use an explicit LoRa PHY header.  The header
    # reports an 81-byte payload, CR 4/5 (represented as 1), and payload CRC.
    # These remain configurable rather than being hidden in the receiver.
    has_explicit_header: bool = True
    expected_payload_bytes: int = 81
    accepted_payload_bytes: tuple[int, ...] = KNOWN_STARLINK_VHF_PAYLOAD_BYTES
    expected_coding_rate: int = 1
    expected_crc_enabled: bool = True
    sync_tolerance_symbols: float = 0.75
    # Deliberately default to forced LDRO for this observed Starlink profile.
    # The generic LoRa symbol-duration rule would leave LDRO disabled for
    # SF8/BW=41.667 kHz (Tsym=6.144 ms), but two independent 21CMA frames
    # decode as 148 symbols with valid payload CRCs only when LDRO is enabled;
    # disabling it gives 113 symbols and CRC failures.  Public Starlink VHF
    # receiver profiles also describe this setting as forced LDRO (fldro=1).
    # Keep it configurable because future Starlink packet profiles may differ.
    force_low_data_rate_optimization: bool = True

    def validate(self) -> None:
        """Validate physical-layer values; return ``None`` on success."""

        if self.bandwidth_hz <= 0:
            raise ValueError("bandwidth_hz must be positive.")
        if not 5 <= self.spreading_factor <= 12:
            raise ValueError("spreading_factor must lie between 5 and 12.")
        if self.preamble_symbols <= 0:
            raise ValueError("preamble_symbols must be positive.")
        if not 0 <= self.sync_word <= 0xFF:
            raise ValueError("sync_word must fit in one byte.")
        if not 1 <= self.expected_payload_bytes <= 255:
            raise ValueError("expected_payload_bytes must lie in [1, 255].")
        if not self.accepted_payload_bytes:
            raise ValueError("accepted_payload_bytes must not be empty.")
        if any(not 1 <= value <= 255 for value in self.accepted_payload_bytes):
            raise ValueError("accepted_payload_bytes values must lie in [1, 255].")
        if len(set(self.accepted_payload_bytes)) != len(self.accepted_payload_bytes):
            raise ValueError("accepted_payload_bytes must not contain duplicates.")
        if self.expected_payload_bytes not in self.accepted_payload_bytes:
            raise ValueError(
                "expected_payload_bytes must be included in accepted_payload_bytes."
            )
        if not 1 <= self.expected_coding_rate <= 4:
            raise ValueError("expected_coding_rate must lie in [1, 4].")
        if not np.isfinite(self.sync_tolerance_symbols) or not (
            0.0 <= self.sync_tolerance_symbols < 64.0
        ):
            raise ValueError("sync_tolerance_symbols must lie in [0, 64).")


@dataclass(frozen=True)
class StarlinkVHFFileConfig:
    """Configuration for decoding one raw-file window at a known RF center."""

    rf_frequency_hz: float
    start_seconds: float
    duration_seconds: float
    dtype: str = "<i2"
    padding_seconds: float = 0.050
    save_iq: bool = False
    diagnostic_packet_index: int = 0
    label: str = "starlink_vhf"
    ddc: StarlinkVHFDDCConfig = field(default_factory=StarlinkVHFDDCConfig)
    lora: StarlinkVHFLoRaConfig = field(default_factory=StarlinkVHFLoRaConfig)

    def validate(self) -> None:
        """Validate the requested file window and nested configurations."""

        self.ddc.validate()
        self.lora.validate()
        if not 0 < self.rf_frequency_hz < self.ddc.fs_in / 2.0:
            raise ValueError("rf_frequency_hz must lie between 0 and fs_in / 2.")
        if self.start_seconds < 0:
            raise ValueError("start_seconds must be non-negative.")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive.")
        if self.padding_seconds < 0:
            raise ValueError("padding_seconds must be non-negative.")
        if self.diagnostic_packet_index < 0:
            raise ValueError("diagnostic_packet_index must be non-negative.")
        if self.lora.bandwidth_hz >= self.ddc.channel_passband_hz * 2.0:
            raise ValueError("LoRa occupied half-band must fit inside the DDC passband.")
        if 2.0 * self.lora.bandwidth_hz > self.ddc.fs_out:
            raise ValueError(
                "DDC channel-IQ rate must cover the configured LoRa occupied "
                "bandwidth before LoRa resampling."
            )
        if not self.label.strip():
            raise ValueError("label must not be empty.")
        np.dtype(self.dtype)
