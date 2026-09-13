"""LoRa physical-layer decoding for the Starlink VHF profile.

Functions and outputs
---------------------
``LoRaPhysicalPacket.payload_start_seconds``
    Compatibility property for ``phy_data_start_seconds``; returns ``float``.
``LoRaPhysicalPacket.payload_bytes``
    Removes the header-declared payload CRC from backend bytes; returns
    ``bytes``.
``LoRaPhysicalPacket.decoded_payload_and_crc_bytes``
    Exposes backend payload-and-CRC bytes under an unambiguous name; returns
    ``bytes``.
``prepare_lora_iq(iq, sample_rate_hz, config)``
    Resamples complex channel IQ to two samples per LoRa chip; returns
    ``(complex_iq, exact_sample_rate_hz)``.
``demodulate_starlink_lora(iq, sample_rate_hz, rf_frequency_hz, config)``
    Detects, synchronizes, demodulates, and CRC-checks all decodable packets;
    returns a list of ``LoRaPhysicalPacket`` records.
``starlink_sync_symbol_offsets(sync_word)``
    Converts a conventional LoRa sync byte to the two dechirped symbol
    offsets used by the decoder; returns ``(high_nibble_offset, low_nibble_offset)``.
``_load_lora_phy()``
    Loads the optional receiver and no-preamble exception; returns both or
    raises ``MissingStarlinkVHFDependencyError`` with installation guidance.
``_install_lora_phy_numpy_compatibility()``
    Installs the narrow NumPy ufunc alias required by lora-phy 0.3.0; returns
    ``None``.
``_symbol_concentration_db(receiver, filtered_iq, payload_start, symbol_count)``
    Measures synchronized-symbol peak concentration; returns a float array in
    decibels.

``lora-phy`` is loaded only when demodulation is requested.  Consequently the
existing FM and airband-AM paths do not require this optional dependency.
The code uses private synchronization metadata from lora-phy 0.3.0 only to
report packet timing and per-symbol diagnostics; decoded bytes and CRC come
from its public ``demodulate`` and ``decode`` methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from importlib.metadata import PackageNotFoundError, version
from typing import Any, List, Tuple

import numpy as np
from scipy import signal

from .config import StarlinkVHFLoRaConfig


class MissingStarlinkVHFDependencyError(ImportError):
    """Indicate that the optional LoRa physical-layer package is unavailable."""


@dataclass(frozen=True)
class LoRaPhysicalPacket:
    """One synchronized LoRa packet and its physical-layer diagnostics."""

    symbols: np.ndarray
    # Bytes returned by lora-phy: LoRa payload followed by the two-byte
    # payload CRC when the PHY header enables CRC.  Preamble, sync, SFD, and
    # explicit PHY-header symbols are represented by metadata, not here.
    frame_bytes: bytes
    calculated_crc: Tuple[int, ...]
    received_crc: Tuple[int, ...]
    crc_valid: bool | None
    has_explicit_header: bool
    header_valid: bool | None
    header_payload_length: int
    coding_rate: int
    crc_enabled: bool
    decode_error: str | None
    carrier_offset_hz: float
    netid_symbol_offsets: Tuple[float, float]
    preamble_start_seconds: float
    phy_data_start_seconds: float
    packet_stop_seconds: float
    concentration_db: np.ndarray

    @property
    def payload_start_seconds(self) -> float:
        """Return the legacy name for the LoRa PHY-data start time."""

        return self.phy_data_start_seconds

    @property
    def payload_bytes(self) -> bytes:
        """Return decoded LoRa payload bytes with the payload CRC removed."""

        return self.frame_bytes[: self.header_payload_length]

    @property
    def decoded_payload_and_crc_bytes(self) -> bytes:
        """Return the backend bytes retained under legacy ``frame_bytes``."""

        return self.frame_bytes


def _install_lora_phy_numpy_compatibility() -> None:
    """Install the narrow NumPy alias required by lora-phy 0.3.0, if absent."""

    # lora-phy 0.3.0 calls np.bitwise_right_shift, an alias absent in recent
    # NumPy releases. np.right_shift has the required ufunc semantics. Keep
    # this compatibility mutation local to optional-backend loading.
    if not hasattr(np, "bitwise_right_shift"):
        np.bitwise_right_shift = np.right_shift  # type: ignore[attr-defined]


def _load_lora_phy() -> Tuple[Any, Any]:
    """Return ``(LoRaReceiver, NoPreambleError)`` or raise an actionable error."""

    _install_lora_phy_numpy_compatibility()
    try:
        from lora_phy import LoRaReceiver
        from lora_phy.errors import NoPreambleError
    except ImportError as error:
        raise MissingStarlinkVHFDependencyError(
            "Starlink VHF decoding needs the optional dependency. Install "
            "it from the LFdemod repository with: "
            "python -m pip install -e '.[starlink-vhf]'"
        ) from error
    try:
        backend_version = version("lora-phy")
    except PackageNotFoundError:
        backend_version = "unknown"
    if backend_version != "0.3.0":
        raise MissingStarlinkVHFDependencyError(
            "The Starlink VHF adapter requires lora-phy==0.3.0 because it "
            f"uses version-specific synchronization metadata; found {backend_version}."
        )
    return LoRaReceiver, NoPreambleError


def starlink_sync_symbol_offsets(sync_word: int) -> Tuple[int, int]:
    """Return decoder symbol offsets corresponding to one LoRa sync byte."""

    if not 0 <= sync_word <= 0xFF:
        raise ValueError("sync_word must fit in one byte.")
    return (((sync_word >> 4) & 0x0F) * 8, (sync_word & 0x0F) * 8)


def prepare_lora_iq(
    iq: np.ndarray,
    sample_rate_hz: float,
    config: StarlinkVHFLoRaConfig = StarlinkVHFLoRaConfig(),
) -> Tuple[np.ndarray, float]:
    """Return complex IQ at exactly twice the configured LoRa bandwidth."""

    config.validate()
    iq = np.asarray(iq)
    if iq.ndim != 1 or iq.size == 0 or not np.iscomplexobj(iq):
        raise ValueError("iq must be a non-empty one-dimensional complex array.")
    if not np.all(np.isfinite(iq)):
        raise ValueError("iq must contain only finite samples.")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")
    target_rate = 2.0 * config.bandwidth_hz
    ratio = Fraction(target_rate / sample_rate_hz).limit_denominator(1_000_000)
    actual_rate = sample_rate_hz * ratio.numerator / ratio.denominator
    if not np.isclose(actual_rate, target_rate, rtol=0.0, atol=1e-6):
        raise ValueError("Could not represent the LoRa resampling ratio accurately.")
    resampled = signal.resample_poly(iq, ratio.numerator, ratio.denominator)
    return np.asarray(resampled, dtype=np.complex64), float(actual_rate)


def _symbol_concentration_db(
    receiver: Any,
    filtered_iq: np.ndarray,
    payload_start: int,
    symbol_count: int,
) -> np.ndarray:
    """Return peak-to-median dechirped FFT concentration for each symbol."""

    values = np.empty(symbol_count, dtype=np.float64)
    tiny = np.finfo(float).tiny
    for index in range(symbol_count):
        left = payload_start + index * receiver._sample_num
        samples = filtered_iq[left : left + receiver._sample_num]
        if samples.size != receiver._sample_num:
            values[index] = np.nan
            continue
        spectrum = np.fft.fft(
            samples * receiver._down_chirp[: samples.size], receiver._fft_len
        )
        combined = np.abs(spectrum[: receiver._bin_num]) + np.abs(
            spectrum[-receiver._bin_num :]
        )
        values[index] = 20.0 * np.log10(
            max(float(np.max(combined)), tiny)
            / max(float(np.median(combined)), tiny)
        )
    return values


def demodulate_starlink_lora(
    iq: np.ndarray,
    sample_rate_hz: float,
    rf_frequency_hz: float,
    config: StarlinkVHFLoRaConfig = StarlinkVHFLoRaConfig(),
) -> List[LoRaPhysicalPacket]:
    """Return every decodable Starlink-profile LoRa packet in channel IQ."""

    config.validate()
    if not np.isfinite(rf_frequency_hz) or rf_frequency_hz <= 0:
        raise ValueError("rf_frequency_hz must be positive and finite.")
    LoRaReceiver, NoPreambleError = _load_lora_phy()
    lora_iq, lora_rate = prepare_lora_iq(iq, sample_rate_hz, config)
    receiver = LoRaReceiver(
        rf_frequency_hz,
        config.spreading_factor,
        config.bandwidth_hz,
        lora_rate,
        has_header=config.has_explicit_header,
        implicit_header_payload_len=config.expected_payload_bytes,
        implicit_header_coding_rate=config.expected_coding_rate,
        implicit_header_enable_crc=config.expected_crc_enabled,
        preamble_len=config.preamble_symbols,
        resample_to_2x=False,
    )
    # Do not substitute the receiver library's automatic duration-based LDRO
    # choice here.  For the currently validated Starlink VHF profile, forced
    # LDRO is supported by repeatable payload-CRC success on two real frames;
    # the same IQ fails CRC when this option is disabled.  The config switch is
    # retained so another observed profile can be tested without changing code.
    receiver._proc.low_data_rate_optimization = (
        config.force_low_data_rate_optimization
    )

    # Filter once here, then bypass the receiver's duplicate low-pass call.
    filtered_iq = np.asarray(receiver.lowpass(lora_iq))
    receiver._fast_mode = True
    pending_payload_starts: List[int] = []
    synchronized_payload_starts: List[int] = []
    header_metadata: List[Tuple[bool | None, int, int, bool]] = []
    original_sync = receiver._sync
    original_parse_header = receiver._parse_header

    def tracked_sync(signal_: np.ndarray, start: int) -> Tuple[int, int, float]:
        result = original_sync(signal_, start)
        pending_payload_starts.append(int(result[0]))
        if not config.has_explicit_header:
            synchronized_payload_starts.append(int(result[0]))
            header_metadata.append(
                (
                    None,
                    config.expected_payload_bytes,
                    config.expected_coding_rate,
                    config.expected_crc_enabled,
                )
            )
        return result

    def tracked_parse_header(symbols_: np.ndarray) -> Tuple[bool, int, int, bool]:
        result = original_parse_header(symbols_)
        if result[0] and pending_payload_starts:
            synchronized_payload_starts.append(pending_payload_starts[-1])
            header_metadata.append(
                (True, int(result[1]), int(result[2]), bool(result[3]))
            )
        return result

    receiver._sync = tracked_sync
    receiver._parse_header = tracked_parse_header
    try:
        symbol_packets, carrier_offsets, netids = receiver.demodulate(filtered_iq)
    except NoPreambleError:
        return []
    finally:
        receiver._sync = original_sync
        receiver._parse_header = original_parse_header

    metadata_lengths = {
        "symbols": len(symbol_packets),
        "carrier_offsets": len(carrier_offsets),
        "netids": len(netids),
        "starts": len(synchronized_payload_starts),
        "headers": len(header_metadata),
    }
    if len(set(metadata_lengths.values())) != 1:
        details = ", ".join(f"{key}={value}" for key, value in metadata_lengths.items())
        raise RuntimeError(
            "lora-phy 0.3.0 returned inconsistent packet metadata: " + details
        )

    results: List[LoRaPhysicalPacket] = []
    symbol_duration = (2**config.spreading_factor) / config.bandwidth_hz
    preamble_equivalents = config.preamble_symbols + 2.0 + 2.25
    for index, symbols in enumerate(symbol_packets):
        header_valid, payload_length, coding_rate, crc_enabled = header_metadata[index]
        decode_error: str | None = None
        try:
            decoded, calculated_crc_array = receiver.decode(symbols)
            frame_bytes = bytes(decoded)
            if calculated_crc_array is None:
                calculated_crc = ()
            else:
                calculated_crc = tuple(
                    int(value) for value in np.asarray(calculated_crc_array).ravel()
                )
            received_crc = (
                tuple(frame_bytes[payload_length : payload_length + len(calculated_crc)])
                if crc_enabled and calculated_crc
                else ()
            )
            if not crc_enabled:
                crc_valid: bool | None = None
            else:
                crc_valid = bool(calculated_crc) and calculated_crc == received_crc
        except Exception as error:  # Preserve other synchronized packets in the window.
            frame_bytes = b""
            calculated_crc = ()
            received_crc = ()
            crc_valid = False if crc_enabled else None
            decode_error = f"{type(error).__name__}: {error}"
        payload_start = synchronized_payload_starts[index]
        payload_start_seconds = payload_start / lora_rate
        preamble_start_seconds = payload_start_seconds - (
            preamble_equivalents * symbol_duration
        )
        packet_stop_seconds = payload_start_seconds + len(symbols) * symbol_duration
        concentration = _symbol_concentration_db(
            receiver, filtered_iq, payload_start, len(symbols)
        )
        results.append(
            LoRaPhysicalPacket(
                symbols=np.asarray(symbols, dtype=np.uint16),
                frame_bytes=frame_bytes,
                calculated_crc=calculated_crc,
                received_crc=received_crc,
                crc_valid=crc_valid,
                has_explicit_header=config.has_explicit_header,
                header_valid=header_valid,
                header_payload_length=payload_length,
                coding_rate=coding_rate,
                crc_enabled=crc_enabled,
                decode_error=decode_error,
                carrier_offset_hz=float(carrier_offsets[index]),
                netid_symbol_offsets=(float(netids[index][0]), float(netids[index][1])),
                preamble_start_seconds=float(preamble_start_seconds),
                phy_data_start_seconds=float(payload_start_seconds),
                packet_stop_seconds=float(packet_stop_seconds),
                concentration_db=concentration,
            )
        )
    return results
