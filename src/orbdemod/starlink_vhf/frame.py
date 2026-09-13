"""Conservative interpretation of decoded Starlink-profile VHF frames.

Functions and outputs
---------------------
``parse_starlink_vhf_payload(payload)``
    Selects a length-based profile and extracts only fields supported by that
    profile; returns a JSON-serializable dictionary. Publicly reported 73,
    81, 87, 104, and 227-byte lengths are accepted, while only the 81-byte
    profile currently has project-validated field interpretation.
``profile_matches(packet, config)``
    Combines length, CRC, sync offsets, and fixed-byte checks; returns ``bool``.
``_cyclic_symbol_distance(observed, expected, modulus)``
    Computes wrapped sync-bin distance; returns ``float``.

Field names ending in ``candidate`` remain reverse-engineered hypotheses.
In particular, the 16-bit spacecraft field is not asserted to be a NORAD ID,
and no public spacecraft name is inferred here.
"""

from __future__ import annotations

from typing import Dict

from .config import KNOWN_STARLINK_VHF_PAYLOAD_BYTES, StarlinkVHFLoRaConfig
from .lora import LoRaPhysicalPacket, starlink_sync_symbol_offsets


def parse_starlink_vhf_payload(payload: bytes) -> Dict[str, object]:
    """Return a conservative length-profile interpretation of payload bytes."""

    payload = bytes(payload)
    payload_length = len(payload)
    known_length = payload_length in KNOWN_STARLINK_VHF_PAYLOAD_BYTES
    result: Dict[str, object] = {
        "payload_length_bytes": payload_length,
        "length_profile": (
            f"starlink_vhf_{payload_length}_byte"
            if known_length
            else "unrecognized_length"
        ),
        "known_length_profile": known_length,
        "field_interpretation_available": payload_length == 81,
        "payload_hex": payload.hex(),
        "interpretation_caution": (
            "Length-profile support does not by itself identify Starlink or a "
            "named spacecraft. Only the 81-byte profile currently has "
            "project-validated candidate field interpretation."
        ),
    }
    if payload_length != 81:
        result.update(
            {
                "fixed_byte_checks": None,
                "fixed_byte_profile_match": None,
            }
        )
        return result

    checks = {
        "byte_5_is_cc": payload[5] == 0xCC,
        "bytes_6_7_are_zero": payload[6:8] == b"\x00\x00",
        "byte_8_is_d0": payload[8] == 0xD0,
        "byte_11_is_45": payload[11] == 0x45,
        "byte_12_is_05": payload[12] == 0x05,
    }
    result.update(
        {
            "message_number_u24_le": int.from_bytes(payload[0:3], "little"),
            "protocol_id_u16_le": int.from_bytes(payload[3:5], "little"),
            # Compatibility name retained for existing result readers. This
            # reverse-engineered protocol value is not a NORAD catalog ID.
            "spacecraft_id_u16_le": int.from_bytes(payload[3:5], "little"),
            "packet_type_byte": payload[5],
            "time_candidate_u32_le": int.from_bytes(payload[14:18], "little"),
            "fixed_byte_checks": checks,
            "fixed_byte_profile_match": all(checks.values()),
            "interpretation_caution": (
                "These 81-byte fields are reverse-engineered protocol values. "
                "protocol_id_u16_le is not assumed to be a NORAD catalog number "
                "or public Starlink name."
            ),
        }
    )
    return result


def _cyclic_symbol_distance(observed: float, expected: float, modulus: int) -> float:
    """Return the shortest absolute distance between two cyclic symbol bins."""

    difference = abs(float(observed) - float(expected)) % modulus
    return min(difference, modulus - difference)


def profile_matches(
    packet: LoRaPhysicalPacket,
    config: StarlinkVHFLoRaConfig = StarlinkVHFLoRaConfig(),
) -> bool:
    """Return whether one physical packet matches the configured profile."""

    payload = packet.payload_bytes
    parsed = parse_starlink_vhf_payload(payload)
    expected_sync = starlink_sync_symbol_offsets(config.sync_word)
    symbol_modulus = 2**config.spreading_factor
    sync_matches = all(
        _cyclic_symbol_distance(observed, expected, symbol_modulus)
        <= config.sync_tolerance_symbols
        for observed, expected in zip(packet.netid_symbol_offsets, expected_sync)
    )
    content_match = parsed["fixed_byte_profile_match"]
    if content_match is None:
        content_match = bool(parsed["known_length_profile"])
    crc_status_matches = (
        packet.crc_valid is True
        if config.expected_crc_enabled
        else packet.crc_valid is None
    )
    return bool(
        packet.decode_error is None
        and crc_status_matches
        and packet.header_payload_length == len(payload)
        and len(payload) in config.accepted_payload_bytes
        and packet.coding_rate == config.expected_coding_rate
        and packet.crc_enabled == config.expected_crc_enabled
        and packet.has_explicit_header == config.has_explicit_header
        and (
            packet.header_valid is True
            if packet.has_explicit_header
            else packet.header_valid is None
        )
        and sync_matches
        and content_match
    )
