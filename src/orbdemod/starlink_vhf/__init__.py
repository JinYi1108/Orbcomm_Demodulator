"""Starlink-profile VHF DDC, LoRa decoding, frame parsing, and file pipeline.

Public outputs are complex channel IQ, ``LoRaPhysicalPacket`` records,
conservative frame dictionaries, or a saved-run summary pointing to JSON,
CSV, PNG, binary-frame, and optional IQ files.
"""

from .config import (
    KNOWN_STARLINK_VHF_PAYLOAD_BYTES,
    StarlinkVHFDDCConfig,
    StarlinkVHFFileConfig,
    StarlinkVHFLoRaConfig,
)
from .ddc import downconvert_starlink_vhf_voltage, make_starlink_vhf_decimation_stages
from .frame import parse_starlink_vhf_payload, profile_matches
from .lora import (
    LoRaPhysicalPacket,
    MissingStarlinkVHFDependencyError,
    demodulate_starlink_lora,
    prepare_lora_iq,
    starlink_sync_symbol_offsets,
)
from .pipeline import demodulate_starlink_vhf_file

__all__ = [
    "LoRaPhysicalPacket",
    "KNOWN_STARLINK_VHF_PAYLOAD_BYTES",
    "MissingStarlinkVHFDependencyError",
    "StarlinkVHFDDCConfig",
    "StarlinkVHFFileConfig",
    "StarlinkVHFLoRaConfig",
    "demodulate_starlink_lora",
    "demodulate_starlink_vhf_file",
    "downconvert_starlink_vhf_voltage",
    "make_starlink_vhf_decimation_stages",
    "parse_starlink_vhf_payload",
    "prepare_lora_iq",
    "profile_matches",
    "starlink_sync_symbol_offsets",
]
