"""Public LFdemod API for Starlink-profile VHF LoRa decoding.

Exports configuration, channel DDC, LoRa physical decoding, conservative
frame parsing, and the raw-file output pipeline.  The optional ``lora-phy``
dependency is not imported until physical decoding is requested.
"""

from orbdemod.starlink_vhf import (
    KNOWN_STARLINK_VHF_PAYLOAD_BYTES,
    LoRaPhysicalPacket,
    MissingStarlinkVHFDependencyError,
    StarlinkVHFDDCConfig,
    StarlinkVHFFileConfig,
    StarlinkVHFLoRaConfig,
    demodulate_starlink_lora,
    demodulate_starlink_vhf_file,
    downconvert_starlink_vhf_voltage,
    make_starlink_vhf_decimation_stages,
    parse_starlink_vhf_payload,
    prepare_lora_iq,
    profile_matches,
    starlink_sync_symbol_offsets,
)

__all__ = [
    "KNOWN_STARLINK_VHF_PAYLOAD_BYTES",
    "LoRaPhysicalPacket",
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
