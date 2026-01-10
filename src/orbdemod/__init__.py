from .logging_config import enable_logging

from .ddc import (
    digital_down_converter,
    normalize_signal,
    frequency_mixing,
    decimate_iir,
    decimate_fir
)

ddc = digital_down_converter

from .cr import (
    v4_freq_offset_estimator,
    carrier_error_recovery    
)

cr = carrier_error_recovery


from .rrc import (
    rrcosfilter,
    apply_rrc_match_filter
)

rrc = apply_rrc_match_filter

from .timing_recovery import (
    symbol_timing_recovery,
    farrow_interpolator,
    gardner_ted,
    update_buffer
)


from .costas import (
    costas_phase_recovery,
    four_quadrant_detector
)

costas = costas_phase_recovery

from .decode import differential_decode

decode = differential_decode

from .packet_utils import(
    OrbcommPacketType,
    find_packet_start,
    bits_to_packets
)

from .fletcher_ecc_save import (
    fletcher_checksum,
    single_bit_fix,
    validate_packet
)

from .plotting import (
    plot_constellation,
    plot_eye_diagram
)


from .pipeline import orbdemod

__version__ = "0.1.0"

__all__ = [
    "enable_logging",

    "digital_down_converter",
    "ddc",
    "normalize_signal",
    "frequency_mixing",
    "decimate_iir",
    "decimate_fir",

    "carrier_error_recovery",
    "cr",
    "v4_freq_offset_estimator",

    "rrcosfilter",
    "apply_rrc_match_filter",
    "rrc",

    "symbol_timing_recovery",
    "farrow_interpolator",
    "gardner_ted",
    "update_buffer",

    "costas_phase_recovery",
    "four_quadrant_detector",
    "costas",

    "differential_decode",
    "decode",

    "OrbcommPacketType",
    "find_packet_start",
    "bits_to_packets",

    "fletcher_checksum",
    "single_bit_fix",
    "validate_packet",

    "plot_constellation",
    "plot_eye_diagram",

    "orbdemod"
]