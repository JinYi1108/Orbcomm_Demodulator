from .logging_config import enable_logging

from .ddc import (
    DecimationStageConfig,
    FilterResponseReport,
    design_butterworth_lowpass,
    design_kaiser_lowpass,
    filter_and_decimate,
    filter_and_decimate_stages,
    format_filter_report,
    measure_filter_response,
    normalize_signal,
    frequency_mixing,
    decimate_iir,
    decimate_fir,
    integer_decimation_factor,
    validate_filter_response,
)

from .orbcomm.orbcomm_ddc import (
    downconvert_orbcomm_voltage,
    make_orbcomm_decimation_stages,
)

ddc = downconvert_orbcomm_voltage

from .orbcomm.cr import (
    v4_freq_offset_estimator,
    carrier_error_recovery    
)

cr = carrier_error_recovery


from .orbcomm.rrc import (
    rrcosfilter,
    apply_rrc_match_filter
)

rrc = apply_rrc_match_filter

from .orbcomm.timing_recovery import (
    symbol_timing_recovery,
    farrow_interpolator,
    gardner_ted,
    update_buffer
)


from .orbcomm.costas import (
    costas_phase_recovery,
    four_quadrant_detector
)

costas = costas_phase_recovery

from .orbcomm.decode import differential_decode

decode = differential_decode

from .orbcomm.packet_utils import(
    OrbcommPacketType,
    find_packet_start,
    bits_to_packets
)

from .orbcomm.fletcher_ecc_save import (
    fletcher_checksum,
    single_bit_fix,
    validate_packet
)

from .orbcomm.plotting import (
    plot_constellation,
    plot_eye_diagram
)


from .orbcomm.pipeline import orbdemod

__version__ = "0.1.0"

__all__ = [
    "enable_logging",

    "downconvert_orbcomm_voltage",
    "DecimationStageConfig",
    "FilterResponseReport",
    "design_butterworth_lowpass",
    "design_kaiser_lowpass",
    "filter_and_decimate",
    "filter_and_decimate_stages",
    "format_filter_report",
    "make_orbcomm_decimation_stages",
    "measure_filter_response",
    "integer_decimation_factor",
    "validate_filter_response",
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
