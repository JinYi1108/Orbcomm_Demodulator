"""Public LFdemod API for known-frequency civil-airband AM voice decoding.

Exports configuration classes, DDC functions, carrier estimation, envelope
demodulation, audio normalization, and the file-to-WAV pipeline.  Each output
type is documented on its implementation function in ``orbdemod.airband_am``.
"""

from orbdemod.airband_am import (
    AirbandAMDDCConfig,
    AirbandAMDemodResult,
    AirbandAMFileConfig,
    demodulate_airband_am,
    demodulate_airband_am_file,
    downconvert_airband_am_voltage,
    estimate_airband_carrier,
    make_airband_am_decimation_stages,
    normalize_airband_audio,
)

__all__ = [
    "AirbandAMDDCConfig",
    "AirbandAMDemodResult",
    "AirbandAMFileConfig",
    "demodulate_airband_am",
    "demodulate_airband_am_file",
    "downconvert_airband_am_voltage",
    "estimate_airband_carrier",
    "make_airband_am_decimation_stages",
    "normalize_airband_audio",
]
