"""Civil-airband AM DDC, envelope voice decoding, metrics, and file pipeline.

Public outputs are complex channel IQ, float32 envelope/audio arrays, or a
saved-run summary pointing to WAV, JSON, and PNG files.
"""

from .config import AirbandAMDDCConfig, AirbandAMFileConfig
from .ddc import downconvert_airband_am_voltage, make_airband_am_decimation_stages
from .demod import (
    AirbandAMDemodResult,
    demodulate_airband_am,
    estimate_airband_carrier,
    normalize_airband_audio,
)
from .pipeline import demodulate_airband_am_file

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
