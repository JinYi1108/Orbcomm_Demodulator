import numpy as np
from typing import Tuple
from .logging_config import get_module_logger

logger = get_module_logger(__name__)

def differential_decode(
    iq_phase_locked: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform differential decoding for SDPSK signals.
    
    Args:
        iq_phase_locked: IQ samples after Costas loop phase recovery.

    Returns:
        decode_bits: final decoded bit sequence.
        phase_diffs_deg: history of phase differences in degrees.
    """
    logger.info(
        "Starting differential decoding."
    )
    iq_phase_locked /= np.median(np.abs(iq_phase_locked))

    symbols = iq_phase_locked

    phase_diffs_rad = np.angle(symbols[1:] * np.conj(symbols[:-1]))

    decoded_bits = (phase_diffs_rad > 0).astype(np.uint8)

    phase_diffs_deg = np.rad2deg(phase_diffs_rad)

    avg_abs_phase = np.mean(np.abs(phase_diffs_deg))
    logger.info(
        "Differential decoding has completed. Bits generated: %d. Mean absolute phase diff: %.4f deg.",
        len(decoded_bits), avg_abs_phase
    )

    return decoded_bits, phase_diffs_deg