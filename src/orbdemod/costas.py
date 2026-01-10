import numpy as np
from typing import Tuple
from .logging_config import get_module_logger

logger = get_module_logger(__name__)


# ===========================
# 1. Basic Functions
# ===========================


def four_quadrant_detector(
    sample: np.complex64
) -> float:
    """
    SDPSK Costas loop phase error detector.

    Args:
        sample: the IQ sample rotated by the current phase estimate.
    
    Returns:
        phase_error: the phase error estimate (proportional to phase mismatch).
    """
    y_I = sample.real
    y_Q = sample.imag

    phase_error = np.sign(y_I) * y_Q - np.sign(y_Q) * y_I

    return phase_error



# ===========================
# 2. Composite Function
# ===========================

def costas_phase_recovery(
    iq_data: np.ndarray,
    alpha: float = 0.03,
    beta_ratio: float = 0.2,
    beta: float = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform carrier phase recovery using a second-order Costas loop for SDPSK signals.

    Args:
        iq_data: input IQ signal after timing recovery.
        alpha: loop proportional gain (phase correction).
        beta_ratio: ratio to calculating integral gain (frequency correction).
        beta: Integral gain (optional).
        
    Returns:
        costas_iq: IQ signal after carrier phase correction.
        phase_est: history of the phase estimate at each input sample.
        freq_out_hist: history of the frequency estimate at each input sample.
    """
    if beta is None:
        beta = beta_ratio * alpha**2

    logger.info(
        "Starting Costas Loop Phase Recovery.Alpha: %f, Beta: %f", 
        alpha, beta
    )
    

    n = len(iq_data)
    phase_est = np.zeros(n + 1)
    frequency_out = 0.0
    freq_out_hist = np.zeros(n)

   

    for i, sample in enumerate(iq_data):
   
        corrected_sample = sample * np.exp(-1j * phase_est[i]) 
        phase_error = four_quadrant_detector(corrected_sample)
              
        frequency_out += beta * phase_error
        phase_est[i+1] = phase_est[i] + alpha * phase_error + frequency_out

        freq_out_hist[i] = frequency_out
        
    costas_iq = iq_data * np.conj(np.exp(1j*phase_est[:-1]))
    
    logger.info("Costas Loop Phase Recovery has completed. Final frequency offset estimate: %.4f rad/symbol.", frequency_out)
    
    return costas_iq, phase_est,freq_out_hist