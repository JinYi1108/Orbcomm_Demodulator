import numpy as np
from typing import Tuple
from .logging_config import get_module_logger

logger = get_module_logger(__name__)

# ===========================
# 1. Basic Functions
# ===========================



def farrow_interpolator(
    data: np.ndarray,
    mu: float
) -> np.complex64:
    """
    Farrow interpolator (3rd order polynomial implementation).
    Estimate the sample value at the fractional delay mu in [0, 1).

    Args:
        data: a buffer containing the most recent 4 samples.
        mu: fractional delay in [0, 1).

    Returns:
        interpolated_sample: the estimated sample value at the fractional delay mu.
    """ 
    x0, x1, x2, x3 = data
    
    c0 = x1
    c1 = 0.5 * (x2 - x0)
    c2 = x0 - (2.5 * x1) + (2 * x2) - (0.5 * x3)
    c3 = (0.5 * (x3 - x0)) + (1.5 * (x1 - x2))

    interpolated_sample = c0 + (c1 * mu) + (c2 * mu**2) + (c3 * mu**3)

    return interpolated_sample


def gardner_ted(
    y_mid: np.complex64,
    y_prev: np.complex64,
    y_current: np.complex64
) -> float:
    """
    Gardner timing error detector.
    
    Args:
        y_mid: the sample at the estimated midpoint (transition) between symbols.
        y_prev: the sample at the previous optimal sampling instant .
        y_current: the sample at the current optimal sampling instant.

    Returns:
        error: the estimated timing error value.
    """
    error = (y_mid.real * (y_prev.real - y_current.real) + y_mid.imag * (y_prev.imag - y_current.imag))

    return error


def update_buffer(
    buf: np.ndarray,
    iq_data: np.ndarray,
    i: int
) -> np.ndarray:
    """
    Update the interpolation buffer with a sliding window from the input data.

    Args:
        buf: the 4-sample buffer to be updated.
        iq_data: input IQ signal.
        i: the current index for the sliding window.

    Returns:
        buf: the updated buffer.
    """
    start_idx = i - 3
    end_idx = i + 1
    buf[:] = iq_data[start_idx:end_idx]

    return buf

# ===========================
# 2. Composite Function
# ===========================


def symbol_timing_recovery(
    iq_data: np.ndarray,
    samples_per_symbol: int = 2,
    alpha: float = 0.0005,
    beta: float = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform symbol timing recovery using a Farrow interpolator controlled by a Gardner TED and a second-order PI loop.

    Args:
        iq_data: input IQ signal after RRC.
        samples_per_symbol: number of samples per symbol in the input signal.
        alpha: proportional gain for the PI loop.
        beta: integral gain for the PI loop. If None, it will be set to 2*sqrt(alpha).
    
    Returns:
        time_recovery_samples: decimated IQ signal (1 sample per symbol).
        tau_vect: History of the fractional delay.
        dtau_vect: History of the integral term (related to frequency correction).
        timing_error: History of the timing error detected by the Gardner TED.
    """    
    logger.info(
        "Starting Symbol Timing Recovery (Gardner/PI Loop). SPS: %d.",
        samples_per_symbol
    )
    if beta is None:
        beta = 2 * np.sqrt(alpha) 
    logger.info(
        "Loop parameters: alpha = %f, beta = %f.",
        alpha, beta
    )

    N = len(iq_data)
    time_recovery_samples = np.zeros(N//samples_per_symbol, dtype=np.complex64)
    dtau_vect = np.zeros(N)
    tau_vect = np.zeros(N)
    timing_error = np.zeros(N)
    
    tau = 0.0
    dtau = 0.0
    counter = 1
    err = 0.0
    buf = np.zeros(4, dtype=np.complex64)


    y_mid = 0.0 + 0.0j
    y_prev = 0.0 + 0.0j

    t = 0
    i = 4
    j = 0
    for x in range(4):
        buf[x] = iq_data[x]
    
    while i < N -1:
        if tau > 1.0:
            buf = update_buffer(buf, iq_data, i + 1)
            t = tau - 1
            current_sample = farrow_interpolator(buf, t)
        elif tau < 0.0:
            buf = update_buffer(buf, iq_data, i -1)
            t = tau + 1
            current_sample = farrow_interpolator(buf, t)
        else:
            buf = update_buffer(buf, iq_data, i)
            t = tau
            current_sample = farrow_interpolator(buf, t)

        counter += 1
        if counter >= samples_per_symbol:
            counter -= samples_per_symbol
            y_current = current_sample
            time_recovery_samples[j] = current_sample
            j += 1

            err = gardner_ted(y_mid, y_prev, y_current)
            
            dtau += alpha * err
            tau += beta * err
            
            y_prev = y_current
        
        else:
            y_mid = current_sample
            err = 0.0
        
        tau = tau + dtau / samples_per_symbol

        dtau_vect[i] = dtau
        tau_vect[i] = tau
        timing_error[i] = err
        i += 1
            

    logger.info("Symbol Timing Recovery has completed.")
    
    return time_recovery_samples, tau_vect, dtau_vect, timing_error








