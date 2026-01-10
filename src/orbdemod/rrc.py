import numpy as np
import scipy.signal as sp_signal
from .logging_config import get_module_logger

logger = get_module_logger(__name__)

# From: https://github.com/veeresht/CommPy/blob/master/commpy/filters.py （same in https://github.com/fbieberly/ORBCOMM-receiver/blob/master/helpers.py）
def rrcosfilter(N, alpha, Ts, Fs):
    """
    Generates a root raised cosine (RRC) filter (FIR) impulse response.

    Parameters
    ----------
    N : int
        Length of the filter in samples.

    alpha: float
        Roll off factor (Valid values are [0, 1]).

    Ts : float
        Symbol period in seconds.

    Fs : float
        Sampling Rate in Hz.

    Returns
    ---------
    h_rrc : 1-D ndarray of floats
        Impulse response of the root raised cosine filter.

    time_idx : 1-D ndarray of floats
        Array containing the time indices, in seconds, for
        the impulse response.
    """
    N = int(N)
    T_delta = 1/float(Fs)
    time_idx = ((np.arange(N)-N/2))*T_delta
    sample_num = np.arange(N)
    h_rrc = np.zeros(N, dtype=float)

    for x in sample_num:
        t = (x-N/2)*T_delta
        if t == 0.0:
            h_rrc[x] = 1.0 - alpha + (4*alpha/np.pi)
        elif alpha != 0 and t == Ts/(4*alpha):
            h_rrc[x] = (alpha/np.sqrt(2))*(((1+2/np.pi)* \
                    (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
        elif alpha != 0 and t == -Ts/(4*alpha):
            h_rrc[x] = (alpha/np.sqrt(2))*(((1+2/np.pi)* \
                    (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
        else:
            h_rrc[x] = (np.sin(np.pi*t*(1-alpha)/Ts) +  \
                    4*alpha*(t/Ts)*np.cos(np.pi*t*(1+alpha)/Ts))/ \
                    (np.pi*t*(1-(4*alpha*t/Ts)*(4*alpha*t/Ts))/Ts)

    return time_idx, h_rrc



def apply_rrc_match_filter(
    iq_data: np.ndarray,
    fs: float,
    baud_rate: float,
    span_symbols: int = 8
) -> np.ndarray:
    """
    ORBCOMM RRC matched filtering.

    Args:
        iq_data: input IQ signal.
        fs: Input signal sampling frequency (Hz).
        baud_rate: Symbol rate (baud).
        span_symbols: Filter half-span in symbols.

    Returns:
        matched_filtered_signal: IQ signal after RRC matched filtering.
    """
    alpha = 0.4
    sps = fs / baud_rate
    rrc_num_taps = int(sps * span_symbols * 2 + 1)

    logger.info(
        "Starting RRC matched filtering. Baud: %.2f, SPS: %.2f, Taps: %d (Span: %d symbols, Alpha: %.2f).",
        baud_rate, sps, rrc_num_taps, span_symbols, alpha
    )

    rrc_taps = rrcosfilter(
        rrc_num_taps,
        alpha,
        Ts=1.0,
        Fs=sps
    )[1]

    matched_filtered_signal = sp_signal.lfilter(rrc_taps, 1.0, iq_data)

    logger.info("RRC matched filtering has completed.")

    return matched_filtered_signal