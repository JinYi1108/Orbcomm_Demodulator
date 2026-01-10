import numpy as np
import scipy.signal as sp_signal
from .ddc import frequency_mixing
from .logging_config import get_module_logger

logger = get_module_logger(__name__)

# ===========================
# 1. Basic Functions
# ===========================


def v4_freq_offset_estimator(
    iq_data: np.ndarray,
    fs: float,
    fft_length: int = 9600,
    search_range_hz: float = 2000.0
) -> float:
    """
    4-th power spectrum method to estimate residual frequency offset for input IQ data.
    4-th power removes the phase modulation of SDPSK signals, resulting in a distinct peak at 4 times the frequency offset in the spectrum.

    Args:
        iq_data: input IQ signal.
        fs: Input signal sampling frequency.
        fft_length: FFT length for Welch's method.
        search_range_hz: Search range for the peak in Hz.

    Returns:
        estimated_offset: Estimated residual frequency offset in Hz.
    """
    iq_power_4 = np.power(iq_data, 4)
    f, pxx = sp_signal.welch(iq_power_4, fs=fs, nperseg=fft_length, return_onesided=False)
    f_shifted = np.fft.fftshift(f)
    pxx_shifted = np.fft.fftshift(pxx)

    resolution = fs / fft_length
    search_range_bins = int(search_range_hz / resolution)
    center_bin = len(f_shifted) // 2

    start_bin = center_bin - search_range_bins
    end_bin = center_bin + search_range_bins
    
    if start_bin < 0 or end_bin >= len(pxx_shifted):
        logger.warning("Search range exceeds spectrum boundaries, searching entire spectrum instead.")
        start_bin, end_bin = 0, len(pxx_shifted)

    peak_index_local = np.argmax(pxx_shifted[start_bin:end_bin])
    peak_index_global = start_bin + peak_index_local
    
    if 0 < peak_index_global < len(f_shifted) - 1:
        L = pxx_shifted[peak_index_global - 1]
        C = pxx_shifted[peak_index_global]
        R = pxx_shifted[peak_index_global + 1]

        denominator = (L - 2.0 * C + R)
        if np.abs(denominator) > 1e-12:
            delta = 0.5 * (L - R) / denominator
            freq_at_peak = f_shifted[peak_index_global] + delta * resolution
        else:
            freq_at_peak = f_shifted[peak_index_global]

    else:
        freq_at_peak = f_shifted[peak_index_global]
    
    
    estimated_offset = freq_at_peak / 4.0
    
    logger.info(f"Estimated residual frequency offset: {estimated_offset:.4f} Hz.")

    return estimated_offset


# ===========================
# 2. Composite Function
# ===========================

def carrier_error_recovery(
    iq_data: np.ndarray,
    fs: float,
    num_points_for_fft: int = 50000,
    fft_length: int = 9600,
    search_range_hz: float = 2000.0
) -> np.ndarray:
    """
    ORBCOMM v4 carrier frequency error recovery using 4-th power spectrum method.
    This method is used to eliminate signficant frequency drift.

    Args:
        iq_data: input IQ signal.
        fs: Input signal sampling frequency (Hz).
        num_points_for_fft: Number of points to use for frequency offset estimation.
        fft_length: FFT length for Welch's method.
        search_range_hz: Search range for the peak in Hz.

    Returns:
        iq_data_mixing: IQ signal after frequency mixing to correct the frequency offset.
    """
    logger.info("Starting carrier frequency error recovery process.")

    num_points_for_fft = min(len(iq_data), num_points_for_fft)
    iq_data_fft = iq_data[:num_points_for_fft]
    estimated_offset = v4_freq_offset_estimator(
        iq_data=iq_data_fft,
        fs=fs,
        fft_length=fft_length,
        search_range_hz=search_range_hz
    )
    iq_data_mixing, _ = frequency_mixing(
        data=iq_data,
        freq_lo=estimated_offset,
        fs=fs   
    )

    logger.info("Carrier frequency error recovery has completed.")

    return iq_data_mixing
