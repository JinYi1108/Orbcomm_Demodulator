import numpy as np
import scipy.signal as sp_signal
from typing import Tuple
from .logging_config import get_module_logger

logger = get_module_logger(__name__)

# ===========================
# 1. Basic Functions
# ===========================

def normalize_signal(
    data: np.ndarray
) -> np.ndarray:
    """
    Normalize the signal.
    """
    data = data.astype(np.float32)
    data_normalized = data / np.median(np.abs(data))

    return data_normalized


def frequency_mixing(
    data: np.ndarray,
    freq_lo: float,
    fs: float,
    initial_phase: float = 0.0
) -> Tuple[np.ndarray, float]:
    """
    Mix the input voltage signal with a local oscillator to shift its frequency.

    Args:
        data: input voltage signal.
        freq_lo: local oscillator frequency (the frequency to shift by).
        fs: sampling frequency of the input signal.
        initial_phase: initial phase of the local oscillator in radians.

    Returns:
        mixed_signal: frequency-shifted IQ signal.
        final_phase: final phase of the local oscillator after processing the input signal, which can be used for next segment processing.
    """
    num_samples = len(data)
    
    dphi = 2 * np.pi * freq_lo / fs
    phase = initial_phase + dphi * np.arange(num_samples)
    lo_signal = np.exp(-1j * phase)
    mixed_signal = data * lo_signal

    final_phase = initial_phase + dphi * num_samples
    final_phase = np.angle(np.exp(1j * final_phase))

    return mixed_signal, final_phase


def decimate_iir(
    data: np.ndarray,
    fs: float,
    fs_target: float,
    cutoff_ratio: float = 2.2,
    order: int = 6
) -> np.ndarray:
    """
    Use an IIR Butterworth SOS filter for anti-aliasing and decimation.
    Fast and efficient, with a nonlinear phase; zero phase can be obtained using sosfiltfilt.
    Intended for first-stage significant downsampling.
    Notice: This method is suitable for offline only. For real-time processing, use sosfilt instead of sosfiltfilt.

    Args:
        data: input IQ signal.
        fs: original sampling frequency of the input signal.
        fs_target: target sampling frequency after decimation.
        cutoff_ratio: ratio to determine the cutoff frequency for the low-pass filter.
        order: order of the Butterworth filter.

    Returns:
        filtered_data: decimated IQ signal.
    """
    decimation_factor = int(fs / fs_target)
    cutoff_freq = fs_target / cutoff_ratio
    sos = sp_signal.butter(order, cutoff_freq, btype='low', fs=fs, output='sos')
    filtered_data = sp_signal.sosfiltfilt(sos, data)

    return filtered_data[::decimation_factor]


def decimate_fir(
    data: np.ndarray,
    fs: float,
    fs_target: float,
    cutoff_ratio: float = 2.4,
    num_taps: int = 201
) -> np.ndarray:
    """
    Use an FIR filter for anti-aliasing and decimation.
    Linear phase, large amount of computation.
    Notice: This method is suitable for offline only. For real-time processing, use lfilter instead of filtfilt.

    Args:
        data: input IQ signal.
        fs: original sampling frequency of the input signal.
        fs_target: target sampling frequency after decimation.
        cutoff_ratio: ratio to determine the cutoff frequency for the low-pass filter.
        num_taps: number of taps (coefficients) in the FIR filter.

    Returns:
        filtered_data: decimated IQ signal.
    """
    decimation_factor = int(fs / fs_target)
    cutoff_freq = fs_target / cutoff_ratio

    fir_filter = sp_signal.firwin(num_taps, cutoff = cutoff_freq, fs=fs)

    filtered_data = sp_signal.filtfilt(fir_filter, 1.0, data)

    return filtered_data[::decimation_factor]


# ===========================
# 2. Composite Function
# ===========================


def digital_down_converter(
    data: np.ndarray,
    freq_lo: float,
    fs_in: float,
    fs_mid: float,
    fs_out: float,
    initial_phase: float = 0.0
) -> Tuple[np.ndarray, float]:
    """
    Perform digital down conversion (DDC) pipeline on the input voltage signal from ORBCOMM satellites.

    Args:
        data: input voltage signal.
        freq_lo: local oscillator frequency (the frequency to shift by).
        fs_in: original sampling frequency of the input signal.
        fs_mid: intermediate sampling frequency after first decimation.
        fs_out: final target sampling frequency after second decimation.
        initial_phase: initial phase of the local oscillator in radians.

    Returns:
        data_final: final down-converted IQ signal.
        next_phase: final phase of the local oscillator after processing the input signal, which can be used for next segment processing.
    """
    logger.info(
        "Starting Digital Down Converter (DDC) pipeline. Fs: %.2f Hz -> %.2f Hz -> %.2f Hz.",
        fs_in, fs_mid, fs_out
    )
    logger.info("Local Oscillator (LO) frequency: %.2f Hz. Initial Phase: %.2f rad.", freq_lo, initial_phase)

    norm_data = normalize_signal(data)

    mixed_data, next_phase = frequency_mixing(norm_data, freq_lo, fs_in, initial_phase)

    dc_offset = np.mean(mixed_data)

    logger.info("DC Offset Removal.")
    mixed_data_center = mixed_data - dc_offset

    logger.info("DDC Step 1/2: Applying IIR Decimation (Fs: %.2f Hz -> %.2f Hz).", fs_in, fs_mid)
    data_mid = decimate_iir(data=mixed_data_center, fs=fs_in, fs_target=fs_mid)

    logger.info("DDC Step 2/2: Applying FIR Decimation (Fs: %.2f Hz -> %.2f Hz).", fs_mid, fs_out)
    data_final = decimate_fir(data=data_mid, fs=fs_mid, fs_target=fs_out)

    logger.info("DDC pipeline has completed. Final Phase: %.2f rad.", next_phase)

    return data_final, next_phase





