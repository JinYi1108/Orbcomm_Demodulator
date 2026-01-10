import numpy as np
from typing import Optional, List
import os
import matplotlib.pyplot as plt

from .logging_config import get_module_logger, enable_logging
from .ddc import digital_down_converter as ddc
from .cr import carrier_error_recovery as cr
from .rrc import apply_rrc_match_filter as rrc
from .timing_recovery import symbol_timing_recovery
from .costas import costas_phase_recovery as costas
from .decode import differential_decode as decode
from .packet_utils import find_packet_start, bits_to_packets
from .fletcher_ecc_save import validate_packet

from .plotting import plot_constellation, plot_eye_diagram

logger = get_module_logger(__name__)

def orbdemod(
    raw_data: np.ndarray,
    freq_orbcomm: float = 137.46e6,
    fs_in: float = 480e6,
    fs_mid: float = 2.4e6,
    fs_out: float = 9600,
    baud_rate: float = 4800.0,
    log_level: str = "INFO",
    plot: bool = False,
    plot_save_dir: Optional[str] = None,
    output_file: Optional[str] = "orbdemod_packets.txt"  
) -> List[str]:
    """
    ORBCOMM Demodulator Pipeline.

    Args:
        raw_data: input voltage signal.
        freq_orbcomm: center frequency of the ORBCOMM signal (local oscillator frequency).
        fs_in: original sampling frequency of the input signal.
        fs_mid: intermediate sampling frequency after first decimation.
        fs_out: final target sampling frequency after second decimation.
        baud_rate: symbol rate (baud).
        log_level: logging level for the pipeline.
        plot: whether to generate and save plots at each stage.
        plot_save_dir: directory to save plots if plotting is enabled.
        output_file: file to save valid decoded hex packets.

    Returns:
        valid_hex_packets: list of valid decoded hex packets.
    """
    enable_logging(level=log_level)
    logger.info("Starting Pipeline with Log Level: %s, Plotting: %s", log_level, plot)

    if plot and plot_save_dir:
        os.makedirs(plot_save_dir, exist_ok=True)

    iq_data, _ = ddc(raw_data, freq_orbcomm, fs_in, fs_mid, fs_out)
    if plot and plot_save_dir:
        plot_constellation(iq_data, sample_skip=1,title="DDC Constellation", save=True, 
                           save_path=os.path.join(plot_save_dir, 'iq_ddc.png'))
    
    iq_cr = cr(iq_data, fs_out)
    if plot and plot_save_dir:
        plot_constellation(iq_cr,  sample_skip=1, title="CR Constellation", save=True,
                         save_path=os.path.join(plot_save_dir, 'iq_cr.png'))
        plot_eye_diagram(iq_cr[::2], title="CR Eye Diagram", save=True,
                         save_path=os.path.join(plot_save_dir, 'iq_cr_eye.png'))
        
    iq_rrc = rrc(iq_cr, fs_out,baud_rate)
    if plot and plot_save_dir:
        plot_constellation(iq_rrc,  sample_skip=1, title="RRC Constellation", save=True,
                         save_path=os.path.join(plot_save_dir, 'iq_rrc.png'))
        plot_eye_diagram(iq_rrc[::2], title="RRC Eye Diagram", save=True,
                         save_path=os.path.join(plot_save_dir, 'iq_rrc_eye.png'))
    
    iq_timed,tau,dtau,error = symbol_timing_recovery(iq_rrc)
    if plot and plot_save_dir:
        plot_constellation(iq_timed,  sample_skip=1, title="Timing Recovery Constellation", save=True,
                         save_path=os.path.join(plot_save_dir, 'iq_timed.png'))
        plot_eye_diagram(iq_timed, title="Timing Recovery Eye Diagram", save=True,
                         save_path=os.path.join(plot_save_dir, 'iq_timed_eye.png'))
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))
        axs[0].plot(tau); axs[0].set_title("Timing Offset (tau)")
        axs[1].plot(dtau); axs[1].set_title("Derivative (Dtau)")
        axs[2].plot(error); axs[2].set_title("Timing Error Signal")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_save_dir, 'str_metrics.png'))
        plt.close()
    
    iq_costas, phase,freq = costas(iq_timed)
    if plot and plot_save_dir:
        plot_constellation(iq_costas, sample_skip=1, title="Costas Loop Constellation", save=True,
                         save_path=os.path.join(plot_save_dir, 'iq_costas.png'))
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        axs[0].plot(phase); axs[0].set_title('Phase output of PLL'); axs[0].grid()
        axs[1].plot(freq); axs[1].set_title('Frequency of PLL'); axs[1].grid()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_save_dir, 'costas_metrics.png'))
        plt.close()

    bit_decode, deg  = decode(iq_costas)
    if plot and plot_save_dir:
        plt.figure()
        plt.title('Differential Decode Angle')
        plt.xlabel('Symbol Number')
        plt.ylabel('Angle (degrees)')
        plt.plot(deg, 'x')
        plt.grid()
        plt.savefig(os.path.join(plot_save_dir, 'angle.png'))
        plt.close()
    
    offset, reverse_order = find_packet_start(bit_decode)

    hex_packets = bits_to_packets(bit_decode, offset, reverse_order)

    valid_hex_packets = validate_packet(hex_packets, output_file=output_file)

    logger.info("Pipeline finished. Valid packets: %d", len(valid_hex_packets))

    return valid_hex_packets