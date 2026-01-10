import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from .logging_config import get_module_logger

logger = get_module_logger(__name__)

def plot_constellation(
    iq_signal: np.ndarray,
    sample_skip: int = 100,
    title: str = 'Constellation Diagram',
    show_plot: bool = False,
    save_path: str = None,
    save: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw the constellation diagram of IQ signal.

    Args:
        iq_signal: IQ signal.
        sample_skip: sampling interval.
        title: chart title.
        show_plot: whether to immediately call plt.show() to display the chart (not display by default).
        save_path: the path to save the chart.
        save: whether to immediately call plt.savefig() to save the chart (not save by default).

    Returns:
        fig: the figure object of matplotlib.
        ax: the axes object of matplotlib.

    """
    plot_data = iq_signal[::sample_skip]

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(plot_data.real, plot_data.imag, s=10, alpha=0.6, marker='.')

    ax.set_title(title)
    ax.set_xlabel('I Component (In-phase)')
    ax.set_ylabel('Q Component (Quadrature)')
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.axhline(0, color='red', linewidth=0.8)
    ax.axvline(0, color='red', linewidth=0.8)

    ax.set_aspect('equal', adjustable='box') 
    


    if save:
        if save_path is None:
            logger.warning("save=True but save_path=None, skipping save.")
        else:
            plt.savefig(save_path)

    if show_plot:
        plt.show()

    logger.info("Constellation plot generated with %d data points (skip=%d).", len(plot_data), sample_skip)
    
    return fig, ax
    
    
def plot_eye_diagram(
    iq_data: np.ndarray,
    sample_offset: int = 400,
    segments: int = 10,
    segment_length: int = 64,
    title: str = 'Eye Diagram (Q Component)',
    show_plot: bool = False,
    save_path: str = None,
    save: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw the eye diagram of IQ signal.

    Args:
        iq_data: IQ signal.
        sample_offset: starting index for eye diagram.
        segments: number of segments to overlay.
        segment_length: number of samples per eye segment.
        title: chart title.
        show_plot: whether to immediately call plt.show() to display the chart (not display by default).
        save_path: the path to save the chart.
        save: whether to immediately call plt.savefig() to save the chart (not save by default).

    Returns:
        fig: the figure object of matplotlib.
        ax: the axes object of matplotlib.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    current_index = sample_offset
    num_plotted = 0

    while current_index + segment_length <= len(iq_data) and num_plotted < segments:

        segment = iq_data[current_index:current_index + segment_length].imag
        ax.plot(segment, alpha=0.7)

        current_index += segment_length
        num_plotted += 1

    ax.set_title(title)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Amplitude (Q Component)')
    ax.grid(True, linestyle='--', alpha=0.5)

    if save:
        if save_path is None:
            logger.warning("save=True but save_path=None, skipping save.")
        else:
            plt.savefig(save_path)

    if show_plot:
        plt.show()

    logger.info("Eye Diagram generated with %d segments (segment_length=%d).", num_plotted, segment_length)

    return fig, ax

