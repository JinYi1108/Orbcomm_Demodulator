"""Command-line adapter for the ``lfdemod fm`` processing pipeline."""

from __future__ import annotations

import argparse
import json

from .fm_ddc import FMDDCConfig
from .pipeline import FMPSDConfig, analyze_fm_psd_file


def add_fm_subparser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Register and return the ``lfdemod fm`` subcommand parser."""

    parser = subparsers.add_parser(
        "fm",
        help="Broadcast-FM DDC, discrimination, and diagnostic plots.",
        description=(
            "Select one broadcast-FM channel from real raw-voltage data, "
            "recover its MPX waveform, and save diagnostic spectra."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "example:\n"
            "  lfdemod fm --input data.dat --rf-frequency 98.3e6 "
            "--start 30 --duration 3\n\n"
            "If neither time-window mode is supplied, the first 0.5 seconds "
            "are analyzed."
        ),
    )

    required = parser.add_argument_group("required input")
    required.add_argument(
        "-i",
        "--input",
        required=True,
        help="Raw real-voltage file.",
    )
    required.add_argument(
        "-f",
        "--rf-frequency",
        required=True,
        type=float,
        help="Target RF frequency in Hz, for example 98.3e6.",
    )

    window = parser.add_argument_group("file window (choose seconds or fraction)")
    window.add_argument(
        "-s",
        "--start",
        type=float,
        help="Start time relative to the raw-file beginning, in seconds.",
    )
    window.add_argument(
        "-d",
        "--duration",
        type=float,
        help="Analysis duration in seconds.",
    )
    window.add_argument(
        "-sf",
        "--start-fraction",
        type=float,
        help="Start position as a fraction of total file duration, from 0 to 1.",
    )
    window.add_argument(
        "-ef",
        "--stop-fraction",
        type=float,
        help="Stop position as a fraction of total file duration, from 0 to 1.",
    )

    ddc = parser.add_argument_group("FM DDC")
    ddc.add_argument(
        "-dt",
        "--dtype",
        default="<i2",
        help="Raw NumPy dtype (default: little-endian signed int16, <i2).",
    )
    ddc.add_argument(
        "-sr",
        "--sample-rate",
        type=float,
        default=480e6,
        help="Raw real-voltage sample rate in samples/s (default: 480e6).",
    )
    ddc.add_argument(
        "-ir",
        "--intermediate-rate",
        type=float,
        default=2.4e6,
        help="First-stage output rate in samples/s (default: 2.4e6).",
    )
    ddc.add_argument(
        "-cr",
        "--channel-rate",
        type=float,
        default=240e3,
        help="Final complex-IQ rate in samples/s (default: 240e3).",
    )
    ddc.add_argument(
        "-pb",
        "--passband",
        type=float,
        default=90e3,
        help="One-sided FM channel passband in Hz (default: 90e3).",
    )
    ddc.add_argument(
        "-sb",
        "--stopband",
        type=float,
        default=120e3,
        help="One-sided final stopband start in Hz (default: 120e3).",
    )
    ddc.add_argument(
        "-sa",
        "--stopband-attenuation",
        type=float,
        default=60.0,
        help="Required stopband attenuation in dB (default: 60).",
    )
    ddc.add_argument(
        "-pd",
        "--padding",
        type=float,
        default=0.020,
        help="Filter-edge padding on each side, in seconds (default: 0.020).",
    )
    ddc.add_argument(
        "-cs",
        "--chunk-samples",
        type=int,
        default=10_000_000,
        help="High-rate input samples processed per block (default: 10000000).",
    )

    waveform = parser.add_argument_group("third diagnostic plot")
    waveform.add_argument(
        "-ws",
        "--waveform-start",
        type=float,
        default=None,
        help=(
            "Start relative to the selected analysis window, in seconds; "
            "omit to center the segment."
        ),
    )
    waveform.add_argument(
        "-wd",
        "--waveform-duration",
        type=float,
        default=0.050,
        help="Displayed time-domain duration in seconds (default: 0.050).",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "-o",
        "--output-dir",
        help=(
            "Explicit output-directory base. If omitted, name it from input, "
            "frequency, start, and duration."
        ),
    )
    output.add_argument(
        "-or",
        "--output-root",
        default="results",
        help="Root for automatic output directories (default: results).",
    )
    output.add_argument(
        "-ow",
        "--overwrite",
        action="store_true",
        help="Replace fixed files in the base directory instead of adding run02.",
    )
    output.add_argument(
        "-l",
        "--label",
        default="direct_fm_test",
        help="Human-readable plot and summary label.",
    )

    parser.set_defaults(command_handler=_run_fm, command_parser=parser)
    return parser


def _run_fm(args: argparse.Namespace) -> int:
    """Translate parsed CLI values to FM configs and run the file pipeline."""

    if all(
        value is None
        for value in (
            args.start,
            args.duration,
            args.start_fraction,
            args.stop_fraction,
        )
    ):
        args.start = 0.0
        args.duration = 0.5

    ddc = FMDDCConfig(
        fs_in=args.sample_rate,
        fs_mid=args.intermediate_rate,
        fs_out=args.channel_rate,
        passband_hz=args.passband,
        stopband_hz=args.stopband,
        stopband_attenuation_db=args.stopband_attenuation,
        chunk_samples=args.chunk_samples,
    )
    config = FMPSDConfig(
        rf_frequency_hz=args.rf_frequency,
        start_seconds=args.start,
        duration_seconds=args.duration,
        start_fraction=args.start_fraction,
        stop_fraction=args.stop_fraction,
        dtype=args.dtype,
        padding_seconds=args.padding,
        waveform_start_seconds=args.waveform_start,
        waveform_duration_seconds=args.waveform_duration,
        label=args.label,
        ddc=ddc,
    )

    try:
        summary = analyze_fm_psd_file(
            args.input,
            args.output_dir,
            config,
            output_root=args.output_root,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError) as error:
        args.command_parser.error(str(error))

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
