"""Command-line registration for the LFdemod civil-airband AM decoder.

Functions and outputs
---------------------
``add_airband_am_subparser(subparsers)``
    Registers ``lfdemod airband-am``; returns its ``ArgumentParser``.
``_run_airband_am(args)``
    Converts parsed options to configs, runs the file pipeline, prints JSON,
    and returns process status ``0`` on success.
"""

from __future__ import annotations

import argparse
import json

from .config import AirbandAMDDCConfig, AirbandAMFileConfig
from .pipeline import demodulate_airband_am_file


def add_airband_am_subparser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Register and return the ``lfdemod airband-am`` command parser."""

    parser = subparsers.add_parser(
        "airband-am",
        help="Decode a known civil-airband AM voice channel to WAV.",
        description=(
            "Decode one known civil-airband AM voice frequency and raw-file "
            "time window. This command does not scan frequencies or diagnose "
            "ADC clipping."
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Raw real-voltage input file.",
    )
    parser.add_argument(
        "-f",
        "--rf-frequency",
        required=True,
        type=float,
        help="Known airband RF carrier to tune, in Hz.",
    )
    parser.add_argument(
        "-s",
        "--start",
        required=True,
        type=float,
        help="Requested start relative to the raw-file beginning, in seconds.",
    )
    parser.add_argument(
        "-d",
        "--duration",
        required=True,
        type=float,
        help="Requested audio duration in seconds; output is not fixed at 2 s.",
    )

    ddc = parser.add_argument_group("Air Band AM DDC")
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
        help="First-stage complex-IQ rate in samples/s (default: 2.4e6).",
    )
    ddc.add_argument(
        "-cr",
        "--channel-rate",
        type=float,
        default=60e3,
        help="Final channel complex-IQ rate in samples/s (default: 60e3).",
    )
    ddc.add_argument(
        "-fp",
        "--first-stage-passband",
        type=float,
        default=100e3,
        help="One-sided first-stage passband edge in Hz (default: 100e3).",
    )
    ddc.add_argument(
        "-fs",
        "--first-stage-stopband",
        type=float,
        default=1.0e6,
        help="One-sided first-stage stopband edge in Hz (default: 1.0e6).",
    )
    ddc.add_argument(
        "-cp",
        "--channel-passband",
        type=float,
        default=5e3,
        help="One-sided final voice-channel passband edge in Hz (default: 5e3).",
    )
    ddc.add_argument(
        "-cb",
        "--channel-stopband",
        type=float,
        default=10e3,
        help="One-sided final channel stopband edge in Hz (default: 10e3).",
    )
    ddc.add_argument(
        "-pr",
        "--passband-ripple",
        type=float,
        default=0.5,
        help="Maximum DDC passband ripple in dB (default: 0.5).",
    )
    ddc.add_argument(
        "-sa",
        "--stopband-attenuation",
        type=float,
        default=70.0,
        help="Required DDC stopband attenuation in dB (default: 70).",
    )
    ddc.add_argument(
        "-cs",
        "--chunk-samples",
        type=int,
        default=5_000_000,
        help="High-rate input samples processed per block (default: 5000000).",
    )
    ddc.add_argument(
        "-pd",
        "--padding",
        type=float,
        default=0.050,
        help="Filter-edge padding on each side, removed from output (default: 0.05 s).",
    )

    audio = parser.add_argument_group("Air Band AM voice audio")
    audio.add_argument(
        "-ar",
        "--audio-rate",
        type=float,
        default=48e3,
        help="Output WAV sample rate in samples/s (default: 48e3).",
    )
    audio.add_argument(
        "-al",
        "--audio-low",
        type=float,
        default=300.0,
        help="Voice-audio high-pass cutoff in Hz (default: 300).",
    )
    audio.add_argument(
        "-ah",
        "--audio-high",
        type=float,
        default=4e3,
        help="Voice-audio low-pass cutoff in Hz (default: 4e3).",
    )
    audio.add_argument(
        "-nn",
        "--no-normalize",
        action="store_true",
        help="Write unnormalized float32 WAV instead of listening-normalized int16.",
    )
    audio.add_argument(
        "-np",
        "--normalization-percentile",
        type=float,
        default=99.5,
        help="Absolute-amplitude percentile used for normalization (default: 99.5).",
    )
    audio.add_argument(
        "-nt",
        "--normalization-target",
        type=float,
        default=0.8,
        help="Full-scale target for the selected percentile (default: 0.8).",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "-o",
        "--output-dir",
        help=(
            "Explicit output-directory base; omit to derive a directory name "
            "from the input, frequency, and time window."
        ),
    )
    output.add_argument(
        "-or",
        "--output-root",
        default="results",
        help="Root for automatically named output directories (default: results).",
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
        default="airband_am",
        help="Human-readable label stored in plots and the JSON summary.",
    )
    parser.set_defaults(command_handler=_run_airband_am, command_parser=parser)
    return parser


def _run_airband_am(args: argparse.Namespace) -> int:
    """Run the configured airband decoder, print its summary, and return 0."""

    ddc = AirbandAMDDCConfig(
        fs_in=args.sample_rate,
        fs_mid=args.intermediate_rate,
        fs_out=args.channel_rate,
        first_stage_passband_hz=args.first_stage_passband,
        first_stage_stopband_hz=args.first_stage_stopband,
        channel_passband_hz=args.channel_passband,
        channel_stopband_hz=args.channel_stopband,
        passband_ripple_db=args.passband_ripple,
        stopband_attenuation_db=args.stopband_attenuation,
        chunk_samples=args.chunk_samples,
    )
    config = AirbandAMFileConfig(
        rf_frequency_hz=args.rf_frequency,
        start_seconds=args.start,
        duration_seconds=args.duration,
        dtype=args.dtype,
        padding_seconds=args.padding,
        audio_sample_rate_hz=args.audio_rate,
        audio_low_hz=args.audio_low,
        audio_high_hz=args.audio_high,
        normalize_audio=not args.no_normalize,
        normalization_percentile=args.normalization_percentile,
        normalization_target=args.normalization_target,
        label=args.label,
        ddc=ddc,
    )
    try:
        summary = demodulate_airband_am_file(
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
