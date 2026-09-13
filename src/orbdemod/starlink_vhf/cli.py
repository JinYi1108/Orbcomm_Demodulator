"""Command-line registration for Starlink-profile VHF LoRa decoding.

Functions and outputs
---------------------
``add_starlink_vhf_subparser(subparsers)``
    Registers ``lfdemod starlink-vhf``; returns its ``ArgumentParser``.
``_run_starlink_vhf(args)``
    Converts arguments to configurations, runs the raw-file pipeline, prints
    its JSON summary, and returns process status ``0`` on success or ``2`` on
    a handled input/dependency error.
``_parse_payload_lengths(value)``
    Parses comma-separated payload lengths for the CLI; returns ``tuple[int]``
    or raises ``ArgumentTypeError``.

Every public long option has a short alias and explanatory help text.
"""

from __future__ import annotations

import argparse
import json

from .config import (
    StarlinkVHFDDCConfig,
    StarlinkVHFFileConfig,
    StarlinkVHFLoRaConfig,
)
from .lora import MissingStarlinkVHFDependencyError
from .pipeline import demodulate_starlink_vhf_file


def _parse_payload_lengths(value: str) -> tuple[int, ...]:
    """Return comma-separated payload lengths as a tuple for argparse."""

    try:
        lengths = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "payload lengths must be comma-separated integers"
        ) from error
    if not lengths:
        raise argparse.ArgumentTypeError("at least one payload length is required")
    return lengths


def add_starlink_vhf_subparser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Register and return the ``lfdemod starlink-vhf`` parser."""

    parser = subparsers.add_parser(
        "starlink-vhf",
        help="Decode a known Starlink-profile VHF LoRa channel.",
        description=(
            "Decode a known RF frequency and raw-file window using the observed "
            "Starlink VHF LoRa profile. This command does not scan frequencies, "
            "analyze cadence, propagate TLEs, or name a spacecraft."
        ),
    )
    parser.add_argument("-i", "--input", required=True, help="Raw real-voltage input file.")
    parser.add_argument(
        "-f", "--rf-frequency", required=True, type=float,
        help="Known RF center to tune, in Hz (for example 137.055e6).",
    )
    parser.add_argument(
        "-s", "--start", required=True, type=float,
        help="Start relative to the raw-file beginning, in seconds.",
    )
    parser.add_argument(
        "-d", "--duration", required=True, type=float,
        help="Duration of the packet-search window, in seconds.",
    )

    ddc = parser.add_argument_group("Starlink VHF DDC")
    ddc.add_argument("-dt", "--dtype", default="<i2", help="Raw NumPy dtype (default: <i2).")
    ddc.add_argument("-sr", "--sample-rate", type=float, default=480e6, help="Raw sample rate in samples/s (default: 480e6).")
    ddc.add_argument("-ir", "--intermediate-rate", type=float, default=2.4e6, help="First-stage IQ rate in samples/s (default: 2.4e6).")
    ddc.add_argument("-cr", "--channel-rate", type=float, default=240e3, help="Final channel-IQ rate in samples/s (default: 240e3).")
    ddc.add_argument("-fp", "--first-stage-passband", type=float, default=150e3, help="One-sided first-stage passband edge in Hz (default: 150e3).")
    ddc.add_argument("-fs", "--first-stage-stopband", type=float, default=1e6, help="One-sided first-stage stopband edge in Hz (default: 1e6).")
    ddc.add_argument("-cp", "--channel-passband", type=float, default=50e3, help="One-sided final-channel passband edge in Hz (default: 50e3).")
    ddc.add_argument("-cb", "--channel-stopband", type=float, default=100e3, help="One-sided final-channel stopband edge in Hz (default: 100e3).")
    ddc.add_argument("-pr", "--passband-ripple", type=float, default=0.5, help="Maximum passband ripple in dB (default: 0.5).")
    ddc.add_argument("-sa", "--stopband-attenuation", type=float, default=70.0, help="Required stopband attenuation in dB (default: 70).")
    ddc.add_argument("-cs", "--chunk-samples", type=int, default=5_000_000, help="High-rate samples processed per block (default: 5000000).")
    ddc.add_argument("-pd", "--padding", type=float, default=0.050, help="Filter/resampling padding retained for edge-packet detection (default: 0.05 s per side).")

    profile = parser.add_argument_group("Starlink VHF LoRa profile")
    profile.add_argument("-bw", "--bandwidth", type=float, default=125_000.0 / 3.0, help="LoRa bandwidth in Hz (default: 125000/3).")
    profile.add_argument("-sf", "--spreading-factor", type=int, default=8, help="LoRa spreading factor (default: 8).")
    profile.add_argument("-pl", "--preamble-symbols", type=int, default=15, help="Expected preamble length in symbols (default: 15).")
    profile.add_argument("-sw", "--sync-word", type=lambda value: int(value, 0), default=0x12, help="Expected LoRa sync byte, decimal or 0x-prefixed (default: 0x12).")
    profile.add_argument("-ih", "--implicit-header", action="store_true", help="Use implicit-header decoding; the validated default is explicit header.")
    profile.add_argument("-eb", "--expected-payload-bytes", type=int, default=81, help="Default/implicit payload length in bytes (default: 81).")
    profile.add_argument("-ab", "--accepted-payload-bytes", type=_parse_payload_lengths, default=(73, 81, 87, 104, 227), help="Comma-separated lengths accepted as known Starlink VHF profiles (default: 73,81,87,104,227).")
    profile.add_argument("-ec", "--expected-coding-rate", type=int, default=1, help="Expected LoRa coding-rate index 1..4, where 1 means 4/5 (default: 1).")
    profile.add_argument("-nc", "--no-expected-crc", action="store_true", help="Expect a profile without payload CRC; validated default expects CRC.")
    profile.add_argument("-st", "--sync-tolerance", type=float, default=0.75, help="Cyclic sync-symbol tolerance in symbol bins (default: 0.75).")
    profile.add_argument("-nl", "--no-forced-ldro", action="store_true", help="Disable the profile's forced low-data-rate optimization flag.")

    output = parser.add_argument_group("output")
    output.add_argument("-o", "--output-dir", help="Explicit output-directory base; omit for automatic naming.")
    output.add_argument("-or", "--output-root", default="results", help="Root for automatically named output directories (default: results).")
    output.add_argument("-ow", "--overwrite", action="store_true", help="Replace fixed outputs in the base directory instead of adding run02.")
    output.add_argument("-iq", "--save-iq", action="store_true", help="Also save trimmed 240 kS/s complex64 channel IQ.")
    output.add_argument("-pi", "--diagnostic-packet-index", type=int, default=0, help="Zero-based decoded packet selected for symbol diagnostics (default: 0).")
    output.add_argument("-l", "--label", default="starlink_vhf", help="Human-readable plot and summary label.")
    parser.set_defaults(command_handler=_run_starlink_vhf, command_parser=parser)
    return parser


def _run_starlink_vhf(args: argparse.Namespace) -> int:
    """Run the configured decoder, print its JSON summary, and return status."""

    ddc = StarlinkVHFDDCConfig(
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
    lora = StarlinkVHFLoRaConfig(
        bandwidth_hz=args.bandwidth,
        spreading_factor=args.spreading_factor,
        preamble_symbols=args.preamble_symbols,
        sync_word=args.sync_word,
        has_explicit_header=not args.implicit_header,
        expected_payload_bytes=args.expected_payload_bytes,
        accepted_payload_bytes=args.accepted_payload_bytes,
        expected_coding_rate=args.expected_coding_rate,
        expected_crc_enabled=not args.no_expected_crc,
        sync_tolerance_symbols=args.sync_tolerance,
        force_low_data_rate_optimization=not args.no_forced_ldro,
    )
    config = StarlinkVHFFileConfig(
        rf_frequency_hz=args.rf_frequency,
        start_seconds=args.start,
        duration_seconds=args.duration,
        dtype=args.dtype,
        padding_seconds=args.padding,
        save_iq=args.save_iq,
        diagnostic_packet_index=args.diagnostic_packet_index,
        label=args.label,
        ddc=ddc,
        lora=lora,
    )
    try:
        summary = demodulate_starlink_vhf_file(
            args.input,
            args.output_dir,
            config,
            output_root=args.output_root,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError, MissingStarlinkVHFDependencyError) as error:
        args.command_parser.error(str(error))
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
