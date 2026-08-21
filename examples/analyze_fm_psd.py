"""Create an FM composite PSD plot from one raw-voltage time window."""

from __future__ import annotations

import argparse
import json

from orbdemod.fm import FMDDCConfig, FMPSDConfig, analyze_fm_psd_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw real-voltage file.")
    parser.add_argument(
        "--rf-frequency",
        required=True,
        type=float,
        help="Target FM station frequency in Hz, for example 100.1e6.",
    )
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds.")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.5,
        help="Analysis-window duration in seconds.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--label", default="direct_fm_test")
    parser.add_argument("--dtype", default="<i2", help="Default: little-endian int16.")
    parser.add_argument("--sample-rate", type=float, default=480e6)
    parser.add_argument("--intermediate-rate", type=float, default=2.4e6)
    parser.add_argument("--channel-rate", type=float, default=240e3)
    parser.add_argument("--passband", type=float, default=90e3)
    parser.add_argument("--stopband", type=float, default=120e3)
    parser.add_argument("--stopband-attenuation", type=float, default=60.0)
    parser.add_argument("--padding", type=float, default=0.020)
    parser.add_argument("--chunk-samples", type=int, default=2_000_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
        dtype=args.dtype,
        padding_seconds=args.padding,
        label=args.label,
        ddc=ddc,
    )
    summary = analyze_fm_psd_file(args.input, args.output_dir, config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
