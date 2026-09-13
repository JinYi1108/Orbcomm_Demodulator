"""Raw-file pipeline for known-frequency Starlink VHF LoRa decoding.

Functions and outputs
---------------------
``_compact_path_number(value)``
    Formats a number for automatic directory names; returns ``str``.
``_prepare_output_dir(input_path, config, output_dir, output_root, overwrite)``
    Creates a non-destructive run directory; returns ``Path``.
``demodulate_starlink_vhf_file(input_path, output_dir, config, ...)``
    Reads one raw-voltage window, applies shared-primitives DDC, LoRa decoding,
    conservative frame parsing, and writes ``summary.json``, ``frames.csv``,
    packet binaries, and ``diagnostic.png``; returns the summary dictionary.

Padding reduces filter/resampling edge effects and remains present during
packet detection; decoded packets are then selected by overlap with the
requested window.  Saved optional IQ remains the trimmed requested window.
Raw inputs are opened read-only.  The pipeline does not scan frequencies,
compare packet cadence, propagate TLEs, or identify a named spacecraft.
"""

from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import StarlinkVHFFileConfig
from .ddc import downconvert_starlink_vhf_voltage
from .frame import parse_starlink_vhf_payload, profile_matches
from .lora import demodulate_starlink_lora
from .plotting import save_starlink_vhf_diagnostic_plot


_MANAGED_FIXED_OUTPUTS = (
    "diagnostic.png",
    "frames.csv",
    "summary.json",
    "channel_iq.c64",
)


def _compact_path_number(value: float) -> str:
    """Return a compact decimal string safe for use in an output path."""

    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def _prepare_output_dir(
    input_path: Path,
    config: StarlinkVHFFileConfig,
    output_dir: str | Path | None,
    output_root: str | Path,
    overwrite: bool,
) -> Path:
    """Create and return a result directory without silently overwriting runs."""

    if output_dir is None:
        frequency = _compact_path_number(config.rf_frequency_hz / 1e6)
        start = _compact_path_number(config.start_seconds)
        duration = _compact_path_number(config.duration_seconds)
        base = Path(output_root).expanduser().resolve() / (
            f"{input_path.stem}_starlink_vhf_{frequency}MHz_"
            f"start{start}s_dur{duration}s"
        )
    else:
        base = Path(output_dir).expanduser().resolve()
    selected = base
    if selected.exists() and not overwrite:
        run_number = 2
        while selected.with_name(f"{base.name}_run{run_number:02d}").exists():
            run_number += 1
        selected = selected.with_name(f"{base.name}_run{run_number:02d}")
    selected.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for filename in _MANAGED_FIXED_OUTPUTS:
            path = selected / filename
            if path.is_file():
                path.unlink()
        for pattern in ("packet_*_payload.bin", "packet_*_frame.bin"):
            for path in selected.glob(pattern):
                if path.is_file():
                    path.unlink()
    return selected


def demodulate_starlink_vhf_file(
    input_path: str | Path,
    output_dir: str | Path | None,
    config: StarlinkVHFFileConfig,
    *,
    output_root: str | Path = "results",
    overwrite: bool = False,
) -> Dict[str, object]:
    """Decode one known Starlink VHF file window and return its run summary."""

    config.validate()
    input_path = Path(input_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    dtype = np.dtype(config.dtype)
    file_bytes = input_path.stat().st_size
    if file_bytes % dtype.itemsize:
        raise ValueError("Input file size is not a whole number of dtype samples.")
    sample_count = file_bytes // dtype.itemsize
    file_duration = sample_count / config.ddc.fs_in
    requested_stop = config.start_seconds + config.duration_seconds
    if requested_stop > file_duration + 1e-12:
        raise ValueError(
            f"Requested window ends at {requested_stop:.6f} s, but the file "
            f"contains only {file_duration:.6f} s."
        )

    padded_start = max(0.0, config.start_seconds - config.padding_seconds)
    padded_stop = min(file_duration, requested_stop + config.padding_seconds)
    start_sample = int(round(padded_start * config.ddc.fs_in))
    stop_sample = int(round(padded_stop * config.ddc.fs_in))
    raw = np.memmap(input_path, dtype=dtype, mode="r")
    initial_phase = float(
        np.remainder(
            2.0 * np.pi * config.rf_frequency_hz * start_sample / config.ddc.fs_in,
            2.0 * np.pi,
        )
    )
    padded_iq, _ = downconvert_starlink_vhf_voltage(
        raw[start_sample:stop_sample],
        config.rf_frequency_hz,
        config.ddc,
        initial_phase,
    )
    trim_seconds = config.start_seconds - padded_start
    iq_start = int(round(trim_seconds * config.ddc.fs_out))
    iq_count = int(round(config.duration_seconds * config.ddc.fs_out))
    trimmed_iq = np.asarray(
        padded_iq[iq_start : iq_start + iq_count], dtype=np.complex64
    )
    if len(trimmed_iq) != iq_count:
        raise ValueError("DDC output was shorter than the requested trimmed window.")

    decoded_packets = demodulate_starlink_lora(
        padded_iq,
        config.ddc.fs_out,
        config.rf_frequency_hz,
        config.lora,
    )
    packets = [
        packet
        for packet in decoded_packets
        if padded_start + packet.packet_stop_seconds > config.start_seconds
        and padded_start + packet.preamble_start_seconds < requested_stop
    ]
    selected_output = _prepare_output_dir(
        input_path, config, output_dir, output_root, overwrite
    )
    packet_summaries: List[Dict[str, object]] = []
    csv_rows: List[Dict[str, object]] = []
    output_files: Dict[str, object] = {}
    for index, packet in enumerate(packets):
        payload = packet.payload_bytes
        try:
            parsed = parse_starlink_vhf_payload(payload)
            parse_error = None
        except Exception as error:
            parsed = {
                "payload_length_bytes": len(payload),
                "payload_hex": payload.hex(),
                "field_interpretation_available": False,
            }
            parse_error = f"{type(error).__name__}: {error}"
        payload_path = selected_output / f"packet_{index:03d}_payload.bin"
        frame_path = selected_output / f"packet_{index:03d}_frame.bin"
        payload_path.write_bytes(payload)
        frame_path.write_bytes(packet.frame_bytes)
        matches = profile_matches(packet, config.lora)
        concentration_finite = packet.concentration_db[np.isfinite(packet.concentration_db)]
        concentration = {
            "minimum_db": float(np.min(concentration_finite)) if concentration_finite.size else None,
            "median_db": float(np.median(concentration_finite)) if concentration_finite.size else None,
            "maximum_db": float(np.max(concentration_finite)) if concentration_finite.size else None,
        }
        packet_summary: Dict[str, object] = {
            "index": index,
            "preamble_start_in_window_seconds": (
                padded_start + packet.preamble_start_seconds - config.start_seconds
            ),
            "preamble_start_in_file_seconds": padded_start + packet.preamble_start_seconds,
            "phy_data_start_in_file_seconds": padded_start + packet.phy_data_start_seconds,
            # Compatibility key: this is the PHY-data/header-symbol start, not
            # necessarily the first user-payload byte.
            "payload_start_in_file_seconds": padded_start + packet.phy_data_start_seconds,
            "packet_stop_in_file_seconds": padded_start + packet.packet_stop_seconds,
            "symbol_count": int(packet.symbols.size),
            "has_explicit_header": packet.has_explicit_header,
            "header_valid": packet.header_valid,
            "header_payload_length": packet.header_payload_length,
            "coding_rate": packet.coding_rate,
            "coding_rate_label": f"4/{packet.coding_rate + 4}",
            "crc_enabled": packet.crc_enabled,
            "carrier_offset_hz": packet.carrier_offset_hz,
            "observed_center_hz": config.rf_frequency_hz + packet.carrier_offset_hz,
            "netid_symbol_offsets": list(packet.netid_symbol_offsets),
            "received_crc_hex": bytes(packet.received_crc).hex(),
            "calculated_crc_hex": bytes(packet.calculated_crc).hex(),
            "crc_valid": packet.crc_valid,
            "decode_error": packet.decode_error,
            "parse_error": parse_error,
            "starlink_profile_match": matches,
            "dechirp_concentration": concentration,
            "decoded_payload": parsed,
            "payload_binary": str(payload_path),
            "frame_binary": str(frame_path),
            "decoded_payload_and_crc_binary": str(frame_path),
        }
        packet_summaries.append(packet_summary)
        csv_rows.append(
            {
                "index": index,
                "preamble_file_s": packet_summary["preamble_start_in_file_seconds"],
                "payload_file_s": packet_summary["payload_start_in_file_seconds"],
                "stop_file_s": packet_summary["packet_stop_in_file_seconds"],
                "carrier_offset_hz": packet.carrier_offset_hz,
                "crc_valid": packet.crc_valid,
                "header_valid": packet.header_valid,
                "header_payload_bytes": packet.header_payload_length,
                "coding_rate": packet.coding_rate,
                "crc_enabled": packet.crc_enabled,
                "profile_match": matches,
                "payload_bytes": len(payload),
                "length_profile": parsed.get("length_profile"),
                "message_number_u24_le": parsed.get("message_number_u24_le"),
                "protocol_id_u16_le": parsed.get("protocol_id_u16_le"),
                "spacecraft_id_u16_le": parsed.get("spacecraft_id_u16_le"),
                "time_candidate_u32_le": parsed.get("time_candidate_u32_le"),
                "decode_error": packet.decode_error,
                "parse_error": parse_error,
            }
        )

    plot_path = selected_output / "diagnostic.png"
    csv_path = selected_output / "frames.csv"
    summary_path = selected_output / "summary.json"
    display_limits_db = save_starlink_vhf_diagnostic_plot(
        plot_path,
        padded_iq,
        config.ddc.fs_out,
        packets,
        rf_frequency_hz=config.rf_frequency_hz,
        window_start_seconds=padded_start,
        requested_start_seconds=config.start_seconds,
        requested_stop_seconds=requested_stop,
        packet_index=config.diagnostic_packet_index,
        label=config.label,
    )
    fieldnames = [
        "index", "preamble_file_s", "payload_file_s", "stop_file_s",
        "carrier_offset_hz", "crc_valid", "header_valid", "header_payload_bytes",
        "coding_rate", "crc_enabled", "profile_match", "payload_bytes",
        "length_profile", "message_number_u24_le", "protocol_id_u16_le",
        "spacecraft_id_u16_le", "time_candidate_u32_le", "decode_error",
        "parse_error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    output_files.update({"diagnostic_png": str(plot_path), "frames_csv": str(csv_path)})
    if config.save_iq:
        iq_path = selected_output / "channel_iq.c64"
        trimmed_iq.tofile(iq_path)
        output_files["channel_iq"] = str(iq_path)
    output_files["summary_json"] = str(summary_path)

    summary: Dict[str, object] = {
        "input_path": str(input_path),
        "output_dir": str(selected_output),
        "label": config.label,
        "requested_start_seconds": config.start_seconds,
        "requested_duration_seconds": config.duration_seconds,
        "padding_seconds": config.padding_seconds,
        "decoded_window_start_seconds": padded_start,
        "decoded_window_duration_seconds": padded_stop - padded_start,
        "diagnostic_magnitude_limits_db": list(display_limits_db),
        "config": asdict(config),
        "packet_count": len(packets),
        "decoded_packet_count_including_padding": len(decoded_packets),
        "crc_valid_packet_count": sum(packet.crc_valid is True for packet in packets),
        "starlink_profile_match_count": sum(
            summary["starlink_profile_match"] for summary in packet_summaries
        ),
        "packets": packet_summaries,
        "output_files": output_files,
        "scope": (
            "Known-frequency Starlink-profile VHF LoRa decoding only; no "
            "frequency scan, cadence inference, TLE propagation, or named-spacecraft identification."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
