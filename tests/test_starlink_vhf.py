"""Synthetic tests for the Starlink-profile VHF decoder.

Test functions and outputs
--------------------------
``StarlinkVHFSyntheticTest.setUp()``
    Creates a small reusable DDC configuration; returns ``None``.
``test_ddc_preserves_configured_channel_tone()``
    Verifies shared-primitives DDC frequency translation and output rate;
    returns no value.
``test_default_filters_meet_specification()``
    Verifies both default anti-alias responses; returns no value.
``test_profile_parser_recovers_known_fields()``
    Verifies conservative parsing of a known CRC-valid payload; returns no
    value.
``test_all_reported_payload_lengths_have_conservative_profiles()``
    Verifies 73/81/87/104/227-byte length routing; returns no value.
``test_unknown_payload_length_remains_saved_but_unrecognized()``
    Verifies raw preservation without invented parsing; returns no value.
``test_lora_rate_and_sync_helpers()``
    Verifies exact resampling and sync-byte conversion; returns no value.
``test_optional_lora_backend_recovers_synthetic_payload()``
    When lora-phy is installed, verifies physical modulation-to-CRC decoding;
    returns no value or is skipped when the optional package is absent.
``test_optional_lora_backend_distinguishes_absent_crc()``
    Verifies tri-state payload-CRC reporting; returns no value or is skipped.
``test_optional_backend_supports_configured_implicit_header()``
    Verifies configurable implicit-header operation; returns no value or is
    skipped.
``test_optional_backend_decodes_all_reported_payload_lengths()``
    Verifies PHY decoding of all reported length profiles; returns no value or
    is skipped.
``test_overwrite_removes_only_managed_starlink_outputs()``
    Verifies stale managed-output cleanup and preservation of unrelated files;
    returns no value.
``test_diagnostic_plot_rejects_empty_iq()``
    Verifies plotting input validation; returns no value.
``test_public_starlink_vhf_api_is_importable()``
    Verifies lowercase public exports without requiring lora-phy; returns no
    value.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import signal

from orbdemod.ddc import (
    design_butterworth_lowpass,
    design_kaiser_lowpass,
    validate_filter_response,
)
from lfdemod.starlink_vhf import (
    KNOWN_STARLINK_VHF_PAYLOAD_BYTES,
    StarlinkVHFDDCConfig,
    StarlinkVHFFileConfig,
    StarlinkVHFLoRaConfig,
    demodulate_starlink_vhf_file,
    downconvert_starlink_vhf_voltage,
    make_starlink_vhf_decimation_stages,
    parse_starlink_vhf_payload,
    prepare_lora_iq,
    profile_matches,
    starlink_sync_symbol_offsets,
)
from orbdemod.starlink_vhf.pipeline import _prepare_output_dir
from orbdemod.starlink_vhf.plotting import save_starlink_vhf_diagnostic_plot


class StarlinkVHFSyntheticTest(unittest.TestCase):
    """Exercise reusable DDC and protocol helpers without private raw data."""

    def setUp(self) -> None:
        """Create and store a compact two-stage DDC configuration."""

        self.ddc = StarlinkVHFDDCConfig(
            fs_in=240e3,
            fs_mid=120e3,
            fs_out=60e3,
            first_stage_passband_hz=30e3,
            first_stage_stopband_hz=50e3,
            channel_passband_hz=15e3,
            channel_stopband_hz=25e3,
            passband_ripple_db=0.5,
            stopband_attenuation_db=60.0,
            chunk_samples=17_003,
        )

    def test_ddc_preserves_configured_channel_tone(self) -> None:
        """Confirm DDC output rate and residual offset for a real RF tone."""

        duration = 0.20
        rf_hz = 70e3
        offset_hz = 3e3
        times = np.arange(int(self.ddc.fs_in * duration)) / self.ddc.fs_in
        raw = np.cos(2.0 * np.pi * (rf_hz + offset_hz) * times)
        iq, _ = downconvert_starlink_vhf_voltage(raw, rf_hz, self.ddc)
        frequencies, psd = signal.welch(
            iq, fs=self.ddc.fs_out, nperseg=4096, return_onesided=False
        )
        recovered = float(frequencies[np.argmax(psd)])
        self.assertEqual(len(iq), int(duration * self.ddc.fs_out))
        self.assertAlmostEqual(recovered, offset_hz, delta=20.0)

    def test_default_filters_meet_specification(self) -> None:
        """Confirm default Starlink VHF DDC stages meet declared limits."""

        first, channel = make_starlink_vhf_decimation_stages()
        self.assertTrue(
            validate_filter_response(design_butterworth_lowpass(first), first).meets_specification
        )
        self.assertTrue(
            validate_filter_response(design_kaiser_lowpass(channel), channel).meets_specification
        )

    def test_profile_parser_recovers_known_fields(self) -> None:
        """Confirm known decoded bytes yield stable conservative field values."""

        payload = bytes.fromhex(
            "2282342717cc0000d0a3504505003966fe6743003104fc340a0021ec044a7303"
            "3a09a19a540120050f029d18ac1a35e6911d344bf6ff52ac0400c37afcff6295"
            "05410e000000000000c08c4400ffffa800"
        )
        parsed = parse_starlink_vhf_payload(payload)
        self.assertEqual(parsed["payload_length_bytes"], 81)
        self.assertEqual(parsed["message_number_u24_le"], 3_441_186)
        self.assertEqual(parsed["spacecraft_id_u16_le"], 5_927)
        self.assertTrue(parsed["fixed_byte_profile_match"])

    def test_all_reported_payload_lengths_have_conservative_profiles(self) -> None:
        """Accept reported lengths without inventing non-81-byte field meanings."""

        self.assertEqual(KNOWN_STARLINK_VHF_PAYLOAD_BYTES, (73, 81, 87, 104, 227))
        for length in KNOWN_STARLINK_VHF_PAYLOAD_BYTES:
            parsed = parse_starlink_vhf_payload(bytes(length))
            self.assertTrue(parsed["known_length_profile"])
            self.assertEqual(parsed["payload_length_bytes"], length)
            self.assertEqual(parsed["field_interpretation_available"], length == 81)
            if length != 81:
                self.assertIsNone(parsed["fixed_byte_profile_match"])

    def test_unknown_payload_length_remains_saved_but_unrecognized(self) -> None:
        """Return raw bytes for unknown lengths instead of raising or guessing."""

        parsed = parse_starlink_vhf_payload(b"short")
        self.assertFalse(parsed["known_length_profile"])
        self.assertEqual(parsed["payload_hex"], b"short".hex())

    def test_lora_rate_and_sync_helpers(self) -> None:
        """Confirm exact two-samples-per-chip rate and 0x12 symbol offsets."""

        config = StarlinkVHFLoRaConfig()
        iq = np.ones(2_400, dtype=np.complex64)
        resampled, rate = prepare_lora_iq(iq, 240e3, config)
        self.assertAlmostEqual(rate, 2.0 * config.bandwidth_hz)
        self.assertEqual(len(resampled), 834)
        self.assertEqual(starlink_sync_symbol_offsets(0x12), (8, 16))

    @unittest.skipUnless(importlib.util.find_spec("lora_phy"), "optional lora-phy is absent")
    def test_optional_lora_backend_recovers_synthetic_payload(self) -> None:
        """Confirm an installed LoRa backend recovers bytes and a valid CRC."""

        if not hasattr(np, "bitwise_right_shift"):
            np.bitwise_right_shift = np.right_shift  # type: ignore[attr-defined]
        from lora_phy import LoRaTransmitter
        from lfdemod.starlink_vhf import demodulate_starlink_lora

        config = StarlinkVHFLoRaConfig()
        lora_rate = 2.0 * config.bandwidth_hz
        payload = np.arange(config.expected_payload_bytes, dtype=np.uint8)
        transmitter = LoRaTransmitter(
            config.spreading_factor,
            config.bandwidth_hz,
            lora_rate,
            has_header=True,
            coding_rate=1,
            enable_crc=True,
            preamble_len=config.preamble_symbols,
        )
        transmitter._proc.low_data_rate_optimization = True
        symbols = transmitter.encode(payload)
        packet_iq = transmitter.modulate(symbols, cfo=-500.0)
        pad = np.zeros(int(round(0.05 * lora_rate)), dtype=np.complex128)
        packets = demodulate_starlink_lora(
            np.r_[pad, packet_iq, pad], lora_rate, 137.055e6, config
        )
        self.assertEqual(len(packets), 1)
        self.assertTrue(packets[0].crc_valid)
        packet = packets[0]
        self.assertEqual(packet.payload_bytes, payload.tobytes())
        self.assertTrue(packet.has_explicit_header)
        self.assertTrue(packet.header_valid)
        self.assertEqual(packet.header_payload_length, 81)
        self.assertEqual(packet.coding_rate, 1)
        self.assertTrue(packet.crc_enabled)
        self.assertIsNone(packet.decode_error)

    @unittest.skipUnless(importlib.util.find_spec("lora_phy"), "optional lora-phy is absent")
    def test_optional_lora_backend_distinguishes_absent_crc(self) -> None:
        """Represent a header-declared no-CRC packet with crc_valid=None."""

        if not hasattr(np, "bitwise_right_shift"):
            np.bitwise_right_shift = np.right_shift  # type: ignore[attr-defined]
        from lora_phy import LoRaTransmitter
        from lfdemod.starlink_vhf import demodulate_starlink_lora

        config = StarlinkVHFLoRaConfig(expected_crc_enabled=False)
        lora_rate = 2.0 * config.bandwidth_hz
        payload = np.arange(config.expected_payload_bytes, dtype=np.uint8)
        transmitter = LoRaTransmitter(
            config.spreading_factor,
            config.bandwidth_hz,
            lora_rate,
            has_header=True,
            coding_rate=config.expected_coding_rate,
            enable_crc=False,
            preamble_len=config.preamble_symbols,
        )
        transmitter._proc.low_data_rate_optimization = True
        signal_iq = transmitter.modulate(transmitter.encode(payload))
        pad = np.zeros(int(round(0.05 * lora_rate)), dtype=np.complex128)
        packets = demodulate_starlink_lora(
            np.r_[pad, signal_iq, pad], lora_rate, 137.055e6, config
        )
        self.assertEqual(len(packets), 1)
        self.assertFalse(packets[0].crc_enabled)
        self.assertIsNone(packets[0].crc_valid)
        self.assertEqual(packets[0].payload_bytes, payload.tobytes())

    @unittest.skipUnless(importlib.util.find_spec("lora_phy"), "optional lora-phy is absent")
    def test_optional_backend_supports_configured_implicit_header(self) -> None:
        """Keep implicit-header operation configurable without changing defaults."""

        if not hasattr(np, "bitwise_right_shift"):
            np.bitwise_right_shift = np.right_shift  # type: ignore[attr-defined]
        from lora_phy import LoRaTransmitter
        from lfdemod.starlink_vhf import demodulate_starlink_lora

        payload = np.frombuffer(
            bytes.fromhex(
                "2282342717cc0000d0a3504505003966fe6743003104fc340a0021ec044a7303"
                "3a09a19a540120050f029d18ac1a35e6911d344bf6ff52ac0400c37afcff6295"
                "05410e000000000000c08c4400ffffa800"
            ),
            dtype=np.uint8,
        )
        config = StarlinkVHFLoRaConfig(
            has_explicit_header=False,
            sync_word=0x34,
        )
        lora_rate = 2.0 * config.bandwidth_hz
        transmitter = LoRaTransmitter(
            config.spreading_factor,
            config.bandwidth_hz,
            lora_rate,
            has_header=False,
            coding_rate=config.expected_coding_rate,
            enable_crc=config.expected_crc_enabled,
            preamble_len=config.preamble_symbols,
        )
        transmitter._proc.low_data_rate_optimization = True
        signal_iq = transmitter.modulate(transmitter.encode(payload))
        pad = np.zeros(int(round(0.05 * lora_rate)), dtype=np.complex128)
        packets = demodulate_starlink_lora(
            np.r_[pad, signal_iq, pad], lora_rate, 137.055e6, config
        )
        self.assertEqual(len(packets), 1)
        self.assertFalse(packets[0].has_explicit_header)
        self.assertIsNone(packets[0].header_valid)
        self.assertEqual(packets[0].payload_bytes, payload.tobytes())
        self.assertTrue(profile_matches(packets[0], config))

    @unittest.skipUnless(importlib.util.find_spec("lora_phy"), "optional lora-phy is absent")
    def test_optional_backend_decodes_all_reported_payload_lengths(self) -> None:
        """Exercise formal PHY/output support for every reported length profile."""

        if not hasattr(np, "bitwise_right_shift"):
            np.bitwise_right_shift = np.right_shift  # type: ignore[attr-defined]
        from lora_phy import LoRaTransmitter
        from lfdemod.starlink_vhf import demodulate_starlink_lora

        known_81 = bytes.fromhex(
            "2282342717cc0000d0a3504505003966fe6743003104fc340a0021ec044a7303"
            "3a09a19a540120050f029d18ac1a35e6911d344bf6ff52ac0400c37afcff6295"
            "05410e000000000000c08c4400ffffa800"
        )
        # lora-phy's synthetic transmitter emits NetID offsets (24, 32),
        # corresponding to conventional sync byte 0x34. Real 21CMA defaults
        # remain 0x12 and are validated separately.
        config = StarlinkVHFLoRaConfig(sync_word=0x34)
        lora_rate = 2.0 * config.bandwidth_hz
        for length in KNOWN_STARLINK_VHF_PAYLOAD_BYTES:
            with self.subTest(length=length):
                payload = (
                    np.frombuffer(known_81, dtype=np.uint8)
                    if length == 81
                    else np.arange(length, dtype=np.uint8)
                )
                transmitter = LoRaTransmitter(
                    config.spreading_factor,
                    config.bandwidth_hz,
                    lora_rate,
                    has_header=True,
                    coding_rate=config.expected_coding_rate,
                    enable_crc=True,
                    preamble_len=config.preamble_symbols,
                )
                transmitter._proc.low_data_rate_optimization = True
                signal_iq = transmitter.modulate(transmitter.encode(payload))
                pad = np.zeros(int(round(0.05 * lora_rate)), dtype=np.complex128)
                packets = demodulate_starlink_lora(
                    np.r_[pad, signal_iq, pad], lora_rate, 137.055e6, config
                )
                self.assertEqual(len(packets), 1)
                self.assertEqual(packets[0].header_payload_length, length)
                self.assertEqual(packets[0].payload_bytes, payload.tobytes())
                self.assertTrue(packets[0].crc_valid)
                self.assertTrue(profile_matches(packets[0], config))

    def test_overwrite_removes_only_managed_starlink_outputs(self) -> None:
        """Prevent stale packet binaries while preserving unrelated user files."""

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "run"
            base.mkdir()
            (base / "packet_009_payload.bin").write_bytes(b"old")
            (base / "packet_009_frame.bin").write_bytes(b"old")
            (base / "summary.json").write_text("old", encoding="utf-8")
            (base / "notes.txt").write_text("keep", encoding="utf-8")
            config = StarlinkVHFFileConfig(
                rf_frequency_hz=137.055e6,
                start_seconds=0.0,
                duration_seconds=1.0,
            )
            selected = _prepare_output_dir(
                Path("input.dat"), config, base, directory, overwrite=True
            )
            self.assertEqual(selected, base.resolve())
            self.assertFalse((base / "packet_009_payload.bin").exists())
            self.assertFalse((base / "summary.json").exists())
            self.assertTrue((base / "notes.txt").exists())

    def test_diagnostic_plot_rejects_empty_iq(self) -> None:
        """Fail with an actionable message before SciPy receives empty IQ."""

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "non-empty"):
                save_starlink_vhf_diagnostic_plot(
                    Path(directory) / "plot.png",
                    np.array([], dtype=np.complex64),
                    240e3,
                    [],
                    rf_frequency_hz=137.055e6,
                    window_start_seconds=0.0,
                    label="test",
                )

    def test_public_starlink_vhf_api_is_importable(self) -> None:
        """Confirm configuration and raw-file pipeline public exports exist."""

        self.assertIsNotNone(StarlinkVHFDDCConfig)
        self.assertTrue(callable(demodulate_starlink_vhf_file))


if __name__ == "__main__":
    unittest.main()
