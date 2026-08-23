"""Synthetic smoke tests for the FM PSD path."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import signal

from orbdemod.ddc import design_butterworth_lowpass, design_kaiser_lowpass
from orbdemod.fm import (
    FMDDCConfig,
    FMPSDConfig,
    analyze_fm_psd_file,
    compute_mpx_psd,
    downconvert_fm_voltage,
    make_fm_decimation_stages,
    quadrature_discriminator,
)
from orbdemod.fm.pipeline import _prepare_output_dir


def synthetic_mpx(time: np.ndarray) -> np.ndarray:
    audio = 0.45 * np.sin(2.0 * np.pi * 1_000.0 * time)
    pilot = 0.09 * np.sin(2.0 * np.pi * 19_000.0 * time)
    stereo = (
        0.18
        * np.sin(2.0 * np.pi * 2_000.0 * time)
        * np.cos(2.0 * np.pi * 38_000.0 * time)
    )
    rds = 0.03 * np.cos(2.0 * np.pi * 57_000.0 * time)
    return audio + pilot + stereo + rds


class FMPSDTest(unittest.TestCase):
    def test_output_directory_naming_collision_and_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "20250415-1940-0.dat"

            first = _prepare_output_dir(
                input_path=input_path,
                output_dir=None,
                output_root=root / "results",
                rf_frequency_hz=98.3e6,
                start_seconds=30.0,
                duration_seconds=3.0,
                overwrite=False,
            )
            self.assertEqual(
                first,
                (root / "results/20250415-1940-0/98.3MHz/30_3").resolve(),
            )

            second = _prepare_output_dir(
                input_path=input_path,
                output_dir=None,
                output_root=root / "results",
                rf_frequency_hz=98.3e6,
                start_seconds=30.0,
                duration_seconds=3.0,
                overwrite=False,
            )
            self.assertEqual(second.name, "30_3_run02")

            more_precise_frequency = _prepare_output_dir(
                input_path=input_path,
                output_dir=None,
                output_root=root / "results",
                rf_frequency_hz=98.33e6,
                start_seconds=30.0,
                duration_seconds=3.0,
                overwrite=False,
            )
            self.assertEqual(more_precise_frequency.parent.name, "98.33MHz")

            explicit = root / "chosen_output"
            self.assertEqual(
                _prepare_output_dir(
                    input_path=input_path,
                    output_dir=explicit,
                    output_root=root / "unused",
                    rf_frequency_hz=98.3e6,
                    start_seconds=30.0,
                    duration_seconds=3.0,
                    overwrite=False,
                ),
                explicit.resolve(),
            )
            self.assertEqual(
                _prepare_output_dir(
                    input_path=input_path,
                    output_dir=explicit,
                    output_root=root / "unused",
                    rf_frequency_hz=98.3e6,
                    start_seconds=30.0,
                    duration_seconds=3.0,
                    overwrite=False,
                ).name,
                "chosen_output_run02",
            )
            self.assertEqual(
                _prepare_output_dir(
                    input_path=input_path,
                    output_dir=explicit,
                    output_root=root / "unused",
                    rf_frequency_hz=98.3e6,
                    start_seconds=30.0,
                    duration_seconds=3.0,
                    overwrite=True,
                ),
                explicit.resolve(),
            )

    def test_window_selection_modes_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one window mode"):
            FMPSDConfig(rf_frequency_hz=98.3e6).validate()
        with self.assertRaisesRegex(ValueError, "exactly one window mode"):
            FMPSDConfig(
                rf_frequency_hz=98.3e6,
                start_seconds=0.0,
                duration_seconds=1.0,
                start_fraction=0.0,
                stop_fraction=0.5,
            ).validate()

        fraction_config = FMPSDConfig(
            rf_frequency_hz=98.3e6,
            start_fraction=0.25,
            stop_fraction=0.50,
        )
        mode, start, duration = fraction_config.resolve_window(8.0)
        self.assertEqual(mode, "fraction")
        self.assertEqual(start, 2.0)
        self.assertEqual(duration, 2.0)

    def test_waveform_window_defaults_to_centre_and_can_be_selected(self) -> None:
        centred_config = FMPSDConfig(
            rf_frequency_hz=98.3e6,
            start_seconds=0.0,
            duration_seconds=1.0,
        )
        start, duration = centred_config.resolve_waveform_window(0.50)
        self.assertAlmostEqual(start, 0.225)
        self.assertAlmostEqual(duration, 0.050)

        selected_config = FMPSDConfig(
            rf_frequency_hz=98.3e6,
            start_seconds=0.0,
            duration_seconds=1.0,
            waveform_start_seconds=0.12,
            waveform_duration_seconds=0.03,
        )
        start, duration = selected_config.resolve_waveform_window(0.50)
        self.assertAlmostEqual(start, 0.12)
        self.assertAlmostEqual(duration, 0.03)

        with self.assertRaisesRegex(ValueError, "inside the selected"):
            selected_config.resolve_waveform_window(0.10)

    def test_default_filter_response_meets_specification(self) -> None:
        config = FMDDCConfig()
        self.assertEqual(config.chunk_samples, 10_000_000)
        first_stage, second_stage = make_fm_decimation_stages(config)
        first_sos = design_butterworth_lowpass(first_stage)
        first_frequencies, first_response = signal.sosfreqz(
            first_sos,
            worN=131_072,
            fs=config.fs_in,
        )
        second_fir = design_kaiser_lowpass(second_stage)
        second_frequencies, second_response = signal.freqz(
            second_fir,
            worN=131_072,
            fs=config.fs_mid,
        )

        first_db = 20.0 * np.log10(np.maximum(np.abs(first_response), 1e-300))
        second_db = 20.0 * np.log10(np.maximum(np.abs(second_response), 1e-300))
        first_pass = first_db[first_frequencies <= config.passband_hz]
        first_stop = first_db[first_frequencies >= config.fs_mid / 2.0]
        second_pass = second_db[second_frequencies <= config.passband_hz]
        second_stop = second_db[second_frequencies >= config.stopband_hz]

        self.assertLessEqual(float(np.ptp(first_pass)), 0.30)
        self.assertLessEqual(float(np.max(first_stop)), -config.stopband_attenuation_db)
        self.assertLessEqual(float(np.ptp(second_pass)), 0.10)
        self.assertLessEqual(float(np.max(second_stop)), -config.stopband_attenuation_db)

    def test_200_fold_first_decimation_is_chunk_invariant(self) -> None:
        fs_in = 48_000_000.0
        fs_mid = 240_000.0
        fs_out = 120_000.0
        rf_frequency_hz = 10_000_000.0
        time = np.arange(int(0.010 * fs_in)) / fs_in
        voltage = np.round(20_000.0 * np.cos(2.0 * np.pi * rf_frequency_hz * time)).astype("<i2")

        common = dict(
            fs_in=fs_in,
            fs_mid=fs_mid,
            fs_out=fs_out,
            passband_hz=45_000.0,
            stopband_hz=60_000.0,
        )
        one_block, _ = downconvert_fm_voltage(
            voltage,
            rf_frequency_hz,
            FMDDCConfig(chunk_samples=len(voltage), **common),
        )
        many_blocks, _ = downconvert_fm_voltage(
            voltage,
            rf_frequency_hz,
            FMDDCConfig(chunk_samples=12_345, **common),
        )

        self.assertEqual(len(one_block), len(many_blocks))
        np.testing.assert_allclose(one_block, many_blocks, rtol=2e-5, atol=2e-5)
        steady_state_level = float(np.median(np.abs(one_block[len(one_block) // 2 :])))
        self.assertGreater(steady_state_level, 5_000.0)

    def test_discriminator_recovers_mpx_components(self) -> None:
        fs = 240_000.0
        time = np.arange(int(0.25 * fs)) / fs
        expected_mpx_hz = 60_000.0 * synthetic_mpx(time)
        phase = 2.0 * np.pi * np.cumsum(expected_mpx_hz) / fs
        iq = np.exp(1j * phase).astype(np.complex64)

        recovered, carrier_offset_hz = quadrature_discriminator(iq, fs)
        frequencies, psd = compute_mpx_psd(recovered, fs)

        self.assertLess(abs(carrier_offset_hz), 100.0)
        for expected_hz in (1_000.0, 19_000.0, 36_000.0, 40_000.0, 57_000.0):
            band = np.abs(frequencies - expected_hz) <= 60.0
            self.assertTrue(np.any(band))
            self.assertGreater(float(np.max(psd[band])), float(np.median(psd)))

    def test_real_voltage_ddc_and_file_pipeline(self) -> None:
        fs_in = 4_800_000.0
        fs_mid = 480_000.0
        fs_out = 240_000.0
        rf_frequency_hz = 1_000_000.0
        duration_seconds = 0.16
        time = np.arange(int(duration_seconds * fs_in)) / fs_in
        deviation_hz = 55_000.0 * synthetic_mpx(time)
        phase = 2.0 * np.pi * rf_frequency_hz * time
        phase += 2.0 * np.pi * np.cumsum(deviation_hz) / fs_in
        voltage = np.round(20_000.0 * np.cos(phase)).astype("<i2")

        ddc = FMDDCConfig(
            fs_in=fs_in,
            fs_mid=fs_mid,
            fs_out=fs_out,
            passband_hz=90_000.0,
            stopband_hz=120_000.0,
            chunk_samples=100_000,
        )
        iq, _ = downconvert_fm_voltage(voltage, rf_frequency_hz, ddc)
        self.assertGreater(len(iq), 1_000)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "synthetic_fm.dat"
            output_dir = root / "results"
            voltage.tofile(input_path)
            config = FMPSDConfig(
                rf_frequency_hz=rf_frequency_hz,
                start_seconds=0.03,
                duration_seconds=0.10,
                padding_seconds=0.01,
                label="synthetic_fm",
                ddc=ddc,
            )
            summary = analyze_fm_psd_file(input_path, output_dir, config)

            self.assertNotIn("classification", summary)
            self.assertEqual(summary["output_dir"], str(output_dir.resolve()))
            self.assertAlmostEqual(
                summary["waveform_duration_seconds"],
                0.05,
            )
            self.assertAlmostEqual(
                summary["waveform_start_seconds"],
                0.024997916666666665,
            )
            self.assertLess(abs(float(summary["pilot_peak_hz"]) - 19_000.0), 60.0)
            self.assertTrue((output_dir / "fm_psd.png").is_file())
            self.assertTrue((output_dir / "fm_arrays.npz").is_file())
            self.assertTrue((output_dir / "summary.json").is_file())
            self.assertTrue((output_dir / "run_config.json").is_file())

            automatic_root = root / "automatic_results"
            fraction_config = FMPSDConfig(
                rf_frequency_hz=rf_frequency_hz,
                start_fraction=0.1875,
                stop_fraction=0.8125,
                padding_seconds=0.01,
                label="synthetic_fm_fraction",
                ddc=ddc,
            )
            fraction_summary = analyze_fm_psd_file(
                input_path,
                None,
                fraction_config,
                output_root=automatic_root,
            )
            fraction_output_dir = (
                automatic_root / "synthetic_fm" / "1MHz" / "0.03_0.1"
            )
            self.assertEqual(fraction_summary["selection_mode"], "fraction")
            self.assertEqual(
                fraction_summary["output_dir"],
                str(fraction_output_dir.resolve()),
            )
            self.assertAlmostEqual(fraction_summary["start_seconds"], 0.03)
            self.assertAlmostEqual(fraction_summary["duration_seconds"], 0.10)
            self.assertLess(
                abs(float(fraction_summary["pilot_peak_hz"]) - 19_000.0),
                60.0,
            )
            self.assertTrue((fraction_output_dir / "fm_psd.png").is_file())


if __name__ == "__main__":
    unittest.main()
