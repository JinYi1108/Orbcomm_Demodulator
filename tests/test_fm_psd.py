"""Synthetic smoke tests for the FM PSD path."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import signal

from orbdemod.fm import (
    FMDDCConfig,
    FMPSDConfig,
    analyze_fm_psd_file,
    compute_mpx_psd,
    downconvert_real_voltage,
    quadrature_discriminator,
)
from orbdemod.fm.ddc import _design_first_stage, _design_second_stage


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
    def test_default_filter_response_meets_specification(self) -> None:
        config = FMDDCConfig()
        first_sos = _design_first_stage(config)
        first_frequencies, first_response = signal.sosfreqz(
            first_sos,
            worN=131_072,
            fs=config.fs_in,
        )
        second_fir = _design_second_stage(config)
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
        one_block, _ = downconvert_real_voltage(
            voltage,
            rf_frequency_hz,
            FMDDCConfig(chunk_samples=len(voltage), **common),
        )
        many_blocks, _ = downconvert_real_voltage(
            voltage,
            rf_frequency_hz,
            FMDDCConfig(chunk_samples=12_345, **common),
        )

        self.assertEqual(len(one_block), len(many_blocks))
        np.testing.assert_allclose(one_block, many_blocks, rtol=2e-5, atol=2e-5)

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
        iq, _ = downconvert_real_voltage(voltage, rf_frequency_hz, ddc)
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

            self.assertEqual(summary["classification"], "not_performed")
            self.assertLess(abs(float(summary["pilot_peak_hz"]) - 19_000.0), 60.0)
            self.assertTrue((output_dir / "fm_psd.png").is_file())
            self.assertTrue((output_dir / "fm_arrays.npz").is_file())
            self.assertTrue((output_dir / "summary.json").is_file())
            self.assertTrue((output_dir / "run_config.json").is_file())


if __name__ == "__main__":
    unittest.main()
