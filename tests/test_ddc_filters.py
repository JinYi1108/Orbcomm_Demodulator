"""Tests for the shared explicit-filter DDC building blocks."""

from __future__ import annotations

import unittest

import numpy as np
from scipy import signal

from orbdemod.ddc import (
    DecimationStageConfig,
    design_butterworth_lowpass,
    design_kaiser_lowpass,
    filter_and_decimate,
    format_filter_report,
    integer_decimation_factor,
    measure_filter_response,
    validate_filter_response,
)
from orbdemod.orbcomm.orbcomm_ddc import (
    downconvert_orbcomm_voltage,
    make_orbcomm_decimation_stages,
)


class DDCFilterTest(unittest.TestCase):
    def test_non_integer_decimation_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            integer_decimation_factor(1_000.0, 333.0)

    def test_default_orbcomm_plan_is_three_stages(self) -> None:
        stages = make_orbcomm_decimation_stages()
        self.assertEqual(
            [(stage.fs_in, stage.fs_out) for stage in stages],
            [
                (480e6, 2.4e6),
                (2.4e6, 96e3),
                (96e3, 9.6e3),
            ],
        )
        self.assertEqual(
            [stage.filter_type for stage in stages],
            ["butterworth", "kaiser_fir", "kaiser_fir"],
        )

    def test_non_default_integer_plan_uses_an_available_factor(self) -> None:
        stages = make_orbcomm_decimation_stages(
            fs_in=10_000.0,
            fs_mid=250.0,
            fs_out=10.0,
            passband_hz=4.0,
        )
        self.assertEqual(
            [(stage.fs_in, stage.fs_out) for stage in stages],
            [(10_000.0, 250.0), (250.0, 50.0), (50.0, 10.0)],
        )

    def test_default_orbcomm_filters_meet_explicit_specifications(self) -> None:
        for stage in make_orbcomm_decimation_stages():
            frequencies = np.concatenate(
                (
                    np.linspace(0.0, stage.passband_hz, 2_001),
                    np.linspace(stage.stopband_hz, stage.fs_in / 2.0, 20_001),
                )
            )
            if stage.filter_type == "butterworth":
                coefficients = design_butterworth_lowpass(stage)
                _, response = signal.sosfreqz(
                    coefficients,
                    worN=frequencies,
                    fs=stage.fs_in,
                )
            else:
                coefficients = design_kaiser_lowpass(stage)
                _, response = signal.freqz(
                    coefficients,
                    worN=frequencies,
                    fs=stage.fs_in,
                )
            response_db = 20.0 * np.log10(
                np.maximum(np.abs(response), 1e-300)
            )
            passband = response_db[:2_001]
            stopband = response_db[2_001:]
            self.assertGreaterEqual(
                float(np.min(passband)),
                -stage.passband_ripple_db - 0.05,
            )
            self.assertLessEqual(
                float(np.max(stopband)),
                -stage.stopband_attenuation_db + 0.05,
            )

    def test_filter_report_accepts_a_generated_fir(self) -> None:
        stage = make_orbcomm_decimation_stages()[-1]
        coefficients = design_kaiser_lowpass(stage)

        report = validate_filter_response(coefficients, stage)

        self.assertTrue(report.meets_specification)
        self.assertTrue(report.coefficients_finite)
        self.assertTrue(report.stable)
        self.assertGreaterEqual(
            report.stopband_attenuation_db,
            stage.stopband_attenuation_db - 0.05,
        )
        self.assertIn("result=PASS", format_filter_report(report))

    def test_filter_report_rejects_an_unfiltered_decimator(self) -> None:
        stage = make_orbcomm_decimation_stages()[-1]
        no_filter = np.array([1.0])

        report = validate_filter_response(no_filter, stage, strict=False)

        self.assertFalse(report.meets_specification)
        self.assertTrue(
            any("stopband attenuation" in reason for reason in report.failure_reasons)
        )
        self.assertIn("result=FAIL", format_filter_report(report))
        with self.assertRaisesRegex(ValueError, "does not meet"):
            validate_filter_response(no_filter, stage)

    def test_zero_phase_iir_is_designed_for_two_effective_passes(self) -> None:
        stage = make_orbcomm_decimation_stages()[0]
        coefficients = design_butterworth_lowpass(stage, processing_passes=2)

        report = measure_filter_response(
            coefficients,
            stage,
            processing_passes=2,
        )

        self.assertTrue(report.meets_specification)
        self.assertEqual(report.processing_passes, 2)
        self.assertTrue(report.stable)

    def test_final_orbcomm_stage_suppresses_first_alias_band(self) -> None:
        stage = make_orbcomm_decimation_stages()[-1]
        duration_seconds = 0.2
        time = np.arange(int(duration_seconds * stage.fs_in)) / stage.fs_in
        wanted = np.exp(2j * np.pi * 1_000.0 * time)
        alias = np.exp(2j * np.pi * (stage.fs_out + 1_000.0) * time)

        wanted_out = filter_and_decimate(wanted, stage)
        alias_out = filter_and_decimate(alias, stage)
        edge = max(1, int(0.01 * len(wanted_out)))
        wanted_level = float(np.sqrt(np.mean(np.abs(wanted_out[edge:-edge]) ** 2)))
        alias_level = float(np.sqrt(np.mean(np.abs(alias_out[edge:-edge]) ** 2)))

        self.assertGreater(wanted_level, 0.95)
        self.assertLess(alias_level / wanted_level, 10.0 ** (-60.0 / 20.0))

    def test_orbcomm_ddc_uses_new_plan_and_preserves_tone(self) -> None:
        fs_in = 100_000.0
        fs_mid = 10_000.0
        fs_out = 100.0
        centre_hz = 20_000.0
        residual_offset_hz = 10.0
        time = np.arange(int(fs_in)) / fs_in
        raw = np.cos(2.0 * np.pi * (centre_hz + residual_offset_hz) * time)

        iq, _ = downconvert_orbcomm_voltage(
            raw,
            centre_hz,
            fs_in,
            fs_mid,
            fs_out,
        )

        self.assertEqual(len(iq), int(fs_out))
        phase_step = np.angle(iq[1:] * np.conj(iq[:-1]))
        measured_offset_hz = float(
            np.median(phase_step[10:-10]) * fs_out / (2.0 * np.pi)
        )
        self.assertAlmostEqual(measured_offset_hz, residual_offset_hz, places=2)


if __name__ == "__main__":
    unittest.main()
