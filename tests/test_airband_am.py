"""Synthetic tests for LFdemod's known-frequency civil-airband AM decoder.

Test functions and outputs
--------------------------
``AirbandAMSyntheticTest.setUp()``
    Creates a small real-valued AM carrier and test configuration; returns
    ``None`` and stores them on the test instance.
``test_ddc_and_demodulator_recover_voice_tone()``
    Verifies IQ rate, carrier offset, audio length, and recovered tone; no
    value is returned.
``test_default_airband_filters_meet_specification()``
    Verifies both default anti-alias filters meet their explicit response
    requirements; no value is returned.
``test_chunk_size_does_not_change_ddc_output()``
    Verifies state continuity across different high-rate block sizes; no value
    is returned.
``test_zero_iq_returns_silent_audio()``
    Verifies an empty-carrier window produces finite silence without crashing;
    no value is returned.
``test_file_pipeline_preserves_requested_duration()``
    Verifies WAV/JSON/PNG creation and exact non-2-second duration; no value is
    returned.
``test_public_airband_api_is_importable()``
    Verifies the lowercase public package exports the decoder; no value is
    returned.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import signal
from scipy.io import wavfile

from orbdemod.ddc import (
    design_butterworth_lowpass,
    design_kaiser_lowpass,
    validate_filter_response,
)
from lfdemod.airband_am import (
    AirbandAMDDCConfig,
    AirbandAMFileConfig,
    demodulate_airband_am,
    demodulate_airband_am_file,
    downconvert_airband_am_voltage,
    estimate_airband_carrier,
    make_airband_am_decimation_stages,
)


class AirbandAMSyntheticTest(unittest.TestCase):
    """Exercise the airband decoder without depending on private 21CMA data."""

    def setUp(self) -> None:
        """Create and store a 1 kHz tone modulating a small synthetic carrier."""

        self.fs_in = 240e3
        self.rf_frequency_hz = 40e3
        self.true_carrier_offset_hz = 200.0
        self.voice_tone_hz = 1e3
        self.file_duration_seconds = 0.50
        times = np.arange(
            int(round(self.file_duration_seconds * self.fs_in)),
            dtype=np.float64,
        ) / self.fs_in
        modulation = 1.0 + 0.65 * np.sin(2.0 * np.pi * self.voice_tone_hz * times)
        voltage = modulation * np.cos(
            2.0
            * np.pi
            * (self.rf_frequency_hz + self.true_carrier_offset_hz)
            * times
        )
        self.raw_float = np.asarray(voltage, dtype=np.float32)
        self.raw_int16 = np.asarray(np.round(12_000.0 * voltage), dtype="<i2")
        self.ddc = AirbandAMDDCConfig(
            fs_in=self.fs_in,
            fs_mid=60e3,
            fs_out=20e3,
            first_stage_passband_hz=6e3,
            first_stage_stopband_hz=20e3,
            channel_passband_hz=2e3,
            channel_stopband_hz=4e3,
            passband_ripple_db=0.5,
            stopband_attenuation_db=60.0,
            chunk_samples=17_003,
        )

    def test_ddc_and_demodulator_recover_voice_tone(self) -> None:
        """Confirm the DDC and envelope decoder recover the synthetic 1 kHz tone."""

        iq, _ = downconvert_airband_am_voltage(
            self.raw_float,
            self.rf_frequency_hz,
            self.ddc,
        )
        self.assertEqual(len(iq), int(self.file_duration_seconds * self.ddc.fs_out))
        offset, _, _ = estimate_airband_carrier(
            iq[int(0.05 * self.ddc.fs_out) :],
            self.ddc.fs_out,
            search_hz=2e3,
        )
        self.assertAlmostEqual(offset, self.true_carrier_offset_hz, delta=8.0)

        result = demodulate_airband_am(
            iq,
            self.ddc.fs_out,
            audio_sample_rate_hz=16e3,
            audio_low_hz=300.0,
            audio_high_hz=1_800.0,
        )
        self.assertEqual(len(result.audio), int(self.file_duration_seconds * 16e3))
        frequencies, psd = signal.welch(
            result.audio[int(0.05 * 16e3) : -int(0.05 * 16e3)],
            fs=16e3,
            nperseg=2048,
        )
        voice = (frequencies >= 500.0) & (frequencies <= 1_500.0)
        recovered_hz = float(frequencies[np.flatnonzero(voice)[np.argmax(psd[voice])]])
        self.assertAlmostEqual(recovered_hz, self.voice_tone_hz, delta=10.0)

    def test_default_airband_filters_meet_specification(self) -> None:
        """Confirm the default IIR and FIR meet their declared filter limits."""

        first_stage, channel_stage = make_airband_am_decimation_stages()
        first_report = validate_filter_response(
            design_butterworth_lowpass(first_stage),
            first_stage,
        )
        channel_report = validate_filter_response(
            design_kaiser_lowpass(channel_stage),
            channel_stage,
        )
        self.assertTrue(first_report.meets_specification)
        self.assertTrue(channel_report.meets_specification)

    def test_chunk_size_does_not_change_ddc_output(self) -> None:
        """Confirm NCO, filter, and decimation state are continuous across chunks."""

        reference, _ = downconvert_airband_am_voltage(
            self.raw_float,
            self.rf_frequency_hz,
            self.ddc,
        )
        one_chunk_config = AirbandAMDDCConfig(
            **{
                **self.ddc.__dict__,
                "chunk_samples": len(self.raw_float) + 1,
            }
        )
        candidate, _ = downconvert_airband_am_voltage(
            self.raw_float,
            self.rf_frequency_hz,
            one_chunk_config,
        )
        np.testing.assert_allclose(candidate, reference, rtol=2e-5, atol=2e-5)

    def test_zero_iq_returns_silent_audio(self) -> None:
        """Confirm zero-valued IQ yields finite zero audio and zero carrier offset."""

        iq = np.zeros(4_000, dtype=np.complex64)
        offset, _, _ = estimate_airband_carrier(iq, 20e3, search_hz=2e3)
        result = demodulate_airband_am(
            iq,
            20e3,
            audio_sample_rate_hz=16e3,
            audio_low_hz=300.0,
            audio_high_hz=1_800.0,
        )
        self.assertEqual(offset, 0.0)
        self.assertTrue(np.all(np.isfinite(result.audio)))
        self.assertTrue(np.all(result.audio == 0.0))

    def test_file_pipeline_preserves_requested_duration(self) -> None:
        """Confirm a 0.273-second request produces exactly 0.273 seconds of WAV."""

        requested_duration = 0.273
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            input_path = temporary_path / "synthetic_airband.dat"
            self.raw_int16.tofile(input_path)
            output_path = temporary_path / "output"
            config = AirbandAMFileConfig(
                rf_frequency_hz=self.rf_frequency_hz,
                start_seconds=0.10,
                duration_seconds=requested_duration,
                padding_seconds=0.03,
                audio_sample_rate_hz=16e3,
                audio_low_hz=300.0,
                audio_high_hz=1_800.0,
                label="synthetic_test",
                ddc=self.ddc,
            )
            summary = demodulate_airband_am_file(
                input_path,
                output_path,
                config,
                overwrite=True,
            )

            wav_rate, wav_audio = wavfile.read(output_path / "audio.wav")
            self.assertEqual(wav_rate, 16_000)
            self.assertEqual(len(wav_audio), int(round(requested_duration * wav_rate)))
            self.assertAlmostEqual(len(wav_audio) / wav_rate, requested_duration)
            self.assertTrue((output_path / "diagnostic.png").is_file())
            self.assertTrue((output_path / "summary.json").is_file())
            saved = json.loads((output_path / "summary.json").read_text())
            self.assertEqual(saved["requested_duration_seconds"], requested_duration)
            self.assertEqual(
                saved["metrics"]["audio_sample_count"],
                int(round(requested_duration * wav_rate)),
            )
            self.assertEqual(summary["output_dir"], str(output_path.resolve()))

    def test_public_airband_api_is_importable(self) -> None:
        """Confirm the public configuration and file pipeline exports exist."""

        self.assertIsNotNone(AirbandAMDDCConfig)
        self.assertTrue(callable(demodulate_airband_am_file))


if __name__ == "__main__":
    unittest.main()
