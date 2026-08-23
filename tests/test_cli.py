"""Tests for the public lowercase ``lfdemod`` interface."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import unittest

from lfdemod import __version__
from lfdemod.cli import main
from lfdemod.fm import FMPSDConfig


class LFdemodCLITest(unittest.TestCase):
    def test_public_fm_api_is_available(self) -> None:
        self.assertIsNotNone(FMPSDConfig)

    def test_no_subcommand_prints_top_level_help(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            status = main([])

        self.assertEqual(status, 0)
        self.assertIn("usage: lfdemod", output.getvalue())
        self.assertIn("fm", output.getvalue())

    def test_fm_help_lists_key_options(self) -> None:
        output = io.StringIO()
        with self.assertRaises(SystemExit) as raised, redirect_stdout(output):
            main(["fm", "--help"])

        self.assertEqual(raised.exception.code, 0)
        help_text = output.getvalue()
        self.assertIn("usage: lfdemod fm", help_text)
        self.assertIn("--rf-frequency", help_text)
        self.assertIn("--waveform-duration", help_text)
        self.assertIn("--output-root", help_text)

    def test_version_uses_lowercase_command_name(self) -> None:
        output = io.StringIO()
        with self.assertRaises(SystemExit) as raised, redirect_stdout(output):
            main(["--version"])

        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(output.getvalue().strip(), f"lfdemod {__version__}")


if __name__ == "__main__":
    unittest.main()
