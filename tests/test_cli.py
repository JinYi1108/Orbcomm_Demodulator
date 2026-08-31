"""Tests for the public lowercase ``lfdemod`` interface.

Test methods return no values. They verify public imports, help text, short
aliases, complete option descriptions, and lowercase version output.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import argparse
import io
import unittest

from lfdemod import __version__
from lfdemod.cli import build_parser, main
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

    def test_airband_am_help_lists_key_options(self) -> None:
        output = io.StringIO()
        with self.assertRaises(SystemExit) as raised, redirect_stdout(output):
            main(["airband-am", "--help"])

        self.assertEqual(raised.exception.code, 0)
        help_text = output.getvalue()
        self.assertIn("usage: lfdemod airband-am", help_text)
        self.assertIn("--rf-frequency", help_text)
        self.assertIn("--duration", help_text)
        self.assertIn("--channel-passband", help_text)
        self.assertIn("--output-root", help_text)

    def test_version_uses_lowercase_command_name(self) -> None:
        output = io.StringIO()
        with self.assertRaises(SystemExit) as raised, redirect_stdout(output):
            main(["--version"])

        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(output.getvalue().strip(), f"lfdemod {__version__}")

    def test_short_version_alias_matches_long_option(self) -> None:
        """Confirm ``-v`` prints the same public LFdemod version string."""

        output = io.StringIO()
        with self.assertRaises(SystemExit) as raised, redirect_stdout(output):
            main(["-v"])

        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(output.getvalue().strip(), f"lfdemod {__version__}")

    def test_common_short_aliases_parse_for_both_decoders(self) -> None:
        """Confirm concise input, frequency, start, and duration aliases parse."""

        parser = build_parser()
        fm = parser.parse_args(
            ["fm", "-i", "fm.dat", "-f", "98.3e6", "-s", "1", "-d", "2"]
        )
        airband = parser.parse_args(
            [
                "airband-am",
                "-i",
                "airband.dat",
                "-f",
                "118.65e6",
                "-s",
                "3",
                "-d",
                "4",
            ]
        )
        self.assertEqual(fm.input, "fm.dat")
        self.assertEqual(fm.rf_frequency, 98.3e6)
        self.assertEqual(fm.start, 1.0)
        self.assertEqual(fm.duration, 2.0)
        self.assertEqual(airband.input, "airband.dat")
        self.assertEqual(airband.rf_frequency, 118.65e6)
        self.assertEqual(airband.start, 3.0)
        self.assertEqual(airband.duration, 4.0)

    def test_every_long_option_has_short_alias_and_explanation(self) -> None:
        """Confirm every public long option has a short spelling and help text."""

        root = build_parser()
        subparsers_action = next(
            action
            for action in root._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        parsers = [root, *subparsers_action.choices.values()]
        for parser in parsers:
            for action in parser._actions:
                long_options = [
                    option for option in action.option_strings if option.startswith("--")
                ]
                if not long_options:
                    continue
                short_options = [
                    option
                    for option in action.option_strings
                    if option.startswith("-") and not option.startswith("--")
                ]
                self.assertTrue(short_options, msg=f"No short alias for {long_options}")
                self.assertNotIn(
                    action.help,
                    (None, argparse.SUPPRESS),
                    msg=f"No help text for {long_options}",
                )


if __name__ == "__main__":
    unittest.main()
