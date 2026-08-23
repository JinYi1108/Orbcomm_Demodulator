"""Top-level command-line interface for LFdemod."""

from __future__ import annotations

import argparse
from typing import Sequence

from orbdemod.fm.cli import add_fm_subparser

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the ``lfdemod`` top-level parser and its subcommands."""

    parser = argparse.ArgumentParser(
        prog="lfdemod",
        description=(
            "Low-frequency radio demodulation and diagnostic processing "
            "from raw voltage data."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        metavar="COMMAND",
    )
    add_fm_subparser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``lfdemod`` CLI and return a process exit status."""

    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "command_handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
