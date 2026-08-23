"""Compatibility wrapper for the canonical ``lfdemod fm`` command."""

from __future__ import annotations

import sys

from lfdemod.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["fm", *sys.argv[1:]]))
