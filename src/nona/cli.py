"""Command-line entry point (`nona`).

Subcommands from docs/design/02-api.md section 4 (`serve`, `scan`, `bench`, `models pull`) are added
in the milestone that implements them, so the CLI never advertises something that does not work.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from nona import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nona",
        description="Pseudonymize personal data before it reaches an LLM, and restore it in the answer.",
    )
    parser.add_argument("--version", action="version", version=f"nona {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
