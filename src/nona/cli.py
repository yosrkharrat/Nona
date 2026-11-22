"""Command-line entry point (`nona`).

Subcommands from docs/design/02-api.md section 4 (`serve`, `scan`, `bench`, `models pull`) are added
in the milestone that implements them, so the CLI never advertises something that does not work.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from nona import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nona",
        description="Pseudonymize personal data before it reaches an LLM, and restore it in the answer.",
    )
    parser.add_argument("--version", action="version", version=f"nona {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="<command>")

    bench = commands.add_parser("bench", help="measure leak and over-masking rates on the benchmark dataset")
    bench.add_argument(
        "--systems", default="identity", help="comma-separated systems to run (default: identity)"
    )
    bench.add_argument("--split", default="dev", help="dataset split, read from <data>/<split>.jsonl")
    bench.add_argument("--data", type=Path, default=Path("bench/data"), help="dataset directory")
    bench.add_argument("--out", type=Path, default=Path("results"), help="output directory")
    return parser


def _bench(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    from nona.bench.dataset import DatasetError
    from nona.bench.runner import format_table, run, write_outputs
    from nona.bench.systems import SYSTEMS

    names = [s.strip() for s in str(args.systems).split(",") if s.strip()]
    unknown = [n for n in names if n not in SYSTEMS]
    if not names or unknown:
        parser.error(f"unknown system(s) {unknown or names}; choose from: {', '.join(sorted(SYSTEMS))}")
    dataset: Path = args.data / f"{args.split}.jsonl"
    if not dataset.is_file() and not args.data.is_absolute():
        from nona.bench.runner import REPO_ROOT

        dataset = REPO_ROOT / dataset  # run from anywhere inside a checkout
    if not dataset.is_file():
        print(f"error: no dataset file at {dataset}", file=sys.stderr)
        return 1
    for i, name in enumerate(names):
        try:
            result = run(SYSTEMS[name](), args.split, dataset)
        except DatasetError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if i:
            print()
        print(format_table(result))
        try:
            print(f"wrote {write_outputs(result, args.out)}")
        except OSError as exc:
            print(f"error: can't write results to {args.out}: {exc}", file=sys.stderr)
            return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "bench":
        return _bench(parser, args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
