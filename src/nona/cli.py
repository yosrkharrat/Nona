"""Command-line entry point (`nona`).

Subcommands from docs/design/02-api.md section 4 (`serve`, `scan`, `bench`, `models pull`) are added
in the milestone that implements them, so the CLI never advertises something that does not work.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path

from nona import __version__

# Frozen at the end of S4 (05 §1, #75): never used for tuning, run once for the release table (06 §1, NFR-11).
# Recognized by its committed hash, not by the split's name: `--split ./test` or `--split Test` (on a
# case-insensitive disk) is the same file, and another folder's `test.jsonl` is not it.
FROZEN_SPLIT = "test"
FROZEN_HASH = Path(__file__).resolve().parents[2] / "bench" / "data" / "test.sha256"


def is_frozen(dataset: Path) -> bool:
    """`dataset` holds the frozen split's bytes (`bench/data/test.sha256`, in a checkout only)."""
    try:
        digest = FROZEN_HASH.read_text(encoding="utf-8").split()[0]
    except (OSError, IndexError):
        return False
    return hashlib.sha256(dataset.read_bytes()).hexdigest() == digest


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
    bench.add_argument(
        "--published",
        action="store_true",
        help=f"required for the frozen split ({FROZEN_SPLIT}): this run gives the published numbers",
    )
    bench.add_argument("--data", type=Path, default=Path("bench/data"), help="dataset directory")
    bench.add_argument("--out", type=Path, default=Path("results"), help="output directory")

    models = commands.add_parser("models", help="manage the NER model (needs the `ner` extra)")
    models_commands = models.add_subparsers(dest="models_command", metavar="<action>", required=True)
    pull = models_commands.add_parser(
        "pull", help="download the pinned NER model (the only command that fetches weights, NFR-3)"
    )
    pull.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="models folder (default: $NONA_MODELS_DIR or ~/.cache/nona/models)",
    )
    return parser


def shell_quote(value: str, *, windows: bool = os.name == "nt") -> str:
    """Quote a value for the user's shell: double quotes on Windows (cmd keeps single ones as part of the
    value), POSIX quoting elsewhere."""
    return f'"{value}"' if windows else shlex.quote(value)


def _models_pull(args: argparse.Namespace) -> int:
    from nona.models import GLINER, ModelError, models_dir, pull

    root = args.dir or models_dir()
    print(f"pulling {GLINER.repo}@{GLINER.revision[:12]} into {root} (about 1.2 GB)")
    try:
        folder = pull(root)
    except ModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"ok: {folder}")
    if root.resolve() != models_dir().resolve():
        # Loading (`nona bench --systems nona-ner`, the proxy) looks in $NONA_MODELS_DIR or the default.
        print(f"to use it, set NONA_MODELS_DIR={shell_quote(str(root.resolve()))}")
    return 0


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

        if (REPO_ROOT / "pyproject.toml").is_file() and (REPO_ROOT / dataset).is_file():
            dataset = REPO_ROOT / dataset  # run from anywhere inside a checkout
    if not dataset.is_file():
        print(f"error: no dataset file at {dataset}", file=sys.stderr)
        return 1
    if not args.published and is_frozen(dataset):
        parser.error(
            f"the {FROZEN_SPLIT} split is frozen: it is never used for tuning (05 §1). Tune on dev; pass "
            "--published only for the run that gives the published numbers"
        )
    from nona.models import ModelError

    for i, name in enumerate(names):
        try:
            result = run(SYSTEMS[name](), args.split, dataset)
        except (DatasetError, ModelError) as exc:
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
    if args.command == "models":
        return _models_pull(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
