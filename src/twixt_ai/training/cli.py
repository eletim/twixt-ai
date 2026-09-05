"""Command-line conversion of self-play artifacts into training shards."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .data import DatasetConfig, build_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="self-play run or match JSON"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=10_000)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", default="0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = DatasetConfig(
            shard_size=args.shard_size,
            validation_fraction=args.validation_fraction,
            split_seed=args.split_seed,
        )
        summary = build_dataset(args.input, args.output_dir, config=config)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(summary.to_json())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
