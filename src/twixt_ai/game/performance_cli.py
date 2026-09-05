"""Command-line interface for canonical engine microbenchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .performance import EngineBenchmarkConfig, run_engine_benchmarks
from .state import BoardDimensions


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--height", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2401)
    parser.add_argument("--ply", type=int, default=160)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = EngineBenchmarkConfig(
        board=BoardDimensions(args.width, args.height),
        seed=args.seed,
        ply=args.ply,
        iterations=args.iterations,
        repeats=args.repeats,
        warmups=args.warmups,
    )
    artifact = json.dumps(run_engine_benchmarks(config), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(artifact, end="")
    else:
        args.output.write_text(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
