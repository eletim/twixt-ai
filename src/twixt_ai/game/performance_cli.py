"""Command-line interface for canonical engine microbenchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .performance import EngineBenchmarkConfig, run_engine_benchmarks
from .experiments import EXPERIMENT_PRESETS, resolve_experiment_board


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", "--board-preset", choices=EXPERIMENT_PRESETS, default="standard")
    parser.add_argument("--width", type=int, help="override preset board width")
    parser.add_argument("--height", type=int, help="override preset board height")
    parser.add_argument("--seed", type=int, default=2401)
    parser.add_argument(
        "--ply",
        type=int,
        help="fixture ply (defaults to 160 scaled by preset board area)",
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    board = resolve_experiment_board(args.preset, width=args.width, height=args.height)
    ply = args.ply
    if ply is None:
        ply = max(1, round(160 * board.width * board.height / (24 * 24)))
    config = EngineBenchmarkConfig(
        board=board,
        seed=args.seed,
        ply=ply,
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
