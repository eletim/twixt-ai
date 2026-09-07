"""Reproduce the Mini Twixt MCTS and self-play performance report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from twixt_ai.game import EXPERIMENT_PRESETS, resolve_experiment_board

from .performance import SelfPlayPerformanceConfig, run_selfplay_performance_benchmark


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=EXPERIMENT_PRESETS, default="mini")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--seed", type=int, default=2401)
    parser.add_argument("--fixture-ply", type=int)
    parser.add_argument("--simulations", type=int, nargs="+", default=[100, 400, 1600])
    parser.add_argument("--move-repeats", type=int, default=3)
    parser.add_argument("--games-per-budget", type=int, default=1)
    parser.add_argument("--scaling-games", type=int, default=4)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--scaling-simulations", type=int, default=100)
    parser.add_argument("--rollout-limit", type=int, default=4)
    parser.add_argument("--engine-iterations", type=int, default=200)
    parser.add_argument("--engine-repeats", type=int, default=5)
    parser.add_argument("--engine-warmups", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        board = resolve_experiment_board(
            args.preset, width=args.width, height=args.height
        )
        fixture_ply = args.fixture_ply
        if fixture_ply is None:
            fixture_ply = max(1, round(8 * board.width * board.height / 100))
        config = SelfPlayPerformanceConfig(
            board=board,
            seed=args.seed,
            fixture_ply=fixture_ply,
            simulation_budgets=tuple(args.simulations),
            move_repeats=args.move_repeats,
            games_per_budget=args.games_per_budget,
            scaling_games=args.scaling_games,
            worker_counts=tuple(args.workers),
            scaling_simulations=args.scaling_simulations,
            rollout_limit=args.rollout_limit,
            engine_iterations=args.engine_iterations,
            engine_repeats=args.engine_repeats,
            engine_warmups=args.engine_warmups,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    artifact = json.dumps(
        run_selfplay_performance_benchmark(config), indent=2, sort_keys=True
    ) + "\n"
    if args.output is None:
        print(artifact, end="")
    else:
        args.output.write_text(artifact, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
