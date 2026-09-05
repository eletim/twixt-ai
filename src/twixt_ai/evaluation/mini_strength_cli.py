"""Evaluate learned Mini MCTS against non-neural baselines."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from .mini_strength import MiniStrengthConfig, run_mini_strength_evaluation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games-per-matchup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=580_100)
    parser.add_argument("--simulations", type=int, default=20)
    parser.add_argument("--rollout-limit", type=int, default=4)
    parser.add_argument("--search-depth", type=int, default=1)
    parser.add_argument("--search-node-budget", type=int, default=10_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must not already exist")
    try:
        report = run_mini_strength_evaluation(
            args.checkpoint,
            config=MiniStrengthConfig(
                games_per_matchup=args.games_per_matchup,
                seed=args.seed,
                simulations=args.simulations,
                rollout_limit=args.rollout_limit,
                search_depth=args.search_depth,
                search_node_budget=args.search_node_budget,
            ),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(args.output)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
