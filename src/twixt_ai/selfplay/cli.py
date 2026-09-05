"""Command-line entry point for batch self-play generation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from functools import partial
from pathlib import Path

from twixt_ai.agents import RandomAgent
from twixt_ai.game import BoardDimensions
from twixt_ai.search import HeuristicSearchAgent, MCTSAgent

from .batch import AgentFactory, BatchConfig, run_batch


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate headless Twixt self-play games")
    parser.add_argument("--games", type=int, required=True, help="number of games")
    parser.add_argument("--workers", type=int, default=1, help="parallel worker processes")
    parser.add_argument("--seed", type=int, default=None, help="reproducible batch seed")
    parser.add_argument("--red", choices=("random", "search", "mcts"), default="random")
    parser.add_argument("--black", choices=("random", "search", "mcts"), default="random")
    parser.add_argument("--width", type=int, default=24, help="board width")
    parser.add_argument("--height", type=int, default=24, help="board height")
    parser.add_argument("--search-depth", type=int, default=1)
    parser.add_argument("--node-budget", type=int, default=10_000)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretty", action="store_true", help="indent JSON artifacts")
    return parser


def _agent_factory(
    name: str, depth: int, node_budget: int, simulations: int
) -> AgentFactory:
    if name == "random":
        return RandomAgent
    if name == "search":
        return partial(HeuristicSearchAgent, depth=depth, node_budget=node_budget)
    return partial(MCTSAgent, simulations=simulations)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = BatchConfig(
            games=args.games,
            workers=args.workers,
            seed=args.seed,
            board=BoardDimensions(args.width, args.height),
            red_agent=args.red,
            black_agent=args.black,
        )
        red_factory = _agent_factory(
            args.red, args.search_depth, args.node_budget, args.simulations
        )
        black_factory = _agent_factory(
            args.black, args.search_depth, args.node_budget, args.simulations
        )
        # Construct once so invalid search limits fail before starting the batch.
        red_factory()
        black_factory()
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    summary = run_batch(
        red_factory,
        black_factory,
        config=config,
        output_dir=args.output_dir,
        pretty=args.pretty,
    )
    print(summary.to_json(indent=2 if args.pretty else None))
    return 1 if summary.failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
