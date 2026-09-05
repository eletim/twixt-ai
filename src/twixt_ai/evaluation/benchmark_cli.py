"""Command-line entry point for reproducible agent benchmarks."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from functools import partial
from importlib.metadata import version
from pathlib import Path

from twixt_ai.agents import RandomAgent
from twixt_ai.game import BoardDimensions
from twixt_ai.search import DEFAULT_ROLLOUT_LIMIT, HeuristicSearchAgent, MCTSAgent

from .benchmark import AgentConfig, AgentFactory, BenchmarkConfig, run_benchmark


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Twixt agents in a paired round robin"
    )
    parser.add_argument(
        "--agent",
        action="append",
        required=True,
        metavar="NAME=TYPE",
        help="entrant name and built-in type (random, search, or mcts); repeat at least twice",
    )
    parser.add_argument("--games-per-pair", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--height", type=int, default=24)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--elo", action="store_true", help="include Elo-style ratings")
    parser.add_argument("--search-depth", type=int, default=1)
    parser.add_argument("--node-budget", type=int, default=10_000)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument(
        "--rollout-limit",
        type=int,
        default=DEFAULT_ROLLOUT_LIMIT,
        help="maximum random moves per MCTS rollout",
    )
    parser.add_argument("--output", type=Path, help="write JSON to a file")
    parser.add_argument("--pretty", action="store_true", help="indent JSON output")
    return parser


def _entrants(
    values: Sequence[str],
    depth: int,
    node_budget: int,
    simulations: int,
    rollout_limit: int,
) -> tuple[tuple[AgentConfig, ...], dict[str, AgentFactory]]:
    package_version = version("twixt-ai")
    configs: list[AgentConfig] = []
    factories: dict[str, AgentFactory] = {}
    for value in values:
        name, separator, agent_type = value.partition("=")
        if not separator:
            agent_type = name
        if not name or agent_type not in ("random", "search", "mcts"):
            raise ValueError("agents must use NAME=random, NAME=search, or NAME=mcts")
        if name in factories:
            raise ValueError(f"duplicate agent name: {name}")
        if agent_type == "random":
            factory: AgentFactory = RandomAgent
            settings: dict[str, object] = {"type": "random"}
        elif agent_type == "search":
            factory = partial(
                HeuristicSearchAgent, depth=depth, node_budget=node_budget
            )
            settings = {
                "type": "search",
                "depth": depth,
                "node_budget": node_budget,
            }
        else:
            factory = partial(
                MCTSAgent, simulations=simulations, rollout_limit=rollout_limit
            )
            settings = {
                "type": "mcts",
                "simulations": simulations,
                "rollout_limit": rollout_limit,
            }
        # Validate constructor settings before starting a potentially long run.
        factory()
        configs.append(AgentConfig(name, package_version, settings))
        factories[name] = factory
    return tuple(configs), factories


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        agents, factories = _entrants(
            args.agent,
            args.search_depth,
            args.node_budget,
            args.simulations,
            args.rollout_limit,
        )
        config = BenchmarkConfig(
            agents=agents,
            games_per_pair=args.games_per_pair,
            board=BoardDimensions(args.width, args.height),
            seed=args.seed,
            confidence_level=args.confidence,
            include_elo=args.elo,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    result = run_benchmark(factories, config=config)
    payload = result.to_json(indent=2 if args.pretty else None) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
