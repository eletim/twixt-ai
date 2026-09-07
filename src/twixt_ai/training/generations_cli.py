"""Run repeatable Mini Twixt self-play training generations."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from .generations import MiniGenerationConfig, run_mini_training_generations


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-champion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--games-per-generation", type=int, default=100)
    parser.add_argument("--dataset-window", type=int, default=5)
    parser.add_argument("--selfplay-simulations", type=int, default=100)
    parser.add_argument("--evaluation-games", type=int, default=20)
    parser.add_argument("--evaluation-simulations", type=int, default=20)
    parser.add_argument("--rollout-limit", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--shard-size", type=int, default=10_000)
    parser.add_argument("--promotion-win-rate", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=590_100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = MiniGenerationConfig(**{
            name: getattr(args, name)
            for name in MiniGenerationConfig.__dataclass_fields__
        })
        report = run_mini_training_generations(
            args.initial_champion, args.output_dir, config=config
        )
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
