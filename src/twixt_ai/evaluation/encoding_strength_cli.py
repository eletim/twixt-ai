"""Compare matched 10-plane and 22-plane learned Mini MCTS agents."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from .encoding_strength import EncodingStrengthConfig, run_encoding_strength_comparison


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ten-plane-checkpoint", type=Path, required=True)
    parser.add_argument("--twenty-two-plane-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games-per-matchup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=760_100)
    parser.add_argument("--simulations", type=int, default=20)
    parser.add_argument("--rollout-limit", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must not already exist")
    try:
        report = run_encoding_strength_comparison(
            args.ten_plane_checkpoint,
            args.twenty_two_plane_checkpoint,
            config=EncodingStrengthConfig(
                games_per_matchup=args.games_per_matchup,
                seed=args.seed,
                simulations=args.simulations,
                rollout_limit=args.rollout_limit,
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
