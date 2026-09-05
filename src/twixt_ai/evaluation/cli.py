"""Command-line entry point for headless matches."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from twixt_ai.agents import RandomAgent
from twixt_ai.game import BoardDimensions

from .match import MatchConfig, run_match


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a headless Twixt agent match")
    parser.add_argument("--red", choices=("random",), default="random", help="red agent")
    parser.add_argument("--black", choices=("random",), default="random", help="black agent")
    parser.add_argument("--width", type=int, default=24, help="board width")
    parser.add_argument("--height", type=int, default=24, help="board height")
    parser.add_argument("--seed", type=int, default=None, help="reproducible match seed")
    parser.add_argument("--output", type=Path, help="write JSON to this file instead of stdout")
    parser.add_argument("--pretty", action="store_true", help="indent JSON output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI configuration, run one match, and emit its JSON artifact."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        board = BoardDimensions(args.width, args.height)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    config = MatchConfig(
        board=board,
        seed=args.seed,
        red_agent=args.red,
        black_agent=args.black,
    )
    result = run_match(RandomAgent(), RandomAgent(), config=config)
    payload = result.to_json(indent=2 if args.pretty else None) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
