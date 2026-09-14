"""Evaluate trained Mini value candidates with the generation-2 protocol."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from .value_candidates import ValueCandidateConfig, run_value_candidate_evaluation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--issue-57", type=Path, required=True)
    parser.add_argument(
        "--candidate", action="append", required=True, metavar="NAME=PATH",
        help="candidate name and checkpoint path; repeat for every candidate",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games-per-matchup", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1_251_300)
    parser.add_argument("--simulations", type=int, default=20)
    parser.add_argument("--rollout-limit", type=int, default=4)
    parser.add_argument("--promotion-win-rate", type=float, default=0.55)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must not already exist")
    candidates: dict[str, Path] = {}
    for value in args.candidate:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            parser.error("candidates must use NAME=PATH")
        if name in candidates:
            parser.error(f"duplicate candidate name: {name}")
        candidates[name] = Path(path)
    try:
        report = run_value_candidate_evaluation(
            args.champion,
            args.issue_57,
            candidates,
            config=ValueCandidateConfig(
                games_per_matchup=args.games_per_matchup,
                seed=args.seed,
                simulations=args.simulations,
                rollout_limit=args.rollout_limit,
                promotion_win_rate=args.promotion_win_rate,
                device=args.device,
            ),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(args.output)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
