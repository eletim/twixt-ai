"""Run a matched search-setting diagnosis for a learned Mini checkpoint."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from .matched_search import (
    DEFAULT_SEARCH_SETTINGS,
    MatchedSearchConfig,
    run_matched_search_analysis,
)
from .mini_strength import GUIDANCE_MODES


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games-per-matchup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1_188_100)
    parser.add_argument("--rollout-limit", type=int, default=4)
    parser.add_argument(
        "--setting",
        action="append",
        choices=tuple(setting.name for setting in DEFAULT_SEARCH_SETTINGS),
        dest="settings",
        help="run only this setting; repeat to select multiple settings",
    )
    parser.add_argument(
        "--guidance-mode",
        action="append",
        choices=GUIDANCE_MODES,
        dest="guidance_modes",
        help="run only this guidance mode; repeat to select multiple modes",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        choices=("heuristic-search", "matched-non-neural-mcts"),
        dest="baselines",
        help="run only this baseline; repeat to select multiple baselines",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must not already exist")
    try:
        schedule = {}
        if args.settings is not None:
            schedule["settings"] = tuple(
                setting
                for setting in DEFAULT_SEARCH_SETTINGS
                if setting.name in args.settings
            )
        if args.guidance_modes is not None:
            schedule["guidance_modes"] = tuple(args.guidance_modes)
        if args.baselines is not None:
            schedule["baselines"] = tuple(args.baselines)
        report = run_matched_search_analysis(
            args.checkpoint,
            config=MatchedSearchConfig(
                games_per_matchup=args.games_per_matchup,
                seed=args.seed,
                rollout_limit=args.rollout_limit,
                device=args.device,
                **schedule,
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
