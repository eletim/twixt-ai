"""Generate a Mini champion self-play dataset without training or evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from .selfplay_dataset import MiniSelfplayDatasetConfig, run_mini_selfplay_dataset


def _parser() -> argparse.ArgumentParser:
    defaults = MiniSelfplayDatasetConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--games", type=int, default=defaults.games)
    parser.add_argument("--simulations", type=int, default=defaults.simulations)
    parser.add_argument("--exploration", type=float, default=defaults.exploration)
    parser.add_argument("--rollout-limit", type=int, default=defaults.rollout_limit)
    parser.add_argument(
        "--progressive-widening-constant",
        type=float,
        default=defaults.progressive_widening_constant,
    )
    parser.add_argument(
        "--progressive-widening-exponent",
        type=float,
        default=defaults.progressive_widening_exponent,
    )
    parser.add_argument("--workers", type=int, default=defaults.workers)
    parser.add_argument(
        "--inference-batch-size", type=int, default=defaults.inference_batch_size
    )
    parser.add_argument(
        "--inference-max-wait-seconds",
        type=float,
        default=defaults.inference_max_wait_seconds,
    )
    parser.add_argument(
        "--validation-fraction", type=float, default=defaults.validation_fraction
    )
    parser.add_argument("--shard-size", type=int, default=defaults.shard_size)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--split-seed")
    parser.add_argument("--workflow-label", default=defaults.workflow_label)
    parser.add_argument(
        "--device", choices=("cpu", "cuda", "auto"), default=defaults.device
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = MiniSelfplayDatasetConfig(**{
            name: getattr(args, name)
            for name in MiniSelfplayDatasetConfig.__dataclass_fields__
        })
        report = run_mini_selfplay_dataset(
            args.champion, args.output_dir, config=config
        )
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
