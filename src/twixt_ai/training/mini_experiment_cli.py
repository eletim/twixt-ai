"""Train and validate the first learned Mini Twixt model."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from .mini_experiment import MiniTrainingExperimentConfig, run_mini_training_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--resume-after-epochs", type=int, default=5)
    parser.add_argument("--tiny-epochs", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = run_mini_training_experiment(
            args.dataset,
            args.output_dir,
            config=MiniTrainingExperimentConfig(
                epochs=args.epochs,
                resume_after_epochs=args.resume_after_epochs,
                tiny_epochs=args.tiny_epochs,
            ),
        )
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
