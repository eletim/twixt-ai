"""Train matched Mini models with the 22-plane and 10-plane encodings."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from .encoding_experiment import (
    MatchedEncodingTrainingConfig,
    run_matched_encoding_training,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=750_100)
    parser.add_argument("--tiny-epochs", type=int, default=300)
    parser.add_argument("--tiny-learning-rate", type=float, default=1e-2)
    parser.add_argument("--tiny-max-loss-ratio", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        report = run_matched_encoding_training(
            args.dataset,
            args.output_dir,
            config=MatchedEncodingTrainingConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                seed=args.seed,
                tiny_epochs=args.tiny_epochs,
                tiny_learning_rate=args.tiny_learning_rate,
                tiny_max_loss_ratio=args.tiny_max_loss_ratio,
                device=args.device,
                torch_threads=args.torch_threads,
            ),
        )
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
