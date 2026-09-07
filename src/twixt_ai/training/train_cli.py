"""Train a reproducible Twixt policy/value network."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from twixt_ai.game import BoardDimensions
from twixt_ai.models import PolicyValueConfig

from .trainer import TrainingConfig, train_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--scheduler", choices=("none", "step"), default="none")
    parser.add_argument("--scheduler-step-size", type=int, default=1)
    parser.add_argument("--scheduler-gamma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--residual-blocks", type=int, default=3)
    parser.add_argument("--value-hidden", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
        raw_board = manifest.get("board", {"width": 24, "height": 24})
        if not isinstance(raw_board, dict) or set(raw_board) != {"width", "height"}:
            raise ValueError("dataset board must contain exactly width and height")
        board = BoardDimensions(raw_board["width"], raw_board["height"])
        summary = train_model(
            args.dataset,
            args.output_dir,
            config=TrainingConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                optimizer=args.optimizer,
                scheduler=args.scheduler,
                scheduler_step_size=args.scheduler_step_size,
                scheduler_gamma=args.scheduler_gamma,
                seed=args.seed,
                device=args.device,
            ),
            model_config=PolicyValueConfig(
                channels=args.channels,
                residual_blocks=args.residual_blocks,
                value_hidden=args.value_hidden,
                board_width=board.width,
                board_height=board.height,
            ),
            resume=args.resume,
        )
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(summary.to_json())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
