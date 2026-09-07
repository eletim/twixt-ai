"""Benchmark and tune Mini Twixt CUDA training and neural self-play."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .cuda_tuning import CudaTuningConfig, run_cuda_tuning_benchmark


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--warmup-games", type=int, default=4)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument(
        "--inference-batch-sizes", type=int, nargs="+", default=[1, 4, 8]
    )
    parser.add_argument(
        "--flush-latencies-seconds", type=float, nargs="+", default=[0.0005, 0.002]
    )
    parser.add_argument("--simulations", type=int, nargs="+", default=[4, 20])
    parser.add_argument(
        "--training-batch-sizes", type=int, nargs="+", default=[32, 64, 128]
    )
    parser.add_argument("--training-epochs", type=int, default=1)
    parser.add_argument("--rollout-limit", type=int, default=4)
    parser.add_argument("--seed", type=int, default=870_100)
    parser.add_argument("--gpu-sample-interval-seconds", type=float, default=0.1)
    try:
        args = parser.parse_args(argv)
        config = CudaTuningConfig(
            games=args.games,
            warmup_games=args.warmup_games,
            worker_counts=tuple(args.workers),
            inference_batch_sizes=tuple(args.inference_batch_sizes),
            flush_latencies_seconds=tuple(args.flush_latencies_seconds),
            simulation_budgets=tuple(args.simulations),
            training_batch_sizes=tuple(args.training_batch_sizes),
            training_epochs=args.training_epochs,
            rollout_limit=args.rollout_limit,
            seed=args.seed,
            gpu_sample_interval_seconds=args.gpu_sample_interval_seconds,
        )
        report = run_cuda_tuning_benchmark(args.dataset, args.checkpoint, config)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
