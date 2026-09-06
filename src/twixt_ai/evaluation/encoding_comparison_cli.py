"""Command-line interface for the v1 versus v2 encoding benchmark."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

import torch

from .encoding_comparison import EncodingComparisonConfig, run_encoding_comparison_benchmark


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Mini 10-plane versus 22-plane paths")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--encoding-repeats", type=int, default=20)
    parser.add_argument("--forward-iterations", type=int, default=100)
    parser.add_argument("--training-steps", type=int, default=50)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--seed", type=int, default=74)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--device",
        action="append",
        dest="devices",
        help="device to measure; repeat for CPU and GPU (default: cpu plus cuda when available)",
    )
    args = parser.parse_args(argv)
    devices = tuple(args.devices or (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]))
    try:
        config = EncodingComparisonConfig(
            dataset_dir=str(args.dataset),
            batch_size=args.batch_size,
            encoding_repeats=args.encoding_repeats,
            forward_iterations=args.forward_iterations,
            training_steps=args.training_steps,
            samples=args.samples,
            warmups=args.warmups,
            seed=args.seed,
            devices=devices,
            torch_threads=args.torch_threads,
        )
        report = run_encoding_comparison_benchmark(config)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
