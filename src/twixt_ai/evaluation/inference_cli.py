"""Command-line entry point for neural inference throughput measurements."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from .inference import InferencePerformanceConfig, run_inference_performance_benchmark


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark batched neural inference")
    parser.add_argument("--requests", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=54)
    parser.add_argument("--max-wait-seconds", type=float, default=0.002)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    values = {
        "requests": args.requests,
        "batch_size": args.batch_size,
        "warmups": args.warmups,
        "seed": args.seed,
        "max_wait_seconds": args.max_wait_seconds,
    }
    if args.device is not None:
        values["device"] = args.device
    try:
        config = InferencePerformanceConfig(**values)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    report = run_inference_performance_benchmark(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
