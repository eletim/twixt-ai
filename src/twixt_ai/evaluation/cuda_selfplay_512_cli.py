"""Command-line entry point for the canonical 512-game CUDA self-play benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

from .cuda_selfplay_512 import (
    BenchmarkOptions,
    BenchmarkTuning,
    CANONICAL_CONTRACT_PATH,
    run_cuda_selfplay_512_benchmark,
)

_DEFAULT_CONTRACT = CANONICAL_CONTRACT_PATH


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _git_branch(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=_DEFAULT_CONTRACT,
        help="canonical v0.0.6 contract or a JSON-identical copy",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory that receives games/ artifacts and summary.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="path to write the benchmark result JSON",
    )
    parser.add_argument(
        "--implementation-label",
        default="v0.0.6-default",
        help="free-text label distinguishing baseline vs. optimized runs",
    )
    parser.add_argument("--gpu-sample-interval-seconds", type=float, default=0.1)
    parser.add_argument("--phase-sample-interval-seconds", type=float, default=0.005)
    parser.add_argument(
        "--detailed-inference-profile",
        action="store_true",
        help=(
            "use the measurement-only CUDA/host phase evaluator; adds diagnostic "
            "overhead and does not represent optimized throughput"
        ),
    )
    parser.add_argument(
        "--worker-concurrency",
        type=int,
        default=None,
        help="override the contract default for concurrent game threads",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=None,
        help="override the contract default maximum shared inference batch",
    )
    parser.add_argument(
        "--queue-flush-max-wait-seconds",
        type=float,
        default=None,
        help="override the contract default inference queue/flush wait",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    try:
        report = run_cuda_selfplay_512_benchmark(
            args.contract,
            args.output_dir,
            checkpoint_path=args.checkpoint,
            repo_root=repo_root,
            options=BenchmarkOptions(
                gpu_sample_interval_seconds=args.gpu_sample_interval_seconds,
                phase_sample_interval_seconds=args.phase_sample_interval_seconds,
                implementation_label=args.implementation_label,
                detailed_inference_profile=args.detailed_inference_profile,
            ),
            tuning=BenchmarkTuning(
                worker_concurrency=args.worker_concurrency,
                inference_batch_size=args.inference_batch_size,
                queue_flush_max_wait_seconds=args.queue_flush_max_wait_seconds,
            ),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
        return 2
    report["source"]["git_commit"] = _git_commit(repo_root)
    report["source"]["branch_base"] = _git_branch(repo_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
