"""Generate the measured Issue 88 Mini Twixt dataset."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .large_experiment import run_large_mini_dataset_experiment


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run_large_mini_dataset_experiment(
            args.output_dir, args.checkpoint, args.benchmark
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
