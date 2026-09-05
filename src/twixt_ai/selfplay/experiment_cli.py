"""Command-line entry point for the first Mini Twixt dataset experiment."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from .experiment import run_mini_dataset_experiment


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run_mini_dataset_experiment(args.output_dir)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    baseline = report["stages"]["baseline"]["selfplay"]["summary"]
    return 1 if baseline["aggregate"]["failed"] else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
