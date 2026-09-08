"""Report value-target balance and checkpoint calibration by game phase."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .value_diagnostics import ValueDiagnosticsConfig, diagnose_value_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ply-bucket-size", type=int, default=16)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must not already exist")
    try:
        report = diagnose_value_model(
            args.dataset,
            args.checkpoint,
            config=ValueDiagnosticsConfig(
                ply_bucket_size=args.ply_bucket_size,
                calibration_bins=args.calibration_bins,
                batch_size=args.batch_size,
                device=args.device,
            ),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(args.output)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
