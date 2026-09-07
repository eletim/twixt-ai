"""Generate a human-readable Mini Twixt training inspection report."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from .inspection import build_mini_inspection_report, render_mini_inspection_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="generation run directory or report.json")
    parser.add_argument("--output", type=Path, help="write Markdown here instead of stdout")
    parser.add_argument("--top-moves", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = build_mini_inspection_report(args.run, top_moves=args.top_moves)
        rendered = render_mini_inspection_report(report)
        if args.output is None:
            print(rendered, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
