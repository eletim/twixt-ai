"""Report whether a requested PyTorch device is usable."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json

from .device import select_device


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    args = parser.parse_args(argv)
    try:
        selection = select_device(args.device)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(selection.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
