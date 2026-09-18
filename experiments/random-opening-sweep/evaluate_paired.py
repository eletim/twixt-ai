"""Evaluate retained random-opening candidates on a common 4-ply suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from twixt_ai.evaluation.paired_opening import run_paired_openings
from twixt_ai.game import experiment_board
from twixt_ai.training.generations import _agent

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[2] / "twixt-ai/experiments/random-opening-sweep"
FROZEN = ROOT.parents[1] / "models/frozen/gen11/best.pt"
NAMES = ("n0", "n2", "n4", "n6")


def checkpoint(name: str) -> Path:
    return FROZEN if name == "frozen" else SOURCE / name / "training/best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", choices=NAMES, required=True)
    parser.add_argument("--right", choices=("frozen", *NAMES), required=True)
    parser.add_argument("--pairs", type=int)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.left == args.right:
        parser.error("players must differ")
    pairs = args.pairs or (200 if args.right == "frozen" else 100)
    output = ROOT / "paired-4ply" / f"{args.left}-vs-{args.right}"
    left, right = checkpoint(args.left), checkpoint(args.right)
    for path in (left, right):
        if not path.is_file():
            parser.error(f"missing checkpoint: {path}")
    # Same 200-seed prefix for every matchup, including the 100-pair screens.
    result = run_paired_openings(
        _agent(str(left), args.simulations, 4, args.device, 0.7, 3.0, 0.5),
        _agent(str(right), args.simulations, 4, args.device, 0.7, 3.0, 0.5),
        board=experiment_board("mini"), pairs=pairs, random_opening_moves=4,
        opening_seed=164904, decision_seed=164900,
        output=output.with_suffix(".pairs.jsonl"),
        candidate_name=args.left, opponent_name=args.right)
    result["config"].update({"simulations": args.simulations,
        "exploration": 0.7, "progressive_widening_constant": 3.0,
        "progressive_widening_exponent": 0.5, "rollout_limit": 4,
        "candidate_sha256": hashlib.sha256(left.read_bytes()).hexdigest(),
        "opponent_sha256": hashlib.sha256(right.read_bytes()).hexdigest(),
        "device": args.device})
    output.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"matchup": f"{args.left}-vs-{args.right}", **result["summary"]}))


if __name__ == "__main__":
    main()
