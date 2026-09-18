"""Paired 40-game policy/value MCTS comparison with fixed seeds and role swaps."""

from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path

from twixt_ai import __version__
from twixt_ai.evaluation import AgentConfig, BenchmarkConfig, run_benchmark
from twixt_ai.game import experiment_board
from twixt_ai.training.generations import _agent


ROOT = Path(__file__).resolve().parent
FROZEN = ROOT.parents[2] / "twixt-ai-frozen-opening/models/frozen/gen11/best.pt"


def checkpoint(name: str) -> Path:
    return FROZEN if name == "frozen" else ROOT / name / "training/best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, choices=("n0", "n2", "n4", "n6"))
    parser.add_argument("--right", required=True, choices=("frozen", "n0", "n2", "n4", "n6"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.left == args.right or args.output.exists():
        parser.error("distinct players and an absent output path are required")
    paths = {name: checkpoint(name) for name in (args.left, args.right)}
    identities = tuple(AgentConfig(name, __version__, {
        "type": "policy-value-mcts", "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "simulations": 64, "exploration": 0.7,
        "progressive_widening_constant": 3.0, "progressive_widening_exponent": 0.5,
        "rollout_limit": 4,
    }) for name, path in paths.items())
    config = BenchmarkConfig(agents=identities, games_per_pair=40,
                             board=experiment_board("mini"), seed=164900,
                             confidence_level=0.95)
    factories = {name: partial(_agent, str(path), 64, 4, "cuda", 0.7, 3.0, 0.5)
                 for name, path in paths.items()}
    report = run_benchmark(factories, config=config).to_dict()
    report["sweep"] = {"left": args.left, "right": args.right,
                       "paired_role_swaps": True, "same_seed_per_pair": True,
                       "promotion_decision": None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
