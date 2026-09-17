"""Evaluate a candidate against the repository's frozen Gen11 Mini baseline."""

from __future__ import annotations

import argparse
from functools import partial
import hashlib
from pathlib import Path

from twixt_ai import __version__
from twixt_ai.evaluation import AgentConfig, BenchmarkConfig, run_benchmark
from twixt_ai.game import experiment_board
from twixt_ai.training.generations import _agent

FROZEN_GEN11 = Path(__file__).resolve().parents[3] / "models/frozen/gen11/best.pt"
FROZEN_SHA256 = "31832286c6493e1bbff77c3357b6787fd17d60e52a5a572483b70d2077fee536"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--seed", type=int, default=11_000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not FROZEN_GEN11.is_file():
        parser.error(f"Frozen Gen11 checkpoint is missing: {FROZEN_GEN11}")
    if hashlib.sha256(FROZEN_GEN11.read_bytes()).hexdigest() != FROZEN_SHA256:
        parser.error("Frozen Gen11 checkpoint SHA-256 mismatch")
    if not args.candidate.is_file():
        parser.error(f"candidate checkpoint is missing: {args.candidate}")
    candidate_sha = hashlib.sha256(args.candidate.read_bytes()).hexdigest()
    settings = {"type": "mcts", "simulations": args.simulations, "guidance": "policy-value"}
    config = BenchmarkConfig(
        agents=(
            AgentConfig("candidate", __version__, {**settings, "checkpoint_sha256": candidate_sha}),
            AgentConfig("frozen_gen11", __version__, {**settings, "checkpoint_sha256": FROZEN_SHA256}),
        ),
        games_per_pair=args.games,
        board=experiment_board("mini"),
        seed=args.seed,
    )
    result = run_benchmark(
        {
            "candidate": partial(_agent, str(args.candidate), args.simulations, 4, args.device),
            "frozen_gen11": partial(_agent, str(FROZEN_GEN11), args.simulations, 4, args.device),
        },
        config=config,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.to_json(indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
