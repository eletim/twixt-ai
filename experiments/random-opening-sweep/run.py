"""Run the fixed Frozen Gen11 random-opening comparison without promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SOURCE = REPO.parent / "twixt-ai-frozen-opening"
FROZEN = SOURCE / "models/frozen/gen11/best.pt"
FROZEN_SHA = "31832286c6493e1bbff77c3357b6787fd17d60e52a5a572483b70d2077fee536"
NS = (0, 2, 4, 6)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(label: str, command: list[str], log: Path, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(label, " ".join(command), flush=True)
    with log.open("w") as stream:
        subprocess.run(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, help="already running N=0 generation process")
    args = parser.parse_args()
    if sha(FROZEN) != FROZEN_SHA:
        raise ValueError("Frozen Gen11 checkpoint hash mismatch")
    source_commit = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    if not source_commit.startswith("98036e1"):
        raise ValueError("source checkout is not the PR #164 implementation")
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONPATH=str(SOURCE / "src"))
    python = sys.executable

    for n in NS:
        condition = ROOT / f"n{n}"
        generation = condition / "generation"
        dataset = generation / "dataset"
        training = condition / "training"
        condition.mkdir(exist_ok=True)
        if not (generation / "report.json").exists():
            if n == 0 and args.wait_pid:
                while not (generation / "report.json").exists():
                    try:
                        os.kill(args.wait_pid, 0)
                    except ProcessLookupError:
                        raise RuntimeError("existing N=0 generation did not complete")
                    time.sleep(30)
            elif generation.exists() and any(generation.iterdir()):
                deadline = time.monotonic() + 7200
                while not (generation / "report.json").exists():
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"N={n} generation did not complete within two hours")
                    time.sleep(30)
            else:
                run(f"N={n} self-play", [
                    python, "-m", "twixt_ai.training.selfplay_dataset_cli",
                    "--champion", str(FROZEN), "--output-dir", str(generation),
                    "--games", "5000", "--simulations", "64", "--random-opening-moves", str(n),
                    "--exploration", "0.7", "--progressive-widening-constant", "3.0",
                    "--progressive-widening-exponent", "0.5", "--workers", "8",
                    "--inference-batch-size", "8", "--inference-max-wait-seconds", "0.002",
                    "--seed", "164000", "--split-seed", "random-opening-sweep-164000",
                    "--workflow-label", f"random-opening-sweep-n{n}", "--device", "cuda",
                ], condition / "generation.log", env)
        report = json.loads((generation / "report.json").read_text())
        if report["selfplay"]["summary"]["aggregate"]["completed"] != 5000:
            raise ValueError(f"N={n} has incomplete self-play")
        if report["champion"]["sha256"] != FROZEN_SHA:
            raise ValueError(f"N={n} used a different initial checkpoint")
        training_config = {
            "epochs": 100, "batch_size": 512, "learning_rate": 0.0003,
            "weight_decay": 0.0001, "optimizer": "adamw", "scheduler": "step",
            "scheduler_step_size": 30, "scheduler_gamma": 0.5, "seed": 164100,
            "device": "cuda", "selection_metric": "total", "channels": 32,
            "residual_blocks": 4, "value_hidden": 256,
            "initial_checkpoint_sha256": FROZEN_SHA,
        }
        (condition / "training-config.json").write_text(json.dumps(training_config, indent=2) + "\n")
        training_complete = ((training / "best.pt").exists() and
                             (training / "latest.pt").exists() and
                             (training / "metrics.jsonl").exists() and
                             len((training / "metrics.jsonl").read_text().splitlines()) == 100)
        if not training_complete:
            if training.exists() and any(training.iterdir()):
                raise RuntimeError(f"N={n} has partial training artifacts; inspect before resuming")
            run(f"N={n} training", [
                python, "-m", "twixt_ai.training.train_cli", "--dataset", str(dataset),
                "--output-dir", str(training), "--epochs", "100", "--batch-size", "512",
                "--learning-rate", "0.0003", "--weight-decay", "0.0001",
                "--optimizer", "adamw", "--scheduler", "step", "--scheduler-step-size", "30",
                "--scheduler-gamma", "0.5", "--seed", "164100", "--device", "cuda",
                "--selection-metric", "total", "--channels", "32", "--residual-blocks", "4",
                "--value-hidden", "256", "--initial-checkpoint", str(FROZEN),
            ], condition / "training.log", env)
        if not (condition / "policy-diagnostics.json").exists():
            run(f"N={n} diagnostics", [
                python, str(ROOT / "model_diagnostics.py"),
                "--dataset", str(dataset), "--output-dir", str(condition),
                "--checkpoint", f"bootstrap={FROZEN}",
                "--checkpoint", f"candidate={training / 'best.pt'}", "--targets",
            ], condition / "diagnostics.log", env)
    for n in NS:
        output = ROOT / f"n{n}" / "strength-evaluation.json"
        if not output.exists():
            run(f"N={n} versus Frozen Gen11", [
                python, str(ROOT / "evaluate.py"), "--left", f"n{n}", "--right", "frozen",
                "--output", str(output),
            ], ROOT / f"n{n}" / "evaluation.log", env)
    summary = ROOT / "summary"
    summary.mkdir(exist_ok=True)
    for i, left in enumerate(NS):
        for right in NS[i + 1:]:
            output = summary / f"n{left}-vs-n{right}.json"
            if not output.exists():
                run(f"N={left} versus N={right}", [
                    python, str(ROOT / "evaluate.py"), "--left", f"n{left}",
                    "--right", f"n{right}", "--output", str(output),
                ], summary / f"n{left}-vs-n{right}.log", env)
    run("summary", [python, str(ROOT / "summarize.py")], summary / "summarize.log", env)


if __name__ == "__main__":
    main()
