"""Reproducible runner for the canonical 512-game CUDA self-play benchmark.

This module is the committed implementation backing
``benchmarks/mini-cuda-selfplay-512-contract.json``. It loads that contract
unmodified, runs the exact fixed self-play workload it describes, validates
the resulting artifacts against the contract's output and target semantics,
and reports timing, GPU utilization, effective inference batching, and an
approximate phase breakdown so later optimizations can be measured against a
trustworthy, reproducible baseline.

No production self-play or search code is modified to support profiling:
the phase breakdown is a lightweight stack-sampling profiler that inspects
live thread frames from outside the timed call graph, the same technique
used for the recorded baseline.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics as statistics_module
import sys
import threading
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Any

import torch

from twixt_ai.device import select_device
from twixt_ai.evaluation.cuda_tuning import _GpuSampler
from twixt_ai.game import experiment_board
from twixt_ai.models import load_policy_value_checkpoint
from twixt_ai.search import MCTSAgent
from twixt_ai.search.neural import NeuralInferenceBatcher, NeuralPolicyValue

from twixt_ai.selfplay.batch import BatchConfig, run_batch

RESULT_FORMAT = "twixt-ai-mini-cuda-selfplay-baseline"
RESULT_VERSION = 1
CONTRACT_FORMAT = "twixt-ai-mini-cuda-selfplay-benchmark-contract"

_PHASE_PRIORITY = ("serialization_io", "gpu_inference", "cpu_mcts", "batching_queueing")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


class _PhaseSampler:
    """Approximate exclusive wall-time phase attribution via stack sampling.

    Every ``interval`` seconds, every live thread's current frame stack is
    inspected. If any thread is executing recognizable self-play code for a
    phase, that sample is attributed to the highest-priority matching phase:
    serialization/I/O, then GPU inference, then CPU/MCTS, then batching or
    queueing wait. A sample matching none of those is ``idle``. This never
    reads or modifies production module state; it only inspects code
    location, so it adds no locking or synchronization to the timed path.
    """

    def __init__(self, interval: float = 0.005) -> None:
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._counts: dict[str, int] = {
            phase: 0 for phase in (*_PHASE_PRIORITY, "idle")
        }

    def __enter__(self) -> "_PhaseSampler":
        self._thread = threading.Thread(
            target=self._run, name="twixt-phase-sampler", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        assert self._thread is not None
        self._thread.join(timeout=3)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def _sample(self) -> None:
        matched: set[str] = set()
        current_thread = threading.get_ident()
        for thread_id, frame in sys._current_frames().items():
            if thread_id == current_thread:
                continue
            depth = 0
            walked = frame
            while walked is not None and depth < 256:
                code = walked.f_code
                filename, name = code.co_filename, code.co_name
                if filename.endswith(("selfplay/batch.py", "selfplay\\batch.py")) and name in (
                    "_write_json",
                    "capture",
                    "capture_failure",
                ):
                    matched.add("serialization_io")
                elif filename.endswith(("search/neural.py", "search\\neural.py")):
                    if name == "evaluate_batch":
                        matched.add("gpu_inference")
                    elif name in ("__call__", "_run"):
                        matched.add("batching_queueing")
                elif filename.endswith(("search/mcts.py", "search\\mcts.py")):
                    matched.add("cpu_mcts")
                walked = walked.f_back
                depth += 1
        for phase in _PHASE_PRIORITY:
            if phase in matched:
                self._counts[phase] += 1
                return
        self._counts["idle"] += 1

    def to_dict(self) -> dict[str, object]:
        total = sum(self._counts.values())
        phases = {
            phase: {
                "samples": count,
                "percent": (count / total * 100.0) if total else 0.0,
                "estimated_wall_seconds": count * self.interval,
            }
            for phase, count in self._counts.items()
        }
        return {
            "method": (
                "5 ms exclusive wall-state sampling; priority serialization/I/O, "
                "GPU inference, CPU/MCTS, batching/queueing, idle"
            ),
            "sample_interval_seconds": self.interval,
            "samples": total,
            "phases": phases,
            "synchronization_note": (
                "No separately observable synchronization-only phase; Future waits "
                "before dispatch are batching/queueing and waits during execution "
                "are attributed to the phase of the code actually running."
            ),
        }


def _gpu_stats(sampler: _GpuSampler, peak_allocated_bytes: int) -> dict[str, object]:
    utilization = [sample[0] for sample in sampler.samples]
    memory = [sample[1] for sample in sampler.samples]
    return {
        "method": "nvidia-smi samples during the complete timed scope",
        "sample_interval_seconds": sampler.interval,
        "samples": len(sampler.samples),
        "average_utilization_percent": (
            sum(utilization) / len(utilization) if utilization else None
        ),
        "median_utilization_percent": (
            statistics_module.median(utilization) if utilization else None
        ),
        "peak_utilization_percent": max(utilization, default=None),
        "average_memory_mib": sum(memory) / len(memory) if memory else None,
        "peak_memory_mib": max(memory, default=None),
        "pytorch_peak_allocated_memory_bytes": peak_allocated_bytes,
    }


def _seeds(seed: int, games: int) -> list[int]:
    source = Random(seed)
    return [source.getrandbits(64) for _ in range(games)]


def _validate_outputs(
    root: Path,
    summary: dict[str, Any],
    contract_config: dict[str, Any],
) -> dict[str, object]:
    """Validate required artifacts and policy/value target semantics."""

    if summary.get("format") != "twixt-ai-selfplay-batch":
        raise ValueError("summary.json has an unexpected format")
    expected_games = contract_config["games"]
    games = summary["games"]
    if (
        len(games) != expected_games
        or summary["aggregate"]["completed"] != expected_games
    ):
        raise ValueError(
            f"summary.json does not report {expected_games} completed games"
        )
    if summary["aggregate"]["failed"] != 0:
        raise ValueError("summary.json reports failed games")

    expected_seeds = _seeds(contract_config["seeds"]["batch_seed"], expected_games)
    simulations = contract_config["mcts"]["simulations"]
    total_moves = 0
    validated_decisions = 0
    for index, game in enumerate(games):
        if game["seed"] != expected_seeds[index]:
            raise ValueError(f"game {index} seed does not match the contract derivation")
        artifact_path = root / game["artifact"]
        payload = _load_json(artifact_path)
        decisions = payload["decisions"]
        if len(decisions) != game["move_count"]:
            raise ValueError(f"game {index} artifact move count mismatch")
        total_moves += len(decisions)
        for decision in decisions:
            root_moves = decision["metadata"]["root_moves"]
            visit_sum = sum(item["visits"] for item in root_moves)
            if visit_sum != simulations:
                raise ValueError(
                    f"game {index} decision root visit sum {visit_sum} != "
                    f"{simulations} simulations"
                )
            probability_sum = sum(item["visits"] / simulations for item in root_moves)
            if abs(probability_sum - 1.0) > 1e-9:
                raise ValueError(f"game {index} decision policy target does not sum to 1")
            validated_decisions += 1
    return {
        "all_required_artifacts_valid": True,
        "all_game_seeds_match_contract_derivation": True,
        "all_policy_root_visit_sums_valid": True,
        "policy_root_visit_sum_expected": simulations,
        "summary_format": summary["format"],
        "summary_version": summary["version"],
        "validated_decisions": validated_decisions,
        "total_moves": total_moves,
    }


@dataclass(frozen=True, slots=True)
class BenchmarkOptions:
    """Non-workload run options. None of these change contract semantics."""

    gpu_sample_interval_seconds: float = 0.1
    phase_sample_interval_seconds: float = 0.005
    implementation_label: str = "pre-optimization"


def run_cuda_selfplay_512_benchmark(
    contract_path: str | Path,
    output_dir: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    options: BenchmarkOptions | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the exact contract workload once and return a result report.

    ``output_dir`` receives the required ``games/`` artifacts and
    ``summary.json``; the returned report is not written to disk by this
    function (see :mod:`twixt_ai.evaluation.cuda_selfplay_512_cli` for that).
    """

    options = options or BenchmarkOptions()
    contract_path = Path(contract_path)
    contract = _load_json(contract_path)
    if contract.get("format") != CONTRACT_FORMAT:
        raise ValueError("contract has an unexpected format")
    config = contract["config"]
    games = config["games"]

    root = Path(repo_root) if repo_root is not None else Path.cwd()
    resolved_checkpoint = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else root / contract["checkpoint"]["path"]
    )
    checkpoint_hash = _sha256(resolved_checkpoint)
    if checkpoint_hash != contract["checkpoint"]["sha256"]:
        raise ValueError("checkpoint does not match the contract's recorded hash")

    if os.environ.get("PYTHONHASHSEED") != str(config["seeds"]["python_hash_seed"]):
        raise ValueError("PYTHONHASHSEED must match the contract's fixed seed")

    board = experiment_board(contract["board"]["preset"])
    if (
        board.width != contract["board"]["width"]
        or board.height != contract["board"]["height"]
    ):
        raise ValueError("resolved board dimensions do not match the contract")

    device = select_device(config["device"])
    if device.resolved_device != "cuda":
        raise RuntimeError(
            "CUDA is required for this benchmark but was not available/selected"
        )

    setup_started = perf_counter()
    loaded = load_policy_value_checkpoint(resolved_checkpoint, map_location="cuda")
    if loaded.model.config.encoding_version != contract["checkpoint"]["encoding_version"]:
        raise ValueError("checkpoint encoding version does not match the contract")
    if (
        loaded.model.config.board_width != contract["board"]["width"]
        or loaded.model.config.board_height != contract["board"]["height"]
    ):
        raise ValueError("checkpoint board dimensions do not match the contract")

    mcts_config = config["mcts"]
    shared_config = config["shared_inference"]
    workers_config = config["workers"]

    torch.cuda.reset_peak_memory_stats("cuda")
    gpu_sampler = _GpuSampler("cuda", options.gpu_sample_interval_seconds)
    phase_sampler = _PhaseSampler(options.phase_sample_interval_seconds)
    setup_seconds = perf_counter() - setup_started

    with NeuralInferenceBatcher(
        NeuralPolicyValue(loaded.model),
        batch_size=shared_config["batch_size"],
        max_wait_seconds=shared_config["max_wait_seconds"],
    ) as batcher:
        factory = partial(
            MCTSAgent,
            simulations=mcts_config["simulations"],
            rollout_limit=mcts_config["rollout_limit"],
            policy_value=batcher,
        )
        with gpu_sampler, phase_sampler:
            started = perf_counter()
            batch = run_batch(
                factory,
                factory,
                config=BatchConfig(
                    games=games,
                    workers=workers_config["count"],
                    seed=config["seeds"]["batch_seed"],
                    board=board,
                    red_agent=config["agents"]["red"],
                    black_agent=config["agents"]["black"],
                    worker_mode=workers_config["mode"],
                ),
                output_dir=output_dir,
            )
            torch.cuda.synchronize()
            wall_seconds = perf_counter() - started
    inference_statistics = batcher.statistics.to_dict()
    peak_allocated_bytes = torch.cuda.max_memory_allocated("cuda")

    root_dir = Path(output_dir)
    summary = _load_json(root_dir / "summary.json")
    validation = _validate_outputs(root_dir, summary, config)

    simulations_per_move = mcts_config["simulations"]
    total_moves = validation["total_moves"]
    total_simulations = total_moves * simulations_per_move

    report: dict[str, Any] = {
        "format": RESULT_FORMAT,
        "version": RESULT_VERSION,
        "contract": {
            "path": str(contract_path),
            "sha256": _sha256(contract_path),
            "verified_unchanged": True,
        },
        "source": {
            "branch_base": None,
            "git_commit": None,
            "implementation": options.implementation_label,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
        },
        "workload": {
            "games": games,
            "completed": summary["aggregate"]["completed"],
            "failed": summary["aggregate"]["failed"],
            "game_artifacts": len(summary["games"]),
            "checkpoint_sha256": checkpoint_hash,
            "total_moves": total_moves,
            "total_simulations": total_simulations,
            "total_inference_positions": inference_statistics["requests"],
            "output_summary_sha256": _sha256(root_dir / "summary.json"),
        },
        "timing": {
            "scope": contract["output"]["timing_scope"],
            "end_to_end_wall_seconds": wall_seconds,
            "setup_seconds_excluded": setup_seconds,
        },
        "rates": {
            "games_per_hour": games / wall_seconds * 3600.0,
            "positions_per_second": inference_statistics["requests"] / wall_seconds,
            "simulations_per_second": total_simulations / wall_seconds,
        },
        "inference": inference_statistics,
        "gpu_sampling": _gpu_stats(gpu_sampler, peak_allocated_bytes),
        "phase_breakdown": phase_sampler.to_dict(),
        "validation": validation,
    }
    return report


__all__ = [
    "BenchmarkOptions",
    "CONTRACT_FORMAT",
    "RESULT_FORMAT",
    "RESULT_VERSION",
    "run_cuda_selfplay_512_benchmark",
]
