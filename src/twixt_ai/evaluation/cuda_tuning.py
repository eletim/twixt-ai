"""End-to-end Mini training and neural self-play tuning benchmarks."""

from __future__ import annotations

import hashlib
import math
import multiprocessing
import os
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import perf_counter
from typing import Any

import torch

from twixt_ai.device import select_device
from twixt_ai.evaluation.match import MatchConfig, run_match
from twixt_ai.game import experiment_board
from twixt_ai.models import MINI_POLICY_VALUE_CONFIG, load_policy_value_checkpoint
from twixt_ai.search import MCTSAgent
from twixt_ai.search.neural import NeuralInferenceBatcher, NeuralPolicyValue
from twixt_ai.selfplay import BatchConfig, run_batch
from twixt_ai.training.trainer import TrainingConfig, train_model

CUDA_TUNING_FORMAT = "twixt-ai-mini-cuda-tuning"
CUDA_TUNING_VERSION = 1


def _positive_tuple(values: tuple[int, ...], name: str) -> tuple[int, ...]:
    values = tuple(values)
    if not values:
        raise ValueError(f"{name} must not be empty")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in values
    ):
        raise ValueError(f"{name} must contain positive integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique")
    return values


@dataclass(frozen=True, slots=True)
class CudaTuningConfig:
    """A compact, reproducible sweep of the important throughput controls."""

    games: int = 4
    worker_counts: tuple[int, ...] = (1, 4, 8)
    inference_batch_sizes: tuple[int, ...] = (1, 4, 8)
    flush_latencies_seconds: tuple[float, ...] = (0.0005, 0.002)
    simulation_budgets: tuple[int, ...] = (4, 20)
    training_batch_sizes: tuple[int, ...] = (32, 64, 128)
    training_epochs: int = 1
    rollout_limit: int = 4
    seed: int = 870_100
    gpu_sample_interval_seconds: float = 0.1

    def __post_init__(self) -> None:
        for name in ("games", "training_epochs", "rollout_limit"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        object.__setattr__(
            self, "worker_counts", _positive_tuple(self.worker_counts, "worker_counts")
        )
        object.__setattr__(
            self,
            "inference_batch_sizes",
            _positive_tuple(self.inference_batch_sizes, "inference_batch_sizes"),
        )
        object.__setattr__(
            self,
            "simulation_budgets",
            _positive_tuple(self.simulation_budgets, "simulation_budgets"),
        )
        object.__setattr__(
            self,
            "training_batch_sizes",
            _positive_tuple(self.training_batch_sizes, "training_batch_sizes"),
        )
        latencies = tuple(float(value) for value in self.flush_latencies_seconds)
        if not latencies or any(
            not math.isfinite(value) or value < 0 for value in latencies
        ):
            raise ValueError(
                "flush_latencies_seconds must contain finite non-negative values"
            )
        if len(set(latencies)) != len(latencies):
            raise ValueError("flush_latencies_seconds must be unique")
        object.__setattr__(self, "flush_latencies_seconds", latencies)
        interval = self.gpu_sample_interval_seconds
        if (
            isinstance(interval, bool)
            or not isinstance(interval, (int, float))
            or not math.isfinite(interval)
            or interval <= 0
        ):
            raise ValueError(
                "gpu_sample_interval_seconds must be a positive finite number"
            )
        object.__setattr__(self, "gpu_sample_interval_seconds", float(interval))

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        for name in (
            "worker_counts",
            "inference_batch_sizes",
            "flush_latencies_seconds",
            "simulation_budgets",
            "training_batch_sizes",
        ):
            value[name] = list(value[name])
        return value


class _GpuSampler:
    """Poll nvidia-smi while a workload is active, retaining time-series summaries."""

    def __init__(self, device: str, interval: float) -> None:
        resolved = torch.device(device)
        self.index = (
            resolved.index
            if resolved.index is not None
            else torch.cuda.current_device()
        )
        self.interval = interval
        self.samples: list[tuple[float, float]] = []
        self._stop = Event()
        self._thread: Thread | None = None

    def __enter__(self) -> _GpuSampler:  # noqa: PYI034
        self._thread = Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={self.index}",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                utilization, memory = result.stdout.strip().splitlines()[0].split(",")
                self.samples.append((float(utilization), float(memory)))
            except (
                FileNotFoundError,
                IndexError,
                subprocess.SubprocessError,
                ValueError,
            ):
                pass
            self._stop.wait(self.interval)

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        assert self._thread is not None
        self._thread.join(timeout=3)

    def to_dict(self) -> dict[str, object]:
        utilization = [sample[0] for sample in self.samples]
        memory = [sample[1] for sample in self.samples]
        return {
            "method": "nvidia-smi samples during workload",
            "sample_interval_seconds": self.interval,
            "samples": len(self.samples),
            "average_utilization_percent": sum(utilization) / len(utilization)
            if utilization
            else None,
            "peak_utilization_percent": max(utilization, default=None),
            "peak_memory_mib": max(memory, default=None),
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cuda_hardware_available() -> bool:
    """Check hardware without initializing CUDA in this process before forks."""

    try:
        return (
            subprocess.run(
                ["nvidia-smi", "-L"],
                check=False,
                capture_output=True,
                timeout=2,
            ).returncode
            == 0
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


def _training_run(
    dataset: Path, root: Path, batch_size: int, device: str, config: CudaTuningConfig
) -> dict[str, object]:
    selected = select_device(device)
    sampler = (
        _GpuSampler(selected.resolved_device, config.gpu_sample_interval_seconds)
        if device == "cuda"
        else None
    )
    started = perf_counter()
    if sampler is None:
        summary = train_model(
            dataset,
            root,
            config=TrainingConfig(
                epochs=config.training_epochs,
                batch_size=batch_size,
                seed=config.seed,
                device=device,
            ),
            model_config=MINI_POLICY_VALUE_CONFIG,
        )
    else:
        torch.cuda.reset_peak_memory_stats(selected.resolved_device)
        with sampler:
            summary = train_model(
                dataset,
                root,
                config=TrainingConfig(
                    epochs=config.training_epochs,
                    batch_size=batch_size,
                    seed=config.seed,
                    device=device,
                ),
                model_config=MINI_POLICY_VALUE_CONFIG,
            )
    return {
        "device": device,
        "batch_size": batch_size,
        "epochs": config.training_epochs,
        "wall_seconds": perf_counter() - started,
        "training_seconds": summary.training_seconds,
        "examples_per_second": summary.examples_per_second,
        "peak_cuda_memory_bytes": summary.peak_cuda_memory_bytes,
        "gpu": sampler.to_dict() if sampler is not None else None,
    }


def _cpu_agent(checkpoint: str, simulations: int, rollout_limit: int) -> MCTSAgent:
    loaded = load_policy_value_checkpoint(checkpoint, map_location="cpu")
    return MCTSAgent(
        simulations=simulations,
        rollout_limit=rollout_limit,
        policy_value=NeuralPolicyValue(loaded.model),
    )


def _cpu_game(arguments: tuple[str, int, int, int]) -> int:
    checkpoint, simulations, rollout_limit, seed = arguments
    result = run_match(
        _cpu_agent(checkpoint, simulations, rollout_limit),
        _cpu_agent(checkpoint, simulations, rollout_limit),
        config=MatchConfig(experiment_board("mini"), seed, "neural", "neural"),
    )
    return len(result.moves)


def _selfplay_run(
    checkpoint: Path,
    root: Path,
    *,
    device: str,
    workers: int,
    batch_size: int,
    latency: float,
    simulations: int,
    config: CudaTuningConfig,
) -> dict[str, object]:
    batch_config = BatchConfig(
        games=config.games,
        workers=workers,
        seed=config.seed,
        board=experiment_board("mini"),
        red_agent="neural",
        black_agent="neural",
        worker_mode="process" if device == "cpu" else "thread",
    )
    sampler: _GpuSampler | None = None
    inference: dict[str, object] | None = None
    started = perf_counter()
    if device == "cpu":
        seed_source = Random(config.seed)
        arguments = [
            (
                str(checkpoint),
                simulations,
                config.rollout_limit,
                seed_source.getrandbits(64),
            )
            for _ in range(config.games)
        ]
        if workers == 1:
            move_counts = [_cpu_game(item) for item in arguments]
        else:
            # Spawn is deliberate: forking a process after importing PyTorch
            # can inherit unsafe CPU/CUDA runtime state and hang the control run.
            with ProcessPoolExecutor(
                max_workers=min(workers, config.games),
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                move_counts = list(pool.map(_cpu_game, arguments))
        completed = len(move_counts)
    else:
        loaded = load_policy_value_checkpoint(checkpoint, map_location="cuda")
        sampler = _GpuSampler("cuda", config.gpu_sample_interval_seconds)
        torch.cuda.reset_peak_memory_stats()
        with (
            sampler,
            NeuralInferenceBatcher(
                NeuralPolicyValue(loaded.model),
                batch_size=batch_size,
                max_wait_seconds=latency,
            ) as batcher,
        ):
            factory = partial(
                MCTSAgent,
                simulations=simulations,
                rollout_limit=config.rollout_limit,
                policy_value=batcher,
            )
            batch = run_batch(factory, factory, config=batch_config, output_dir=root)
        inference = batcher.statistics.to_dict()
        if batch.failed:
            raise RuntimeError(f"self-play benchmark had {batch.failed} failed games")
        completed = batch.completed
        move_counts = [game.move_count or 0 for game in batch.games]
    wall_seconds = perf_counter() - started
    moves = sum(move_counts)
    return {
        "device": device,
        "mode": "cpu-process"
        if device == "cpu"
        else ("synchronous-cuda" if batch_size == 1 else "shared-batched-cuda"),
        "workers": workers,
        "inference_batch_size": batch_size if device == "cuda" else 1,
        "flush_latency_seconds": latency if device == "cuda" else 0.0,
        "simulations_per_move": simulations,
        "games": completed,
        "moves": moves,
        "wall_seconds": wall_seconds,
        "games_per_hour": completed / wall_seconds * 3600.0,
        "move_latency_seconds": wall_seconds / moves,
        "simulations_per_second": simulations * moves / wall_seconds,
        "inference": inference,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated()
        if device == "cuda"
        else None,
        "gpu": sampler.to_dict() if sampler is not None else None,
    }


def _projections(games_per_hour: float) -> dict[str, float]:
    return {
        f"{games}_games_hours": games / games_per_hour for games in (1000, 5000, 10000)
    }


def run_cuda_tuning_benchmark(
    dataset: str | Path, checkpoint: str | Path, config: CudaTuningConfig | None = None
) -> dict[str, Any]:
    """Run training and representative neural self-play sweeps and select defaults."""

    config = config or CudaTuningConfig()
    if not isinstance(config, CudaTuningConfig):
        raise TypeError("config must be a CudaTuningConfig")
    dataset, checkpoint = Path(dataset), Path(checkpoint)
    if not _cuda_hardware_available():
        raise RuntimeError("CUDA is required for the CPU-versus-CUDA tuning benchmark")
    training: list[dict[str, object]] = []
    selfplay: list[dict[str, object]] = []
    with TemporaryDirectory(prefix="twixt-ai-issue-87-") as temporary:
        temporary_root = Path(temporary)
        # Complete all process-pool work before training initializes PyTorch's
        # CPU thread pools or the CUDA runtime. Forking after either runtime is
        # active can deadlock the otherwise CPU-only self-play workers.
        for simulations in config.simulation_budgets:
            for workers in config.worker_counts:
                selfplay.append(
                    _selfplay_run(
                        checkpoint,
                        temporary_root / f"selfplay-cpu-{simulations}-{workers}",
                        device="cpu",
                        workers=workers,
                        batch_size=1,
                        latency=0.0,
                        simulations=simulations,
                        config=config,
                    )
                )

        if not torch.cuda.is_available():
            raise RuntimeError("installed PyTorch cannot use the available CUDA device")
        for batch_size in config.training_batch_sizes:
            training.append(
                _training_run(
                    dataset,
                    temporary_root / f"train-cpu-{batch_size}",
                    batch_size,
                    "cpu",
                    config,
                )
            )
        for batch_size in config.training_batch_sizes:
            training.append(
                _training_run(
                    dataset,
                    temporary_root / f"train-cuda-{batch_size}",
                    batch_size,
                    "cuda",
                    config,
                )
            )
        # Batch size one is independent of a flush timeout, so measure it only
        # once per worker/simulation pair in the CUDA Cartesian sweep.
        for simulations in config.simulation_budgets:
            for workers in config.worker_counts:
                for batch_size in config.inference_batch_sizes:
                    latencies = (
                        (0.0,) if batch_size == 1 else config.flush_latencies_seconds
                    )
                    for latency in latencies:
                        selfplay.append(
                            _selfplay_run(
                                checkpoint,
                                temporary_root
                                / f"selfplay-cuda-{simulations}-{workers}-{batch_size}-{latency}",
                                device="cuda",
                                workers=workers,
                                batch_size=batch_size,
                                latency=latency,
                                simulations=simulations,
                                config=config,
                            )
                        )
    best_training = max(training, key=lambda item: item["examples_per_second"])
    best_cuda_selfplay = max(
        (item for item in selfplay if item["device"] == "cuda"),
        key=lambda item: item["games_per_hour"],
    )
    matching_cpu = max(
        (
            item
            for item in selfplay
            if item["device"] == "cpu"
            and item["simulations_per_move"]
            == best_cuda_selfplay["simulations_per_move"]
        ),
        key=lambda item: item["games_per_hour"],
    )
    best_selfplay = max(
        (best_cuda_selfplay, matching_cpu),
        key=lambda item: item["games_per_hour"],
    )
    inference = best_cuda_selfplay.get("inference") or {}
    gpu = best_cuda_selfplay.get("gpu") or {}
    gpu_utilization = gpu.get("average_utilization_percent")
    requests = inference.get("requests")
    batches = inference.get("batches")
    avg_batch = (
        requests / batches
        if isinstance(requests, int) and isinstance(batches, int) and batches
        else None
    )
    bottleneck = (
        "cpu_game_and_mcts"
        if isinstance(gpu_utilization, (int, float)) and gpu_utilization < 70
        else "gpu_inference"
    )
    return {
        "format": CUDA_TUNING_FORMAT,
        "version": CUDA_TUNING_VERSION,
        "config": config.to_dict(),
        "inputs": {
            "dataset": str(dataset),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "available_cpus": len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count(),
        },
        "training_sweep": training,
        "selfplay_sweep": selfplay,
        "recommendation": {
            "training": {
                "device": best_training["device"],
                "batch_size": best_training["batch_size"],
                "examples_per_second": best_training["examples_per_second"],
                "peak_cuda_memory_bytes": best_training["peak_cuda_memory_bytes"],
            },
            "selfplay": {
                "device": best_selfplay["device"],
                "workers": best_selfplay["workers"],
                "inference_batch_size": best_selfplay["inference_batch_size"],
                "inference_max_wait_seconds": best_selfplay["flush_latency_seconds"],
                "simulations": best_selfplay["simulations_per_move"],
                "games_per_hour": best_selfplay["games_per_hour"],
                "estimated_runtime": _projections(
                    float(best_selfplay["games_per_hour"])
                ),
            },
            "cuda_selfplay": {
                "workers": best_cuda_selfplay["workers"],
                "inference_batch_size": best_cuda_selfplay["inference_batch_size"],
                "inference_max_wait_seconds": best_cuda_selfplay[
                    "flush_latency_seconds"
                ],
                "simulations": best_cuda_selfplay["simulations_per_move"],
                "games_per_hour": best_cuda_selfplay["games_per_hour"],
                "estimated_runtime": _projections(
                    float(best_cuda_selfplay["games_per_hour"])
                ),
            },
            "cpu_fallback": {
                "workers": matching_cpu["workers"],
                "simulations": matching_cpu["simulations_per_move"],
                "games_per_hour": matching_cpu["games_per_hour"],
                "estimated_runtime": _projections(
                    float(matching_cpu["games_per_hour"])
                ),
            },
        },
        "bottleneck": {
            "classification": bottleneck,
            "reason": "Average GPU utilization below 70% indicates that CPU game rules and MCTS do not submit inference fast enough to saturate the GPU."
            if bottleneck == "cpu_game_and_mcts"
            else "Sustained GPU utilization of at least 70% indicates GPU inference is the limiting resource.",
            "evidence": {
                "average_gpu_utilization_percent": gpu_utilization,
                "average_inference_batch_size": avg_batch,
                "configured_inference_batch_size": best_cuda_selfplay[
                    "inference_batch_size"
                ],
                "cuda_games_per_hour": best_cuda_selfplay["games_per_hour"],
                "cpu_games_per_hour": matching_cpu["games_per_hour"],
            },
        },
    }


__all__ = [
    "CUDA_TUNING_FORMAT",
    "CUDA_TUNING_VERSION",
    "CudaTuningConfig",
    "run_cuda_tuning_benchmark",
]
