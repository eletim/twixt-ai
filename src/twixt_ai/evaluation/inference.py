"""Reproducible synchronous-versus-batched neural inference measurement."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import math
import os
import platform
import subprocess
from time import perf_counter, process_time

import torch

from twixt_ai.game import BoardDimensions, GameState, legal_peg_placements
from twixt_ai.models import MINI_POLICY_VALUE_CONFIG, PolicyValueNetwork
from twixt_ai.search.neural import NeuralInferenceBatcher, NeuralPolicyValue


INFERENCE_PERFORMANCE_FORMAT = "twixt-ai-inference-performance"
INFERENCE_PERFORMANCE_VERSION = 1


@dataclass(frozen=True, slots=True)
class InferencePerformanceConfig:
    """Settings sufficient to reproduce a Mini inference comparison."""

    requests: int = 256
    batch_size: int = 16
    warmups: int = 2
    seed: int = 54
    max_wait_seconds: float = 0.002
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        for name in ("requests", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.warmups, bool)
            or not isinstance(self.warmups, int)
            or self.warmups < 0
        ):
            raise ValueError("warmups must be a non-negative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if (
            isinstance(self.max_wait_seconds, bool)
            or not isinstance(self.max_wait_seconds, (int, float))
            or not math.isfinite(self.max_wait_seconds)
            or self.max_wait_seconds < 0
        ):
            raise ValueError("max_wait_seconds must be a finite non-negative number")
        if not isinstance(self.device, str):
            raise TypeError("device must be a string")
        try:
            torch.empty(0, device=self.device)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(f"device is unavailable: {self.device}") from exc

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-POSIX fallback
        return os.cpu_count() or 1


def _synchronize(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _gpu_utilization(device: str) -> float | None:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return None
    index = (
        resolved.index if resolved.index is not None else torch.cuda.current_device()
    )
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return float(result.stdout.strip().splitlines()[0])
    except (FileNotFoundError, IndexError, subprocess.SubprocessError, ValueError):
        return None


def _measure(
    operation: Callable[[], object], requests: int, device: str
) -> dict[str, float | None]:
    _synchronize(device)
    cpu_started = process_time()
    wall_started = perf_counter()
    operation()
    _synchronize(device)
    wall_seconds = perf_counter() - wall_started
    cpu_seconds = max(0.0, process_time() - cpu_started)
    average_cores = cpu_seconds / wall_seconds if wall_seconds else 0.0
    return {
        "wall_seconds": wall_seconds,
        "positions_per_second": requests / wall_seconds,
        "cpu_seconds": cpu_seconds,
        "cpu_utilization_percent": average_cores * 100.0,
        "available_cpu_percent": average_cores * 100.0 / _cpu_count(),
        "gpu_utilization_percent": _gpu_utilization(device),
    }


def run_inference_performance_benchmark(
    config: InferencePerformanceConfig | None = None,
) -> dict[str, object]:
    """Compare one-position forwards with concurrent dynamically batched calls."""

    config = config or InferencePerformanceConfig()
    if not isinstance(config, InferencePerformanceConfig):
        raise TypeError("config must be an InferencePerformanceConfig")
    torch.manual_seed(config.seed)
    model = PolicyValueNetwork(MINI_POLICY_VALUE_CONFIG).to(config.device)
    policy_value = NeuralPolicyValue(model)
    state = GameState.initial(BoardDimensions(10, 10))
    moves = legal_peg_placements(state)

    for _ in range(config.warmups):
        policy_value(state, moves)

    synchronous = _measure(
        lambda: [policy_value(state, moves) for _ in range(config.requests)],
        config.requests,
        config.device,
    )
    with NeuralInferenceBatcher(
        policy_value,
        batch_size=config.batch_size,
        max_wait_seconds=config.max_wait_seconds,
    ) as batcher:

        def batched_workload() -> None:
            with ThreadPoolExecutor(max_workers=config.batch_size) as pool:
                list(pool.map(lambda _: batcher(state, moves), range(config.requests)))

        batched = _measure(batched_workload, config.requests, config.device)
        batch_statistics = asdict(batcher.statistics)

    batched["speedup"] = (
        batched["positions_per_second"] / synchronous["positions_per_second"]
    )
    device = torch.device(config.device)
    accelerator: dict[str, object] = {
        "type": device.type,
        "cuda_available": torch.cuda.is_available(),
        "utilization_measurement": "nvidia-smi post-workload sample"
        if device.type == "cuda"
        else "not applicable",
    }
    if device.type == "cuda":
        accelerator.update(
            name=torch.cuda.get_device_name(device),
            peak_memory_bytes=torch.cuda.max_memory_allocated(device),
        )
    return {
        "format": INFERENCE_PERFORMANCE_FORMAT,
        "version": INFERENCE_PERFORMANCE_VERSION,
        "config": config.to_dict(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "available_cpus": _cpu_count(),
            "accelerator": accelerator,
        },
        "synchronous": synchronous,
        "batched": batched,
        "batch_statistics": batch_statistics,
    }


__all__ = [
    "INFERENCE_PERFORMANCE_FORMAT",
    "INFERENCE_PERFORMANCE_VERSION",
    "InferencePerformanceConfig",
    "run_inference_performance_benchmark",
]
