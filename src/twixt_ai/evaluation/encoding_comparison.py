"""Reproducible v1 (22-plane) versus v2 (10-plane) Mini benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import json
import os
import platform
from pathlib import Path
from statistics import median
import subprocess
from time import perf_counter

import torch
from torch import Tensor
from torch.nn import functional as F

from twixt_ai.game import GameState
from twixt_ai.models import (
    ENCODING_VERSION,
    MINI_ENCODING_VERSION,
    MINI_NUM_CHANNELS,
    MINI_POLICY_VALUE_CONFIG,
    PolicyValueConfig,
    PolicyValueNetwork,
    encode_position_for_version,
)
from twixt_ai.training import trainer as trainer_module


ENCODING_COMPARISON_FORMAT = "twixt-ai-encoding-comparison"
ENCODING_COMPARISON_VERSION = 2


@dataclass(frozen=True, slots=True)
class EncodingComparisonConfig:
    """Workload and model settings recorded in an encoding comparison."""

    dataset_dir: str
    batch_size: int = 32
    encoding_repeats: int = 20
    forward_iterations: int = 100
    training_steps: int = 50
    samples: int = 7
    warmups: int = 5
    seed: int = 74
    devices: tuple[str, ...] = ("cpu",)
    torch_threads: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_dir, str) or not self.dataset_dir:
            raise ValueError("dataset_dir must be a non-empty string")
        for name in (
            "batch_size",
            "encoding_repeats",
            "forward_iterations",
            "training_steps",
            "samples",
            "torch_threads",
        ):
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
        if not isinstance(self.devices, tuple) or not self.devices:
            raise ValueError("devices must be a non-empty tuple")
        resolved_devices: set[str] = set()
        for device in self.devices:
            if not isinstance(device, str) or not device:
                raise ValueError("devices must contain non-empty strings")
            try:
                resolved = torch.device(device)
            except (RuntimeError, TypeError) as exc:
                raise ValueError(f"invalid device: {device}") from exc
            if resolved.type not in {"cpu", "cuda"}:
                raise ValueError("benchmark devices must be CPU or CUDA")
            if resolved.type == "cpu":
                if resolved.index is not None:
                    raise ValueError("CPU device must be specified as 'cpu'")
                identity = "cpu"
            else:
                if not torch.cuda.is_available():
                    raise ValueError(f"device is unavailable: {device}")
                index = (
                    resolved.index
                    if resolved.index is not None
                    else torch.cuda.current_device()
                )
                if not 0 <= index < torch.cuda.device_count():
                    raise ValueError(f"device is unavailable: {device}")
                identity = f"cuda:{index}"
            if identity in resolved_devices:
                raise ValueError(f"duplicate device: {device}")
            try:
                torch.empty(1, device=device)
                _synchronize(device)
            except (RuntimeError, TypeError) as exc:
                raise ValueError(f"device is unavailable: {device}") from exc
            resolved_devices.add(identity)

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["devices"] = list(self.devices)
        return value


def _synchronize(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _measure(operation: Callable[[], object], count: int, device: str) -> dict[str, float]:
    _synchronize(device)
    started = perf_counter()
    operation()
    _synchronize(device)
    seconds = perf_counter() - started
    return {
        "wall_seconds": seconds,
        "latency_seconds": seconds / count,
        "positions_per_second": count / seconds,
    }


def _summarize(samples: list[dict[str, float]]) -> dict[str, object]:
    if not samples:
        raise ValueError("at least one measurement sample is required")
    metrics = ("wall_seconds", "latency_seconds", "positions_per_second")
    summary: dict[str, object] = {"sample_count": len(samples), "samples": samples}
    dispersion: dict[str, object] = {}
    for metric in metrics:
        values = [sample[metric] for sample in samples]
        center = median(values)
        summary[metric] = center
        dispersion[metric] = {
            "minimum": min(values),
            "maximum": max(values),
            "median_absolute_deviation": median(
                [abs(value - center) for value in values]
            ),
        }
    summary["dispersion"] = dispersion
    return summary


def _summarize_values(values: list[float]) -> dict[str, object]:
    center = median(values)
    return {
        "sample_count": len(values),
        "samples": values,
        "median": center,
        "minimum": min(values),
        "maximum": max(values),
        "median_absolute_deviation": median(
            [abs(value - center) for value in values]
        ),
    }


def _repeat(operation: Callable[[], object], iterations: int) -> None:
    for _ in range(iterations):
        operation()


def _tensor_bytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _module_bytes(model: PolicyValueNetwork) -> int:
    return sum(_tensor_bytes(item) for item in (*model.parameters(), *model.buffers()))


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    except OSError:  # pragma: no cover - non-Linux fallback
        pass
    return platform.processor() or "unknown"


def _git_metadata() -> dict[str, object]:
    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ("git", *args), check=True, capture_output=True, text=True, timeout=2
            )
            return result.stdout.strip()
        except (FileNotFoundError, subprocess.SubprocessError):
            return None

    revision = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {"revision": revision, "dirty": bool(status) if status is not None else None}


def _environment() -> dict[str, object]:
    accelerator: dict[str, object] = {
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "devices": [],
    }
    if torch.cuda.is_available():
        accelerator["devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
        try:
            driver = subprocess.run(
                ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout.splitlines()[0].strip()
        except (FileNotFoundError, IndexError, subprocess.SubprocessError):
            driver = None
        accelerator["driver"] = driver
    try:
        available_cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-POSIX fallback
        available_cpus = os.cpu_count() or 1
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu_model(),
        "available_cpus": available_cpus,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "accelerator": accelerator,
        "git": _git_metadata(),
    }


def _model_config(encoding_version: int) -> PolicyValueConfig:
    base = MINI_POLICY_VALUE_CONFIG
    return PolicyValueConfig(
        channels=base.channels,
        residual_blocks=base.residual_blocks,
        value_hidden=base.value_hidden,
        board_width=base.board_width,
        board_height=base.board_height,
        input_channels=(
            base.input_channels
            if encoding_version == ENCODING_VERSION
            else MINI_NUM_CHANNELS
        ),
        encoding_version=encoding_version,
    )


def _batch(
    items: list[tuple[Tensor, Tensor, Tensor]], size: int
) -> tuple[Tensor, Tensor, Tensor]:
    selected = [items[index % len(items)] for index in range(size)]
    return tuple(torch.stack(values) for values in zip(*selected))  # type: ignore[return-value]


def _peak_memory(device: str) -> int | None:
    return (
        torch.cuda.max_memory_allocated(device)
        if torch.device(device).type == "cuda"
        else None
    )


def _benchmark_forward_sample(
    model_config: PolicyValueConfig,
    batch: tuple[Tensor, Tensor, Tensor],
    config: EncodingComparisonConfig,
    device: str,
    *,
    sample: int,
    batched: bool,
) -> tuple[dict[str, float], int | None]:
    torch.manual_seed(config.seed + sample)
    model = PolicyValueNetwork(model_config).to(device)
    inputs = batch[0].to(device)
    selected = inputs if batched else inputs[:1]
    model.eval()
    with torch.no_grad():
        for _ in range(config.warmups):
            model(selected)

        if torch.device(device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        measurement = _measure(
            lambda: _repeat(lambda: model(selected), config.forward_iterations),
            config.forward_iterations * len(selected),
            device,
        )
    return measurement, _peak_memory(device)


def _benchmark_training_sample(
    model_config: PolicyValueConfig,
    batch: tuple[Tensor, Tensor, Tensor],
    config: EncodingComparisonConfig,
    device: str,
    *,
    sample: int,
) -> tuple[dict[str, float], int | None]:
    torch.manual_seed(config.seed + sample)
    model = PolicyValueNetwork(model_config).to(device)
    inputs, policies, outcomes = (value.to(device) for value in batch)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        logits, values = model(inputs)
        policy_loss = -(policies * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        loss = policy_loss + F.mse_loss(values, outcomes)
        loss.backward()
        optimizer.step()

    model.train()
    for _ in range(config.warmups):
        step()
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training = _measure(
        lambda: _repeat(step, config.training_steps),
        config.training_steps * config.batch_size,
        device,
    )
    training["steps_per_second"] = training["positions_per_second"] / config.batch_size
    training["seconds_per_step"] = training["latency_seconds"] * config.batch_size
    return training, _peak_memory(device)


def _sample_order(sample: int) -> tuple[str, str]:
    order = ("22_plane_v1", "10_plane_v2")
    return order if sample % 2 == 0 else tuple(reversed(order))


def _percent_reduction(old: float | int, new: float | int) -> float:
    return (float(old) - float(new)) * 100.0 / float(old)


def run_encoding_comparison_benchmark(
    config: EncodingComparisonConfig,
) -> dict[str, object]:
    """Measure v1 and v2 on identical Mini positions and matched networks."""

    if not isinstance(config, EncodingComparisonConfig):
        raise TypeError("config must be an EncodingComparisonConfig")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(config.torch_threads)
    try:
        dataset = Path(config.dataset_dir)
        manifest_v1, digest_v1, board, examples_v1, _ = trainer_module._load_dataset(
            dataset, encoding_version=ENCODING_VERSION
        )
        manifest_v2, digest_v2, board_v2, examples_v2, _ = trainer_module._load_dataset(
            dataset, encoding_version=MINI_ENCODING_VERSION
        )
        if board.width != 10 or board.height != 10:
            raise ValueError("encoding comparison requires a 10x10 Mini dataset")
        if (
            board_v2 != board
            or digest_v2 != digest_v1
            or len(examples_v2) != len(examples_v1)
        ):
            raise ValueError("encoding paths did not load the same dataset")

        states = []
        for shard in manifest_v1["splits"]["train"]["shards"]:
            for line in (dataset / shard["path"]).read_text(encoding="utf-8").splitlines():
                states.append(GameState.from_dict(json.loads(line)["position"]))
        if not states:
            raise ValueError("training split must contain at least one position")

        versions = {
            "22_plane_v1": (ENCODING_VERSION, examples_v1),
            "10_plane_v2": (MINI_ENCODING_VERSION, examples_v2),
        }
        results: dict[str, dict[str, object]] = {}
        for name, (version, examples) in versions.items():
            for _ in range(config.warmups):
                for state in states:
                    encode_position_for_version(state, version)
            one = encode_position_for_version(states[0], version)
            model_config = _model_config(version)
            model = PolicyValueNetwork(model_config)
            results[name] = {
                "encoding_version": version,
                "planes": model_config.input_channels,
                "encoding": {
                    "bytes_per_position": _tensor_bytes(one),
                    "batch_bytes": _tensor_bytes(one) * config.batch_size,
                },
                "model": {
                    "config": model_config.to_dict(),
                    "parameter_count": sum(item.numel() for item in model.parameters()),
                    "parameter_and_buffer_bytes": _module_bytes(model),
                },
                "devices": {},
            }

        encoding_samples: dict[str, list[dict[str, float]]] = {
            name: [] for name in versions
        }
        for sample in range(config.samples):
            for name in _sample_order(sample):
                version = versions[name][0]
                encoding_samples[name].append(
                    _measure(
                        lambda: _repeat(
                            lambda: [
                                encode_position_for_version(state, version)
                                for state in states
                            ],
                            config.encoding_repeats,
                        ),
                        config.encoding_repeats * len(states),
                        "cpu",
                    )
                )
        for name in versions:
            results[name]["encoding"].update(_summarize(encoding_samples[name]))  # type: ignore[union-attr]

        batches = {
            name: _batch(examples.items, config.batch_size)
            for name, (_, examples) in versions.items()
        }
        for device in config.devices:
            measurements = {
                name: {"single": [], "batched": [], "training": []}
                for name in versions
            }
            peaks = {
                name: {"forward": [], "training": []} for name in versions
            }
            for sample in range(config.samples):
                order = _sample_order(sample)
                for name in order:
                    measured, peak = _benchmark_forward_sample(
                        _model_config(versions[name][0]),
                        batches[name],
                        config,
                        device,
                        sample=sample,
                        batched=False,
                    )
                    measurements[name]["single"].append(measured)
                    if peak is not None:
                        peaks[name]["forward"].append(peak)
                for name in order:
                    measured, peak = _benchmark_forward_sample(
                        _model_config(versions[name][0]),
                        batches[name],
                        config,
                        device,
                        sample=sample,
                        batched=True,
                    )
                    measurements[name]["batched"].append(measured)
                    if peak is not None:
                        peaks[name]["forward"].append(peak)
                for name in order:
                    measured, peak = _benchmark_training_sample(
                        _model_config(versions[name][0]),
                        batches[name],
                        config,
                        device,
                        sample=sample,
                    )
                    measurements[name]["training"].append(measured)
                    if peak is not None:
                        peaks[name]["training"].append(peak)

            for name in versions:
                training = _summarize(measurements[name]["training"])
                training["steps_per_second"] = (
                    training["positions_per_second"] / config.batch_size  # type: ignore[operator]
                )
                training["seconds_per_step"] = (
                    training["latency_seconds"] * config.batch_size  # type: ignore[operator]
                )
                results[name]["devices"][device] = {  # type: ignore[index]
                    "single_position_forward": _summarize(
                        measurements[name]["single"]
                    ),
                    "batched_forward": _summarize(measurements[name]["batched"]),
                    "training": training,
                    "memory": {
                        "input_batch_bytes": _tensor_bytes(batches[name][0]),
                        "forward_peak_allocated_bytes": (
                            max(peaks[name]["forward"])
                            if peaks[name]["forward"]
                            else None
                        ),
                        "training_peak_allocated_bytes": (
                            max(peaks[name]["training"])
                            if peaks[name]["training"]
                            else None
                        ),
                    },
                }

        old, new = results["22_plane_v1"], results["10_plane_v2"]
        comparisons: dict[str, object] = {
            "encoding_bytes_reduction_percent": _percent_reduction(
                old["encoding"]["bytes_per_position"],  # type: ignore[index]
                new["encoding"]["bytes_per_position"],  # type: ignore[index]
            ),
            "encoding_latency_reduction_percent": _summarize_values(
                [
                    _percent_reduction(
                        old_sample["latency_seconds"],
                        new_sample["latency_seconds"],
                    )
                    for old_sample, new_sample in zip(
                        old["encoding"]["samples"],  # type: ignore[index]
                        new["encoding"]["samples"],  # type: ignore[index]
                    )
                ]
            ),
            "model_parameters_reduction_percent": _percent_reduction(
                old["model"]["parameter_count"],  # type: ignore[index]
                new["model"]["parameter_count"],  # type: ignore[index]
            ),
            "devices": {},
        }
        for device in config.devices:
            old_device = old["devices"][device]  # type: ignore[index]
            new_device = new["devices"][device]  # type: ignore[index]
            def throughput_changes(workload: str) -> dict[str, object]:
                return _summarize_values(
                    [
                        -_percent_reduction(
                            old_sample["positions_per_second"],
                            new_sample["positions_per_second"],
                        )
                        for old_sample, new_sample in zip(
                            old_device[workload]["samples"],
                            new_device[workload]["samples"],
                        )
                    ]
                )

            comparisons["devices"][device] = {  # type: ignore[index]
                "single_forward_throughput_change_percent": throughput_changes(
                    "single_position_forward"
                ),
                "batched_forward_throughput_change_percent": throughput_changes(
                    "batched_forward"
                ),
                "training_throughput_change_percent": throughput_changes(
                    "training"
                ),
            }
        return {
            "format": ENCODING_COMPARISON_FORMAT,
            "version": ENCODING_COMPARISON_VERSION,
            "config": config.to_dict(),
            "dataset": {
                "manifest": str(dataset / "manifest.json"),
                "manifest_sha256": digest_v1,
                "training_examples": len(states),
                "encoding_positions_per_repeat": len(states),
                "model_batch_examples": min(config.batch_size, len(states)),
                "model_batch_selection": (
                    "first training examples; cycle in order only when batch size "
                    "exceeds the training split"
                ),
                "board": board.to_dict(),
            },
            "environment": _environment(),
            "methodology": {
                "summary_statistic": "median",
                "dispersion": "minimum, maximum, and median absolute deviation",
                "pairing": "each sample measures both encodings on the same workload",
                "sample_order": [list(_sample_order(sample)) for sample in range(config.samples)],
                "fresh_model_per_timed_sample": True,
            },
            "results": results,
            "comparison": comparisons,
            "interpretation": "Speed and memory measurements do not measure playing strength.",
        }
    finally:
        torch.set_num_threads(old_threads)


__all__ = [
    "ENCODING_COMPARISON_FORMAT",
    "ENCODING_COMPARISON_VERSION",
    "EncodingComparisonConfig",
    "run_encoding_comparison_benchmark",
]
