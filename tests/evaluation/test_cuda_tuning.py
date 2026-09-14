from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from twixt_ai.evaluation import cuda_tuning
from twixt_ai.evaluation.cuda_tuning import (
    CUDA_TUNING_FORMAT,
    CudaTuningConfig,
    run_cuda_tuning_benchmark,
)


def test_config_rejects_invalid_sweeps() -> None:
    with pytest.raises(ValueError, match="worker_counts"):
        CudaTuningConfig(worker_counts=())
    with pytest.raises(ValueError, match="flush_latencies"):
        CudaTuningConfig(flush_latencies_seconds=(-0.1,))
    with pytest.raises(ValueError, match="gpu_sample_interval"):
        CudaTuningConfig(gpu_sample_interval_seconds=0)
    with pytest.raises(ValueError, match="warmup_games"):
        CudaTuningConfig(games=2, warmup_games=1, worker_counts=(1, 2))
    with pytest.raises(ValueError, match="must not exceed games"):
        CudaTuningConfig(games=4, warmup_games=8, worker_counts=(1, 8))


def test_default_workers_are_all_exercised_by_default_games() -> None:
    config = CudaTuningConfig()

    assert config.worker_counts == (1, 4)
    assert max(config.worker_counts) <= config.games


def test_gpu_sampler_uses_physical_uuid_for_logical_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    sampler: cuda_tuning._GpuSampler

    class Uuid:
        def __str__(self) -> str:
            return "01234567-89ab-cdef-0123-456789abcdef"

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        sampler._stop.set()
        return SimpleNamespace(stdout="17, 256\n")

    monkeypatch.setattr(
        cuda_tuning.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(uuid=Uuid()),
    )
    monkeypatch.setattr(cuda_tuning.subprocess, "run", run)
    sampler = cuda_tuning._GpuSampler("cuda:0", 0.1)

    sampler._poll()

    assert calls == [
        [
            "nvidia-smi",
            "--id=GPU-01234567-89ab-cdef-0123-456789abcdef",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
    ]
    assert sampler.samples == [(17.0, 256.0)]


@pytest.mark.parametrize(
    ("failure", "expected_detail"),
    (
        (FileNotFoundError("nvidia-smi unavailable"), "FileNotFoundError"),
        (
            subprocess.CalledProcessError(
                1, ["nvidia-smi"], stderr="driver communication failed"
            ),
            "driver communication failed",
        ),
    ),
)
def test_gpu_sampler_fails_when_nvidia_smi_produces_no_valid_sample(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    expected_detail: str,
) -> None:
    sampler: cuda_tuning._GpuSampler

    def run(*args: object, **kwargs: object) -> None:
        sampler._stop.set()
        raise failure

    monkeypatch.setattr(
        cuda_tuning.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(uuid="GPU-test"),
    )
    monkeypatch.setattr(cuda_tuning.subprocess, "run", run)
    sampler = cuda_tuning._GpuSampler("cuda:0", 0.1)

    sampler._poll()

    with pytest.raises(RuntimeError, match="no valid nvidia-smi sample") as caught:
        sampler.to_dict()
    assert expected_detail in str(caught.value)
    assert sampler.failure_count == 1


def test_report_selects_measured_settings_and_projects_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = CudaTuningConfig(
        games=2,
        warmup_games=2,
        worker_counts=(1, 2),
        inference_batch_sizes=(1, 2),
        flush_latencies_seconds=(0.001,),
        simulation_budgets=(2,),
        training_batch_sizes=(8, 16),
    )
    observed_selfplay: list[tuple[str, int, int, float, int]] = []

    def training_run(
        dataset: Path,
        root: Path,
        batch_size: int,
        device: str,
        config: CudaTuningConfig,
    ) -> dict[str, object]:
        rate = float(batch_size * (10 if device == "cuda" else 1))
        return {
            "device": device,
            "batch_size": batch_size,
            "examples_per_second": rate,
            "peak_cuda_memory_bytes": 1024 if device == "cuda" else None,
        }

    def selfplay_run(
        checkpoint: Path,
        *,
        device: str,
        workers: int,
        batch_size: int,
        latency: float,
        simulations: int,
        config: CudaTuningConfig,
    ) -> dict[str, object]:
        observed_selfplay.append((device, workers, batch_size, latency, simulations))
        rate = float(100 * workers + (50 * batch_size if device == "cuda" else 0))
        return {
            "device": device,
            "workers": workers,
            "inference_batch_size": batch_size,
            "flush_latency_seconds": latency,
            "simulations_per_move": simulations,
            "steady_state_games_per_hour": rate,
            "setup_and_warmup_seconds": 10.0,
            "inference": {"requests": 8, "batches": 4},
            "gpu": {"average_utilization_percent": 40.0},
        }

    monkeypatch.setattr(cuda_tuning, "_training_run", training_run)
    monkeypatch.setattr(cuda_tuning, "_selfplay_run", selfplay_run)
    monkeypatch.setattr(cuda_tuning, "_sha256", lambda path: "digest")
    monkeypatch.setattr(cuda_tuning, "_cuda_hardware_available", lambda: True)
    monkeypatch.setattr(cuda_tuning.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda_tuning.torch.cuda, "get_device_name", lambda: "RTX test")

    report = run_cuda_tuning_benchmark(
        tmp_path / "dataset", tmp_path / "model.pt", config
    )

    assert report["format"] == CUDA_TUNING_FORMAT
    assert report["recommendation"]["training"]["batch_size"] == 16
    selected = report["recommendation"]["selfplay"]
    assert selected["workers"] == 2
    assert selected["device"] == "cuda"
    assert selected["inference_batch_size"] == 2
    assert selected["estimated_runtime"]["5000_games_hours"] == pytest.approx(
        10 / 3600 + 5000 / 300
    )
    assert report["bottleneck"]["classification"] == "cpu_game_and_mcts"
    assert report["bottleneck"]["evidence"]["average_inference_batch_size"] == 2
    # Per worker: one CPU run, one synchronous CUDA run, and one batched run.
    assert len(observed_selfplay) == 6
    assert [device for device, *_ in observed_selfplay] == [
        "cpu",
        "cpu",
        "cuda",
        "cuda",
        "cuda",
        "cuda",
    ]


def test_cuda_is_required_for_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cuda_tuning, "_cuda_hardware_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is required"):
        run_cuda_tuning_benchmark(tmp_path, tmp_path / "model.pt")


def test_missing_gpu_samples_do_not_claim_a_bottleneck() -> None:
    classification, reason = cuda_tuning._bottleneck_classification(None)

    assert classification == "unknown"
    assert "could not be sampled" in reason


def test_selfplay_metrics_separate_throughput_from_experienced_latency() -> None:
    metrics = cuda_tuning._selfplay_metrics(
        [
            {"moves": 10, "elapsed_seconds": 2.0},
            {"moves": 20, "elapsed_seconds": 6.0},
        ],
        wall_seconds=6.0,
        setup_seconds=4.0,
        simulations=8,
    )

    assert metrics["aggregate_moves_per_second"] == pytest.approx(5.0)
    assert metrics["per_game_move_latency_seconds"] == {
        "minimum": pytest.approx(0.2),
        "median": pytest.approx(0.25),
        "p95": pytest.approx(0.3),
        "maximum": pytest.approx(0.3),
    }
    assert metrics["setup_and_warmup_seconds"] == 4.0
