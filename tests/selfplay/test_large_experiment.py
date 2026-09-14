"""Tests for staged, neural Mini dataset generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from twixt_ai.models import (
    MINI_POLICY_VALUE_CONFIG,
    PolicyValueNetwork,
    save_policy_value_checkpoint,
)
from twixt_ai.selfplay.large_experiment import (
    LargeMiniDatasetConfig,
    LargeStageConfig,
    run_large_mini_dataset_experiment,
)


def _inputs(root: Path) -> tuple[Path, Path]:
    checkpoint = root / "model.pt"
    save_policy_value_checkpoint(
        checkpoint, PolicyValueNetwork(MINI_POLICY_VALUE_CONFIG)
    )
    benchmark = root / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            {
                "recommendation": {
                    "cuda_selfplay": {
                        "simulations": 1,
                        "workers": 1,
                        "inference_batch_size": 1,
                        "inference_max_wait_seconds": 0.0,
                        "estimated_runtime": {"1_games_hours": 0.0},
                    }
                }
            }
        )
    )
    return checkpoint, benchmark


def test_staged_experiment_validates_targets_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 123)

    class Selection:
        resolved_device = "cpu"

        def to_dict(self) -> dict[str, str]:
            return {"requested_device": "cuda", "resolved_device": "cpu"}

    monkeypatch.setattr(
        "twixt_ai.selfplay.large_experiment.select_device", lambda request: Selection()
    )
    checkpoint, benchmark = _inputs(tmp_path)
    output = tmp_path / "output"
    config = LargeMiniDatasetConfig(
        simulations=1,
        workers=1,
        inference_batch_size=1,
        inference_max_wait_seconds=0.0,
        stages=(LargeStageConfig("tiny", 1, 88, "split"),),
    )

    first = run_large_mini_dataset_experiment(
        output, checkpoint, benchmark, config=config
    )
    second = run_large_mini_dataset_experiment(
        output, checkpoint, benchmark, config=config
    )

    validation = first["stages"]["tiny"]["dataset"]["validation"]
    assert validation["valid"] is True
    assert validation["examples"] == validation["policy_examples"] > 0
    manifest = output / "tiny" / "dataset" / "manifest.json"
    assert (
        validation["manifest_sha256"]
        == hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    assert second == first


def test_requires_matching_benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    checkpoint, benchmark = _inputs(tmp_path)
    config = LargeMiniDatasetConfig(stages=(LargeStageConfig("tiny", 1, 88, "split"),))
    with pytest.raises(ValueError, match="benchmark recommendation"):
        run_large_mini_dataset_experiment(
            tmp_path / "output", checkpoint, benchmark, config=config
        )
