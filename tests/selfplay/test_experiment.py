"""Tests for the reproducible Mini dataset experiment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from twixt_ai.selfplay import experiment
from twixt_ai.selfplay.experiment import (
    MiniDatasetExperimentConfig,
    StageConfig,
    run_mini_dataset_experiment,
)


def test_experiment_records_runtime_config_and_valid_datasets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = MiniDatasetExperimentConfig(
        simulations=2,
        workers=1,
        smoke=StageConfig("smoke", 1, 1, "smoke"),
        baseline=StageConfig("baseline", 2, 2, "baseline"),
    )

    report = run_mini_dataset_experiment(tmp_path, config=config)

    assert report["config"]["board"] == {"width": 10, "height": 10}
    assert report["config"]["mcts"]["simulations"] == 2
    for name, games in (("smoke", 1), ("baseline", 2)):
        stage = report["stages"][name]
        assert stage["selfplay"]["wall_seconds"] > 0
        assert stage["selfplay"]["games_per_hour"] > 0
        assert stage["selfplay"]["summary"]["aggregate"]["completed"] == games
        manifest = stage["dataset"]["manifest"]
        assert manifest["source_games"] == games
        assert manifest["examples"] > 0
        assert manifest["config"]["metadata"]["mcts"]["simulations"] == 2
        assert json.loads(
            (tmp_path / name / "dataset" / "manifest.json").read_text()
        ) == manifest
    assert json.loads((tmp_path / "report.json").read_text()) == report


def test_experiment_uses_cpu_count_without_sched_getaffinity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.delattr(experiment.os, "sched_getaffinity")
    monkeypatch.setattr(experiment.os, "cpu_count", lambda: 7)
    config = MiniDatasetExperimentConfig(
        simulations=1,
        workers=1,
        smoke=StageConfig("smoke", 1, 1, "smoke"),
        baseline=StageConfig("baseline", 1, 2, "baseline"),
    )

    report = run_mini_dataset_experiment(tmp_path, config=config)

    assert report["environment"]["available_cpus"] == 7


def test_experiment_requires_reproducible_hash_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)

    with pytest.raises(ValueError, match="PYTHONHASHSEED must be 0"):
        run_mini_dataset_experiment(tmp_path)


def test_experiment_refuses_nonempty_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    (tmp_path / "existing").write_text("keep")

    with pytest.raises(ValueError, match="must be empty"):
        run_mini_dataset_experiment(tmp_path)
