"""Tests for matched v1 and v2 Mini training."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from twixt_ai.models import load_policy_value_checkpoint
from twixt_ai.training.encoding_experiment import (
    MATCHED_ENCODING_TRAINING_FORMAT,
    MatchedEncodingTrainingConfig,
    run_matched_encoding_training,
)
from twixt_ai.training.encoding_experiment_cli import main


DATASET = Path(__file__).parents[2] / "experiments/issue-56/smoke/dataset"


def _config() -> MatchedEncodingTrainingConfig:
    return MatchedEncodingTrainingConfig(
        epochs=2,
        batch_size=8,
        learning_rate=0.01,
        tiny_epochs=30,
        tiny_learning_rate=0.02,
        tiny_max_loss_ratio=0.9,
        torch_threads=1,
    )


def test_trains_both_encodings_with_matched_conditions(tmp_path: Path) -> None:
    output = tmp_path / "comparison"
    report = run_matched_encoding_training(DATASET, output, config=_config())

    assert report["format"] == MATCHED_ENCODING_TRAINING_FORMAT
    assert report["dataset"]["train_examples"] == 33
    old = report["results"]["22_plane_v1"]
    new = report["results"]["10_plane_v2"]
    assert (old["planes"], new["planes"]) == (22, 10)
    assert old["model_config"]["channels"] == new["model_config"]["channels"]
    assert old["model_config"]["residual_blocks"] == new["model_config"]["residual_blocks"]
    for name, result in (("22-plane-v1", old), ("10-plane-v2", new)):
        assert result["tiny_overfit"]["passed"] is True
        training = result["training"]
        assert len(training["summary"]["history"]) == 2
        assert len(training["checkpoint_history"]) == 2
        assert training["examples_per_second"] > 0
        for kind in ("latest", "best"):
            checkpoint = training["checkpoints"][kind]
            path = output / name / "training" / checkpoint["path"]
            assert load_policy_value_checkpoint(path).model.config.encoding_version == result["encoding_version"]
            assert checkpoint["finite_weights"] is True
    assert json.loads((output / "report.json").read_text(encoding="utf-8")) == report


def test_refuses_nonempty_output(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "keep").write_text("existing")
    with pytest.raises(ValueError, match="must be empty"):
        run_matched_encoding_training(DATASET, output, config=_config())


def test_cli_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "comparison"
    assert main([
        "--dataset", str(DATASET), "--output-dir", str(output),
        "--epochs", "1", "--batch-size", "8", "--tiny-epochs", "20",
        "--tiny-learning-rate", "0.02", "--tiny-max-loss-ratio", "0.95",
    ]) == 0
    assert json.loads((output / "report.json").read_text())["format"] == MATCHED_ENCODING_TRAINING_FORMAT


@pytest.mark.parametrize("kwargs", [
    {"epochs": 0},
    {"tiny_epochs": 0},
    {"tiny_max_loss_ratio": 0},
    {"tiny_max_loss_ratio": 1},
    {"torch_threads": 0},
])
def test_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        MatchedEncodingTrainingConfig(**kwargs)  # type: ignore[arg-type]
