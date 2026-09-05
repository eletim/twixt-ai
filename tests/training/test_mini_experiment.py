"""Tests for the first learned Mini Twixt training experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from twixt_ai.game import BoardDimensions, GameState
from twixt_ai.models import load_policy_value_checkpoint
from twixt_ai.training.mini_experiment import (
    MiniTrainingExperimentConfig,
    run_mini_training_experiment,
)


def _dataset(root: Path) -> Path:
    board = BoardDimensions(10, 10)
    root.mkdir()
    (root / "train").mkdir()
    (root / "validation").mkdir()

    def example(identifier: str, x: int, outcome: int) -> bytes:
        return (
            json.dumps({
                "format": "twixt-ai-training-example",
                "version": 1,
                "id": identifier,
                "position": GameState.initial(board).to_dict(),
                "action": {"x": x, "y": 1},
                "outcome": outcome,
                "source": {},
            }, sort_keys=True).encode()
            + b"\n"
        )

    contents = {
        "train/shard-00000.jsonl": example("a", 1, 1) + example("b", 2, 1),
        "validation/shard-00000.jsonl": example("c", 3, 1),
    }
    for relative, content in contents.items():
        (root / relative).write_bytes(content)
    manifest = {
        "format": "twixt-ai-training-dataset",
        "version": 1,
        "board": board.to_dict(),
        "splits": {
            name: {
                "examples": content.count(b"\n"),
                "shards": [{
                    "path": relative,
                    "examples": content.count(b"\n"),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }],
            }
            for name, relative, content in (
                ("train", "train/shard-00000.jsonl", contents["train/shard-00000.jsonl"]),
                (
                    "validation",
                    "validation/shard-00000.jsonl",
                    contents["validation/shard-00000.jsonl"],
                ),
            )
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    return root


def test_experiment_overfits_resumes_and_loads_for_mcts(tmp_path: Path) -> None:
    config = MiniTrainingExperimentConfig(
        epochs=4,
        resume_after_epochs=2,
        batch_size=2,
        learning_rate=0.01,
        tiny_epochs=40,
        tiny_learning_rate=0.02,
        tiny_max_loss_ratio=0.8,
    )

    report = run_mini_training_experiment(
        _dataset(tmp_path / "dataset"), tmp_path / "experiment", config=config
    )

    assert report["format"] == "twixt-ai-mini-training-experiment"
    assert report["tiny_overfit"]["passed"] is True
    assert report["tiny_overfit"]["loss_ratio"] < 0.8
    training = report["training"]
    assert training["resume_verified"] is True
    assert training["examples_per_second"] > 0
    assert len(training["summary"]["history"]) == 4
    assert len(training["checkpoint_history"]) == 4
    assert report["sanity_checks"]["inference"]["prior_total"] == pytest.approx(1)
    best = tmp_path / "experiment" / "baseline" / "best.pt"
    assert load_policy_value_checkpoint(best).model.config.board_width == 10
    assert json.loads((tmp_path / "experiment" / "report.json").read_text()) == report


def test_experiment_refuses_nonempty_output(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "keep").write_text("existing")

    with pytest.raises(ValueError, match="must be empty"):
        run_mini_training_experiment(_dataset(tmp_path / "dataset"), output)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 1, "resume_after_epochs": 1},
        {"tiny_max_loss_ratio": 0},
        {"tiny_max_loss_ratio": 1},
    ],
)
def test_experiment_config_rejects_invalid_thresholds(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        MiniTrainingExperimentConfig(**kwargs)  # type: ignore[arg-type]
