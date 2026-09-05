"""End-to-end tests for policy/value training and checkpoint recovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from twixt_ai.game import GameState
from twixt_ai.models import PolicyValueConfig, load_policy_value_checkpoint
from twixt_ai.training import TrainingConfig, train_model
from twixt_ai.training import trainer as trainer_module
from twixt_ai.training.train_cli import main


def _dataset(root: Path) -> Path:
    root.mkdir()
    (root / "train").mkdir()
    (root / "validation").mkdir()

    def example(identifier: str, x: int, outcome: int, policy: bool = False) -> dict[str, object]:
        value: dict[str, object] = {
            "format": "twixt-ai-training-example",
            "version": 1,
            "id": identifier,
            "position": GameState.initial().to_dict(),
            "action": {"x": x, "y": 1},
            "outcome": outcome,
            "source": {},
        }
        if policy:
            value["policy"] = [
                {"coordinate": {"x": x, "y": 1}, "probability": 0.75},
                {"coordinate": {"x": x + 1, "y": 1}, "probability": 0.25},
            ]
        return value

    contents = {
        "train/shard-00000.jsonl": b"".join(
            json.dumps(item, sort_keys=True).encode() + b"\n"
            for item in (example("a", 1, 1, True), example("b", 2, -1))
        ),
        "validation/shard-00000.jsonl": (
            json.dumps(example("c", 3, 0), sort_keys=True).encode() + b"\n"
        ),
    }
    for relative, content in contents.items():
        (root / relative).write_bytes(content)
    manifest = {
        "format": "twixt-ai-training-dataset",
        "version": 1,
        "splits": {
            "train": {
                "examples": 2,
                "shards": [{
                    "path": "train/shard-00000.jsonl", "examples": 2,
                    "sha256": hashlib.sha256(contents["train/shard-00000.jsonl"]).hexdigest(),
                }],
            },
            "validation": {
                "examples": 1,
                "shards": [{
                    "path": "validation/shard-00000.jsonl", "examples": 1,
                    "sha256": hashlib.sha256(contents["validation/shard-00000.jsonl"]).hexdigest(),
                }],
            },
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    return root


def _config(epochs: int) -> TrainingConfig:
    return TrainingConfig(
        epochs=epochs, batch_size=2, learning_rate=0.01, optimizer="sgd",
        scheduler="step", scheduler_gamma=0.5, seed=19,
    )


def test_trains_fixture_and_identifies_loadable_checkpoints(tmp_path: Path) -> None:
    output = tmp_path / "run"
    model_config = PolicyValueConfig(channels=2, residual_blocks=1, value_hidden=4)

    summary = train_model(
        _dataset(tmp_path / "dataset"), output,
        config=_config(2), model_config=model_config,
    )

    assert summary.completed_epochs == 2
    assert summary.best_epoch in {1, 2}
    assert len(summary.history) == 2
    assert [item.learning_rate for item in summary.history] == [0.01, 0.005]
    assert load_policy_value_checkpoint(output / "latest.pt").model.config == model_config
    metadata = load_policy_value_checkpoint(output / "best.pt").metadata
    assert metadata["training_config"]["seed"] == 19  # type: ignore[index]
    assert len((output / "metrics.jsonl").read_text().splitlines()) == 2
    assert json.loads((output / "summary.json").read_text()) == summary.to_dict()


def test_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "dataset")
    model_config = PolicyValueConfig(channels=2, residual_blocks=1, value_hidden=4)
    resumed = tmp_path / "resumed"
    train_model(dataset, resumed, config=_config(1), model_config=model_config)
    resumed_summary = train_model(
        dataset, resumed, config=_config(2), model_config=model_config, resume=True
    )
    uninterrupted = tmp_path / "uninterrupted"
    direct_summary = train_model(
        dataset, uninterrupted, config=_config(2), model_config=model_config
    )

    assert resumed_summary.history == direct_summary.history
    resumed_model = load_policy_value_checkpoint(resumed / "latest.pt").model
    direct_model = load_policy_value_checkpoint(uninterrupted / "latest.pt").model
    for resumed_weight, direct_weight in zip(
        resumed_model.state_dict().values(), direct_model.state_dict().values()
    ):
        assert torch.equal(resumed_weight, direct_weight)


def test_interruption_before_latest_does_not_commit_new_best(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset(tmp_path / "dataset")
    output = tmp_path / "run"
    model_config = PolicyValueConfig(channels=2, residual_blocks=1, value_hidden=4)
    train_model(dataset, output, config=_config(1), model_config=model_config)
    original_save = trainer_module._save

    def fixed_loss(*args: object, **kwargs: object) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    def interrupt_latest(path: Path, payload: object) -> None:
        if path.name == "latest.pt":
            raise RuntimeError("simulated interruption")
        original_save(path, payload)

    monkeypatch.setattr(trainer_module, "_epoch", fixed_loss)
    monkeypatch.setattr(trainer_module, "_save", interrupt_latest)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train_model(
            dataset, output, config=_config(2),
            model_config=model_config, resume=True,
        )

    # best.pt may be ahead, but latest.pt still identifies the last fully
    # committed epoch and therefore causes epoch two to be replayed.
    assert load_policy_value_checkpoint(output / "latest.pt").metadata["epoch"] == 1
    assert load_policy_value_checkpoint(output / "best.pt").metadata["epoch"] == 2

    monkeypatch.setattr(trainer_module, "_save", original_save)
    summary = train_model(
        dataset, output, config=_config(2), model_config=model_config, resume=True
    )
    assert summary.completed_epochs == summary.best_epoch == 2
    assert load_policy_value_checkpoint(output / "latest.pt").metadata["best_epoch"] == 2
    assert load_policy_value_checkpoint(output / "best.pt").metadata["epoch"] == 2


def test_resume_repairs_metrics_from_final_checkpoint(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "dataset")
    output = tmp_path / "run"
    model_config = PolicyValueConfig(channels=2, residual_blocks=1, value_hidden=4)
    original = train_model(
        dataset, output, config=_config(1), model_config=model_config
    )
    (output / "metrics.jsonl").write_text("stale\n")

    resumed = train_model(
        dataset, output, config=_config(1), model_config=model_config, resume=True
    )

    assert resumed.history == original.history
    metrics = [
        json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()
    ]
    assert metrics == [item.to_dict() for item in original.history]


def test_cli_emits_summary(tmp_path: Path, capsys: object) -> None:
    dataset = _dataset(tmp_path / "dataset")
    output = tmp_path / "run"

    assert main([
        "--dataset", str(dataset), "--output-dir", str(output),
        "--epochs", "1", "--batch-size", "2", "--channels", "2",
        "--residual-blocks", "1", "--value-hidden", "4", "--seed", "7",
    ]) == 0

    emitted = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert emitted["completed_epochs"] == 1
    assert emitted["config"]["seed"] == 7
