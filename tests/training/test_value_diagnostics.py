"""Focused regression coverage for the value-target perspective contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from twixt_ai.agents import AgentRequest, RandomAgent
from twixt_ai.evaluation import MatchConfig, run_match
from twixt_ai.game import BoardDimensions, GameState, PegPlacement, Player
from twixt_ai.models import (
    BoardSymmetry,
    PolicyValueConfig,
    PolicyValueNetwork,
    save_policy_value_checkpoint,
    transform_state,
)
from twixt_ai.search import MCTSAgent, PolicyValueEstimate
from twixt_ai.training import DatasetConfig, build_dataset
from twixt_ai.training.value_diagnostics import (
    ValueDiagnosticsConfig,
    diagnose_value_model,
)


def _decisive_dataset(root: Path) -> tuple[Path, object]:
    match = run_match(
        RandomAgent(),
        RandomAgent(),
        config=MatchConfig(BoardDimensions(6, 6), 0),
    )
    assert match.winner is Player.RED
    source = root / "match.json"
    source.write_text(match.to_json(), encoding="utf-8")
    dataset = root / "dataset"
    summary = build_dataset(
        source,
        dataset,
        config=DatasetConfig(validation_fraction=0, split_seed="perspective"),
    )
    return dataset, summary


def test_value_perspective_survives_artifact_dataset_and_augmentation(
    tmp_path: Path,
) -> None:
    dataset, summary = _decisive_dataset(tmp_path)
    shard = summary.to_dict()["splits"]["train"]["shards"][0]["path"]
    examples = [
        json.loads(line)
        for line in (dataset / shard).read_text(encoding="utf-8").splitlines()
    ]

    assert [example["outcome"] for example in examples] == [
        1 if index % 2 == 0 else -1 for index in range(len(examples))
    ]
    for example in examples:
        # Dataset positions are exactly the pre-move self-play states. An
        # axis-swapping symmetry exchanges colors and goals, so winner and
        # side-to-move both change and the relative value stays invariant.
        state = transform_state(
            GameState.from_dict(example["position"]), BoardSymmetry.TRANSPOSE
        )
        transformed_winner = Player.BLACK
        expected = 1 if transformed_winner is state.side_to_move else -1
        assert expected == example["outcome"]


def test_diagnostics_measure_loaded_targets_checkpoint_and_mse(tmp_path: Path) -> None:
    dataset, summary = _decisive_dataset(tmp_path)
    config = PolicyValueConfig(
        channels=2,
        residual_blocks=1,
        value_hidden=2,
        board_width=6,
        board_height=6,
    )
    model = PolicyValueNetwork(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    checkpoint = tmp_path / "zero.pt"
    save_policy_value_checkpoint(checkpoint, model, metadata={"purpose": "test"})

    report = diagnose_value_model(
        dataset,
        checkpoint,
        config=ValueDiagnosticsConfig(
            ply_bucket_size=4,
            calibration_bins=4,
            batch_size=3,
            device="cpu",
        ),
    )

    expected_examples = summary.train_examples
    assert report["splits"]["train"]["targets"]["counts"] == {
        "-1": expected_examples // 2,
        "0": 0,
        "1": (expected_examples + 1) // 2,
    }
    metrics = report["splits"]["train"]["metrics"]
    assert metrics["mean_prediction"] == 0
    assert metrics["mse"] == 1
    assert metrics["mae"] == 1
    assert report["splits"]["validation"]["metrics"]["mse"] is None
    assert report["generalization"]["mse_gap"] is None


def test_target_only_diagnostics_use_default_config(tmp_path: Path) -> None:
    dataset, summary = _decisive_dataset(tmp_path)

    report = diagnose_value_model(dataset)

    assert report["splits"]["train"]["targets"]["examples"] == summary.train_examples
    assert "checkpoint" not in report


def test_diagnostics_reject_checkpoint_from_another_dataset(tmp_path: Path) -> None:
    dataset, _ = _decisive_dataset(tmp_path)
    config = PolicyValueConfig(
        channels=2,
        residual_blocks=1,
        value_hidden=2,
        board_width=6,
        board_height=6,
    )
    checkpoint = tmp_path / "other-dataset.pt"
    save_policy_value_checkpoint(
        checkpoint,
        PolicyValueNetwork(config),
        metadata={"dataset_sha256": "not-this-dataset"},
    )

    with pytest.raises(ValueError, match="different dataset"):
        diagnose_value_model(dataset, checkpoint)


def test_mcts_converts_child_side_value_to_root_perspective() -> None:
    def certain_for_side_to_move(
        state: GameState, moves: tuple[PegPlacement, ...]
    ) -> PolicyValueEstimate:
        del state
        return PolicyValueEstimate({move: 1 for move in moves}, 1)

    agent = MCTSAgent(simulations=1, policy_value=certain_for_side_to_move)
    result = agent.choose_move(
        AgentRequest(
            GameState.initial(BoardDimensions(6, 6)),
            seed=89,
        )
    )

    selected = next(item for item in result.metadata["root_moves"] if item["visits"])
    assert selected["value"] == -1


def test_cli_writes_report_once(tmp_path: Path) -> None:
    from twixt_ai.training.value_diagnostics_cli import main

    dataset, _ = _decisive_dataset(tmp_path)
    output = tmp_path / "report.json"
    arguments = ["--dataset", str(dataset), "--output", str(output), "--device", "cpu"]

    assert main(arguments) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["version"] == 1
    with pytest.raises(SystemExit):
        main(arguments)
