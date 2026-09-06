"""Tests for the matched Mini encoding strength comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from twixt_ai.evaluation.encoding_strength import (
    EncodingStrengthConfig,
    run_encoding_strength_comparison,
)
from twixt_ai.game import BoardDimensions
from twixt_ai.models import (
    PolicyValueConfig,
    PolicyValueNetwork,
    save_policy_value_checkpoint,
)


def _checkpoint(path: Path, board: BoardDimensions, encoding_version: int) -> Path:
    input_channels = 10 if encoding_version == 2 else 22
    model = PolicyValueNetwork(
        PolicyValueConfig(
            channels=2,
            residual_blocks=1,
            value_hidden=2,
            board_width=board.width,
            board_height=board.height,
            input_channels=input_channels,
            encoding_version=encoding_version,
        )
    )
    save_policy_value_checkpoint(
        path,
        model,
        metadata={
            "epoch": encoding_version,
            "dataset_sha256": "matched-dataset",
            "training_config": {"seed": 75},
        },
    )
    return path


def test_comparison_runs_all_matched_guidance_schedules(tmp_path: Path) -> None:
    board = BoardDimensions(4, 4)
    ten = _checkpoint(tmp_path / "ten.pt", board, 2)
    twenty_two = _checkpoint(tmp_path / "twenty-two.pt", board, 1)
    config = EncodingStrengthConfig(
        board=board,
        games_per_matchup=2,
        seed=76,
        simulations=1,
        rollout_limit=1,
    )

    report = run_encoding_strength_comparison(ten, twenty_two, config=config)

    assert report["format"] == "twixt-ai-encoding-strength-comparison"
    assert set(report["checkpoints"]) == {"10-plane", "22-plane"}
    assert report["methodology"]["equal_mcts_simulation_budgets"] is True
    assert len(report["matchups"]) == 9
    assert {item["guidance"] for item in report["matchups"]} == {
        "policy-value",
        "policy-only",
        "value-only",
    }
    assert {tuple(item["comparison"]) for item in report["matchups"]} == {
        ("10-plane", "22-plane"),
        ("10-plane", "non-neural-mcts"),
        ("22-plane", "non-neural-mcts"),
    }
    seed_schedules = {
        tuple(game["seed"] for game in item["games"])
        for item in report["matchups"]
    }
    assert len(seed_schedules) == 1
    for matchup in report["matchups"]:
        assert matchup["strength"]["games"] == 2
        assert matchup["runtime"]["wall_seconds"] >= 0
        assert matchup["runtime"]["seconds_per_move"] >= 0
        assert matchup["games"][0]["agents"] != matchup["games"][1]["agents"]


def test_comparison_rejects_unmatched_checkpoints(tmp_path: Path) -> None:
    board = BoardDimensions(4, 4)
    ten = _checkpoint(tmp_path / "ten.pt", board, 2)
    another_ten = _checkpoint(tmp_path / "another-ten.pt", board, 2)

    with pytest.raises(ValueError, match="22-plane checkpoint"):
        run_encoding_strength_comparison(
            ten,
            another_ten,
            config=EncodingStrengthConfig(
                board=board,
                games_per_matchup=2,
                simulations=1,
                rollout_limit=1,
            ),
        )

    with pytest.raises(ValueError, match="must be even"):
        EncodingStrengthConfig(games_per_matchup=3)


def test_cli_writes_once_without_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import twixt_ai.evaluation.encoding_strength_cli as cli

    output = tmp_path / "nested" / "report.json"
    expected = {"format": "test-report"}
    monkeypatch.setattr(
        cli, "run_encoding_strength_comparison", lambda *args, **kwargs: expected
    )
    arguments = [
        "--ten-plane-checkpoint",
        str(tmp_path / "ten.pt"),
        "--twenty-two-plane-checkpoint",
        str(tmp_path / "twenty-two.pt"),
        "--output",
        str(output),
    ]

    assert cli.main(arguments) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected
    with pytest.raises(SystemExit):
        cli.main(arguments)
