"""Tests for learned Mini MCTS strength evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from twixt_ai.evaluation.mini_strength import (
    AblatedPolicyValue,
    BASELINES,
    GUIDANCE_MODES,
    MiniStrengthConfig,
    run_mini_strength_evaluation,
)
from twixt_ai.game import BoardDimensions
from twixt_ai.game import create_game, legal_peg_placements
from twixt_ai.models import (
    PolicyValueConfig,
    PolicyValueNetwork,
    save_policy_value_checkpoint,
)
from twixt_ai.search.neural import NeuralPolicyValue


def _checkpoint(path: Path, board: BoardDimensions) -> Path:
    model = PolicyValueNetwork(PolicyValueConfig(
        channels=2,
        residual_blocks=1,
        value_hidden=2,
        board_width=board.width,
        board_height=board.height,
    ))
    save_policy_value_checkpoint(path, model, metadata={"epoch": 2, "seed": 57})
    return path


def test_strength_evaluation_runs_complete_reproducible_schedule(tmp_path: Path) -> None:
    board = BoardDimensions(4, 4)
    checkpoint = _checkpoint(tmp_path / "model.pt", board)
    config = MiniStrengthConfig(
        board=board,
        games_per_matchup=2,
        seed=58,
        simulations=1,
        rollout_limit=1,
        search_node_budget=20,
    )

    report = run_mini_strength_evaluation(checkpoint, config=config)

    assert report["format"] == "twixt-ai-mini-strength-evaluation"
    assert report["checkpoint"]["metadata"] == {"epoch": 2, "seed": 57}
    assert report["methodology"]["equal_mcts_simulation_budgets"] is True
    assert len(report["matchups"]) == len(GUIDANCE_MODES) * len(BASELINES)
    assert {item["candidate"] for item in report["matchups"]} == {
        f"learned-{mode}" for mode in GUIDANCE_MODES
    }
    assert {item["baseline"] for item in report["matchups"]} == set(BASELINES)
    shared_seeds = {
        tuple(game["seed"] for game in item["games"])
        for item in report["matchups"]
    }
    assert len(shared_seeds) == 1
    for matchup in report["matchups"]:
        assert matchup["strength"]["games"] == 2
        assert matchup["runtime"]["wall_seconds"] >= 0
        assert matchup["runtime"]["moves"] == sum(
            game["result"]["move_count"] for game in matchup["games"]
        )
        assert matchup["runtime"]["seconds_per_move"] >= 0
        assert len(matchup["games"]) == 2
        assert matchup["games"][0]["agents"] != matchup["games"][1]["agents"]


def test_policy_and_value_ablations_disable_only_the_requested_output() -> None:
    board = BoardDimensions(4, 4)
    model = PolicyValueNetwork(PolicyValueConfig(
        channels=2,
        residual_blocks=1,
        value_hidden=2,
        board_width=board.width,
        board_height=board.height,
    ))
    state = create_game(board)
    moves = legal_peg_placements(state)
    neural = NeuralPolicyValue(model)
    combined = AblatedPolicyValue(neural, "policy-value")(state, moves)
    policy_only = AblatedPolicyValue(neural, "policy-only")(state, moves)
    value_only = AblatedPolicyValue(neural, "value-only")(state, moves)

    assert combined.value is not None
    assert policy_only.priors == combined.priors
    assert policy_only.value is None
    assert value_only.priors == {}
    assert value_only.value == combined.value


def test_config_and_checkpoint_board_are_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be even"):
        MiniStrengthConfig(games_per_matchup=3)

    checkpoint = _checkpoint(tmp_path / "model.pt", BoardDimensions(4, 4))
    with pytest.raises(ValueError, match="checkpoint board dimensions"):
        run_mini_strength_evaluation(checkpoint)


def test_cli_writes_once_without_overwriting(tmp_path: Path) -> None:
    from twixt_ai.evaluation.mini_strength_cli import main

    checkpoint = _checkpoint(tmp_path / "model.pt", BoardDimensions(10, 10))
    output = tmp_path / "nested" / "report.json"
    arguments = [
        "--checkpoint", str(checkpoint),
        "--output", str(output),
        "--games-per-matchup", "2",
        "--simulations", "1",
        "--rollout-limit", "1",
        "--search-node-budget", "20",
    ]

    assert main(arguments) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["config"]["seed"] == 580_100
    with pytest.raises(SystemExit):
        main(arguments)
