"""Tests for fixed-protocol value-candidate evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from twixt_ai.evaluation.value_candidates import (
    ValueCandidateConfig,
    run_value_candidate_evaluation,
)
from twixt_ai.game import BoardDimensions
from twixt_ai.models import PolicyValueConfig, PolicyValueNetwork, save_policy_value_checkpoint


def _checkpoint(path: Path, board: BoardDimensions) -> Path:
    model = PolicyValueNetwork(PolicyValueConfig(
        channels=2,
        residual_blocks=1,
        value_hidden=2,
        board_width=board.width,
        board_height=board.height,
    ))
    save_policy_value_checkpoint(path, model, metadata={"epoch": 1})
    return path


def test_evaluates_every_candidate_and_fixed_opponent(tmp_path: Path) -> None:
    board = BoardDimensions(4, 4)
    champion = _checkpoint(tmp_path / "champion.pt", board)
    issue_57 = _checkpoint(tmp_path / "issue-57.pt", board)
    candidates = {
        "first": _checkpoint(tmp_path / "first.pt", board),
        "second": _checkpoint(tmp_path / "second.pt", board),
    }
    report = run_value_candidate_evaluation(
        champion,
        issue_57,
        candidates,
        config=ValueCandidateConfig(
            board=board,
            games_per_matchup=2,
            simulations=1,
            rollout_limit=1,
            heuristic_search_node_budget=20,
            promotion_win_rate=0.55,
            device="cpu",
        ),
    )

    assert report["format"] == "twixt-ai-value-candidate-evaluation"
    assert len(report["matchups"]) == 8
    assert len(report["promotion_decisions"]) == 2
    assert not report["methodology"]["generation_2_budget_and_rules_unchanged"]
    assert {
        matchup["opponent"] for matchup in report["matchups"]
    } == {
        "generation-2-champion",
        "issue-57",
        "matched-non-neural-mcts",
        "heuristic-search",
    }
    schedules = {
        tuple(game["seed"] for game in matchup["games"])
        for matchup in report["matchups"]
    }
    assert len(schedules) == 1
    for matchup in report["matchups"]:
        assert matchup["games"][0]["seed"] == matchup["games"][1]["seed"]
        assert matchup["games"][0]["agents"] != matchup["games"][1]["agents"]
        if matchup["opponent"] == "heuristic-search":
            assert matchup["opponent_search"] == {
                "type": "search", "depth": 1, "node_budget": 20,
            }


def test_config_rejects_protocol_ambiguity() -> None:
    with pytest.raises(ValueError, match="must be even"):
        ValueCandidateConfig(games_per_matchup=3)
    with pytest.raises(ValueError, match="promotion_win_rate"):
        ValueCandidateConfig(promotion_win_rate=1.0)


def test_rejects_reserved_candidate_name(tmp_path: Path) -> None:
    board = BoardDimensions(4, 4)
    checkpoint = _checkpoint(tmp_path / "model.pt", board)
    with pytest.raises(ValueError, match="reserved"):
        run_value_candidate_evaluation(
            checkpoint,
            checkpoint,
            {"issue-57": checkpoint},
            config=ValueCandidateConfig(board=board, games_per_matchup=2),
        )


def test_cli_writes_once_without_overwriting(tmp_path: Path) -> None:
    from twixt_ai.evaluation import value_candidates_cli

    board = BoardDimensions(10, 10)
    champion = _checkpoint(tmp_path / "champion.pt", board)
    issue_57 = _checkpoint(tmp_path / "issue-57.pt", board)
    candidate = _checkpoint(tmp_path / "candidate.pt", board)
    output = tmp_path / "nested" / "report.json"
    original = value_candidates_cli.ValueCandidateConfig

    def tiny_config(**kwargs: object) -> ValueCandidateConfig:
        kwargs["heuristic_search_node_budget"] = 20
        return original(**kwargs)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(value_candidates_cli, "ValueCandidateConfig", tiny_config)
    arguments = [
        "--champion", str(champion),
        "--issue-57", str(issue_57),
        "--candidate", f"candidate={candidate}",
        "--output", str(output),
        "--games-per-matchup", "2",
        "--simulations", "1",
        "--rollout-limit", "1",
        "--device", "cpu",
    ]
    try:
        assert value_candidates_cli.main(arguments) == 0
        assert json.loads(output.read_text(encoding="utf-8"))["config"]["seed"] == 1_251_300
        with pytest.raises(SystemExit):
            value_candidates_cli.main(arguments)
    finally:
        monkeypatch.undo()
