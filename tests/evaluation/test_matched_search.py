"""Tests for the matched Mini search-setting diagnosis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from twixt_ai.evaluation.matched_search import (
    MatchedSearchConfig,
    SearchSetting,
    run_matched_search_analysis,
)
from twixt_ai.game import BoardDimensions
from twixt_ai.models import (
    PolicyValueConfig,
    PolicyValueNetwork,
    save_policy_value_checkpoint,
)


def _checkpoint(path: Path, board: BoardDimensions) -> Path:
    model = PolicyValueNetwork(PolicyValueConfig(
        channels=2,
        residual_blocks=1,
        value_hidden=2,
        board_width=board.width,
        board_height=board.height,
    ))
    save_policy_value_checkpoint(path, model, metadata={"epoch": 2})
    return path


def test_every_setting_mode_and_baseline_uses_one_matched_schedule(
    tmp_path: Path,
) -> None:
    board = BoardDimensions(4, 4)
    checkpoint = _checkpoint(tmp_path / "model.pt", board)
    settings = (
        SearchSetting("small", 1, 0.7, 1.5, 0.5),
        SearchSetting("wide", 2, 1.4, 3.0, 0.75),
    )
    report = run_matched_search_analysis(
        checkpoint,
        config=MatchedSearchConfig(
            board=board,
            games_per_matchup=2,
            seed=118,
            rollout_limit=1,
            heuristic_search_node_budget=20,
            settings=settings,
            device="cpu",
        ),
    )

    assert report["format"] == "twixt-ai-matched-search-analysis"
    assert len(report["matchups"]) == 2 * 3 * 2
    assert report["methodology"]["heuristic_configuration_fixed_across_every_matchup"]
    assert report["methodology"]["non_neural_mcts_settings_match_candidate"]
    seed_schedules = {
        tuple(game["seed"] for game in matchup["games"])
        for matchup in report["matchups"]
    }
    assert len(seed_schedules) == 1
    for matchup in report["matchups"]:
        games = matchup["games"]
        assert games[0]["seed"] == games[1]["seed"]
        assert games[0]["agents"] != games[1]["agents"]
        if matchup["baseline"] == "matched-non-neural-mcts":
            candidate = dict(matchup["candidate_search"])
            candidate.pop("guidance", None)
            candidate.pop("checkpoint_sha256", None)
            assert matchup["baseline_search"] == candidate
        else:
            assert matchup["baseline_search"] == {
                "type": "search",
                "depth": 1,
                "node_budget": 20,
            }


def test_config_rejects_invalid_or_ambiguous_schedules() -> None:
    with pytest.raises(ValueError, match="must be even"):
        MatchedSearchConfig(games_per_matchup=3)
    duplicate = SearchSetting("duplicate", 1, 0.0, 1.0, 0.5)
    with pytest.raises(ValueError, match="names must be unique"):
        MatchedSearchConfig(settings=(duplicate, duplicate))
    with pytest.raises(ValueError, match="simulations"):
        SearchSetting("invalid", 0, 1.0, 1.0, 0.5)


def test_cli_writes_once_without_overwriting(tmp_path: Path) -> None:
    from twixt_ai.evaluation import matched_search_cli

    checkpoint = _checkpoint(tmp_path / "model.pt", BoardDimensions(10, 10))
    output = tmp_path / "nested" / "report.json"
    tiny_setting = (SearchSetting("tiny", 1, 0.7, 1.5, 0.5),)
    original = matched_search_cli.MatchedSearchConfig

    def tiny_config(**kwargs: object) -> MatchedSearchConfig:
        return original(settings=tiny_setting, **kwargs)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(matched_search_cli, "MatchedSearchConfig", tiny_config)
    arguments = [
        "--checkpoint", str(checkpoint),
        "--output", str(output),
        "--games-per-matchup", "2",
        "--rollout-limit", "1",
        "--device", "cpu",
    ]
    try:
        assert matched_search_cli.main(arguments) == 0
        assert json.loads(output.read_text(encoding="utf-8"))["config"]["seed"] == 1_188_100
        with pytest.raises(SystemExit):
            matched_search_cli.main(arguments)
    finally:
        monkeypatch.undo()
