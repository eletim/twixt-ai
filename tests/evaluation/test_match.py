"""Tests for complete headless agent matches and their artifacts."""

from __future__ import annotations

import json

import pytest

from twixt_ai.agents import AgentRequest, AgentResult, RandomAgent
from twixt_ai.evaluation import MatchConfig, run_match
from twixt_ai.evaluation.cli import main
from twixt_ai.game import BoardDimensions, Player


class FirstLegalAgent:
    def __init__(self, label: str) -> None:
        self.label = label
        self.requests: list[AgentRequest] = []

    def choose_move(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        return AgentResult(request.legal_moves[0], {"agent": self.label})


class InvalidMetadataAgent:
    def choose_move(self, request: AgentRequest) -> AgentResult:
        return AgentResult(request.legal_moves[0], {"diagnostic": object()})


def test_match_runs_to_completion_with_explicit_side_assignment() -> None:
    red = FirstLegalAgent("r")
    black = FirstLegalAgent("b")
    config = MatchConfig(
        BoardDimensions(4, 4),
        seed=19,
        red_agent="first-r",
        black_agent="first-b",
    )

    result = run_match(red, black, config=config)

    assert result.final_state.is_terminal
    assert result.record.replay() == result.final_state
    assert len(result.moves) == len(result.decisions) == len(red.requests) + len(black.requests)
    assert all(request.state.side_to_move is Player.RED for request in red.requests)
    assert all(request.state.side_to_move is Player.BLACK for request in black.requests)
    assert [item.metadata["agent"] for item in result.decisions] == [
        "r" if item.move.player is Player.RED else "b" for item in result.decisions
    ]


def test_seeded_random_matches_are_reproducible() -> None:
    config = MatchConfig(
        BoardDimensions(6, 6),
        seed=8675309,
        red_agent="random",
        black_agent="random",
    )

    first = run_match(RandomAgent(), RandomAgent(), config=config)
    second = run_match(RandomAgent(), RandomAgent(), config=config)

    assert first == second
    assert first.to_json() == second.to_json()
    assert all(decision.seed is not None for decision in first.decisions)


def test_convenience_api_captures_board_seed_and_agent_types() -> None:
    result = run_match(
        RandomAgent(), RandomAgent(), board=BoardDimensions(4, 4), seed=31
    )

    assert result.config == MatchConfig(
        BoardDimensions(4, 4),
        seed=31,
        red_agent="RandomAgent",
        black_agent="RandomAgent",
    )


def test_artifact_captures_config_history_and_final_result() -> None:
    config = MatchConfig(BoardDimensions(4, 4), seed=7, red_agent="red-id", black_agent="black-id")
    result = run_match(FirstLegalAgent("red"), FirstLegalAgent("black"), config=config)

    artifact = json.loads(result.to_json())

    assert artifact["format"] == "twixt-ai-match"
    assert artifact["version"] == 1
    assert artifact["config"] == {
        "agents": {"black": "black-id", "red": "red-id"},
        "board": {"height": 4, "width": 4},
        "seed": 7,
    }
    assert artifact["result"]["status"] == result.game_result.value
    assert artifact["result"]["move_count"] == len(result.moves)
    assert [entry["coordinate"] for entry in artifact["decisions"]] == [
        move.coordinate.to_dict() for move in result.record.moves
    ]


def test_match_rejects_non_json_agent_metadata_when_capturing_decision() -> None:
    with pytest.raises(
        TypeError,
        match=r"metadata\.diagnostic contains non-JSON-compatible value of type object",
    ):
        run_match(
            InvalidMetadataAgent(),
            FirstLegalAgent("black"),
            board=BoardDimensions(4, 4),
        )


def test_cli_emits_batch_friendly_json(capsys: object) -> None:
    assert main(["--width", "4", "--height", "4", "--seed", "3"]) == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    artifact = json.loads(output)

    assert artifact["config"]["seed"] == 3
    assert artifact["config"]["agents"] == {"red": "random", "black": "random"}
    assert artifact["result"]["status"] != "in_progress"
