"""Tests for versioned position and game-record persistence."""

from __future__ import annotations

import json

import pytest

from twixt_ai.game import (
    BoardDimensions,
    Coordinate,
    GameRecord,
    GameState,
    PegPlacement,
    Player,
)


def sample_record() -> GameRecord:
    initial = GameState.initial(BoardDimensions(6, 6))
    moves = (
        PegPlacement(Player.RED, Coordinate(2, 1)),
        PegPlacement(Player.BLACK, Coordinate(1, 2)),
        PegPlacement(Player.RED, Coordinate(3, 3)),
        PegPlacement(Player.BLACK, Coordinate(3, 2)),
    )
    return GameRecord.from_moves(initial, moves)


def test_record_round_trip_is_deterministic_and_replays_final_state() -> None:
    record = sample_record()
    restored = GameRecord.from_json(record.to_json())

    assert restored == record
    assert restored.replay() == record.final_state
    assert restored.to_json() == record.to_json()
    assert json.loads(record.to_json())["version"] == 1


def test_record_preserves_ordered_move_history() -> None:
    record = sample_record()

    assert [move.player for move in record.moves] == [
        Player.RED,
        Player.BLACK,
        Player.RED,
        Player.BLACK,
    ]
    assert record.to_dict()["moves"][0] == {
        "player": "red",
        "coordinate": {"x": 2, "y": 1},
    }


def test_record_rejects_unsupported_versions() -> None:
    serialized = sample_record().to_dict()
    serialized["version"] = 2

    with pytest.raises(ValueError, match="unsupported game record version: 2"):
        GameRecord.from_dict(serialized)


def test_record_rejects_final_state_that_does_not_match_replay() -> None:
    serialized = sample_record().to_dict()
    serialized["final_state"] = serialized["initial_state"]

    with pytest.raises(ValueError, match="does not match"):
        GameRecord.from_dict(serialized)


def test_record_loading_rejects_an_illegal_history() -> None:
    serialized = sample_record().to_dict()
    serialized["moves"][1]["player"] = "red"

    with pytest.raises(ValueError, match="wrong_turn"):
        GameRecord.from_dict(serialized)
