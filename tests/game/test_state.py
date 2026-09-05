"""Tests for the canonical Twixt state values."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from twixt_ai.game import BoardDimensions, Coordinate, GameResult, GameState, Link, Peg, Player


def populated_state(*, reverse: bool = False) -> GameState:
    red_a = Peg(Player.RED, Coordinate(2, 1))
    red_b = Peg(Player.RED, Coordinate(3, 3))
    black = Peg(Player.BLACK, Coordinate(5, 5))
    pegs = (black, red_b, red_a) if reverse else (red_a, red_b, black)
    return GameState(
        board=BoardDimensions(8, 7),
        pegs=pegs,
        links=(Link(Player.RED, red_b.coordinate, red_a.coordinate),),
        side_to_move=Player.BLACK,
        result=GameResult.IN_PROGRESS,
    )


def test_initial_state_uses_standard_board_and_red_to_move() -> None:
    state = GameState.initial()

    assert state.board == BoardDimensions(24, 24)
    assert state.side_to_move is Player.RED
    assert state.pegs == ()
    assert state.links == ()
    assert not state.is_terminal
    assert state.winner is None


def test_state_is_immutable_and_normalized() -> None:
    state = populated_state(reverse=True)

    assert [peg.coordinate for peg in state.pegs] == [Coordinate(2, 1), Coordinate(3, 3), Coordinate(5, 5)]
    assert state.links[0].start == Coordinate(2, 1)
    with pytest.raises(FrozenInstanceError):
        state.side_to_move = Player.RED  # type: ignore[misc]


def test_copy_and_serialization_are_deterministic() -> None:
    state = populated_state()
    reordered = populated_state(reverse=True)

    assert state.copy() == state
    assert state.copy() is not state
    assert state.to_dict() == reordered.to_dict()
    assert state.to_json() == reordered.to_json()
    assert GameState.from_dict(state.to_dict()) == state
    assert GameState.from_json(state.to_json()) == state
    assert json.loads(state.to_json())["format"] == "twixt-ai-state"
    assert json.loads(state.to_json())["version"] == 1
    assert json.loads(state.to_json())["board"] == {"height": 7, "width": 8}


@pytest.mark.parametrize(
    "factory, match",
    [
        (lambda: BoardDimensions(0, 24), "positive"),
        (lambda: Coordinate(-1, 0), "negative"),
        (lambda: Link(Player.RED, Coordinate(0, 0), Coordinate(1, 1)), "knight"),
        (
            lambda: GameState(board=BoardDimensions(3, 3), pegs=(Peg(Player.RED, Coordinate(3, 1)),)),
            "outside",
        ),
        (
            lambda: GameState(
                pegs=(Peg(Player.RED, Coordinate(1, 1)), Peg(Player.BLACK, Coordinate(1, 1)))
            ),
            "multiple pegs",
        ),
        (
            lambda: GameState(
                pegs=(Peg(Player.RED, Coordinate(1, 1)), Peg(Player.BLACK, Coordinate(2, 3))),
                links=(Link(Player.RED, Coordinate(1, 1), Coordinate(2, 3)),),
            ),
            "owned",
        ),
    ],
)
def test_invalid_states_are_rejected(factory: object, match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        factory()  # type: ignore[operator]


def test_result_exposes_terminal_state_and_winner() -> None:
    state = GameState(result=GameResult.BLACK_WINS)

    assert state.is_terminal
    assert state.winner is Player.BLACK
    assert GameState(result=GameResult.DRAW).winner is None


def test_peg_owner_lookup_checks_board_bounds() -> None:
    state = populated_state()

    assert state.peg_owner_at(Coordinate(2, 1)) is Player.RED
    assert state.peg_owner_at(Coordinate(0, 0)) is None
    with pytest.raises(ValueError, match="outside"):
        state.peg_owner_at(Coordinate(8, 0))


def test_deserialization_rejects_unknown_fields_and_invalid_ownership() -> None:
    serialized = populated_state().to_dict()
    serialized["extra"] = True
    with pytest.raises(ValueError, match="exactly"):
        GameState.from_dict(serialized)

    serialized = populated_state().to_dict()
    serialized["pegs"][0]["owner"] = "green"  # type: ignore[index]
    with pytest.raises(ValueError, match="green"):
        GameState.from_dict(serialized)


def test_deserialization_rejects_unsupported_format_version() -> None:
    serialized = populated_state().to_dict()
    serialized["version"] = 2

    with pytest.raises(ValueError, match="unsupported state format version: 2"):
        GameState.from_dict(serialized)
