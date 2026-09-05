"""Tests for the authoritative v0.0.1 game transition API."""

from __future__ import annotations

import pytest

from twixt_ai.game import (
    BoardDimensions,
    Coordinate,
    GameResult,
    GameState,
    IllegalMoveError,
    IllegalPlacementReason,
    Link,
    Peg,
    PegPlacement,
    Player,
    apply_move,
    create_game,
    reset_game,
)


def test_apply_move_places_peg_links_it_and_advances_turn() -> None:
    neighbor = Peg(Player.RED, Coordinate(2, 1))
    state = GameState(pegs=(neighbor,))
    move = PegPlacement(Player.RED, Coordinate(3, 3))

    updated = apply_move(state, move)

    assert state.pegs == (neighbor,)
    assert updated.pegs == (neighbor, Peg(Player.RED, move.coordinate))
    assert updated.links == (Link(Player.RED, neighbor.coordinate, move.coordinate),)
    assert updated.side_to_move is Player.BLACK
    assert updated.result is GameResult.IN_PROGRESS


def test_apply_move_updates_winning_result_before_advancing_play() -> None:
    coordinates = tuple(
        Coordinate(x, y)
        for x, y in ((1, 0), (2, 2), (4, 3), (3, 5), (5, 6))
    )
    state = GameState(
        board=BoardDimensions(8, 8),
        pegs=tuple(Peg(Player.RED, coordinate) for coordinate in coordinates),
        links=tuple(
            Link(Player.RED, start, end)
            for start, end in zip(coordinates, coordinates[1:])
        ),
    )

    updated = apply_move(
        state, PegPlacement(Player.RED, Coordinate(3, 7))
    )

    assert updated.result is GameResult.RED_WINS
    assert updated.winner is Player.RED
    assert updated.side_to_move is Player.BLACK


@pytest.mark.parametrize(
    "state, move, reason",
    [
        (
            GameState.initial(BoardDimensions(4, 4)),
            PegPlacement(Player.RED, Coordinate(0, 1)),
            IllegalPlacementReason.FORBIDDEN_BORDER,
        ),
        (
            GameState(result=GameResult.RED_WINS),
            PegPlacement(Player.RED, Coordinate(2, 2)),
            IllegalPlacementReason.GAME_OVER,
        ),
    ],
)
def test_apply_move_rejects_illegal_and_post_game_moves(
    state: GameState,
    move: PegPlacement,
    reason: IllegalPlacementReason,
) -> None:
    with pytest.raises(IllegalMoveError) as error:
        apply_move(state, move)

    assert error.value.reason is reason


def test_no_legal_reply_results_in_draw() -> None:
    state = create_game(BoardDimensions(3, 2))

    updated = apply_move(
        state, PegPlacement(Player.RED, Coordinate(1, 0))
    )

    assert updated.result is GameResult.DRAW


def test_reapplying_a_move_to_equal_states_is_deterministic() -> None:
    state = create_game(BoardDimensions(6, 6))
    restored = GameState.from_json(state.to_json())
    move = PegPlacement(Player.RED, Coordinate(2, 2))

    assert apply_move(state, move) == apply_move(restored, move)


def test_create_and_reset_return_fresh_games() -> None:
    board = BoardDimensions(8, 7)
    state = apply_move(
        create_game(board), PegPlacement(Player.RED, Coordinate(2, 2))
    )

    assert reset_game(state) == GameState.initial(board)
    assert reset_game() == create_game() == GameState.initial()
    assert reset_game(state) is not state
