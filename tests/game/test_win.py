"""Tests for player-specific winning-path detection."""

from __future__ import annotations

import pytest

from twixt_ai.game import (
    BoardDimensions,
    Coordinate,
    GameState,
    Link,
    Peg,
    Player,
    has_winning_path,
    winning_path,
)


def state_with_path(player: Player, coordinates: tuple[Coordinate, ...]) -> GameState:
    pegs = tuple(Peg(player, coordinate) for coordinate in coordinates)
    links = tuple(
        Link(player, start, end)
        for start, end in zip(coordinates, coordinates[1:])
    )
    return GameState(board=BoardDimensions(8, 8), pegs=pegs, links=links)


def test_red_wins_by_connecting_north_to_south() -> None:
    coordinates = tuple(
        Coordinate(x, y)
        for x, y in ((1, 0), (2, 2), (4, 3), (3, 5), (5, 6), (3, 7))
    )
    state = state_with_path(Player.RED, coordinates)

    assert winning_path(state, Player.RED) == coordinates
    assert has_winning_path(state, Player.RED)
    assert not has_winning_path(state, Player.BLACK)


def test_black_wins_by_connecting_west_to_east() -> None:
    coordinates = tuple(
        Coordinate(x, y)
        for x, y in ((0, 1), (2, 2), (3, 4), (5, 3), (7, 4))
    )
    state = state_with_path(Player.BLACK, coordinates)

    assert winning_path(state, Player.BLACK) == coordinates
    assert has_winning_path(state, Player.BLACK)


def test_pegs_on_both_target_borders_do_not_win_when_disconnected() -> None:
    coordinates = tuple(
        Coordinate(x, y)
        for x, y in ((1, 0), (2, 2), (4, 3), (5, 7), (3, 6))
    )
    pegs = tuple(Peg(Player.RED, coordinate) for coordinate in coordinates)
    state = GameState(
        board=BoardDimensions(8, 8),
        pegs=pegs,
        links=(
            Link(Player.RED, coordinates[0], coordinates[1]),
            Link(Player.RED, coordinates[1], coordinates[2]),
            Link(Player.RED, coordinates[3], coordinates[4]),
        ),
    )

    assert winning_path(state, Player.RED) is None
    assert not has_winning_path(state, Player.RED)


@pytest.mark.parametrize(
    "player, coordinates",
    [
        (
            Player.RED,
            tuple(
                Coordinate(x, y)
                for x, y in ((0, 1), (2, 2), (4, 3), (6, 2), (7, 4))
            ),
        ),
        (
            Player.BLACK,
            tuple(
                Coordinate(x, y)
                for x, y in ((1, 0), (2, 2), (4, 3), (3, 5), (5, 6), (3, 7))
            ),
        ),
    ],
)
def test_connecting_non_target_borders_is_not_a_win(
    player: Player, coordinates: tuple[Coordinate, ...]
) -> None:
    state = state_with_path(player, coordinates)

    assert not has_winning_path(state, player)


def test_cycles_and_unconnected_corner_pegs_do_not_create_false_positive() -> None:
    coordinates = tuple(
        Coordinate(x, y)
        for x, y in ((1, 0), (2, 2), (4, 3), (3, 1), (0, 0), (7, 7))
    )
    pegs = tuple(Peg(Player.RED, coordinate) for coordinate in coordinates)
    state = GameState(
        board=BoardDimensions(8, 8),
        pegs=pegs,
        links=(
            Link(Player.RED, coordinates[0], coordinates[1]),
            Link(Player.RED, coordinates[1], coordinates[2]),
            Link(Player.RED, coordinates[2], coordinates[3]),
            Link(Player.RED, coordinates[3], coordinates[0]),
        ),
    )

    assert not has_winning_path(state, Player.RED)


@pytest.mark.parametrize(
    "player, coordinates",
    [
        (
            Player.RED,
            tuple(
                Coordinate(x, y)
                for x, y in ((0, 0), (1, 2), (3, 3), (2, 5), (4, 6), (2, 7))
            ),
        ),
        (
            Player.RED,
            tuple(
                Coordinate(x, y)
                for x, y in ((2, 0), (3, 2), (5, 3), (6, 5), (7, 7))
            ),
        ),
        (
            Player.BLACK,
            tuple(
                Coordinate(x, y)
                for x, y in ((0, 0), (2, 1), (3, 3), (5, 2), (7, 3))
            ),
        ),
        (
            Player.BLACK,
            tuple(
                Coordinate(x, y)
                for x, y in ((0, 3), (2, 2), (4, 3), (5, 1), (7, 0))
            ),
        ),
    ],
)
def test_connected_corner_does_not_count_as_a_goal_border_point(
    player: Player, coordinates: tuple[Coordinate, ...]
) -> None:
    state = state_with_path(player, coordinates)

    assert winning_path(state, player) is None
    assert not has_winning_path(state, player)


@pytest.mark.parametrize("value", [None, "red", 1])
def test_win_detection_rejects_invalid_players(value: object) -> None:
    with pytest.raises(TypeError, match="player"):
        winning_path(GameState.initial(), value)  # type: ignore[arg-type]
