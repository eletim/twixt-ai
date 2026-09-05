"""Tests for authoritative v0.0.1 peg-placement rules."""

from __future__ import annotations

import pytest

from twixt_ai.game import (
    BoardDimensions,
    Coordinate,
    GameState,
    IllegalPlacementReason,
    Link,
    Peg,
    PegPlacement,
    Player,
    automatic_links_for_placement,
    check_peg_placement,
    knight_move_neighbors,
    legal_peg_placements,
    links_cross,
)


def placement(player: Player, x: int, y: int) -> PegPlacement:
    return PegPlacement(player, Coordinate(x, y))


def test_interior_placement_is_legal() -> None:
    result = check_peg_placement(GameState.initial(), placement(Player.RED, 4, 5))

    assert result.is_legal
    assert bool(result)
    assert result.reason is None


@pytest.mark.parametrize(
    "state, move, reason",
    [
        (
            GameState.initial(BoardDimensions(4, 5)),
            placement(Player.RED, 4, 2),
            IllegalPlacementReason.OUT_OF_BOUNDS,
        ),
        (
            GameState.initial(BoardDimensions(4, 5)),
            placement(Player.BLACK, 1, 1),
            IllegalPlacementReason.WRONG_TURN,
        ),
        (
            GameState(
                board=BoardDimensions(4, 5),
                pegs=(Peg(Player.BLACK, Coordinate(1, 2)),),
            ),
            placement(Player.RED, 1, 2),
            IllegalPlacementReason.OCCUPIED,
        ),
        (
            GameState.initial(BoardDimensions(4, 5)),
            placement(Player.RED, 0, 2),
            IllegalPlacementReason.FORBIDDEN_BORDER,
        ),
        (
            GameState(board=BoardDimensions(4, 5), side_to_move=Player.BLACK),
            placement(Player.BLACK, 2, 4),
            IllegalPlacementReason.FORBIDDEN_BORDER,
        ),
    ],
)
def test_illegal_placements_return_clear_reasons(
    state: GameState,
    move: PegPlacement,
    reason: IllegalPlacementReason,
) -> None:
    result = check_peg_placement(state, move)

    assert not result.is_legal
    assert not result
    assert result.reason is reason


def test_player_borders_are_orientation_specific() -> None:
    red_state = GameState.initial(BoardDimensions(4, 5))
    black_state = GameState(board=BoardDimensions(4, 5), side_to_move=Player.BLACK)

    assert check_peg_placement(red_state, placement(Player.RED, 2, 0)).is_legal
    assert check_peg_placement(red_state, placement(Player.RED, 2, 4)).is_legal
    assert check_peg_placement(black_state, placement(Player.BLACK, 0, 2)).is_legal
    assert check_peg_placement(black_state, placement(Player.BLACK, 3, 2)).is_legal


def test_legal_placements_are_complete_and_row_major() -> None:
    state = GameState(
        board=BoardDimensions(4, 3),
        pegs=(Peg(Player.BLACK, Coordinate(2, 0)), Peg(Player.RED, Coordinate(1, 1))),
    )

    moves = legal_peg_placements(state)

    assert moves == (
        placement(Player.RED, 1, 0),
        placement(Player.RED, 2, 1),
        placement(Player.RED, 1, 2),
        placement(Player.RED, 2, 2),
    )
    assert all(check_peg_placement(state, move).is_legal for move in moves)


def test_corners_are_forbidden_to_both_players() -> None:
    board = BoardDimensions(4, 3)

    for player in Player:
        state = GameState(board=board, side_to_move=player)
        moves = legal_peg_placements(state)
        corners = {Coordinate(0, 0), Coordinate(3, 0), Coordinate(0, 2), Coordinate(3, 2)}
        assert all(move.coordinate not in corners for move in moves)


def test_only_peg_placement_action_is_exported() -> None:
    import twixt_ai.game as game

    assert "swap" not in {name.lower() for name in game.__all__}
    assert "pie" not in {name.lower() for name in game.__all__}


def test_knight_move_neighbors_include_only_same_player_pegs() -> None:
    center = Peg(Player.RED, Coordinate(4, 4))
    same_player_neighbors = (
        Peg(Player.RED, Coordinate(2, 3)),
        Peg(Player.RED, Coordinate(3, 2)),
        Peg(Player.RED, Coordinate(3, 6)),
        Peg(Player.RED, Coordinate(5, 2)),
        Peg(Player.RED, Coordinate(5, 6)),
        Peg(Player.RED, Coordinate(6, 3)),
        Peg(Player.RED, Coordinate(6, 5)),
    )
    opponent_neighbor = Peg(Player.BLACK, Coordinate(2, 5))
    non_neighbor = Peg(Player.RED, Coordinate(4, 5))
    state = GameState(pegs=(*reversed(same_player_neighbors), opponent_neighbor, non_neighbor))

    assert knight_move_neighbors(state, center) == same_player_neighbors


@pytest.mark.parametrize(
    "first, second, expected",
    [
        (
            Link(Player.RED, Coordinate(1, 1), Coordinate(2, 3)),
            Link(Player.BLACK, Coordinate(1, 2), Coordinate(3, 1)),
            True,
        ),
        (
            Link(Player.RED, Coordinate(1, 1), Coordinate(2, 3)),
            Link(Player.RED, Coordinate(1, 1), Coordinate(3, 2)),
            False,
        ),
        (
            Link(Player.RED, Coordinate(1, 1), Coordinate(2, 3)),
            Link(Player.BLACK, Coordinate(3, 1), Coordinate(4, 3)),
            False,
        ),
    ],
)
def test_link_crossing_uses_segment_interiors(first: Link, second: Link, expected: bool) -> None:
    assert links_cross(first, second) is expected
    assert links_cross(second, first) is expected


def test_automatic_links_are_same_owner_complete_and_deterministic() -> None:
    placed = Peg(Player.RED, Coordinate(4, 4))
    neighbors = (
        Peg(Player.RED, Coordinate(6, 5)),
        Peg(Player.BLACK, Coordinate(2, 5)),
        Peg(Player.RED, Coordinate(3, 2)),
        Peg(Player.RED, Coordinate(2, 3)),
    )
    state = GameState(pegs=neighbors)

    assert automatic_links_for_placement(state, placed) == (
        Link(Player.RED, Coordinate(2, 3), Coordinate(4, 4)),
        Link(Player.RED, Coordinate(3, 2), Coordinate(4, 4)),
        Link(Player.RED, Coordinate(4, 4), Coordinate(6, 5)),
    )


def test_automatic_links_never_cross_existing_links() -> None:
    blocker_start = Peg(Player.BLACK, Coordinate(1, 2))
    blocker_end = Peg(Player.BLACK, Coordinate(3, 1))
    blocked_neighbor = Peg(Player.RED, Coordinate(1, 1))
    clear_neighbor = Peg(Player.RED, Coordinate(4, 4))
    state = GameState(
        pegs=(blocker_start, blocker_end, blocked_neighbor, clear_neighbor),
        links=(Link(Player.BLACK, blocker_start.coordinate, blocker_end.coordinate),),
    )
    placed = Peg(Player.RED, Coordinate(2, 3))

    assert automatic_links_for_placement(state, placed) == (
        Link(Player.RED, placed.coordinate, clear_neighbor.coordinate),
    )


def test_automatic_link_generation_rejects_invalid_placement_context() -> None:
    state = GameState(
        board=BoardDimensions(5, 5),
        pegs=(Peg(Player.RED, Coordinate(2, 2)),),
    )

    with pytest.raises(ValueError, match="occupied"):
        automatic_links_for_placement(state, Peg(Player.RED, Coordinate(2, 2)))
    with pytest.raises(ValueError, match="outside"):
        automatic_links_for_placement(state, Peg(Player.RED, Coordinate(5, 2)))
