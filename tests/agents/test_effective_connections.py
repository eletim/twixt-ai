"""Issue #158 fixed patterns affect heuristic progress only."""

from dataclasses import replace

import pytest

import twixt_ai.agents.heuristic as heuristic
from twixt_ai.agents import HeuristicWeights, evaluate_position, position_features
from twixt_ai.game import (
    BoardDimensions,
    Coordinate,
    GameResult,
    GameState,
    Link,
    Peg,
    PegPlacement,
    Player,
    apply_move,
    automatic_links_for_placement,
    has_winning_path,
    legal_peg_placements,
)


# Independently transcribed from the authoritative issue matrices.
PATTERNS = (
    ("001000", "021100", "111110", "011111", "011120", "000100"),
    ("000100", "001120", "011111", "111110", "021110", "001000"),
    ("00000", "01210", "01110", "01110", "01110", "01210", "00000"),
    ("0000000", "0111110", "0211120", "0111110", "0000000"),
)
PROGRESS_ONLY = HeuristicWeights(progress=1, connectivity=0, threats=0, blocking=0)


def cells(pattern: tuple[str, ...], value: str) -> tuple[Coordinate, ...]:
    return tuple(
        Coordinate(x + 2, y + 2)
        for y, row in enumerate(pattern)
        for x, cell in enumerate(row)
        if cell == value
    )


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("player", list(Player))
def test_fixed_patterns_and_every_required_empty_cell(
    pattern: tuple[str, ...], player: Player
) -> None:
    endpoints = cells(pattern, "2")
    state = GameState(
        board=BoardDimensions(12, 13),
        pegs=tuple(Peg(player, point) for point in endpoints),
    )
    extent = state.board.height - 1 if player is Player.RED else state.board.width - 1
    axis = (lambda p: p.y) if player is Player.RED else (lambda p: p.x)
    expected = abs(axis(endpoints[1]) - axis(endpoints[0])) / extent
    assert position_features(state, player).progress == pytest.approx(expected)
    assert position_features(state, player).connectivity == 0
    assert evaluate_position(
        state, player, weights=PROGRESS_ONLY
    ) == pytest.approx(expected)
    assert evaluate_position(
        state, player.opponent, weights=PROGRESS_ONLY
    ) == pytest.approx(-expected)
    assert evaluate_position(state, player) == -evaluate_position(state, player.opponent)

    for point in cells(pattern, "1"):
        for blocker in Player:
            blocked = replace(state, pegs=state.pegs + (Peg(blocker, point),))
            # A same-owner blocker may form a new effective pair: measure the
            # original endpoints directly rather than assuming zero progress.
            snapshot = blocked.to_json()
            components = heuristic._components(blocked, player, effective=True)
            assert not any(set(endpoints) <= component for component in components)
            evaluate_position(blocked, player)
            assert blocked.to_json() == snapshot
            assert blocked.links == ()
            assert not has_winning_path(blocked, player)

    for point in cells(pattern, "0"):
        for owner in Player:
            occupied_zero = replace(state, pegs=state.pegs + (Peg(owner, point),))
            assert position_features(occupied_zero, player).progress >= expected

    mixed = replace(
        state, pegs=(Peg(player, endpoints[0]), Peg(player.opponent, endpoints[1]))
    )
    assert position_features(mixed, player).progress == 0
    assert position_features(mixed, player.opponent).progress == 0


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("player", list(Player))
def test_evaluation_does_not_change_rules_or_generate_links(
    pattern: tuple[str, ...], player: Player
) -> None:
    first, second = cells(pattern, "2")
    before = GameState(
        board=BoardDimensions(12, 13), pegs=(Peg(player, first),), side_to_move=player,
    )
    assert automatic_links_for_placement(before, Peg(player, second)) == ()
    state = apply_move(before, PegPlacement(player, second))
    snapshot = state.to_json()
    legal = legal_peg_placements(state)
    next_state = apply_move(state, legal[0])
    evaluate_position(state, player)
    assert state.to_json() == snapshot
    assert legal_peg_placements(state) == legal
    assert apply_move(state, legal[0]) == next_state
    assert state.links == ()
    assert state.result is GameResult.IN_PROGRESS
    assert not has_winning_path(state, player)


def test_real_links_and_effective_connections_join_transitively_without_a_win() -> None:
    points = tuple(Coordinate(2, y) for y in (0, 4, 8)) + (Coordinate(3, 10),)
    state = GameState(
        board=BoardDimensions(8, 11),
        pegs=tuple(Peg(Player.RED, point) for point in points),
        links=(Link(Player.RED, points[2], points[3]),),
    )
    assert position_features(state, Player.RED).progress == 2
    assert position_features(state, Player.RED).connectivity == 2
    assert not has_winning_path(state, Player.RED)
    assert state.result is GameResult.IN_PROGRESS
    assert len(state.links) == 1


def test_required_cells_outside_board_do_not_match_but_zero_cells_can() -> None:
    # Vertical B at the top edge: its all-zero first row lies off-board.
    state = GameState(
        board=BoardDimensions(8, 8),
        pegs=(Peg(Player.RED, Coordinate(2, 0)), Peg(Player.RED, Coordinate(2, 4))),
    )
    assert position_features(state, Player.RED).progress == pytest.approx(4 / 7 + 0.5)
    # B's required cells to the left now lie off-board.
    outside = replace(
        state,
        pegs=(Peg(Player.RED, Coordinate(0, 0)), Peg(Player.RED, Coordinate(0, 4))),
    )
    assert position_features(outside, Player.RED).progress == 0.5


def test_progress_is_the_difference_of_both_players_effective_spans() -> None:
    state = GameState(
        board=BoardDimensions(12, 13),
        pegs=(
            Peg(Player.RED, Coordinate(2, 2)),
            Peg(Player.RED, Coordinate(2, 6)),
            Peg(Player.BLACK, Coordinate(5, 9)),
            Peg(Player.BLACK, Coordinate(9, 9)),
        ),
    )
    expected = 4 / 12 - 4 / 11
    assert evaluate_position(
        state, Player.RED, weights=PROGRESS_ONLY
    ) == pytest.approx(expected)
    assert evaluate_position(
        state, Player.BLACK, weights=PROGRESS_ONLY
    ) == pytest.approx(-expected)
