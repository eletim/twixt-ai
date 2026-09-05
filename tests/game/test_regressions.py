"""Black-box regression fixtures for the canonical engine interfaces."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from pathlib import Path

import pytest

from twixt_ai.game import (
    BoardDimensions,
    Coordinate,
    GameRecord,
    GameState,
    IllegalPlacementReason,
    Link,
    Peg,
    PegPlacement,
    Player,
    check_peg_placement,
    has_winning_path,
    legal_peg_placements,
    links_cross,
    winning_path,
)


FIXTURES = Path(__file__).parents[1] / "fixtures"


@pytest.mark.parametrize("player", tuple(Player))
def test_every_board_point_has_the_expected_placement_legality(player: Player) -> None:
    """Exercise borders, corners, and occupancy as one exhaustive fixture."""

    board = BoardDimensions(5, 6)
    occupied = Coordinate(2, 2)
    state = GameState(
        board=board,
        pegs=(Peg(player.opponent, occupied),),
        side_to_move=player,
    )
    enumerated = {move.coordinate for move in legal_peg_placements(state)}

    for y in range(board.height):
        for x in range(board.width):
            coordinate = Coordinate(x, y)
            legality = check_peg_placement(state, PegPlacement(player, coordinate))
            forbidden = (
                x in {0, board.width - 1}
                if player is Player.RED
                else y in {0, board.height - 1}
            )
            expected_reason = (
                IllegalPlacementReason.OCCUPIED
                if coordinate == occupied
                else IllegalPlacementReason.FORBIDDEN_BORDER
                if forbidden
                else None
            )

            assert legality.reason is expected_reason
            assert (coordinate in enumerated) is (expected_reason is None)


def _all_links(board: BoardDimensions) -> tuple[Link, ...]:
    links: set[Link] = set()
    for y in range(board.height):
        for x in range(board.width):
            start = Coordinate(x, y)
            for dx, dy in ((1, 2), (2, 1), (-1, 2), (-2, 1)):
                end_x, end_y = x + dx, y + dy
                if 0 <= end_x < board.width and 0 <= end_y < board.height:
                    links.add(Link(Player.RED, start, Coordinate(end_x, end_y)))
    return tuple(sorted(links, key=lambda link: (link.start, link.end)))


def _reference_links_cross(first: Link, second: Link) -> bool:
    """Compute proper segment intersection using exact parametric fractions."""

    if {first.start, first.end} & {second.start, second.end}:
        return False
    first_dx = first.end.x - first.start.x
    first_dy = first.end.y - first.start.y
    second_dx = second.end.x - second.start.x
    second_dy = second.end.y - second.start.y
    denominator = first_dx * second_dy - first_dy * second_dx
    if denominator == 0:
        return False
    offset_x = second.start.x - first.start.x
    offset_y = second.start.y - first.start.y
    first_parameter = Fraction(
        offset_x * second_dy - offset_y * second_dx, denominator
    )
    second_parameter = Fraction(
        offset_x * first_dy - offset_y * first_dx, denominator
    )
    return 0 < first_parameter < 1 and 0 < second_parameter < 1


def test_all_small_board_link_pairs_match_exact_crossing_geometry() -> None:
    """Cover crossing, touching, parallel, and disjoint knight-link pairs."""

    links = _all_links(BoardDimensions(6, 6))

    assert len(links) == 80
    for first, second in combinations(links, 2):
        expected = _reference_links_cross(first, second)
        assert links_cross(first, second) is expected
        assert links_cross(second, first) is expected


@pytest.mark.parametrize(
    "player, coordinates, expected_path",
    [
        (
            Player.RED,
            ((1, 0), (2, 2), (4, 3), (3, 5), (5, 6), (3, 7)),
            True,
        ),
        (
            Player.BLACK,
            ((0, 1), (2, 2), (3, 4), (5, 3), (7, 4)),
            True,
        ),
        (
            Player.RED,
            ((0, 0), (1, 2), (3, 3), (2, 5), (4, 6), (2, 7)),
            False,
        ),
        (
            Player.BLACK,
            ((0, 1), (2, 2), (4, 1), (6, 2), (7, 0)),
            False,
        ),
    ],
)
def test_win_path_regression_fixtures(
    player: Player,
    coordinates: tuple[tuple[int, int], ...],
    expected_path: bool,
) -> None:
    points = tuple(Coordinate(x, y) for x, y in coordinates)
    state = GameState(
        board=BoardDimensions(8, 8),
        pegs=tuple(Peg(player, point) for point in points),
        links=tuple(Link(player, start, end) for start, end in zip(points, points[1:])),
    )

    assert has_winning_path(state, player) is expected_path
    assert winning_path(state, player) == (points if expected_path else None)


def test_version_one_game_record_golden_fixture_replays_byte_for_byte() -> None:
    persisted = (FIXTURES / "game-record-v1.json").read_text(encoding="utf-8").strip()

    record = GameRecord.from_json(persisted)

    assert record.replay() == record.final_state
    assert record.to_json() == persisted
    assert record.final_state.links == (
        Link(Player.BLACK, Coordinate(0, 2), Coordinate(2, 3)),
        Link(Player.RED, Coordinate(2, 0), Coordinate(3, 2)),
        Link(Player.BLACK, Coordinate(2, 3), Coordinate(4, 2)),
    )
