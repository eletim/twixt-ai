"""Authoritative peg-placement and automatic-link rules for v0.0.1."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .state import Coordinate, GameState, Link, Peg, Player


_KNIGHT_OFFSETS = (
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
)


@dataclass(frozen=True, slots=True)
class PegPlacement:
    """The only player action in v0.0.1: placing one peg."""

    player: Player
    coordinate: Coordinate

    def __post_init__(self) -> None:
        if not isinstance(self.player, Player):
            raise TypeError("placement player must be a Player")
        if not isinstance(self.coordinate, Coordinate):
            raise TypeError("placement coordinate must be a Coordinate")


class IllegalPlacementReason(str, Enum):
    """Stable reason codes for rejected peg placements."""

    OUT_OF_BOUNDS = "out_of_bounds"
    WRONG_TURN = "wrong_turn"
    OCCUPIED = "occupied"
    FORBIDDEN_BORDER = "forbidden_border"


@dataclass(frozen=True, slots=True)
class PlacementLegality:
    """The result of checking a placement without changing the state."""

    reason: IllegalPlacementReason | None = None

    @property
    def is_legal(self) -> bool:
        return self.reason is None

    def __bool__(self) -> bool:
        return self.is_legal


def _require_state_and_placement(state: GameState, placement: PegPlacement) -> None:
    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState")
    if not isinstance(placement, PegPlacement):
        raise TypeError("placement must be a PegPlacement")


def _is_forbidden_border(state: GameState, placement: PegPlacement) -> bool:
    coordinate = placement.coordinate
    if placement.player is Player.RED:
        return coordinate.x == 0 or coordinate.x == state.board.width - 1
    return coordinate.y == 0 or coordinate.y == state.board.height - 1


def check_peg_placement(state: GameState, placement: PegPlacement) -> PlacementLegality:
    """Return whether *placement* is legal, with a reason when it is not.

    Checks have a deterministic precedence so callers always receive one stable
    reason: bounds, turn ownership, occupancy, then player-specific borders.
    """

    _require_state_and_placement(state, placement)
    coordinate = placement.coordinate
    if not state.board.contains(coordinate):
        return PlacementLegality(IllegalPlacementReason.OUT_OF_BOUNDS)
    if placement.player is not state.side_to_move:
        return PlacementLegality(IllegalPlacementReason.WRONG_TURN)
    if any(peg.coordinate == coordinate for peg in state.pegs):
        return PlacementLegality(IllegalPlacementReason.OCCUPIED)
    if _is_forbidden_border(state, placement):
        return PlacementLegality(IllegalPlacementReason.FORBIDDEN_BORDER)
    return PlacementLegality()


def legal_peg_placements(state: GameState) -> tuple[PegPlacement, ...]:
    """Enumerate legal placements for the side to move in row-major order."""

    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState")

    occupied = {peg.coordinate for peg in state.pegs}
    player = state.side_to_move
    placements: list[PegPlacement] = []
    for y in range(state.board.height):
        for x in range(state.board.width):
            coordinate = Coordinate(x, y)
            placement = PegPlacement(player, coordinate)
            if coordinate not in occupied and not _is_forbidden_border(state, placement):
                placements.append(placement)
    return tuple(placements)


def knight_move_neighbors(state: GameState, peg: Peg) -> tuple[Peg, ...]:
    """Return same-player pegs a knight's move from *peg*.

    The returned pegs use the canonical coordinate order, independent of the
    order in which the state's pegs were supplied.
    """

    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState")
    if not isinstance(peg, Peg):
        raise TypeError("peg must be a Peg")

    neighbor_coordinates = {
        Coordinate(peg.coordinate.x + dx, peg.coordinate.y + dy)
        for dx, dy in _KNIGHT_OFFSETS
        if peg.coordinate.x + dx >= 0 and peg.coordinate.y + dy >= 0
    }
    return tuple(
        candidate
        for candidate in state.pegs
        if candidate.owner is peg.owner and candidate.coordinate in neighbor_coordinates
    )


def _orientation(start: Coordinate, end: Coordinate, point: Coordinate) -> int:
    """Return the signed area of the triangle formed by three points."""

    return (end.x - start.x) * (point.y - start.y) - (end.y - start.y) * (
        point.x - start.x
    )


def links_cross(first: Link, second: Link) -> bool:
    """Return whether two links intersect in their interiors.

    Links incident to the same peg meet at an endpoint and do not cross.  The
    integer orientation test avoids floating-point geometry at board scale.
    """

    if not isinstance(first, Link) or not isinstance(second, Link):
        raise TypeError("links must be Link values")
    if {first.start, first.end} & {second.start, second.end}:
        return False

    first_start_side = _orientation(first.start, first.end, second.start)
    first_end_side = _orientation(first.start, first.end, second.end)
    second_start_side = _orientation(second.start, second.end, first.start)
    second_end_side = _orientation(second.start, second.end, first.end)
    return (
        first_start_side * first_end_side < 0
        and second_start_side * second_end_side < 0
    )


def automatic_links_for_placement(state: GameState, peg: Peg) -> tuple[Link, ...]:
    """Return the legal links created by adding *peg* to *state*.

    ``state`` is the position immediately before placement. Existing links
    take precedence: a candidate crossing any of them is omitted. The result
    contains newly created links only, in canonical order, so a transition can
    append it to ``state.links`` without exposing a separate link action.
    """

    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState")
    if not isinstance(peg, Peg):
        raise TypeError("peg must be a Peg")
    if not state.board.contains(peg.coordinate):
        raise ValueError("placed peg is outside the board")
    if any(existing.coordinate == peg.coordinate for existing in state.pegs):
        raise ValueError("placed peg coordinate is occupied")

    generated = (
        Link(peg.owner, peg.coordinate, neighbor.coordinate)
        for neighbor in knight_move_neighbors(state, peg)
    )
    return tuple(
        candidate
        for candidate in generated
        if not any(links_cross(candidate, existing) for existing in state.links)
    )


__all__ = [
    "IllegalPlacementReason",
    "PegPlacement",
    "PlacementLegality",
    "automatic_links_for_placement",
    "check_peg_placement",
    "knight_move_neighbors",
    "legal_peg_placements",
    "links_cross",
]
