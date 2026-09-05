"""Authoritative peg-placement rules for the v0.0.1 ruleset."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .state import Coordinate, GameState, Player


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


__all__ = [
    "IllegalPlacementReason",
    "PegPlacement",
    "PlacementLegality",
    "check_peg_placement",
    "legal_peg_placements",
]
