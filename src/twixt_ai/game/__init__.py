"""Canonical game state, rules, transitions, and serialization."""

from .rules import (
    IllegalPlacementReason,
    PegPlacement,
    PlacementLegality,
    check_peg_placement,
    legal_peg_placements,
)
from .state import BoardDimensions, Coordinate, GameResult, GameState, Link, Peg, Player, Position

__all__ = [
    "BoardDimensions",
    "Coordinate",
    "GameResult",
    "GameState",
    "IllegalPlacementReason",
    "Link",
    "Peg",
    "PegPlacement",
    "PlacementLegality",
    "Player",
    "Position",
    "check_peg_placement",
    "legal_peg_placements",
]
