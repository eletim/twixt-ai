"""Canonical game state, rules, transitions, and serialization."""

from .rules import (
    IllegalPlacementReason,
    PegPlacement,
    PlacementLegality,
    automatic_links_for_placement,
    check_peg_placement,
    knight_move_neighbors,
    legal_peg_placements,
    links_cross,
)
from .state import BoardDimensions, Coordinate, GameResult, GameState, Link, Peg, Player, Position
from .win import has_winning_path, winning_path

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
    "automatic_links_for_placement",
    "check_peg_placement",
    "has_winning_path",
    "knight_move_neighbors",
    "legal_peg_placements",
    "links_cross",
    "winning_path",
]
