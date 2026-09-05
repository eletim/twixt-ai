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
from .record import GAME_RECORD_FORMAT, GAME_RECORD_VERSION, GameRecord
from .state import (
    STATE_FORMAT,
    STATE_FORMAT_VERSION,
    BoardDimensions,
    Coordinate,
    GameResult,
    GameState,
    Link,
    Peg,
    Player,
    Position,
)
from .transitions import IllegalMoveError, apply_move, create_game, reset_game
from .win import has_winning_path, winning_path

__all__ = [
    "BoardDimensions",
    "Coordinate",
    "GameResult",
    "GameRecord",
    "GameState",
    "GAME_RECORD_FORMAT",
    "GAME_RECORD_VERSION",
    "IllegalPlacementReason",
    "IllegalMoveError",
    "Link",
    "Peg",
    "PegPlacement",
    "PlacementLegality",
    "Player",
    "Position",
    "STATE_FORMAT",
    "STATE_FORMAT_VERSION",
    "automatic_links_for_placement",
    "apply_move",
    "check_peg_placement",
    "create_game",
    "has_winning_path",
    "knight_move_neighbors",
    "legal_peg_placements",
    "links_cross",
    "reset_game",
    "winning_path",
]
