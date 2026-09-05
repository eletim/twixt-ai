"""Authoritative state transitions for the v0.0.1 ruleset."""

from __future__ import annotations

from .rules import (
    IllegalPlacementReason,
    PegPlacement,
    automatic_links_for_placement,
    check_peg_placement,
    legal_peg_placements,
)
from .state import BoardDimensions, GameResult, GameState, Peg, Player
from .win import has_winning_path


class IllegalMoveError(ValueError):
    """Raised when a move cannot be applied to a position."""

    def __init__(self, reason: IllegalPlacementReason) -> None:
        self.reason = reason
        super().__init__(f"illegal peg placement: {reason.value}")


def create_game(board: BoardDimensions | None = None) -> GameState:
    """Create a fresh game, using standard dimensions when none are supplied."""

    if board is not None and not isinstance(board, BoardDimensions):
        raise TypeError("board must be BoardDimensions or None")
    return GameState.initial(board)


def reset_game(state: GameState | None = None) -> GameState:
    """Return a fresh game, preserving *state*'s board when supplied."""

    if state is not None and not isinstance(state, GameState):
        raise TypeError("state must be a GameState or None")
    return create_game(state.board if state is not None else None)


def apply_move(state: GameState, move: PegPlacement) -> GameState:
    """Validate and apply the sole v0.0.1 player action.

    This is the public transition path for game mutation. It places the peg,
    derives every permitted link, updates the terminal result, and advances the
    side to move. The input state is immutable and is never modified.
    """

    legality = check_peg_placement(state, move)
    if not legality:
        # ``reason`` is necessarily populated for an illegal placement.
        assert legality.reason is not None
        raise IllegalMoveError(legality.reason)

    peg = Peg(move.player, move.coordinate)
    next_side = move.player.opponent
    position = GameState(
        board=state.board,
        pegs=(*state.pegs, peg),
        links=(*state.links, *automatic_links_for_placement(state, peg)),
        side_to_move=next_side,
    )

    if has_winning_path(position, move.player):
        result = (
            GameResult.RED_WINS
            if move.player is Player.RED
            else GameResult.BLACK_WINS
        )
    elif not legal_peg_placements(position):
        result = GameResult.DRAW
    else:
        result = GameResult.IN_PROGRESS

    if result is GameResult.IN_PROGRESS:
        return position
    return GameState(
        board=position.board,
        pegs=position.pegs,
        links=position.links,
        side_to_move=position.side_to_move,
        result=result,
    )


__all__ = ["IllegalMoveError", "apply_move", "create_game", "reset_game"]
