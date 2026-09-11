"""Authoritative state transitions for the v0.0.1 ruleset."""

from __future__ import annotations

from bisect import bisect_left

from .rules import (
    _automatic_links_for_placement,
    _has_legal_peg_placement,
    IllegalPlacementReason,
    PegPlacement,
    check_peg_placement,
    legal_peg_placements,
)
from .state import BoardDimensions, GameResult, GameState, Link, Peg, Player
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
    state = GameState.initial(board)
    if legal_peg_placements(state):
        return state
    return GameState(board=state.board, result=GameResult.DRAW)


def reset_game(state: GameState | None = None) -> GameState:
    """Return a fresh game, preserving *state*'s board when supplied."""

    if state is not None and not isinstance(state, GameState):
        raise TypeError("state must be a GameState or None")
    return create_game(state.board if state is not None else None)


def _peg_key(peg: Peg) -> tuple[int, int, str]:
    return (peg.coordinate.x, peg.coordinate.y, peg.owner.value)


def _link_key(link: Link) -> tuple[int, int, int, int, str]:
    return (
        link.start.x,
        link.start.y,
        link.end.x,
        link.end.y,
        link.owner.value,
    )


def _apply_legal_move(state: GameState, move: PegPlacement) -> GameState:
    """Apply a placement already obtained from this state's legal move list."""

    peg = Peg(move.player, move.coordinate)
    next_side = move.player.opponent
    peg_index = bisect_left(state.pegs, _peg_key(peg), key=_peg_key)
    pegs = (
        state.pegs[:peg_index]
        + (peg,)
        + state.pegs[peg_index:]
    )
    new_links = _automatic_links_for_placement(state, peg)
    links = tuple(
        sorted(
            (*state.links, *new_links),
            key=_link_key,
        )
    )
    occupied = state._occupied | {move.coordinate}
    position = GameState._from_canonical(
        state.board,
        pegs,
        links,
        next_side,
        occupied=occupied,
    )

    if has_winning_path(position, move.player):
        result = (
            GameResult.RED_WINS
            if move.player is Player.RED
            else GameResult.BLACK_WINS
        )
    elif not _has_legal_peg_placement(position):
        result = GameResult.DRAW
    else:
        result = GameResult.IN_PROGRESS

    if result is GameResult.IN_PROGRESS:
        return position
    return GameState._from_canonical(
        position.board,
        position.pegs,
        position.links,
        position.side_to_move,
        result,
        occupied=occupied,
    )


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
    return _apply_legal_move(state, move)


__all__ = ["IllegalMoveError", "apply_move", "create_game", "reset_game"]
