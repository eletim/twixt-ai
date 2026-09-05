"""Reusable black-box contract tests for current and future Twixt agents.

Concrete agent test classes can inherit :class:`AgentContract` and provide an
``agent_factory``. Keeping this suite in tests avoids coupling the game engine
to agent implementations while giving every agent the same acceptance checks.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from twixt_ai.game import (
    BoardDimensions,
    Coordinate,
    GameState,
    Peg,
    PegPlacement,
    Player,
    check_peg_placement,
)


class AgentLike(Protocol):
    """The minimal behavior exercised by this black-box contract."""

    def choose_move(self, state: GameState) -> PegPlacement:
        """Choose one legal move from ``state``."""


class AgentContract:
    """Mixin containing behavior shared by all agent implementations."""

    agent_factory: Callable[[], AgentLike]

    def test_returns_a_legal_move_without_mutating_the_position(self) -> None:
        state = GameState.initial(BoardDimensions(6, 6))
        snapshot = state.to_json()

        move = self.agent_factory().choose_move(state)

        assert isinstance(move, PegPlacement)
        assert check_peg_placement(state, move).is_legal
        assert state.to_json() == snapshot

    def test_honors_the_only_available_legal_move(self) -> None:
        board = BoardDimensions(4, 3)
        only_move = Coordinate(2, 2)
        occupied = tuple(
            Coordinate(x, y)
            for y in range(board.height)
            for x in (1, 2)
            if Coordinate(x, y) != only_move
        )
        state = GameState(
            board=board,
            pegs=tuple(Peg(Player.BLACK, coordinate) for coordinate in occupied),
        )

        assert self.agent_factory().choose_move(state) == PegPlacement(
            Player.RED, only_move
        )
