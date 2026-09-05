"""Deterministic, bounded minimax search over the shared position heuristic."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import math

from twixt_ai.agents import (
    TERMINAL_SCORE,
    AgentContractError,
    AgentRequest,
    AgentResult,
    evaluate_position,
)
from twixt_ai.game import GameState, PegPlacement, Player, apply_move, legal_peg_placements


EvaluationFunction = Callable[[GameState, Player], float]
"""Return a finite non-terminal score from the supplied perspective."""


_MAX_HEURISTIC_SCORE = math.nextafter(TERMINAL_SCORE, 0.0)


MoveOrderer = Callable[
    [GameState, tuple[PegPlacement, ...]], Iterable[PegPlacement]
]
"""Order a position's legal moves without adding or removing any move."""


def _coordinate_order(
    state: GameState, moves: tuple[PegPlacement, ...]
) -> Iterable[PegPlacement]:
    del state
    return sorted(moves, key=lambda move: (move.coordinate.y, move.coordinate.x))


class HeuristicSearchAgent:
    """Choose moves with depth- and node-bounded alpha-beta minimax.

    Non-terminal leaf positions are scored with the configured evaluator and
    bounded inside the terminal score range. Terminal wins and losses therefore
    always dominate heuristic values, and custom evaluators do not need
    terminal-state behavior. ``move_orderer`` is invoked at every node and can
    put tactically promising moves first to improve alpha-beta pruning. It must
    return each supplied legal move exactly once.

    Nodes count applied child positions. Consequently the budget is a hard
    bound on state transitions as well as a deterministic search limit.
    """

    def __init__(
        self,
        *,
        depth: int = 1,
        node_budget: int = 10_000,
        evaluator: EvaluationFunction = evaluate_position,
        move_orderer: MoveOrderer | None = None,
    ) -> None:
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("depth must be a positive integer")
        if (
            isinstance(node_budget, bool)
            or not isinstance(node_budget, int)
            or node_budget < 1
        ):
            raise ValueError("node_budget must be a positive integer")
        if not callable(evaluator):
            raise TypeError("evaluator must be callable")
        if move_orderer is not None and not callable(move_orderer):
            raise TypeError("move_orderer must be callable or None")
        self.depth = depth
        self.node_budget = node_budget
        self.evaluator = evaluator
        self.move_orderer = move_orderer or _coordinate_order
        self._nodes = 0

    def _ordered(
        self, state: GameState, moves: tuple[PegPlacement, ...]
    ) -> tuple[PegPlacement, ...]:
        ordered = tuple(self.move_orderer(state, moves))
        if len(ordered) != len(moves) or set(ordered) != set(moves):
            raise ValueError("move_orderer must return each legal move exactly once")
        return ordered

    def _evaluate(self, state: GameState, root: Player) -> float:
        if state.is_terminal:
            if state.winner is root:
                return TERMINAL_SCORE
            if state.winner is root.opponent:
                return -TERMINAL_SCORE
            return 0.0

        score = self.evaluator(state, root)
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError("evaluator must return a number")
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("evaluator must return a finite score")
        return max(-_MAX_HEURISTIC_SCORE, min(_MAX_HEURISTIC_SCORE, score))

    def _score(
        self,
        state: GameState,
        depth: int,
        root: Player,
        alpha: float,
        beta: float,
    ) -> float:
        if state.result.is_terminal or depth == 0 or self._nodes >= self.node_budget:
            return self._evaluate(state, root)
        moves = legal_peg_placements(state)
        if not moves:
            return self._evaluate(state, root)
        maximizing = state.side_to_move is root
        best = -math.inf if maximizing else math.inf
        for move in self._ordered(state, moves):
            if self._nodes >= self.node_budget:
                break
            self._nodes += 1
            score = self._score(
                apply_move(state, move), depth - 1, root, alpha, beta
            )
            if maximizing:
                best = max(best, score)
                alpha = max(alpha, best)
            else:
                best = min(best, score)
                beta = min(beta, best)
            if beta <= alpha:
                break
        return best

    def choose_move(self, request: AgentRequest) -> AgentResult:
        """Return the best legal move found within the configured limits."""

        if not isinstance(request, AgentRequest):
            raise TypeError("request must be an AgentRequest")
        if not request.legal_moves:
            raise AgentContractError("cannot select a move when no legal moves exist")
        self._nodes = 0
        root = request.state.side_to_move
        ordered_moves = self._ordered(request.state, request.legal_moves)
        best_move = ordered_moves[0]
        best_score = -math.inf
        alpha = -math.inf
        for move in ordered_moves:
            if self._nodes >= self.node_budget:
                break
            self._nodes += 1
            score = self._score(
                apply_move(request.state, move),
                self.depth - 1,
                root,
                alpha,
                math.inf,
            )
            if score > best_score:
                best_move = move
                best_score = score
            alpha = max(alpha, best_score)
        return AgentResult(
            best_move,
            {"depth": self.depth, "nodes": self._nodes, "score": best_score},
        )


# Preserve the UI-era public name while exposing the more descriptive name.
SearchAgent = HeuristicSearchAgent


__all__ = [
    "EvaluationFunction",
    "HeuristicSearchAgent",
    "MoveOrderer",
    "SearchAgent",
]
