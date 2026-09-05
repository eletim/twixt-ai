"""Small deterministic lookahead agent suitable for interactive play."""

from __future__ import annotations

from twixt_ai.agents import AgentContractError, AgentRequest, AgentResult
from twixt_ai.game import GameState, PegPlacement, Player, apply_move, legal_peg_placements


_WIN_SCORE = 1_000_000


class SearchAgent:
    """Choose moves with bounded minimax search and a lightweight heuristic.

    The default single-ply search stays responsive on a standard browser board.
    Deeper searches remain bounded by ``node_budget`` and use deterministic move
    ordering, making the agent useful without introducing UI-specific policy.
    """

    def __init__(self, *, depth: int = 1, node_budget: int = 10_000) -> None:
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("depth must be a positive integer")
        if (
            isinstance(node_budget, bool)
            or not isinstance(node_budget, int)
            or node_budget < 1
        ):
            raise ValueError("node_budget must be a positive integer")
        self.depth = depth
        self.node_budget = node_budget
        self._nodes = 0

    @staticmethod
    def _ordered(moves: tuple[PegPlacement, ...]) -> tuple[PegPlacement, ...]:
        return tuple(
            sorted(moves, key=lambda move: (move.coordinate.y, move.coordinate.x))
        )

    @staticmethod
    def _evaluate(state: GameState, root: Player) -> int:
        winner = state.result.winner
        if winner is root:
            return _WIN_SCORE
        if winner is root.opponent:
            return -_WIN_SCORE
        if state.result.is_terminal:
            return 0
        own_pegs = sum(peg.owner is root for peg in state.pegs)
        opposing_pegs = len(state.pegs) - own_pegs
        own_links = sum(link.owner is root for link in state.links)
        opposing_links = len(state.links) - own_links
        return (own_links - opposing_links) * 10 + own_pegs - opposing_pegs

    def _score(self, state: GameState, depth: int, root: Player) -> int:
        if state.result.is_terminal or depth == 0 or self._nodes >= self.node_budget:
            return self._evaluate(state, root)
        moves = self._ordered(legal_peg_placements(state))
        if not moves:
            return self._evaluate(state, root)
        maximizing = state.side_to_move is root
        best = -_WIN_SCORE if maximizing else _WIN_SCORE
        for move in moves:
            if self._nodes >= self.node_budget:
                break
            self._nodes += 1
            score = self._score(apply_move(state, move), depth - 1, root)
            best = max(best, score) if maximizing else min(best, score)
            if (maximizing and best == _WIN_SCORE) or (
                not maximizing and best == -_WIN_SCORE
            ):
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
        best_move = self._ordered(request.legal_moves)[0]
        best_score = -_WIN_SCORE
        for move in self._ordered(request.legal_moves):
            if self._nodes >= self.node_budget:
                break
            self._nodes += 1
            score = self._score(apply_move(request.state, move), self.depth - 1, root)
            if score > best_score:
                best_move = move
                best_score = score
            if best_score == _WIN_SCORE:
                break
        return AgentResult(
            best_move,
            {"depth": self.depth, "nodes": self._nodes, "score": best_score},
        )
