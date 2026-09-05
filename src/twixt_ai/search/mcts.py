"""Monte Carlo tree search with optional policy/value guidance."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
import math
from random import Random

from twixt_ai.agents import (
    AgentContractError,
    AgentRequest,
    AgentResult,
    evaluate_position,
)
from twixt_ai.game import GameState, PegPlacement, Player, apply_move, legal_peg_placements


@dataclass(frozen=True, slots=True)
class PolicyValueEstimate:
    """Policy priors and an optional value for one position.

    ``value`` is in ``[-1, 1]`` from the position's side-to-move perspective.
    Returning ``None`` requests an ordinary seeded rollout. Priors may omit
    legal moves (they receive zero weight), but may not contain illegal moves.
    """

    priors: Mapping[PegPlacement, float]
    value: float | None = None


PolicyValueFunction = Callable[
    [GameState, tuple[PegPlacement, ...]], PolicyValueEstimate
]
"""Extension point for learned policy priors and value estimates."""


DEFAULT_ROLLOUT_LIMIT = 4
"""Default playout horizon, chosen to keep standard-board decisions practical."""

_HEURISTIC_VALUE_SCALE = 100.0
_PROGRESSIVE_WIDENING_CONSTANT = 1.5
_PROGRESSIVE_WIDENING_EXPONENT = 0.5


RolloutEvaluationFunction = Callable[[GameState, Player], float]
"""Return a finite cutoff value in ``[-1, 1]`` from a player's perspective."""


def heuristic_rollout_value(state: GameState, player: Player) -> float:
    """Map the shared position heuristic to a bounded MCTS cutoff value."""

    return math.tanh(evaluate_position(state, player) / _HEURISTIC_VALUE_SCALE)


@dataclass(frozen=True, slots=True)
class MCTSMoveStatistics:
    """Search measurements for one legal move at the root."""

    move: PegPlacement
    visits: int
    value: float
    prior: float


@dataclass(frozen=True, slots=True)
class MCTSSearchStatistics:
    """Inspectable summary of a completed MCTS decision."""

    simulations: int
    rollout_limit: int | None
    nodes: int
    rollout_moves: int
    maximum_depth: int
    moves: tuple[MCTSMoveStatistics, ...]


class _Node:
    __slots__ = (
        "state",
        "move",
        "prior",
        "parent",
        "children",
        "unexpanded",
        "priors",
        "estimated_value",
        "visits",
        "value_sum",
    )

    def __init__(
        self,
        state: GameState,
        *,
        move: PegPlacement | None = None,
        prior: float = 1.0,
        parent: _Node | None = None,
    ) -> None:
        self.state = state
        self.move = move
        self.prior = prior
        self.parent = parent
        self.children: list[_Node] = []
        self.unexpanded: list[PegPlacement] = []
        self.priors: dict[PegPlacement, float] = {}
        self.estimated_value: float | None = None
        self.visits = 0
        self.value_sum = 0.0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def _uniform_priors(moves: tuple[PegPlacement, ...]) -> dict[PegPlacement, float]:
    if not moves:
        return {}
    probability = 1.0 / len(moves)
    return {move: probability for move in moves}


class MCTSAgent:
    """Choose moves using bounded Monte Carlo tree search.

    Each simulation performs selection, one-node expansion, simulation (or a
    supplied value estimate), and backpropagation. The simulation count is the
    primary reproducible budget. Rollouts default to a short fixed horizon so
    decisions remain practical on a standard board. A non-terminal cutoff is
    scored by ``rollout_evaluator``; explicitly pass ``rollout_limit=None`` to
    run playouts to completion.

    A ``policy_value`` callback supplies normalized search priors and/or leaf
    values without changing the tree orchestration used by this baseline.
    Progressive widening admits new actions as a node's visit count grows, so
    PUCT can revisit children even when the legal action space is much larger
    than the simulation budget.
    """

    def __init__(
        self,
        *,
        simulations: int = 100,
        exploration: float = math.sqrt(2.0),
        rollout_limit: int | None = DEFAULT_ROLLOUT_LIMIT,
        rollout_evaluator: RolloutEvaluationFunction = heuristic_rollout_value,
        policy_value: PolicyValueFunction | None = None,
    ) -> None:
        if (
            isinstance(simulations, bool)
            or not isinstance(simulations, int)
            or simulations < 1
        ):
            raise ValueError("simulations must be a positive integer")
        if (
            isinstance(exploration, bool)
            or not isinstance(exploration, (int, float))
            or not math.isfinite(exploration)
            or exploration < 0
        ):
            raise ValueError("exploration must be a finite non-negative number")
        if rollout_limit is not None and (
            isinstance(rollout_limit, bool)
            or not isinstance(rollout_limit, int)
            or rollout_limit < 1
        ):
            raise ValueError("rollout_limit must be a positive integer or None")
        if policy_value is not None and not callable(policy_value):
            raise TypeError("policy_value must be callable or None")
        if not callable(rollout_evaluator):
            raise TypeError("rollout_evaluator must be callable")
        self.simulations = simulations
        self.exploration = float(exploration)
        self.rollout_limit = rollout_limit
        self.rollout_evaluator = rollout_evaluator
        self.policy_value = policy_value
        self.last_statistics: MCTSSearchStatistics | None = None
        self._rollout_moves = 0
        self._maximum_depth = 0

    def _initialize(self, node: _Node) -> None:
        moves = legal_peg_placements(node.state)
        node.unexpanded = list(moves)
        node.priors = _uniform_priors(moves)
        if self.policy_value is None or node.state.is_terminal:
            return

        estimate = self.policy_value(node.state, moves)
        if not isinstance(estimate, PolicyValueEstimate):
            raise TypeError("policy_value must return a PolicyValueEstimate")
        if not isinstance(estimate.priors, Mapping):
            raise TypeError("policy priors must be a mapping")
        if any(not isinstance(move, PegPlacement) for move in estimate.priors):
            raise TypeError("policy priors must use PegPlacement keys")
        illegal = set(estimate.priors) - set(moves)
        if illegal:
            raise ValueError("policy priors must not contain illegal moves")
        weights: dict[PegPlacement, float] = {}
        for move in moves:
            weight = estimate.priors.get(move, 0.0)
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(weight)
                or weight < 0
            ):
                raise ValueError("policy priors must be finite non-negative numbers")
            weights[move] = float(weight)
        total = sum(weights.values())
        if total > 0:
            node.priors = {move: weight / total for move, weight in weights.items()}

        value = estimate.value
        if value is not None:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not -1 <= value <= 1
            ):
                raise ValueError("policy value must be a finite number in [-1, 1] or None")
            node.estimated_value = float(value)

    @staticmethod
    def _terminal_value(state: GameState, root_player: Player) -> float:
        if state.winner is root_player:
            return 1.0
        if state.winner is root_player.opponent:
            return -1.0
        return 0.0

    def _expand(self, node: _Node, random: Random) -> _Node:
        weights = [node.priors[move] for move in node.unexpanded]
        if sum(weights) > 0:
            index = random.choices(range(len(node.unexpanded)), weights=weights, k=1)[0]
        else:
            index = random.randrange(len(node.unexpanded))
        move = node.unexpanded.pop(index)
        child = _Node(
            apply_move(node.state, move),
            move=move,
            prior=node.priors[move],
            parent=node,
        )
        self._initialize(child)
        node.children.append(child)
        return child

    def _select(self, node: _Node) -> _Node:
        maximizing = node.state.side_to_move is self._root_player
        scale = math.sqrt(node.visits)

        def score(child: _Node) -> tuple[float, int, int]:
            exploitation = child.mean_value if maximizing else -child.mean_value
            exploration = (
                self.exploration * child.prior * scale / (1 + child.visits)
            )
            assert child.move is not None
            # Stable coordinate tie-breaking keeps behavior reproducible even
            # when a caller provides zero exploration.
            return (
                exploitation + exploration,
                -child.move.coordinate.y,
                -child.move.coordinate.x,
            )

        return max(node.children, key=score)

    @staticmethod
    def _can_expand(node: _Node) -> bool:
        """Return whether progressive widening admits another action."""

        if not node.unexpanded:
            return False
        if not node.children:
            return True
        child_limit = math.ceil(
            _PROGRESSIVE_WIDENING_CONSTANT
            * node.visits**_PROGRESSIVE_WIDENING_EXPONENT
        )
        return len(node.children) < child_limit

    def _rollout(self, state: GameState, random: Random) -> float:
        steps = 0
        while not state.is_terminal and (
            self.rollout_limit is None or steps < self.rollout_limit
        ):
            moves = legal_peg_placements(state)
            if not moves:
                break
            state = apply_move(state, random.choice(moves))
            steps += 1
        self._rollout_moves += steps
        if state.is_terminal:
            return self._terminal_value(state, self._root_player)
        value = self.rollout_evaluator(state, self._root_player)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not -1 <= value <= 1
        ):
            raise ValueError(
                "rollout_evaluator must return a finite number in [-1, 1]"
            )
        return float(value)

    def _leaf_value(self, node: _Node, random: Random) -> float:
        if node.state.is_terminal:
            return self._terminal_value(node.state, self._root_player)
        if node.estimated_value is not None:
            return (
                node.estimated_value
                if node.state.side_to_move is self._root_player
                else -node.estimated_value
            )
        return self._rollout(node.state, random)

    def choose_move(self, request: AgentRequest) -> AgentResult:
        """Run exactly the configured simulations and return a legal move."""

        if not isinstance(request, AgentRequest):
            raise TypeError("request must be an AgentRequest")
        if not request.legal_moves:
            raise AgentContractError("cannot select a move when no legal moves exist")

        random = Random(request.seed)
        self._root_player = request.state.side_to_move
        self._rollout_moves = 0
        self._maximum_depth = 0
        root = _Node(request.state)
        self._initialize(root)

        for _ in range(self.simulations):
            node = root
            path = [root]
            while not node.state.is_terminal:
                if self._can_expand(node):
                    node = self._expand(node, random)
                    path.append(node)
                    break
                if not node.children:
                    break
                node = self._select(node)
                path.append(node)
            self._maximum_depth = max(self._maximum_depth, len(path) - 1)
            value = self._leaf_value(node, random)
            for visited in path:
                visited.visits += 1
                visited.value_sum += value

        # At least one root child exists because simulations is positive.
        best = max(
            root.children,
            key=lambda child: (
                child.visits,
                child.mean_value,
                -child.move.coordinate.y,  # type: ignore[union-attr]
                -child.move.coordinate.x,  # type: ignore[union-attr]
            ),
        )
        move_statistics = tuple(
            MCTSMoveStatistics(move, 0, 0.0, root.priors[move])
            for move in request.legal_moves
        )
        by_move = {item.move: item for item in root.children}
        move_statistics = tuple(
            MCTSMoveStatistics(
                item.move,
                by_move[item.move].visits,
                by_move[item.move].mean_value,
                item.prior,
            )
            if item.move in by_move
            else item
            for item in move_statistics
        )
        statistics = MCTSSearchStatistics(
            simulations=self.simulations,
            rollout_limit=self.rollout_limit,
            nodes=1 + sum(1 for _ in self._walk(root)),
            rollout_moves=self._rollout_moves,
            maximum_depth=self._maximum_depth,
            moves=move_statistics,
        )
        self.last_statistics = statistics
        assert best.move is not None
        metadata = {
            "simulations": statistics.simulations,
            "rollout_limit": statistics.rollout_limit,
            "nodes": statistics.nodes,
            "rollout_moves": statistics.rollout_moves,
            "maximum_depth": statistics.maximum_depth,
            "root_moves": [
                {
                    "x": item.move.coordinate.x,
                    "y": item.move.coordinate.y,
                    "visits": item.visits,
                    "value": item.value,
                    "prior": item.prior,
                }
                for item in statistics.moves
            ],
        }
        return AgentResult(best.move, metadata)

    @staticmethod
    def _walk(root: _Node) -> Iterator[_Node]:
        pending = list(root.children)
        while pending:
            node = pending.pop()
            yield node
            pending.extend(node.children)


__all__ = [
    "DEFAULT_ROLLOUT_LIMIT",
    "MCTSAgent",
    "MCTSMoveStatistics",
    "MCTSSearchStatistics",
    "PolicyValueEstimate",
    "PolicyValueFunction",
    "RolloutEvaluationFunction",
    "heuristic_rollout_value",
]
