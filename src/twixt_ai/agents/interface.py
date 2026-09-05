"""Common, headless contract for Twixt move-selection agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Protocol, runtime_checkable

from twixt_ai.game import GameState, PegPlacement, legal_peg_placements


ThinkingMetadata = Mapping[str, object]
"""Optional agent-specific diagnostics such as depth, nodes, or scores."""


def _require_seed(seed: int | None) -> None:
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None")


@dataclass(frozen=True, slots=True)
class AgentRequest:
    """Everything an agent may use to select one move.

    Legal moves are derived once from the canonical immutable state. ``seed``
    is optional because deterministic agents do not need it; stochastic agents
    can use it without requiring orchestration code to know their concrete type.
    """

    state: GameState
    seed: int | None = None
    legal_moves: tuple[PegPlacement, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.state, GameState):
            raise TypeError("state must be a GameState")
        _require_seed(self.seed)
        object.__setattr__(self, "legal_moves", legal_peg_placements(self.state))

    @property
    def position(self) -> GameState:
        """Alias for callers that use position terminology."""

        return self.state


@dataclass(frozen=True, slots=True)
class AgentResult:
    """An agent's proposed move and optional, read-only thinking details."""

    move: PegPlacement
    metadata: ThinkingMetadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.move, PegPlacement):
            raise TypeError("move must be a PegPlacement")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        if any(not isinstance(key, str) for key in self.metadata):
            raise TypeError("metadata keys must be strings")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@runtime_checkable
class Agent(Protocol):
    """Structural protocol shared by human, search, and automated agents."""

    def choose_move(self, request: AgentRequest) -> AgentResult:
        """Select one move from ``request.legal_moves`` without changing state."""


class AgentContractError(ValueError):
    """Raised when an agent cannot produce a valid result for a request."""


def select_agent_move(
    agent: Agent, state: GameState, *, seed: int | None = None
) -> AgentResult:
    """Run and validate an agent through the common orchestration path.

    The engine remains authoritative: terminal positions are rejected before
    invoking the agent, and an agent result must contain one of the supplied
    legal moves. The immutable state can then be advanced with
    :func:`twixt_ai.game.apply_move` by a match runner or UI boundary.
    """

    if not isinstance(agent, Agent):
        raise TypeError("agent must implement choose_move(request)")
    request = AgentRequest(state=state, seed=seed)
    if not request.legal_moves:
        raise AgentContractError("cannot select a move when no legal moves exist")

    result = agent.choose_move(request)
    if not isinstance(result, AgentResult):
        raise AgentContractError("agent must return an AgentResult")
    if result.move not in request.legal_moves:
        raise AgentContractError("agent returned a move outside request.legal_moves")
    return result


__all__ = [
    "Agent",
    "AgentContractError",
    "AgentRequest",
    "AgentResult",
    "ThinkingMetadata",
    "select_agent_move",
]
