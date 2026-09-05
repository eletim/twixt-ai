"""Tests for the common agent interface and its reusable contract."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from twixt_ai.agents import (
    Agent,
    AgentContractError,
    AgentRequest,
    AgentResult,
    select_agent_move,
)
from twixt_ai.game import (
    BoardDimensions,
    Coordinate,
    GameResult,
    GameState,
    PegPlacement,
    Player,
)

from .contract import AgentContract


class FirstLegalAgent:
    """Small test double proving that the shared contract is executable."""

    def choose_move(self, request: AgentRequest) -> AgentResult:
        return AgentResult(
            request.legal_moves[0],
            {"candidate_count": len(request.legal_moves)},
        )


class TestFirstLegalAgentContract(AgentContract):
    agent_factory = FirstLegalAgent


def test_request_exposes_state_legal_moves_and_optional_seed() -> None:
    state = GameState.initial(BoardDimensions(4, 3))

    request = AgentRequest(state, seed=42)

    assert request.state is request.position is state
    assert request.seed == 42
    assert request.legal_moves == (
        PegPlacement(Player.RED, Coordinate(1, 0)),
        PegPlacement(Player.RED, Coordinate(2, 0)),
        PegPlacement(Player.RED, Coordinate(1, 1)),
        PegPlacement(Player.RED, Coordinate(2, 1)),
        PegPlacement(Player.RED, Coordinate(1, 2)),
        PegPlacement(Player.RED, Coordinate(2, 2)),
    )


@pytest.mark.parametrize("seed", [True, 1.5, "1"])
def test_request_rejects_non_integer_seeds(seed: object) -> None:
    with pytest.raises(TypeError, match="seed must be an integer or None"):
        AgentRequest(GameState.initial(), seed=seed)  # type: ignore[arg-type]


def test_result_copies_metadata_into_a_read_only_mapping() -> None:
    source = {"nodes": 12}
    result = AgentResult(
        PegPlacement(Player.RED, Coordinate(1, 1)), metadata=source
    )
    source["nodes"] = 20

    assert isinstance(result.metadata, MappingProxyType)
    assert result.metadata == {"nodes": 12}
    with pytest.raises(TypeError):
        result.metadata["nodes"] = 30  # type: ignore[index]


def test_reference_agent_satisfies_runtime_protocol() -> None:
    assert isinstance(FirstLegalAgent(), Agent)


def test_selection_forwards_seed_and_preserves_metadata() -> None:
    class SeedAgent:
        def choose_move(self, request: AgentRequest) -> AgentResult:
            return AgentResult(request.legal_moves[0], {"seed": request.seed})

    result = select_agent_move(SeedAgent(), GameState.initial(), seed=73)

    assert result.metadata == {"seed": 73}


def test_selection_rejects_a_non_result_or_illegal_move() -> None:
    class NonResultAgent:
        def choose_move(self, request: AgentRequest) -> object:
            return request.legal_moves[0]

    class IllegalAgent:
        def choose_move(self, request: AgentRequest) -> AgentResult:
            return AgentResult(PegPlacement(Player.BLACK, Coordinate(0, 0)))

    state = GameState.initial(BoardDimensions(4, 4))
    with pytest.raises(AgentContractError, match="must return an AgentResult"):
        select_agent_move(NonResultAgent(), state)  # type: ignore[arg-type]
    with pytest.raises(AgentContractError, match="outside request.legal_moves"):
        select_agent_move(IllegalAgent(), state)


def test_selection_does_not_invoke_agent_without_legal_moves() -> None:
    class UnexpectedAgent:
        def choose_move(self, request: AgentRequest) -> AgentResult:
            raise AssertionError("agent should not be called")

    terminal = GameState(result=GameResult.RED_WINS)

    with pytest.raises(AgentContractError, match="no legal moves"):
        select_agent_move(UnexpectedAgent(), terminal)
