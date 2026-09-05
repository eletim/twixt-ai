"""Tests for the uniform-random baseline agent."""

from __future__ import annotations

from random import Random

import pytest

from twixt_ai.agents import (
    Agent,
    AgentContractError,
    AgentRequest,
    RandomAgent,
    select_agent_move,
)
from twixt_ai.game import BoardDimensions, GameState, apply_move

from .contract import AgentContract


class TestRandomAgentContract(AgentContract):
    agent_factory = RandomAgent


def test_seeded_selection_matches_a_local_uniform_choice() -> None:
    state = GameState.initial(BoardDimensions(6, 6))
    request = AgentRequest(state, seed=8675309)

    result = RandomAgent().choose_move(request)

    assert result.move == Random(request.seed).choice(request.legal_moves)


def test_same_seeds_reproduce_a_complete_game() -> None:
    def play(seed_source: Random) -> tuple[object, ...]:
        state = GameState.initial(BoardDimensions(6, 6))
        moves = []
        agent: Agent = RandomAgent()
        while not state.is_terminal:
            result = select_agent_move(
                agent,
                state,
                seed=seed_source.randrange(2**64),
            )
            moves.append(result.move)
            state = apply_move(state, result.move)
        return tuple(moves)

    assert play(Random(12345)) == play(Random(12345))


def test_direct_selection_rejects_a_position_without_legal_moves() -> None:
    request = AgentRequest(GameState.initial(BoardDimensions(1, 1)))

    with pytest.raises(AgentContractError, match="no legal moves"):
        RandomAgent().choose_move(request)


def test_selection_does_not_change_global_random_state() -> None:
    import random

    state = GameState.initial(BoardDimensions(4, 4))
    before = random.getstate()

    select_agent_move(RandomAgent(), state, seed=7)

    assert random.getstate() == before
