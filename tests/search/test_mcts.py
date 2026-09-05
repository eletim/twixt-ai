from __future__ import annotations

import json

import pytest

from agents.contract import AgentContract
from twixt_ai.agents import AgentRequest
from twixt_ai.game import BoardDimensions, Coordinate, GameState, PegPlacement, Player
from twixt_ai.evaluation.benchmark_cli import _entrants
from twixt_ai.search import DEFAULT_ROLLOUT_LIMIT, MCTSAgent, PolicyValueEstimate
from twixt_ai.selfplay.cli import _agent_factory


class TestMCTSAgentContract(AgentContract):
    agent_factory = lambda self: MCTSAgent(simulations=1, rollout_limit=1)


def test_seeded_search_is_reproducible_and_reports_root_statistics() -> None:
    request = AgentRequest(GameState.initial(BoardDimensions(4, 4)), seed=1729)

    first = MCTSAgent(simulations=12).choose_move(request)
    second = MCTSAgent(simulations=12).choose_move(request)

    assert first.move == second.move
    assert first.metadata == second.metadata
    assert first.metadata["simulations"] == 12
    assert first.metadata["nodes"] <= 13
    assert sum(item["visits"] for item in first.metadata["root_moves"]) == 12
    json.dumps(dict(first.metadata))


def test_simulation_budget_is_hard_even_with_a_large_tree() -> None:
    agent = MCTSAgent(simulations=7, rollout_limit=1)

    result = agent.choose_move(AgentRequest(GameState.initial(), seed=4))

    assert result.move in AgentRequest(GameState.initial()).legal_moves
    assert result.metadata["simulations"] == 7
    assert result.metadata["nodes"] == 8
    assert agent.last_statistics is not None
    assert agent.last_statistics.maximum_depth == 1


def test_default_rollout_is_bounded_and_uses_cutoff_evaluation() -> None:
    evaluated: list[tuple[GameState, Player]] = []

    def cutoff_value(state: GameState, player: Player) -> float:
        evaluated.append((state, player))
        return 0.75

    agent = MCTSAgent(simulations=1, rollout_evaluator=cutoff_value)
    result = agent.choose_move(AgentRequest(GameState.initial(), seed=12))

    assert result.metadata["rollout_limit"] == DEFAULT_ROLLOUT_LIMIT
    assert result.metadata["rollout_moves"] == DEFAULT_ROLLOUT_LIMIT
    selected = next(
        item
        for item in result.metadata["root_moves"]
        if (item["x"], item["y"])
        == (result.move.coordinate.x, result.move.coordinate.y)
    )
    assert selected["value"] == 0.75
    assert len(evaluated) == 1
    assert evaluated[0][1] is Player.RED


def test_policy_and_value_hook_guides_the_same_tree() -> None:
    preferred = PegPlacement(Player.RED, Coordinate(2, 2))

    def guidance(
        state: GameState, moves: tuple[PegPlacement, ...]
    ) -> PolicyValueEstimate:
        del state
        priors = {move: float(move == preferred) for move in moves}
        return PolicyValueEstimate(priors, 0.25)

    result = MCTSAgent(simulations=1, policy_value=guidance).choose_move(
        AgentRequest(GameState.initial(BoardDimensions(4, 4)), seed=9)
    )

    assert result.move == preferred
    assert result.metadata["rollout_moves"] == 0
    preferred_stats = next(
        item
        for item in result.metadata["root_moves"]
        if (item["x"], item["y"]) == (2, 2)
    )
    assert preferred_stats == {
        "x": 2,
        "y": 2,
        "visits": 1,
        "value": -0.25,
        "prior": 1.0,
    }


@pytest.mark.parametrize("simulations", [0, -1, True, 1.5])
def test_simulation_budget_must_be_a_positive_integer(simulations: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        MCTSAgent(simulations=simulations)  # type: ignore[arg-type]


def test_policy_hook_rejects_illegal_priors() -> None:
    illegal = PegPlacement(Player.RED, Coordinate(0, 0))

    def guidance(
        state: GameState, moves: tuple[PegPlacement, ...]
    ) -> PolicyValueEstimate:
        del state, moves
        return PolicyValueEstimate({illegal: 1.0})

    with pytest.raises(ValueError, match="illegal moves"):
        MCTSAgent(simulations=1, policy_value=guidance).choose_move(
            AgentRequest(GameState.initial(BoardDimensions(4, 4)))
        )


def test_rollout_limit_is_recorded_by_benchmark_and_selfplay_clis() -> None:
    configs, factories = _entrants(("baseline=random", "candidate=mcts"), 1, 10, 3, 7)

    assert configs[1].configuration == {
        "type": "mcts",
        "simulations": 3,
        "rollout_limit": 7,
    }
    assert factories["candidate"]().rollout_limit == 7
    assert _agent_factory("mcts", 1, 10, 3, 9)().rollout_limit == 9
