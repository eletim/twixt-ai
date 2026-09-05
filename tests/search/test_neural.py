from __future__ import annotations

import pytest

from twixt_ai.agents import AgentRequest
from twixt_ai.game import GameState, legal_peg_placements
from twixt_ai.models import PolicyValueConfig, PolicyValueNetwork
from twixt_ai.search import MCTSAgent
from twixt_ai.search.neural import NeuralPolicyValue


def test_neural_inference_masks_policy_and_preserves_training_mode() -> None:
    model = PolicyValueNetwork(
        PolicyValueConfig(channels=4, residual_blocks=1, value_hidden=8)
    )
    model.train()
    state = GameState.initial()
    moves = legal_peg_placements(state)

    estimate = NeuralPolicyValue(model)(state, moves)

    assert model.training
    assert tuple(estimate.priors) == moves
    assert sum(estimate.priors.values()) == pytest.approx(1.0)
    assert all(probability >= 0 for probability in estimate.priors.values())
    assert estimate.value is not None and -1 <= estimate.value <= 1


def test_neural_policy_value_plugs_directly_into_mcts() -> None:
    model = PolicyValueNetwork(
        PolicyValueConfig(channels=4, residual_blocks=1, value_hidden=8)
    )
    result = MCTSAgent(
        simulations=1,
        policy_value=NeuralPolicyValue(model),
    ).choose_move(AgentRequest(GameState.initial(), seed=5))

    assert result.move in legal_peg_placements(GameState.initial())
    assert result.metadata["rollout_moves"] == 0
