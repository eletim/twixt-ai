from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event, Lock

import pytest
import torch
from torch import nn

from twixt_ai.agents import AgentRequest
from twixt_ai.game import (
    BoardDimensions,
    GameState,
    PegPlacement,
    legal_peg_placements,
)
from twixt_ai.models import PolicyValueConfig, PolicyValueNetwork
from twixt_ai.search import MCTSAgent
from twixt_ai.search.neural import NeuralInferenceBatcher, NeuralPolicyValue


class BlockingNetwork(PolicyValueNetwork):
    def __init__(self) -> None:
        super().__init__(
            PolicyValueConfig(channels=4, residual_blocks=1, value_hidden=8)
        )
        self.started = Event()
        self.release = Event()
        self.counter_lock = Lock()
        self.active = 0
        self.maximum_active = 0

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with self.counter_lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        self.started.set()
        assert self.release.wait(timeout=2)
        try:
            return super().forward(inputs)
        finally:
            with self.counter_lock:
                self.active -= 1


def _assert_next_request_waits_for_batch(
    batcher: NeuralInferenceBatcher,
    state: GameState,
    moves: tuple[PegPlacement, ...],
) -> None:
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(batcher, state, moves)
        with batcher._condition:
            assert batcher._condition.wait_for(
                lambda: len(batcher._queue) == 1 or first.done(), timeout=2
            )
        assert not first.done()
        with pytest.raises(TimeoutError):
            first.result(timeout=0.05)
        second = pool.submit(batcher, state, moves)
        first.result(timeout=2)
        second.result(timeout=2)


def test_neural_inference_masks_policy_and_preserves_mixed_training_modes() -> None:
    model = PolicyValueNetwork(
        PolicyValueConfig(channels=4, residual_blocks=1, value_hidden=8)
    )
    model.train()
    frozen_batch_norm = next(
        module for module in model.modules() if isinstance(module, nn.BatchNorm2d)
    )
    frozen_batch_norm.eval()
    state = GameState.initial()
    moves = legal_peg_placements(state)

    estimate = NeuralPolicyValue(model)(state, moves)

    assert model.training
    assert not frozen_batch_norm.training
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


def test_batched_inference_matches_synchronous_semantics() -> None:
    torch.manual_seed(54)
    model = PolicyValueNetwork(
        PolicyValueConfig(
            channels=4,
            residual_blocks=1,
            value_hidden=8,
            board_width=10,
            board_height=10,
        )
    )
    policy_value = NeuralPolicyValue(model)
    states = [GameState.initial(BoardDimensions(10, 10)) for _ in range(4)]
    moves = [legal_peg_placements(state) for state in states]
    expected = [policy_value(state, legal) for state, legal in zip(states, moves)]

    with NeuralInferenceBatcher(
        policy_value, batch_size=4, max_wait_seconds=1.0
    ) as batcher:
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(batcher, states, moves))
        statistics = batcher.statistics

    assert statistics.requests == 4
    assert statistics.batches == 1
    assert statistics.maximum_batch_size == 4
    for synchronous, batched in zip(expected, actual):
        assert batched.value == pytest.approx(synchronous.value, abs=1e-6)
        assert batched.priors == pytest.approx(synchronous.priors, abs=1e-7)


def test_batch_size_one_is_a_synchronous_debugging_path() -> None:
    model = PolicyValueNetwork(
        PolicyValueConfig(channels=4, residual_blocks=1, value_hidden=8)
    )
    state = GameState.initial()
    moves = legal_peg_placements(state)
    batcher = NeuralInferenceBatcher(NeuralPolicyValue(model), batch_size=1)

    estimate = batcher(state, moves)
    batcher.close()

    assert tuple(estimate.priors) == moves
    assert batcher.statistics.requests == 1
    with pytest.raises(RuntimeError, match="closed"):
        batcher(state, moves)


def test_batch_size_one_serializes_callers_and_close_waits_for_inference() -> None:
    model = BlockingNetwork()
    model.train()
    state = GameState.initial()
    moves = legal_peg_placements(state)
    batcher = NeuralInferenceBatcher(NeuralPolicyValue(model), batch_size=1)

    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(batcher, state, moves)
        assert model.started.wait(timeout=2)
        second = pool.submit(batcher, state, moves)
        with batcher._condition:
            assert batcher._condition.wait_for(
                lambda: len(batcher._queue) == 1, timeout=2
            )
        closing = pool.submit(batcher.close)
        with pytest.raises(TimeoutError):
            closing.result(timeout=0.05)
        model.release.set()
        first.result(timeout=2)
        second.result(timeout=2)
        closing.result(timeout=2)

    assert model.maximum_active == 1
    assert model.training
    assert batcher.statistics.requests == 2


def test_empty_flush_does_not_force_the_next_batch() -> None:
    model = PolicyValueNetwork(
        PolicyValueConfig(channels=4, residual_blocks=1, value_hidden=8)
    )
    state = GameState.initial()
    moves = legal_peg_placements(state)

    with NeuralInferenceBatcher(
        NeuralPolicyValue(model), batch_size=2, max_wait_seconds=1.0
    ) as batcher:
        batcher.flush()
        _assert_next_request_waits_for_batch(batcher, state, moves)


def test_active_batch_flush_does_not_force_the_next_batch() -> None:
    model = BlockingNetwork()
    state = GameState.initial()
    moves = legal_peg_placements(state)
    with NeuralInferenceBatcher(
        NeuralPolicyValue(model), batch_size=2, max_wait_seconds=1.0
    ) as batcher:
        with ThreadPoolExecutor(max_workers=3) as pool:
            first = pool.submit(batcher, state, moves)
            second = pool.submit(batcher, state, moves)
            assert model.started.wait(timeout=2)
            flushing = pool.submit(batcher.flush)
            with batcher._condition:
                assert batcher._condition.wait_for(
                    lambda: batcher._flushing, timeout=2
                )
            model.release.set()
            first.result(timeout=2)
            second.result(timeout=2)
            flushing.result(timeout=2)

        _assert_next_request_waits_for_batch(batcher, state, moves)
