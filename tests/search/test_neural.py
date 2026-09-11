from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
import sys
from threading import Event, Lock

import pytest
import torch
from torch import nn

import twixt_ai.search.neural as neural
from twixt_ai.agents import AgentRequest
from twixt_ai.game import (
    BoardDimensions,
    Coordinate,
    GameState,
    PegPlacement,
    Peg,
    Player,
    legal_peg_placements,
)
from twixt_ai.models import (
    ENCODING_VERSION,
    MINI_ENCODING_VERSION,
    MINI_NUM_CHANNELS,
    NUM_CHANNELS,
    PolicyValueConfig,
    PolicyValueNetwork,
    batched_action_indices_for_version,
    batched_legal_move_mask,
    legal_move_mask_for_version,
    move_to_action_index_for_version,
)
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


class RecordingBatcherObserver:
    def __init__(self) -> None:
        self.queue_samples: list[tuple[float, float]] = []
        self.dispatch_samples: list[tuple[str, float, float, float, float]] = []
        self.completion_samples: list[tuple[float, float]] = []

    def queue_submission(
        self, lock_wait_seconds: float, critical_section_seconds: float
    ) -> None:
        self.queue_samples.append((lock_wait_seconds, critical_section_seconds))

    def batch_dispatch(
        self,
        flush_reason: str,
        formation_seconds: float,
        lock_wait_seconds: float,
        critical_section_seconds: float,
        condition_wait_deadline_overshoot_seconds: float,
    ) -> None:
        self.dispatch_samples.append(
            (
                flush_reason,
                formation_seconds,
                lock_wait_seconds,
                critical_section_seconds,
                condition_wait_deadline_overshoot_seconds,
            )
        )

    def batch_completion(
        self, lock_wait_seconds: float, critical_section_seconds: float
    ) -> None:
        self.completion_samples.append(
            (lock_wait_seconds, critical_section_seconds)
        )


@pytest.mark.parametrize(
    "encoding_version", (ENCODING_VERSION, MINI_ENCODING_VERSION)
)
@pytest.mark.parametrize("dimensions", ((5, 5), (10, 10), (24, 24)))
def test_batched_action_preparation_matches_versioned_reference(
    encoding_version: int, dimensions: tuple[int, int]
) -> None:
    width, height = dimensions
    states = tuple(
        GameState(board=BoardDimensions(width, height), side_to_move=player)
        for player in Player
    )
    move_batches = tuple(legal_peg_placements(state) for state in states)

    actual_indices = batched_action_indices_for_version(
        move_batches,
        encoding_version,
        board_width=width,
        board_height=height,
    )
    expected_indices = [
        [
            move_to_action_index_for_version(
                move,
                encoding_version,
                board_width=width,
                board_height=height,
            )
            for move in moves
        ]
        for moves in move_batches
    ]
    actual_masks = batched_legal_move_mask(actual_indices, width * height)
    assert actual_indices == expected_indices
    for actual_mask, moves in zip(actual_masks, move_batches):
        assert torch.equal(
            actual_mask,
            legal_move_mask_for_version(
                moves,
                encoding_version,
                board_width=width,
                board_height=height,
            ),
        )


@pytest.mark.parametrize("encoding_version", (ENCODING_VERSION, MINI_ENCODING_VERSION))
def test_batched_action_preparation_rejects_out_of_board_coordinates(
    encoding_version: int,
) -> None:
    move = PegPlacement(Player.RED, Coordinate(10, 2))

    with pytest.raises(ValueError) as expected:
        move_to_action_index_for_version(
            move,
            encoding_version,
            board_width=10,
            board_height=10,
        )
    with pytest.raises(ValueError, match=str(expected.value)):
        batched_action_indices_for_version(
            ((move,),),
            encoding_version,
            board_width=10,
            board_height=10,
        )


def test_batched_action_preparation_maps_each_moves_player() -> None:
    moves = (
        PegPlacement(Player.RED, Coordinate(2, 3)),
        PegPlacement(Player.BLACK, Coordinate(2, 3)),
    )

    actual = batched_action_indices_for_version(
        (moves,),
        MINI_ENCODING_VERSION,
        board_width=10,
        board_height=10,
    )

    assert actual == [
        [
            move_to_action_index_for_version(
                move,
                MINI_ENCODING_VERSION,
                board_width=10,
                board_height=10,
            )
            for move in moves
        ]
    ]


@pytest.mark.parametrize("action_count", (0, -1))
def test_batched_legal_move_mask_rejects_non_positive_action_count(
    action_count: int,
) -> None:
    with pytest.raises(ValueError, match="action_count must be a positive integer"):
        batched_legal_move_mask(((0,),), action_count)


@pytest.mark.parametrize("action_index", (-1, 4))
def test_batched_legal_move_mask_rejects_out_of_range_indices(
    action_index: int,
) -> None:
    with pytest.raises(ValueError, match=r"action indices must be in \[0, 4\)"):
        batched_legal_move_mask(((action_index,),), 4)


@pytest.mark.parametrize("action_index", (1.0, True, "1"))
def test_batched_legal_move_mask_rejects_non_integer_indices(
    action_index: object,
) -> None:
    with pytest.raises(TypeError, match="action indices must be integers"):
        batched_legal_move_mask(((action_index,),), 4)  # type: ignore[list-item]


@pytest.mark.parametrize(
    ("encoding_version", "input_channels"),
    (
        (ENCODING_VERSION, NUM_CHANNELS),
        (MINI_ENCODING_VERSION, MINI_NUM_CHANNELS),
    ),
)
@pytest.mark.parametrize("board_size", (5, 10, 24))
def test_batched_inference_matches_independent_versioned_reference(
    encoding_version: int, input_channels: int, board_size: int
) -> None:
    class DeterministicPolicyNetwork(PolicyValueNetwork):
        def forward(
            self, inputs: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            logits = torch.linspace(
                -2.0,
                2.0,
                self.action_count,
                dtype=inputs.dtype,
                device=inputs.device,
            ).expand(len(inputs), -1)
            values = torch.linspace(
                -0.375,
                0.625,
                len(inputs),
                dtype=inputs.dtype,
                device=inputs.device,
            )
            return logits, values

    model = DeterministicPolicyNetwork(
        PolicyValueConfig(
            channels=2,
            residual_blocks=1,
            value_hidden=4,
            board_width=board_size,
            board_height=board_size,
            input_channels=input_channels,
            encoding_version=encoding_version,
        )
    )
    states = tuple(
        GameState(
            board=BoardDimensions(board_size, board_size),
            pegs=(Peg(player.opponent, Coordinate(2, 2)),),
            side_to_move=player,
        )
        for player in Player
    )
    move_batches = tuple(legal_peg_placements(state) for state in states)

    actual = NeuralPolicyValue(model).evaluate_batch(
        tuple(zip(states, move_batches))
    )
    reference_logits = torch.linspace(-2.0, 2.0, model.action_count)
    reference_values = torch.linspace(-0.375, 0.625, len(states)).tolist()

    for estimate, state, moves, expected_value in zip(
        actual, states, move_batches, reference_values
    ):
        reference_mask = legal_move_mask_for_version(
            moves,
            encoding_version,
            board_width=board_size,
            board_height=board_size,
        )
        reference_probabilities = torch.softmax(
            reference_logits.masked_fill(~reference_mask, -torch.inf), dim=0
        ).tolist()
        expected_priors = {
            move: reference_probabilities[
                move_to_action_index_for_version(
                    move,
                    encoding_version,
                    board_width=state.board.width,
                    board_height=state.board.height,
                )
            ]
            for move in moves
        }

        assert tuple(estimate.priors) == moves
        assert estimate.priors == pytest.approx(expected_priors, abs=1e-8)
        assert estimate.value == pytest.approx(expected_value, abs=1e-8)


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


def test_v2_black_inference_uses_normalized_inputs_and_policy_indices() -> None:
    class IndexedPolicyNetwork(PolicyValueNetwork):
        captured_inputs: torch.Tensor | None = None

        def forward(
            self, inputs: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            self.captured_inputs = inputs.detach().clone()
            logits = torch.arange(
                self.action_count, dtype=inputs.dtype, device=inputs.device
            ).expand(len(inputs), -1)
            return logits, torch.zeros(len(inputs), device=inputs.device)

    model = IndexedPolicyNetwork(
        PolicyValueConfig(
            channels=2,
            residual_blocks=1,
            value_hidden=4,
            board_width=10,
            board_height=10,
            input_channels=MINI_NUM_CHANNELS,
            encoding_version=MINI_ENCODING_VERSION,
        )
    )
    state = GameState(
        board=BoardDimensions(10, 10),
        pegs=(Peg(Player.BLACK, Coordinate(2, 3)),),
        side_to_move=Player.BLACK,
    )
    moves = (
        PegPlacement(Player.BLACK, Coordinate(1, 2)),
        PegPlacement(Player.BLACK, Coordinate(2, 1)),
    )

    estimate = NeuralPolicyValue(model)(state, moves)

    assert model.captured_inputs is not None
    assert model.captured_inputs.shape == (1, 10, 10, 10)
    assert model.captured_inputs[0, 0, 2, 3] == 1
    # Black transposition maps the actions to indices 12 and 21. An
    # unnormalized lookup would reverse their order.
    assert estimate.priors[moves[1]] > estimate.priors[moves[0]]
    assert sum(estimate.priors.values()) == pytest.approx(1.0)


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
    assert statistics.batch_size_distribution == {4: 1}
    assert statistics.full_batch_flushes == 1
    assert statistics.latency_flushes == 0
    assert statistics.requests == sum(
        size * count for size, count in statistics.batch_size_distribution.items()
    )
    persisted = statistics.to_dict()
    assert persisted["batch_size_distribution"] == {"4": 1}
    assert persisted["positions_per_second"] > 0
    for synchronous, batched in zip(expected, actual):
        assert batched.value == pytest.approx(synchronous.value, abs=1e-6)
        assert batched.priors == pytest.approx(synchronous.priors, abs=1e-7)


def test_batcher_observer_separates_formation_and_condition_timings() -> None:
    model = PolicyValueNetwork(
        PolicyValueConfig(
            channels=4,
            residual_blocks=1,
            value_hidden=8,
            board_width=10,
            board_height=10,
        )
    )
    state = GameState.initial(BoardDimensions(10, 10))
    moves = legal_peg_placements(state)
    observer = RecordingBatcherObserver()

    with NeuralInferenceBatcher(
        NeuralPolicyValue(model),
        batch_size=2,
        max_wait_seconds=1.0,
        observer=observer,
    ) as batcher:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(batcher, (state, state), (moves, moves)))

    assert len(results) == 2
    assert len(observer.queue_samples) == 2
    assert len(observer.dispatch_samples) == 1
    assert observer.dispatch_samples[0][0] == "full_batch"
    assert len(observer.completion_samples) == 1
    for sample in (
        *observer.queue_samples,
        observer.dispatch_samples[0][1:],
        *observer.completion_samples,
    ):
        assert all(duration >= 0 for duration in sample)


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


def test_timed_batchers_bound_and_restore_the_thread_switch_interval() -> None:
    original_interval = sys.getswitchinterval()
    first = second = synchronous = None
    try:
        sys.setswitchinterval(0.005)
        policy_value = NeuralPolicyValue(
            PolicyValueNetwork(
                PolicyValueConfig(channels=4, residual_blocks=1, value_hidden=8)
            )
        )

        first = NeuralInferenceBatcher(
            policy_value, batch_size=2, max_wait_seconds=0.002
        )
        second = NeuralInferenceBatcher(
            policy_value, batch_size=2, max_wait_seconds=0.002
        )
        assert sys.getswitchinterval() == pytest.approx(0.001)

        first.close()
        assert sys.getswitchinterval() == pytest.approx(0.001)
        second.close()
        assert sys.getswitchinterval() == pytest.approx(0.005)

        synchronous = NeuralInferenceBatcher(
            policy_value, batch_size=1, max_wait_seconds=0.002
        )
        assert sys.getswitchinterval() == pytest.approx(0.005)
    finally:
        for batcher in (first, second, synchronous):
            if batcher is not None:
                batcher.close()
        sys.setswitchinterval(original_interval)


def test_timed_batcher_preserves_an_external_switch_interval_change() -> None:
    original_interval = sys.getswitchinterval()
    batcher = None
    try:
        sys.setswitchinterval(0.005)
        model = PolicyValueNetwork(
            PolicyValueConfig(channels=4, residual_blocks=1, value_hidden=8)
        )
        batcher = NeuralInferenceBatcher(
            NeuralPolicyValue(model), batch_size=2, max_wait_seconds=0.002
        )
        assert sys.getswitchinterval() == pytest.approx(0.001)

        sys.setswitchinterval(0.002)
        batcher.close()

        assert sys.getswitchinterval() == pytest.approx(0.002)
    finally:
        if batcher is not None:
            batcher.close()
        sys.setswitchinterval(original_interval)


def test_thread_construction_failure_does_not_acquire_switch_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_interval = sys.getswitchinterval()
    try:
        sys.setswitchinterval(0.005)
        model = PolicyValueNetwork(
            PolicyValueConfig(channels=4, residual_blocks=1, value_hidden=8)
        )

        def fail_thread_construction(**kwargs: object) -> None:
            raise RuntimeError("thread construction failed")

        monkeypatch.setattr(neural, "Thread", fail_thread_construction)
        with pytest.raises(RuntimeError, match="thread construction failed"):
            NeuralInferenceBatcher(
                NeuralPolicyValue(model), batch_size=2, max_wait_seconds=0.002
            )

        assert sys.getswitchinterval() == pytest.approx(0.005)
    finally:
        sys.setswitchinterval(original_interval)


def test_interrupted_close_retries_join_and_releases_switch_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_interval = sys.getswitchinterval()
    batcher = None
    try:
        sys.setswitchinterval(0.005)
        model = PolicyValueNetwork(
            PolicyValueConfig(channels=4, residual_blocks=1, value_hidden=8)
        )
        batcher = NeuralInferenceBatcher(
            NeuralPolicyValue(model), batch_size=2, max_wait_seconds=0.002
        )
        real_join = batcher._worker.join
        attempts = 0

        def interrupted_join() -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise KeyboardInterrupt
            real_join()

        monkeypatch.setattr(batcher._worker, "join", interrupted_join)
        with pytest.raises(KeyboardInterrupt):
            batcher.close()
        assert sys.getswitchinterval() == pytest.approx(0.001)

        batcher.close()

        assert attempts == 2
        assert sys.getswitchinterval() == pytest.approx(0.005)
    finally:
        if batcher is not None:
            batcher.close()
        sys.setswitchinterval(original_interval)


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
