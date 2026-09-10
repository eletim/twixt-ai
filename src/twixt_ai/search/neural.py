"""Synchronous and dynamically batched policy/value inference for MCTS."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
import math
import sys
from threading import Condition, Lock, Thread
from time import monotonic, perf_counter
from typing import Protocol

import torch

from twixt_ai.game import GameState, PegPlacement
from twixt_ai.models import (
    PolicyValueNetwork,
    encode_positions_for_version,
    mask_policy_logits,
    move_to_action_index_for_version,
)

from .mcts import PolicyValueEstimate


_BATCHER_SWITCH_INTERVAL_SECONDS = 0.001
_switch_interval_lock = Lock()
_switch_interval_users = 0
_saved_switch_interval_seconds = 0.0
_installed_switch_interval_seconds = 0.0


def _acquire_batcher_switch_interval() -> None:
    """Bound GIL scheduling latency while a timed batch worker is active."""

    global _installed_switch_interval_seconds, _saved_switch_interval_seconds
    global _switch_interval_users
    with _switch_interval_lock:
        if not _switch_interval_users:
            _saved_switch_interval_seconds = sys.getswitchinterval()
            _installed_switch_interval_seconds = min(
                _saved_switch_interval_seconds,
                _BATCHER_SWITCH_INTERVAL_SECONDS,
            )
            sys.setswitchinterval(_installed_switch_interval_seconds)
        _switch_interval_users += 1


def _release_batcher_switch_interval() -> None:
    """Restore the process setting after the final timed batcher closes."""

    global _switch_interval_users
    with _switch_interval_lock:
        _switch_interval_users -= 1
        if (
            not _switch_interval_users
            and sys.getswitchinterval() == _installed_switch_interval_seconds
        ):
            sys.setswitchinterval(_saved_switch_interval_seconds)


class NeuralInferenceObserver(Protocol):
    """Optional measurement boundary for the source-of-truth inference path."""

    def start_host_phase(self, phase: str) -> object: ...

    def finish_host_phase(self, phase: str, token: object) -> None: ...

    def start_cuda_phase(self, phase: str) -> object: ...

    def finish_cuda_phase(self, phase: str, token: object) -> None: ...

    def prepare_result_tensors(
        self, probabilities: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def batch_completed(self, positions: int, total_seconds: float) -> None: ...


class InferenceBatcherObserver(Protocol):
    """Optional measurement boundary for batching and condition contention."""

    def queue_submission(
        self, lock_wait_seconds: float, critical_section_seconds: float
    ) -> None: ...

    def batch_dispatch(
        self,
        flush_reason: str,
        formation_seconds: float,
        lock_wait_seconds: float,
        critical_section_seconds: float,
        condition_wait_deadline_overshoot_seconds: float,
    ) -> None: ...

    def batch_completion(
        self, lock_wait_seconds: float, critical_section_seconds: float
    ) -> None: ...


class NeuralPolicyValue:
    """Callable inference hook for :class:`~twixt_ai.search.MCTSAgent`.

    Inference is performed without gradients and with the model in evaluation
    mode. See :meth:`evaluate_batch` for why input/output tensors are staged
    on the CPU rather than built directly on the model's device.
    """

    def __init__(
        self,
        model: PolicyValueNetwork,
        *,
        observer: NeuralInferenceObserver | None = None,
    ) -> None:
        if not isinstance(model, PolicyValueNetwork):
            raise TypeError("model must be a PolicyValueNetwork")
        self.model = model
        self.observer = observer

    def __call__(
        self, state: GameState, moves: tuple[PegPlacement, ...]
    ) -> PolicyValueEstimate:
        """Evaluate one position through the simple synchronous path."""

        return self.evaluate_batch(((state, moves),))[0]

    def evaluate_batch(
        self,
        requests: Sequence[tuple[GameState, tuple[PegPlacement, ...]]],
    ) -> tuple[PolicyValueEstimate, ...]:
        """Evaluate compatible positions in one model forward pass."""

        if not requests:
            return ()
        observer = self.observer
        total_started = perf_counter() if observer is not None else 0.0
        phase_token = (
            observer.start_host_phase("request_validation")
            if observer is not None
            else None
        )
        states: list[GameState] = []
        move_batches: list[tuple[PegPlacement, ...]] = []
        for request in requests:
            if not isinstance(request, tuple) or len(request) != 2:
                raise TypeError("requests must contain (state, moves) tuples")
            state, moves = request
            if not isinstance(state, GameState):
                raise TypeError("state must be a GameState")
            if not isinstance(moves, tuple):
                raise TypeError("moves must be a tuple")
            if any(not isinstance(move, PegPlacement) for move in moves):
                raise TypeError("moves must contain only PegPlacement values")
            states.append(state)
            move_batches.append(moves)
        if observer is not None:
            observer.finish_host_phase("request_validation", phase_token)

        parameter = next(self.model.parameters())
        device = parameter.device
        config = self.model.config
        if any(
            state.board.width != config.board_width
            or state.board.height != config.board_height
            for state in states
        ):
            raise ValueError(
                "state board dimensions do not match the policy/value model"
            )
        # Build every position's encoding and legal-move mask on the CPU, then
        # move each stacked batch tensor to the model's device in one
        # transfer each way. Populating a CUDA tensor one element at a time
        # (the previous behavior of passing ``device=device`` into the
        # encoding/masking calls below) issues one kernel launch per peg,
        # link, and legal move instead of two batched host-to-device copies,
        # and dominated wall time in the fixed Issue #98 self-play benchmark
        # despite using little actual GPU compute. Each move's action index
        # is computed once here and reused for both the mask and the result
        # below, rather than recomputed on extraction.
        phase_token = (
            observer.start_host_phase("action_index_construction")
            if observer is not None
            else None
        )
        action_indices = [
            [
                move_to_action_index_for_version(
                    move,
                    config.encoding_version,
                    board_width=config.board_width,
                    board_height=config.board_height,
                )
                for move in moves
            ]
            for moves in move_batches
        ]
        if observer is not None:
            observer.finish_host_phase("action_index_construction", phase_token)
        phase_token = (
            observer.start_host_phase("cpu_encoding_and_stack")
            if observer is not None
            else None
        )
        cpu_inputs = encode_positions_for_version(
            states, config.encoding_version
        )
        if observer is not None:
            observer.finish_host_phase("cpu_encoding_and_stack", phase_token)
            phase_token = observer.start_cuda_phase("host_to_device")
        inputs = cpu_inputs.to(device)
        if observer is not None:
            observer.finish_cuda_phase("host_to_device", phase_token)
        phase_token = (
            observer.start_host_phase("cpu_mask_construction")
            if observer is not None
            else None
        )
        action_count = config.board_width * config.board_height
        masks = torch.zeros((len(move_batches), action_count), dtype=torch.bool)
        for row, indices in zip(masks, action_indices):
            row[indices] = True
        if observer is not None:
            observer.finish_host_phase("cpu_mask_construction", phase_token)
            phase_token = observer.start_cuda_phase("host_to_device")
        masks = masks.to(device)
        if observer is not None:
            observer.finish_cuda_phase("host_to_device", phase_token)
            phase_token = observer.start_host_phase("model_mode_management")
        training_modes = tuple(
            (module, module.training) for module in self.model.modules()
        )
        self.model.eval()
        if observer is not None:
            observer.finish_host_phase("model_mode_management", phase_token)
        try:
            with torch.inference_mode():
                if observer is not None:
                    phase_token = observer.start_cuda_phase("model")
                logits, values = self.model(inputs)
                if observer is not None:
                    observer.finish_cuda_phase("model", phase_token)
                    phase_token = observer.start_cuda_phase("policy_mask")
                masked = mask_policy_logits(logits, masks)
                if observer is not None:
                    observer.finish_cuda_phase("policy_mask", phase_token)
                    phase_token = observer.start_cuda_phase("softmax")
                probabilities = torch.softmax(masked, dim=-1)
                if observer is not None:
                    observer.finish_cuda_phase("softmax", phase_token)
        finally:
            # Calling ``model.train(...)`` here would recursively overwrite
            # mixed configurations such as intentionally frozen BatchNorm
            # layers. Restore each module's exact pre-inference mode instead.
            if observer is not None:
                phase_token = observer.start_host_phase("model_mode_management")
            for module, was_training in training_modes:
                module.training = was_training
            if observer is not None:
                observer.finish_host_phase("model_mode_management", phase_token)
        # Convert each result tensor to plain Python values in one transfer
        # instead of calling ``.item()`` per legal move per position: each
        # ``.item()`` on a CUDA tensor is its own host/device
        # synchronization, and a position can have dozens of legal moves.
        if observer is not None:
            probabilities, values = observer.prepare_result_tensors(
                probabilities, values
            )
            phase_token = observer.start_host_phase("python_result_extraction")
        probabilities_list = probabilities.tolist()
        values_list = values.tolist()
        estimates = tuple(
            PolicyValueEstimate(
                {
                    move: position_probabilities[action_index]
                    for move, action_index in zip(moves, indices)
                },
                value,
            )
            for moves, indices, position_probabilities, value in zip(
                move_batches, action_indices, probabilities_list, values_list
            )
        )
        if observer is not None:
            observer.finish_host_phase("python_result_extraction", phase_token)
            observer.batch_completed(len(requests), perf_counter() - total_started)
        return estimates


@dataclass(frozen=True, slots=True)
class InferenceBatchStatistics:
    """Snapshot of work completed by :class:`NeuralInferenceBatcher`."""

    requests: int
    batches: int
    maximum_batch_size: int
    batch_size_distribution: dict[int, int] = field(default_factory=dict)
    full_batch_flushes: int = 0
    latency_flushes: int = 0
    forced_flushes: int = 0
    total_queue_wait_seconds: float = 0.0
    inference_seconds: float = 0.0

    @property
    def positions_per_second(self) -> float:
        """Model throughput, excluding time spent waiting to form a batch."""

        return (
            self.requests / self.inference_seconds
            if self.inference_seconds
            else 0.0
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-native metrics artifact."""

        return {
            "requests": self.requests,
            "batches": self.batches,
            "maximum_batch_size": self.maximum_batch_size,
            "batch_size_distribution": {
                str(size): count
                for size, count in sorted(self.batch_size_distribution.items())
            },
            "flushes": {
                "full_batch": self.full_batch_flushes,
                "latency": self.latency_flushes,
                "forced": self.forced_flushes,
            },
            "total_queue_wait_seconds": self.total_queue_wait_seconds,
            "average_queue_wait_seconds": (
                self.total_queue_wait_seconds / self.requests if self.requests else 0.0
            ),
            "inference_seconds": self.inference_seconds,
            "positions_per_second": self.positions_per_second,
        }


class NeuralInferenceBatcher:
    """Coalesce concurrent policy/value calls into bounded model batches.

    Callers retain the ordinary synchronous ``PolicyValueFunction`` interface:
    each call blocks until its estimate is available. A background worker
    flushes when ``batch_size`` requests arrive or ``max_wait_seconds`` elapses
    after the first queued request. ``batch_size=1`` is the serialized,
    deterministic synchronous path intended for tests and debugging.
    """

    def __init__(
        self,
        policy_value: NeuralPolicyValue,
        *,
        batch_size: int = 16,
        max_wait_seconds: float = 0.002,
        observer: InferenceBatcherObserver | None = None,
    ) -> None:
        if not isinstance(policy_value, NeuralPolicyValue):
            raise TypeError("policy_value must be a NeuralPolicyValue")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        if (
            isinstance(max_wait_seconds, bool)
            or not isinstance(max_wait_seconds, (int, float))
            or not math.isfinite(max_wait_seconds)
            or max_wait_seconds < 0
        ):
            raise ValueError("max_wait_seconds must be a finite non-negative number")
        self.policy_value = policy_value
        self.batch_size = batch_size
        self.max_wait_seconds = float(max_wait_seconds)
        self.observer = observer
        self._uses_short_switch_interval = batch_size > 1 and max_wait_seconds > 0
        self._switch_interval_acquired = False
        self._close_lock = Lock()
        self._close_complete = False
        self._condition = Condition()
        self._queue: deque[
            tuple[
                GameState,
                tuple[PegPlacement, ...],
                Future[PolicyValueEstimate],
                float,
            ]
        ] = deque()
        self._closed = False
        self._flushing = False
        self._active = False
        self._requests = 0
        self._batches = 0
        self._maximum_batch_size = 0
        self._batch_size_distribution: dict[int, int] = {}
        self._full_batch_flushes = 0
        self._latency_flushes = 0
        self._forced_flushes = 0
        self._total_queue_wait_seconds = 0.0
        self._inference_seconds = 0.0
        self._worker = Thread(
            target=self._run,
            name="twixt-neural-inference",
            daemon=True,
        )
        if self._uses_short_switch_interval:
            _acquire_batcher_switch_interval()
            self._switch_interval_acquired = True
        try:
            self._worker.start()
        except BaseException:
            if self._switch_interval_acquired:
                _release_batcher_switch_interval()
                self._switch_interval_acquired = False
            raise

    @property
    def statistics(self) -> InferenceBatchStatistics:
        with self._condition:
            return InferenceBatchStatistics(
                self._requests,
                self._batches,
                self._maximum_batch_size,
                dict(self._batch_size_distribution),
                self._full_batch_flushes,
                self._latency_flushes,
                self._forced_flushes,
                self._total_queue_wait_seconds,
                self._inference_seconds,
            )

    def __enter__(self) -> NeuralInferenceBatcher:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __call__(
        self, state: GameState, moves: tuple[PegPlacement, ...]
    ) -> PolicyValueEstimate:
        future: Future[PolicyValueEstimate] = Future()
        observer = self.observer
        lock_started = perf_counter() if observer is not None else 0.0
        with self._condition:
            lock_acquired = perf_counter() if observer is not None else 0.0
            if self._closed:
                raise RuntimeError("inference batcher is closed")
            self._queue.append((state, moves, future, monotonic()))
            self._condition.notify_all()
            critical_finished = perf_counter() if observer is not None else 0.0
        if observer is not None:
            observer.queue_submission(
                lock_acquired - lock_started,
                critical_finished - lock_acquired,
            )
        return future.result()

    def flush(self) -> None:
        """Immediately submit queued work and wait until it has completed."""

        with self._condition:
            self._flushing = True
            self._condition.notify_all()
            self._condition.wait_for(lambda: not self._queue and not self._active)
            if not self._closed:
                self._flushing = False

    def close(self) -> None:
        """Flush pending requests and stop the background worker."""

        with self._close_lock:
            if self._close_complete:
                return
            with self._condition:
                if not self._closed:
                    self._closed = True
                    self._flushing = True
                    self._condition.notify_all()
            self._worker.join()
            if self._switch_interval_acquired:
                _release_batcher_switch_interval()
                self._switch_interval_acquired = False
            self._close_complete = True

    def _run(self) -> None:
        while True:
            observer = self.observer
            lock_started = perf_counter() if observer is not None else 0.0
            with self._condition:
                lock_acquired = perf_counter() if observer is not None else 0.0
                self._condition.wait_for(lambda: self._queue or self._closed)
                if not self._queue and self._closed:
                    return
                active_critical_started = (
                    perf_counter() if observer is not None else 0.0
                )
                condition_wait_seconds = 0.0
                condition_wait_overshoot_seconds = 0.0
                formation_started = monotonic()
                deadline = monotonic() + self.max_wait_seconds
                while (
                    len(self._queue) < self.batch_size
                    and not self._closed
                    and not self._flushing
                ):
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        break
                    wait_started = perf_counter() if observer is not None else 0.0
                    self._condition.wait(remaining)
                    if observer is not None:
                        wait_seconds = perf_counter() - wait_started
                        condition_wait_seconds += wait_seconds
                        condition_wait_overshoot_seconds += max(
                            0.0, wait_seconds - remaining
                        )
                if len(self._queue) >= self.batch_size:
                    flush_reason = "full_batch"
                elif self._flushing or self._closed:
                    flush_reason = "forced"
                else:
                    flush_reason = "latency"
                batch = [
                    self._queue.popleft()
                    for _ in range(min(self.batch_size, len(self._queue)))
                ]
                dispatched_at = monotonic()
                if not self._queue:
                    self._flushing = False
                self._active = True
                critical_finished = perf_counter() if observer is not None else 0.0

            if observer is not None:
                observer.batch_dispatch(
                    flush_reason,
                    dispatched_at - formation_started,
                    lock_acquired - lock_started,
                    critical_finished
                    - active_critical_started
                    - condition_wait_seconds,
                    condition_wait_overshoot_seconds,
                )

            try:
                inference_started = perf_counter()
                estimates = self.policy_value.evaluate_batch(
                    tuple((state, moves) for state, moves, _, _ in batch)
                )
                inference_seconds = perf_counter() - inference_started
                if len(estimates) != len(batch):
                    raise RuntimeError(
                        "batch inference returned the wrong result count"
                    )
            except BaseException as exc:
                inference_seconds = perf_counter() - inference_started
                for _, _, future, _ in batch:
                    future.set_exception(exc)
            else:
                for estimate, (_, _, future, _) in zip(estimates, batch):
                    future.set_result(estimate)
            finally:
                completion_lock_started = (
                    perf_counter() if observer is not None else 0.0
                )
                with self._condition:
                    completion_lock_acquired = (
                        perf_counter() if observer is not None else 0.0
                    )
                    size = len(batch)
                    self._requests += size
                    self._batches += 1
                    self._maximum_batch_size = max(self._maximum_batch_size, size)
                    self._batch_size_distribution[size] = (
                        self._batch_size_distribution.get(size, 0) + 1
                    )
                    if flush_reason == "full_batch":
                        self._full_batch_flushes += 1
                    elif flush_reason == "forced":
                        self._forced_flushes += 1
                    else:
                        self._latency_flushes += 1
                    self._total_queue_wait_seconds += sum(
                        dispatched_at - queued_at for _, _, _, queued_at in batch
                    )
                    self._inference_seconds += inference_seconds
                    self._active = False
                    self._condition.notify_all()
                    completion_finished = (
                        perf_counter() if observer is not None else 0.0
                    )
                if observer is not None:
                    observer.batch_completion(
                        completion_lock_acquired - completion_lock_started,
                        completion_finished - completion_lock_acquired,
                    )


__all__ = [
    "InferenceBatchStatistics",
    "InferenceBatcherObserver",
    "NeuralInferenceObserver",
    "NeuralInferenceBatcher",
    "NeuralPolicyValue",
]
