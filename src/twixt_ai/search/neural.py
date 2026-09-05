"""Synchronous and dynamically batched policy/value inference for MCTS."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import dataclass
import math
from threading import Condition, Thread
from time import monotonic

import torch

from twixt_ai.game import GameState, PegPlacement
from twixt_ai.models import (
    PolicyValueNetwork,
    encode_position,
    legal_move_mask,
    mask_policy_logits,
    move_to_action_index,
)

from .mcts import PolicyValueEstimate


class NeuralPolicyValue:
    """Callable inference hook for :class:`~twixt_ai.search.MCTSAgent`.

    Inference is performed without gradients and with the model in evaluation
    mode. The model's device determines where inputs and masks are allocated.
    """

    def __init__(self, model: PolicyValueNetwork) -> None:
        if not isinstance(model, PolicyValueNetwork):
            raise TypeError("model must be a PolicyValueNetwork")
        self.model = model

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
        inputs = torch.stack(
            [encode_position(state, device=device) for state in states]
        )
        masks = torch.stack(
            [
                legal_move_mask(
                    moves,
                    board_width=config.board_width,
                    board_height=config.board_height,
                    device=device,
                )
                for moves in move_batches
            ]
        )
        training_modes = tuple(
            (module, module.training) for module in self.model.modules()
        )
        self.model.eval()
        try:
            with torch.inference_mode():
                logits, values = self.model(inputs)
                masked = mask_policy_logits(logits, masks)
                probabilities = torch.softmax(masked, dim=-1)
        finally:
            # Calling ``model.train(...)`` here would recursively overwrite
            # mixed configurations such as intentionally frozen BatchNorm
            # layers. Restore each module's exact pre-inference mode instead.
            for module, was_training in training_modes:
                module.training = was_training
        return tuple(
            PolicyValueEstimate(
                {
                    move: float(probabilities[index, move_to_action_index(
                        move,
                        board_width=config.board_width,
                        board_height=config.board_height,
                    )].item())
                    for move in moves
                },
                float(values[index].item()),
            )
            for index, moves in enumerate(move_batches)
        )


@dataclass(frozen=True, slots=True)
class InferenceBatchStatistics:
    """Snapshot of work completed by :class:`NeuralInferenceBatcher`."""

    requests: int
    batches: int
    maximum_batch_size: int


class NeuralInferenceBatcher:
    """Coalesce concurrent policy/value calls into bounded model batches.

    Callers retain the ordinary synchronous ``PolicyValueFunction`` interface:
    each call blocks until its estimate is available. A background worker
    flushes when ``batch_size`` requests arrive or ``max_wait_seconds`` elapses
    after the first queued request. ``batch_size=1`` bypasses the worker and is
    the deterministic synchronous path intended for tests and debugging.
    """

    def __init__(
        self,
        policy_value: NeuralPolicyValue,
        *,
        batch_size: int = 16,
        max_wait_seconds: float = 0.002,
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
        self._condition = Condition()
        self._queue: deque[
            tuple[GameState, tuple[PegPlacement, ...], Future[PolicyValueEstimate]]
        ] = deque()
        self._closed = False
        self._flushing = False
        self._active = False
        self._requests = 0
        self._batches = 0
        self._maximum_batch_size = 0
        self._worker: Thread | None = None
        if batch_size > 1:
            self._worker = Thread(
                target=self._run,
                name="twixt-neural-inference",
                daemon=True,
            )
            self._worker.start()

    @property
    def statistics(self) -> InferenceBatchStatistics:
        with self._condition:
            return InferenceBatchStatistics(
                self._requests, self._batches, self._maximum_batch_size
            )

    def __enter__(self) -> NeuralInferenceBatcher:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __call__(
        self, state: GameState, moves: tuple[PegPlacement, ...]
    ) -> PolicyValueEstimate:
        if self.batch_size == 1:
            with self._condition:
                if self._closed:
                    raise RuntimeError("inference batcher is closed")
            estimate = self.policy_value(state, moves)
            with self._condition:
                self._requests += 1
                self._batches += 1
                self._maximum_batch_size = 1
            return estimate

        future: Future[PolicyValueEstimate] = Future()
        with self._condition:
            if self._closed:
                raise RuntimeError("inference batcher is closed")
            self._queue.append((state, moves, future))
            self._condition.notify_all()
        return future.result()

    def flush(self) -> None:
        """Immediately submit queued work and wait until it has completed."""

        if self.batch_size == 1:
            return
        with self._condition:
            self._flushing = True
            self._condition.notify_all()
            self._condition.wait_for(lambda: not self._queue and not self._active)

    def close(self) -> None:
        """Flush pending requests and stop the background worker."""

        if self.batch_size == 1:
            with self._condition:
                self._closed = True
            return
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._flushing = True
            self._condition.notify_all()
        assert self._worker is not None
        self._worker.join()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._queue or self._closed)
                if not self._queue and self._closed:
                    return
                deadline = monotonic() + self.max_wait_seconds
                while (
                    len(self._queue) < self.batch_size
                    and not self._closed
                    and not self._flushing
                ):
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(remaining)
                batch = [
                    self._queue.popleft()
                    for _ in range(min(self.batch_size, len(self._queue)))
                ]
                if not self._queue:
                    self._flushing = False
                self._active = True

            try:
                estimates = self.policy_value.evaluate_batch(
                    tuple((state, moves) for state, moves, _ in batch)
                )
                if len(estimates) != len(batch):
                    raise RuntimeError(
                        "batch inference returned the wrong result count"
                    )
            except BaseException as exc:
                for _, _, future in batch:
                    future.set_exception(exc)
            else:
                for estimate, (_, _, future) in zip(estimates, batch):
                    future.set_result(estimate)
            finally:
                with self._condition:
                    size = len(batch)
                    self._requests += size
                    self._batches += 1
                    self._maximum_batch_size = max(self._maximum_batch_size, size)
                    self._active = False
                    self._condition.notify_all()


__all__ = [
    "InferenceBatchStatistics",
    "NeuralInferenceBatcher",
    "NeuralPolicyValue",
]
