"""Measurement-only phase profiler for the canonical CUDA inference path.

This module deliberately lives outside :mod:`twixt_ai.search`: it reproduces
``NeuralPolicyValue.evaluate_batch`` with timing boundaries, but does not alter
the production evaluator.  The profile synchronizes once at the same result
extraction boundary where ``Tensor.tolist`` necessarily synchronizes, then
spells its implicit device-to-host copy out so transfer and Python extraction
can be measured separately.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from time import perf_counter
from typing import TypeVar

import torch

from twixt_ai.game import GameState, PegPlacement
from twixt_ai.models import (
    PolicyValueNetwork,
    encode_position_for_version,
    mask_policy_logits,
    move_to_action_index_for_version,
)
from twixt_ai.search.mcts import PolicyValueEstimate
from twixt_ai.search.neural import NeuralPolicyValue


_T = TypeVar("_T")

_HOST_PHASES = (
    "request_validation",
    "action_index_construction",
    "cpu_encoding_and_stack",
    "cpu_mask_construction",
    "host_to_device_submission",
    "model_submission",
    "policy_mask_submission",
    "softmax_submission",
    "cuda_synchronization_wait",
    "device_to_host_transfer",
    "python_result_extraction",
    "model_mode_management",
)

_CUDA_PHASES = (
    "host_to_device_transfer",
    "model_execution",
    "policy_mask",
    "softmax",
    "device_to_host_transfer",
)


class CudaInferencePhaseProfile:
    """Accumulate host wall time and CUDA event time for inference batches."""

    def __init__(self) -> None:
        self.batches = 0
        self.positions = 0
        self.total_evaluate_batch_seconds = 0.0
        self.host_seconds = {phase: 0.0 for phase in _HOST_PHASES}
        self.cuda_seconds = {phase: 0.0 for phase in _CUDA_PHASES}

    def host_call(self, phase: str, operation: Callable[[], _T]) -> _T:
        started = perf_counter()
        try:
            return operation()
        finally:
            self.host_seconds[phase] += perf_counter() - started

    def cuda_call(
        self, phase: str, operation: Callable[[], _T]
    ) -> tuple[_T, torch.cuda.Event, torch.cuda.Event]:
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        host_started = perf_counter()
        started.record()
        result = operation()
        finished.record()
        self.host_seconds[f"{phase}_submission"] += perf_counter() - host_started
        return result, started, finished

    def record_cuda_elapsed(
        self, phase: str, started: torch.cuda.Event, finished: torch.cuda.Event
    ) -> None:
        self.cuda_seconds[phase] += started.elapsed_time(finished) / 1000.0

    def to_dict(self) -> dict[str, object]:
        accounted_host = sum(self.host_seconds.values())
        unaccounted = max(0.0, self.total_evaluate_batch_seconds - accounted_host)
        host = dict(self.host_seconds)
        host["unaccounted_profiler_overhead"] = unaccounted
        ranked = sorted(host, key=host.__getitem__, reverse=True)
        return {
            "method": (
                "benchmark-only perf_counter host spans plus CUDA events; an explicit "
                "CUDA event synchronization at the existing Tensor.tolist result "
                "boundary separates pending device work, device-to-host copies, and "
                "Python extraction"
            ),
            "batches": self.batches,
            "positions": self.positions,
            "total_evaluate_batch_seconds": self.total_evaluate_batch_seconds,
            "host_seconds": host,
            "host_percent_of_evaluate_batch": {
                phase: (
                    seconds / self.total_evaluate_batch_seconds * 100.0
                    if self.total_evaluate_batch_seconds
                    else 0.0
                )
                for phase, seconds in host.items()
            },
            "host_milliseconds_per_batch": {
                phase: seconds / self.batches * 1000.0 if self.batches else 0.0
                for phase, seconds in host.items()
            },
            "cuda_seconds": dict(self.cuda_seconds),
            "cuda_milliseconds_per_batch": {
                phase: seconds / self.batches * 1000.0 if self.batches else 0.0
                for phase, seconds in self.cuda_seconds.items()
            },
            "ranked_host_phases": [
                {
                    "rank": rank,
                    "phase": phase,
                    "seconds": host[phase],
                    "percent_of_evaluate_batch": (
                        host[phase] / self.total_evaluate_batch_seconds * 100.0
                        if self.total_evaluate_batch_seconds
                        else 0.0
                    ),
                }
                for rank, phase in enumerate(ranked, start=1)
            ],
            "interpretation": {
                "host_and_cuda_times_are_not_additive": (
                    "CUDA kernels and copies are asynchronous and can overlap host wall "
                    "time; event durations identify device work rather than another "
                    "exclusive wall-time partition."
                ),
                "profiling_overhead": (
                    "CUDA events and phase clocks add overhead, so profile wall time is "
                    "diagnostic and must not be compared as an optimized throughput run."
                ),
            },
        }


class ProfiledCudaNeuralPolicyValue(NeuralPolicyValue):
    """Semantics-equivalent CUDA evaluator with detailed timing boundaries."""

    def __init__(
        self, model: PolicyValueNetwork, profile: CudaInferencePhaseProfile
    ) -> None:
        super().__init__(model)
        self.profile = profile

    def evaluate_batch(
        self,
        requests: Sequence[tuple[GameState, tuple[PegPlacement, ...]]],
    ) -> tuple[PolicyValueEstimate, ...]:
        if not requests:
            return ()
        total_started = perf_counter()
        profile = self.profile

        def validate_requests() -> tuple[
            list[GameState], list[tuple[PegPlacement, ...]]
        ]:
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
            return states, move_batches

        states, move_batches = profile.host_call(
            "request_validation", validate_requests
        )
        parameter = next(self.model.parameters())
        device = parameter.device
        if device.type != "cuda":
            raise ValueError("the CUDA inference phase profiler requires a CUDA model")
        config = self.model.config
        if any(
            state.board.width != config.board_width
            or state.board.height != config.board_height
            for state in states
        ):
            raise ValueError(
                "state board dimensions do not match the policy/value model"
            )

        action_indices = profile.host_call(
            "action_index_construction",
            lambda: [
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
            ],
        )
        cpu_inputs = profile.host_call(
            "cpu_encoding_and_stack",
            lambda: torch.stack(
                [
                    encode_position_for_version(state, config.encoding_version)
                    for state in states
                ]
            ),
        )

        def construct_masks() -> torch.Tensor:
            masks = torch.zeros(
                (len(move_batches), config.board_width * config.board_height),
                dtype=torch.bool,
            )
            for row, indices in zip(masks, action_indices):
                row[indices] = True
            return masks

        cpu_masks = profile.host_call("cpu_mask_construction", construct_masks)
        inputs, input_start, input_end = profile.cuda_call(
            "host_to_device", lambda: cpu_inputs.to(device)
        )
        masks, mask_start, mask_end = profile.cuda_call(
            "host_to_device", lambda: cpu_masks.to(device)
        )

        mode_started = perf_counter()
        training_modes = tuple(
            (module, module.training) for module in self.model.modules()
        )
        self.model.eval()
        profile.host_seconds["model_mode_management"] += perf_counter() - mode_started
        try:
            with torch.inference_mode():
                (logits, values), model_start, model_end = profile.cuda_call(
                    "model", lambda: self.model(inputs)
                )
                masked, policy_mask_start, policy_mask_end = profile.cuda_call(
                    "policy_mask", lambda: mask_policy_logits(logits, masks)
                )
                probabilities, softmax_start, softmax_end = profile.cuda_call(
                    "softmax", lambda: torch.softmax(masked, dim=-1)
                )
        finally:
            mode_started = perf_counter()
            for module, was_training in training_modes:
                module.training = was_training
            profile.host_seconds["model_mode_management"] += (
                perf_counter() - mode_started
            )

        sync_started = perf_counter()
        softmax_end.synchronize()
        profile.host_seconds["cuda_synchronization_wait"] += (
            perf_counter() - sync_started
        )
        for phase, started, finished in (
            ("host_to_device_transfer", input_start, input_end),
            ("host_to_device_transfer", mask_start, mask_end),
            ("model_execution", model_start, model_end),
            ("policy_mask", policy_mask_start, policy_mask_end),
            ("softmax", softmax_start, softmax_end),
        ):
            profile.record_cuda_elapsed(phase, started, finished)

        transfer_started = perf_counter()
        copy_start = torch.cuda.Event(enable_timing=True)
        copy_end = torch.cuda.Event(enable_timing=True)
        copy_start.record()
        cpu_probabilities = probabilities.to("cpu")
        cpu_values = values.to("cpu")
        copy_end.record()
        copy_end.synchronize()
        profile.host_seconds["device_to_host_transfer"] += (
            perf_counter() - transfer_started
        )
        profile.record_cuda_elapsed("device_to_host_transfer", copy_start, copy_end)

        def extract_results() -> tuple[PolicyValueEstimate, ...]:
            probabilities_list = cpu_probabilities.tolist()
            values_list = cpu_values.tolist()
            return tuple(
                PolicyValueEstimate(
                    {
                        move: position_probabilities[action_index]
                        for move, action_index in zip(moves, indices)
                    },
                    value,
                )
                for moves, indices, position_probabilities, value in zip(
                    move_batches,
                    action_indices,
                    probabilities_list,
                    values_list,
                )
            )

        estimates = profile.host_call("python_result_extraction", extract_results)
        profile.batches += 1
        profile.positions += len(requests)
        profile.total_evaluate_batch_seconds += perf_counter() - total_started
        return estimates


__all__ = ["CudaInferencePhaseProfile", "ProfiledCudaNeuralPolicyValue"]
