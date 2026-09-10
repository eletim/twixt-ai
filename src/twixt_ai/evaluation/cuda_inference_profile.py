"""CUDA and contention measurements for the canonical inference observers."""

from __future__ import annotations

from threading import Lock
from time import perf_counter

import torch


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

_CUDA_NAMES = {
    "host_to_device": "host_to_device_transfer",
    "model": "model_execution",
    "policy_mask": "policy_mask",
    "softmax": "softmax",
}


def _percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int((len(ordered) - 1) * percent)
    return ordered[index]


def _duration_summary(values: list[float], scale: float) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "total_seconds": sum(values),
        "average": sum(values) / len(values) * scale if values else 0.0,
        "p50": _percentile(values, 0.50) * scale,
        "p95": _percentile(values, 0.95) * scale,
        "p99": _percentile(values, 0.99) * scale,
        "maximum": max(values, default=0.0) * scale,
    }


class CudaInferencePhaseProfile:
    """Implement inference and batcher observers with diagnostic timing."""

    def __init__(self) -> None:
        self.batches = 0
        self.positions = 0
        self.total_evaluate_batch_seconds = 0.0
        self.host_seconds = {phase: 0.0 for phase in _HOST_PHASES}
        self.cuda_seconds = {phase: 0.0 for phase in _CUDA_PHASES}
        self._pending_cuda_events: list[
            tuple[str, torch.cuda.Event, torch.cuda.Event]
        ] = []
        self._contention_lock = Lock()
        self._queue_lock_waits: list[float] = []
        self._queue_critical_sections: list[float] = []
        self._dispatch_lock_waits: list[float] = []
        self._dispatch_critical_sections: list[float] = []
        self._completion_lock_waits: list[float] = []
        self._completion_critical_sections: list[float] = []
        self._formation_by_reason: dict[str, list[float]] = {
            "full_batch": [],
            "latency": [],
            "forced": [],
        }

    def start_host_phase(self, phase: str) -> object:
        return perf_counter()

    def finish_host_phase(self, phase: str, token: object) -> None:
        self.host_seconds[phase] += perf_counter() - float(token)

    def start_cuda_phase(self, phase: str) -> object:
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        host_started = perf_counter()
        started.record()
        return phase, host_started, started, finished

    def finish_cuda_phase(self, phase: str, token: object) -> None:
        token_phase, host_started, started, finished = token  # type: ignore[misc]
        if token_phase != phase:
            raise RuntimeError("CUDA inference observer phase mismatch")
        finished.record()
        self.host_seconds[f"{phase}_submission"] += perf_counter() - host_started
        self._pending_cuda_events.append((_CUDA_NAMES[phase], started, finished))

    def prepare_result_tensors(
        self, probabilities: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._pending_cuda_events:
            raise RuntimeError("CUDA inference observer has no pending device work")
        sync_started = perf_counter()
        self._pending_cuda_events[-1][2].synchronize()
        self.host_seconds["cuda_synchronization_wait"] += (
            perf_counter() - sync_started
        )
        for phase, started, finished in self._pending_cuda_events:
            self.cuda_seconds[phase] += started.elapsed_time(finished) / 1000.0
        self._pending_cuda_events.clear()

        transfer_started = perf_counter()
        copy_start = torch.cuda.Event(enable_timing=True)
        copy_end = torch.cuda.Event(enable_timing=True)
        copy_start.record()
        cpu_probabilities = probabilities.to("cpu")
        cpu_values = values.to("cpu")
        copy_end.record()
        copy_end.synchronize()
        self.host_seconds["device_to_host_transfer"] += (
            perf_counter() - transfer_started
        )
        self.cuda_seconds["device_to_host_transfer"] += (
            copy_start.elapsed_time(copy_end) / 1000.0
        )
        return cpu_probabilities, cpu_values

    def batch_completed(self, positions: int, total_seconds: float) -> None:
        self.batches += 1
        self.positions += positions
        self.total_evaluate_batch_seconds += total_seconds

    def queue_submission(
        self, lock_wait_seconds: float, critical_section_seconds: float
    ) -> None:
        with self._contention_lock:
            self._queue_lock_waits.append(lock_wait_seconds)
            self._queue_critical_sections.append(critical_section_seconds)

    def batch_dispatch(
        self,
        flush_reason: str,
        formation_seconds: float,
        lock_wait_seconds: float,
        critical_section_seconds: float,
    ) -> None:
        with self._contention_lock:
            self._formation_by_reason[flush_reason].append(formation_seconds)
            self._dispatch_lock_waits.append(lock_wait_seconds)
            self._dispatch_critical_sections.append(critical_section_seconds)

    def batch_completion(
        self, lock_wait_seconds: float, critical_section_seconds: float
    ) -> None:
        with self._contention_lock:
            self._completion_lock_waits.append(lock_wait_seconds)
            self._completion_critical_sections.append(critical_section_seconds)

    def contention_to_dict(self) -> dict[str, object]:
        with self._contention_lock:
            queue_waits = list(self._queue_lock_waits)
            queue_critical = list(self._queue_critical_sections)
            dispatch_waits = list(self._dispatch_lock_waits)
            dispatch_critical = list(self._dispatch_critical_sections)
            completion_waits = list(self._completion_lock_waits)
            completion_critical = list(self._completion_critical_sections)
            formations = {
                reason: list(values)
                for reason, values in self._formation_by_reason.items()
            }
        all_formations = [value for values in formations.values() for value in values]
        return {
            "method": (
                "direct perf_counter timing of Condition acquisition and lock-held "
                "critical sections; batch formation is timed separately from first "
                "queued request until dispatch"
            ),
            "units": {
                "condition_measurements": "microseconds",
                "batch_formation_measurements": "milliseconds",
            },
            "producer_queue_condition_lock_acquisition": _duration_summary(
                queue_waits, 1_000_000.0
            ),
            "producer_queue_condition_critical_section": _duration_summary(
                queue_critical, 1_000_000.0
            ),
            "worker_dispatch_condition_lock_acquisition": _duration_summary(
                dispatch_waits, 1_000_000.0
            ),
            "worker_dispatch_condition_critical_section_excluding_wait": (
                _duration_summary(dispatch_critical, 1_000_000.0)
            ),
            "worker_completion_condition_lock_acquisition": _duration_summary(
                completion_waits, 1_000_000.0
            ),
            "worker_completion_condition_critical_section": _duration_summary(
                completion_critical, 1_000_000.0
            ),
            "batch_formation_delay": {
                "all": _duration_summary(all_formations, 1_000.0),
                "by_flush_reason": {
                    reason: _duration_summary(values, 1_000.0)
                    for reason, values in formations.items()
                },
            },
            "interpretation": {
                "summed_thread_times_are_not_wall_time": (
                    "Producer acquisition and critical-section totals sum concurrent "
                    "requests and must not be added to end-to-end wall time."
                ),
                "observer_overhead": (
                    "Recording each sample uses a separate profiler lock after the "
                    "production Condition is released; percentiles are diagnostic."
                ),
            },
        }

    def to_dict(self) -> dict[str, object]:
        accounted_host = sum(self.host_seconds.values())
        unaccounted = max(0.0, self.total_evaluate_batch_seconds - accounted_host)
        host = dict(self.host_seconds)
        host["unaccounted_profiler_overhead"] = unaccounted
        ranked = sorted(host, key=host.__getitem__, reverse=True)
        return {
            "method": (
                "observer spans on NeuralPolicyValue.evaluate_batch plus CUDA events; "
                "an explicit CUDA event synchronization at its Tensor.tolist result "
                "boundary separates pending device work, device-to-host copies, and "
                "Python extraction"
            ),
            "source_of_truth": (
                "The observer measures NeuralPolicyValue.evaluate_batch directly; no "
                "second evaluator or inference implementation is used."
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
            "batching_and_contention": self.contention_to_dict(),
            "interpretation": {
                "host_and_cuda_times_are_not_additive": (
                    "CUDA kernels and copies are asynchronous and can overlap host wall "
                    "time; event durations identify device work rather than another "
                    "exclusive wall-time partition."
                ),
                "profiling_overhead": (
                    "CUDA events, observer calls, and phase clocks add overhead, so "
                    "profile wall time is diagnostic, not optimized throughput."
                ),
            },
        }


__all__ = ["CudaInferencePhaseProfile"]
