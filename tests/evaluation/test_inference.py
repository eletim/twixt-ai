from __future__ import annotations

import pytest

from twixt_ai.evaluation import inference
from twixt_ai.evaluation.inference import (
    INFERENCE_PERFORMANCE_FORMAT,
    InferencePerformanceConfig,
    run_inference_performance_benchmark,
)


def test_inference_benchmark_reports_throughput_utilization_and_batching() -> None:
    config = InferencePerformanceConfig(
        requests=4,
        batch_size=2,
        warmups=0,
        max_wait_seconds=0.01,
        device="cpu",
    )

    report = run_inference_performance_benchmark(config)

    assert report["format"] == INFERENCE_PERFORMANCE_FORMAT
    assert report["config"] == config.to_dict()
    assert report["environment"]["accelerator"]["type"] == "cpu"
    assert report["environment"]["device"]["requested_device"] == "cpu"
    assert report["environment"]["device"]["resolved_device"] == "cpu"
    assert report["synchronous"]["positions_per_second"] > 0
    assert report["batched"]["positions_per_second"] > 0
    assert report["batched"]["cpu_utilization_percent"] >= 0
    assert report["batch_statistics"]["requests"] == 4
    assert report["batch_statistics"]["maximum_batch_size"] == 2


def test_inference_benchmark_snapshots_statistics_after_batcher_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DelayedStatistics:
        def __init__(self, batcher: DelayedStatisticsBatcher) -> None:
            self.batcher = batcher

        def to_dict(self) -> dict[str, int]:
            return {"requests": int(self.batcher.closed)}

    class DelayedStatisticsBatcher:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.closed = False

        @property
        def statistics(self) -> DelayedStatistics:
            return DelayedStatistics(self)

        def __enter__(self) -> DelayedStatisticsBatcher:
            return self

        def __exit__(self, *args: object) -> None:
            self.closed = True

        def __call__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(
        inference, "NeuralInferenceBatcher", DelayedStatisticsBatcher
    )
    config = InferencePerformanceConfig(
        requests=1,
        batch_size=1,
        warmups=0,
        device="cpu",
    )

    report = run_inference_performance_benchmark(config)

    assert report["batch_statistics"] == {"requests": 1}
