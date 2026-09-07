from __future__ import annotations

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
    assert report["synchronous"]["positions_per_second"] > 0
    assert report["batched"]["positions_per_second"] > 0
    assert report["batched"]["cpu_utilization_percent"] >= 0
    assert report["batch_statistics"]["requests"] == 4
    assert report["batch_statistics"]["maximum_batch_size"] == 2
