"""Tests for the matched Mini encoding cost benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from twixt_ai.evaluation.encoding_comparison import (
    ENCODING_COMPARISON_FORMAT,
    EncodingComparisonConfig,
    run_encoding_comparison_benchmark,
)
from twixt_ai.evaluation.encoding_comparison_cli import main


DATASET = Path(__file__).parents[2] / "experiments/issue-56/smoke/dataset"


def _small_config() -> EncodingComparisonConfig:
    return EncodingComparisonConfig(
        dataset_dir=str(DATASET),
        batch_size=2,
        encoding_repeats=1,
        forward_iterations=1,
        training_steps=1,
        warmups=0,
        devices=("cpu",),
        torch_threads=1,
    )


def test_report_separates_encoding_forward_training_and_memory() -> None:
    report = run_encoding_comparison_benchmark(_small_config())

    assert report["format"] == ENCODING_COMPARISON_FORMAT
    assert report["dataset"]["training_examples"] == 33
    assert report["dataset"]["encoding_positions_per_repeat"] == 33
    assert report["dataset"]["model_batch_examples"] == 2
    assert report["interpretation"].endswith("playing strength.")
    old = report["results"]["22_plane_v1"]
    new = report["results"]["10_plane_v2"]
    assert (old["planes"], new["planes"]) == (22, 10)
    assert old["encoding"]["bytes_per_position"] == 22 * 10 * 10 * 4
    assert new["encoding"]["bytes_per_position"] == 10 * 10 * 10 * 4
    assert old["model"]["config"]["channels"] == new["model"]["config"]["channels"]
    for result in (old, new):
        device = result["devices"]["cpu"]
        assert device["single_position_forward"]["positions_per_second"] > 0
        assert device["batched_forward"]["positions_per_second"] > 0
        assert device["training"]["steps_per_second"] > 0
    assert report["comparison"]["encoding_bytes_reduction_percent"] == pytest.approx(
        100 * 12 / 22
    )
    json.dumps(report)


def test_cli_writes_versioned_report(tmp_path: Path) -> None:
    output = tmp_path / "comparison.json"

    assert main([
        "--dataset", str(DATASET), "--output", str(output), "--batch-size", "2",
        "--encoding-repeats", "1", "--forward-iterations", "1",
        "--training-steps", "1", "--warmups", "0", "--device", "cpu",
    ]) == 0

    assert json.loads(output.read_text(encoding="utf-8"))["format"] == ENCODING_COMPARISON_FORMAT


@pytest.mark.parametrize("name", ["batch_size", "encoding_repeats", "training_steps"])
def test_config_rejects_non_positive_workloads(name: str) -> None:
    values = {"dataset_dir": str(DATASET), name: 0}
    with pytest.raises(ValueError, match=name):
        EncodingComparisonConfig(**values)  # type: ignore[arg-type]
