"""Tests for reproducible canonical engine microbenchmarks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from twixt_ai.game import (
    ENGINE_BENCHMARK_FORMAT,
    BoardDimensions,
    EngineBenchmarkConfig,
    run_engine_benchmarks,
)
from twixt_ai.game.performance_cli import main


def _small_config() -> EngineBenchmarkConfig:
    return EngineBenchmarkConfig(
        board=BoardDimensions(8, 8),
        seed=24,
        ply=8,
        iterations=1,
        repeats=1,
        warmups=0,
    )


def test_engine_benchmark_covers_canonical_hot_paths() -> None:
    artifact = run_engine_benchmarks(_small_config())

    assert artifact["format"] == ENGINE_BENCHMARK_FORMAT
    assert artifact["config"] == _small_config().to_dict()
    assert artifact["fixture"]["pegs"] == 8
    assert set(artifact["benchmarks"]) == {
        "legal_move_generation",
        "automatic_link_updates",
        "win_check",
        "move_application",
    }
    assert artifact["positions_per_second"] > 0
    for result in artifact["benchmarks"].values():
        assert result["iterations_per_repeat"] == 1
        assert result["calls_per_second"] > 0


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"ply": -1}, "ply"),
        ({"iterations": 0}, "iterations"),
        ({"repeats": 0}, "repeats"),
        ({"warmups": -1}, "warmups"),
    ],
)
def test_engine_benchmark_rejects_invalid_workloads(
    changes: dict[str, int], message: str
) -> None:
    values = {"ply": 1, "iterations": 1, "repeats": 1, "warmups": 0}
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        EngineBenchmarkConfig(**values)


def test_engine_benchmark_cli_writes_machine_readable_artifact(tmp_path: Path) -> None:
    output = tmp_path / "engine-benchmark.json"

    assert main(
        [
            "--width", "8", "--height", "8", "--seed", "24", "--ply", "8",
            "--iterations", "1", "--repeats", "1", "--warmups", "0",
            "--output", str(output),
        ]
    ) == 0

    artifact = json.loads(output.read_text())
    assert artifact["format"] == ENGINE_BENCHMARK_FORMAT
    assert artifact["config"]["seed"] == 24
