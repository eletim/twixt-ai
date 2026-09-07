"""Tests for measured Mini Twixt throughput reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from twixt_ai.evaluation import (
    SELFPLAY_PERFORMANCE_FORMAT,
    SelfPlayPerformanceConfig,
    run_selfplay_performance_benchmark,
)
from twixt_ai.evaluation.performance_cli import main
from twixt_ai.game import BoardDimensions


def _small_config() -> SelfPlayPerformanceConfig:
    return SelfPlayPerformanceConfig(
        board=BoardDimensions(4, 4),
        seed=52,
        fixture_ply=2,
        simulation_budgets=(1, 2),
        move_repeats=1,
        games_per_budget=1,
        scaling_games=2,
        worker_counts=(1, 2),
        scaling_simulations=1,
        rollout_limit=1,
        engine_iterations=1,
        engine_repeats=1,
        engine_warmups=0,
    )


def test_report_covers_engine_search_games_scaling_and_costs() -> None:
    report = run_selfplay_performance_benchmark(_small_config())

    assert report["format"] == SELFPLAY_PERFORMANCE_FORMAT
    assert report["config"] == _small_config().to_dict()
    assert report["environment"]["available_cpus"] >= 1
    baseline = report["nn_free_baseline"]
    assert baseline["policy_value_network"] is False
    assert set(baseline["engine"]["benchmarks"]) >= {
        "legal_move_generation",
        "move_application",
    }
    assert baseline["engine"]["cpu_utilization"]["aggregate_percent"] >= 0
    assert [item["simulations"] for item in baseline["mcts_moves"]] == [1, 2]
    assert [item["workers"] for item in baseline["worker_scaling"]] == [1, 2]
    assert baseline["worker_scaling"][0]["speedup_vs_sequential"] == 1
    for item in baseline["mcts_moves"]:
        assert item["simulations_per_second"] > 0
        assert item["nodes_per_second"] > 0
        assert item["move_latency_seconds"] > 0
        assert item["cpu_utilization"]["aggregate_percent"] >= 0
    for item in baseline["full_games"]:
        assert item["seconds_per_game"] > 0
        assert item["games_per_hour"] > 0
        assert item["estimated_wall_clock_hours"]["1000_games"] > 0
        assert item["estimated_wall_clock_hours"]["10000_games"] > 0
    assert {item["area"] for item in report["identified_bottlenecks"]} == {
        "engine",
        "mcts",
        "selfplay_scaling",
    }
    json.dumps(report)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"simulation_budgets": ()}, "must not be empty"),
        ({"simulation_budgets": (1, 1)}, "unique"),
        ({"worker_counts": (2, 1)}, "begin with 1"),
        ({"fixture_ply": -1}, "fixture_ply"),
        ({"move_repeats": 0}, "move_repeats"),
    ],
)
def test_config_rejects_invalid_workloads(
    changes: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "simulation_budgets": (1,),
        "worker_counts": (1,),
    }
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        SelfPlayPerformanceConfig(**values)  # type: ignore[arg-type]


def test_cli_defaults_to_mini_and_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "mini-performance.json"

    assert main(
        [
            "--simulations", "1", "--move-repeats", "1",
            "--games-per-budget", "1", "--scaling-games", "1",
            "--workers", "1", "--scaling-simulations", "1",
            "--rollout-limit", "1", "--engine-iterations", "1",
            "--engine-repeats", "1", "--engine-warmups", "0",
            "--output", str(output),
        ]
    ) == 0

    report = json.loads(output.read_text())
    assert report["config"]["board"] == {"width": 10, "height": 10}
    assert report["format"] == SELFPLAY_PERFORMANCE_FORMAT
