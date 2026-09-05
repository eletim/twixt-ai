"""Tests for parallel, persisted self-play batches."""

from __future__ import annotations

import json
from pathlib import Path

from twixt_ai.agents import AgentRequest, RandomAgent
from twixt_ai.game import BoardDimensions
from twixt_ai.selfplay import BatchConfig, run_batch
from twixt_ai.selfplay.cli import main


class BrokenAgent:
    def choose_move(self, request: AgentRequest) -> object:
        del request
        raise RuntimeError("intentional failure")


def test_seeded_parallel_batch_persists_games_and_summary(tmp_path: Path) -> None:
    config = BatchConfig(
        games=4,
        workers=2,
        seed=2024,
        board=BoardDimensions(4, 4),
        red_agent="random-a",
        black_agent="random-b",
    )

    summary = run_batch(RandomAgent, RandomAgent, config=config, output_dir=tmp_path)

    assert summary.completed == 4
    assert summary.failed == 0
    assert [report.index for report in summary.games] == list(range(4))
    assert len({report.seed for report in summary.games}) == 4
    persisted = json.loads((tmp_path / "summary.json").read_text())
    assert persisted == summary.to_dict()
    for report in summary.games:
        artifact = json.loads((tmp_path / report.artifact).read_text())
        assert artifact["format"] == "twixt-ai-match"
        assert artifact["config"]["seed"] == report.seed
        assert len(artifact["decisions"]) == report.move_count


def test_worker_failures_are_isolated_and_reported(tmp_path: Path) -> None:
    config = BatchConfig(
        games=3,
        workers=2,
        seed=7,
        board=BoardDimensions(4, 4),
        red_agent="broken",
        black_agent="random",
    )

    summary = run_batch(BrokenAgent, RandomAgent, config=config, output_dir=tmp_path)

    assert summary.completed == 0
    assert summary.failed == 3
    for report in summary.games:
        assert report.error_type == "RuntimeError"
        artifact = json.loads((tmp_path / report.artifact).read_text())
        assert artifact["format"] == "twixt-ai-selfplay-failure"
        assert artifact["error"]["message"] == "intentional failure"


def test_seeded_output_is_independent_of_worker_count(tmp_path: Path) -> None:
    common = dict(
        games=3,
        seed=99,
        board=BoardDimensions(4, 4),
        red_agent="random",
        black_agent="random",
    )
    sequential = run_batch(
        RandomAgent,
        RandomAgent,
        config=BatchConfig(workers=1, **common),
        output_dir=tmp_path / "sequential",
    )
    parallel = run_batch(
        RandomAgent,
        RandomAgent,
        config=BatchConfig(workers=2, **common),
        output_dir=tmp_path / "parallel",
    )

    assert [report.seed for report in sequential.games] == [
        report.seed for report in parallel.games
    ]
    for left, right in zip(sequential.games, parallel.games):
        assert (tmp_path / "sequential" / left.artifact).read_text() == (
            tmp_path / "parallel" / right.artifact
        ).read_text()


def test_cli_writes_machine_readable_summary(tmp_path: Path, capsys: object) -> None:
    output = tmp_path / "run"

    assert main([
        "--games", "2", "--workers", "1", "--seed", "5",
        "--width", "4", "--height", "4", "--output-dir", str(output),
    ]) == 0

    emitted = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert emitted["aggregate"]["completed"] == 2
    assert emitted == json.loads((output / "summary.json").read_text())
