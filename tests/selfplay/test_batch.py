"""Tests for parallel, persisted self-play batches."""

from __future__ import annotations

from functools import partial
import json
import os
from pathlib import Path

from twixt_ai.agents import AgentRequest, AgentResult, RandomAgent
from twixt_ai.game import BoardDimensions
from twixt_ai.models import PolicyValueConfig, PolicyValueNetwork
from twixt_ai.search import MCTSAgent
from twixt_ai.search.neural import NeuralInferenceBatcher, NeuralPolicyValue
from twixt_ai.selfplay import BatchConfig, run_batch
from twixt_ai.selfplay.cli import main


class BrokenAgent:
    def choose_move(self, request: AgentRequest) -> object:
        del request
        raise RuntimeError("intentional failure")


class CrashOnceAgent:
    """Terminate one worker once, then behave normally after pool recovery."""

    def __init__(self, marker: str) -> None:
        self.marker = marker

    def choose_move(self, request: AgentRequest) -> AgentResult:
        try:
            descriptor = os.open(
                self.marker,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError:
            return RandomAgent().choose_move(request)
        os.close(descriptor)
        os._exit(17)


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


def test_abrupt_worker_exit_still_produces_complete_manifest(tmp_path: Path) -> None:
    output = tmp_path / "run"
    config = BatchConfig(
        games=5,
        workers=2,
        seed=11,
        board=BoardDimensions(4, 4),
        red_agent="crash-once",
        black_agent="random",
    )

    summary = run_batch(
        partial(CrashOnceAgent, str(tmp_path / "crashed")),
        RandomAgent,
        config=config,
        output_dir=output,
    )

    assert summary.failed >= 1
    assert summary.completed >= 1
    assert summary.failed + summary.completed == config.games
    assert len(tuple((output / "games").glob("*.json"))) == config.games
    assert json.loads((output / "summary.json").read_text()) == summary.to_dict()


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


def test_thread_workers_support_shared_in_process_state(tmp_path: Path) -> None:
    calls: list[int] = []

    def factory() -> RandomAgent:
        calls.append(1)
        return RandomAgent()

    config = BatchConfig(
        games=3,
        workers=2,
        seed=54,
        board=BoardDimensions(4, 4),
        worker_mode="thread",
    )

    summary = run_batch(factory, factory, config=config, output_dir=tmp_path)

    assert summary.completed == 3
    assert len(calls) == 6
    assert summary.to_dict()["config"]["worker_mode"] == "thread"


def test_concurrent_selfplay_games_share_neural_batches(tmp_path: Path) -> None:
    model = PolicyValueNetwork(
        PolicyValueConfig(
            channels=2,
            residual_blocks=1,
            value_hidden=4,
            board_width=4,
            board_height=4,
        )
    )
    config = BatchConfig(
        games=2,
        workers=2,
        seed=54,
        board=BoardDimensions(4, 4),
        red_agent="batched-mcts",
        black_agent="batched-mcts",
        worker_mode="thread",
    )

    with NeuralInferenceBatcher(
        NeuralPolicyValue(model), batch_size=2, max_wait_seconds=0.05
    ) as batcher:
        factory = partial(MCTSAgent, simulations=1, policy_value=batcher)
        summary = run_batch(
            factory,
            factory,
            config=config,
            output_dir=tmp_path,
        )
        statistics = batcher.statistics

    assert summary.completed == 2
    assert statistics.requests > 2
    assert statistics.maximum_batch_size == 2


def test_cli_writes_machine_readable_summary(tmp_path: Path, capsys: object) -> None:
    output = tmp_path / "run"

    assert main([
        "--games", "2", "--workers", "1", "--seed", "5",
        "--width", "4", "--height", "4", "--output-dir", str(output),
    ]) == 0

    emitted = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert emitted["aggregate"]["completed"] == 2
    assert emitted == json.loads((output / "summary.json").read_text())
