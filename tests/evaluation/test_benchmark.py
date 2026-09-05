"""Tests for reproducible paired agent benchmarks."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from twixt_ai.agents import RandomAgent
from twixt_ai.evaluation import (
    AgentConfig,
    BenchmarkConfig,
    BenchmarkGame,
    BenchmarkResult,
    run_benchmark,
)
from twixt_ai.evaluation.benchmark_cli import main
from twixt_ai.game import BoardDimensions, Player


def _config(*, agents: tuple[AgentConfig, ...] | None = None, elo: bool = False) -> BenchmarkConfig:
    return BenchmarkConfig(
        agents=agents
        or (
            AgentConfig("alpha", "1.0", {"policy": "random"}),
            AgentConfig("beta", "2.0", {"policy": "random"}),
        ),
        games_per_pair=2,
        board=BoardDimensions(4, 4),
        seed=1234,
        include_elo=elo,
    )


def test_head_to_head_swaps_roles_using_the_same_seed() -> None:
    result = run_benchmark(
        {"alpha": RandomAgent, "beta": RandomAgent}, config=_config()
    )

    assert len(result.games) == 2
    first, second = result.games
    assert first.seed == second.seed
    assert (first.red_agent, first.black_agent) == ("alpha", "beta")
    assert (second.red_agent, second.black_agent) == ("beta", "alpha")
    assert first.pair_index == second.pair_index == 0
    assert first.pair_round == second.pair_round == 0


def test_round_robin_is_complete_and_seeded_results_are_reproducible() -> None:
    agents = (
        AgentConfig("alpha", "1", {}),
        AgentConfig("beta", "1", {}),
        AgentConfig("gamma", "1", {}),
    )
    config = _config(agents=agents)
    factories = {agent.name: RandomAgent for agent in agents}

    first = run_benchmark(factories, config=config)
    second = run_benchmark(factories, config=config)

    assert len(first.games) == 6
    assert first == second
    assert first.to_json() == second.to_json()
    assert {frozenset((game.red_agent, game.black_agent)) for game in first.games} == {
        frozenset(("alpha", "beta")),
        frozenset(("alpha", "gamma")),
        frozenset(("beta", "gamma")),
    }


def test_artifact_records_configs_statistics_confidence_and_first_player_bias() -> None:
    result = run_benchmark(
        {"alpha": RandomAgent, "beta": RandomAgent}, config=_config(elo=True)
    )
    artifact = result.to_dict()

    assert artifact["format"] == "twixt-ai-benchmark"
    assert artifact["config"]["agents"] == [
        {"name": "alpha", "version": "1.0", "configuration": {"policy": "random"}},
        {"name": "beta", "version": "2.0", "configuration": {"policy": "random"}},
    ]
    summary = artifact["summary"]
    alpha = summary["agents"]["alpha"]
    assert alpha["overall"]["games"] == 2
    assert alpha["red"]["games"] == alpha["black"]["games"] == 1
    assert 0.0 <= alpha["overall"]["win_rate"]["lower"] <= 1.0
    assert 0.0 <= alpha["overall"]["win_rate"]["upper"] <= 1.0
    assert summary["first_player"]["side"] == "red"
    assert summary["first_player"]["games"] == 2
    assert set(summary["elo"]["ratings"]) == {"alpha", "beta"}
    assert abs(sum(summary["elo"]["ratings"].values()) - 3000.0) < 1e-9


def test_agent_configuration_is_copied_and_recursively_immutable() -> None:
    source = {"layers": [32, 16]}
    agent = AgentConfig("agent", "v1", source)
    source["layers"].append(8)

    assert agent.to_dict()["configuration"] == {"layers": [32, 16]}
    with pytest.raises(TypeError):
        agent.configuration["new"] = True  # type: ignore[index]
    with pytest.raises(AttributeError):
        agent.configuration["layers"].append(8)  # type: ignore[union-attr]


def test_config_requires_balanced_games_and_exact_factory_names() -> None:
    with pytest.raises(ValueError, match="must be even"):
        BenchmarkConfig(
            agents=(AgentConfig("a", "1"), AgentConfig("b", "1")),
            games_per_pair=3,
        )
    with pytest.raises(ValueError, match="exactly match"):
        run_benchmark({"alpha": RandomAgent}, config=_config())


def _replace_assignment(
    game: BenchmarkGame, red: str, black: str, **changes: object
) -> BenchmarkGame:
    winner = (
        red
        if game.winning_side is Player.RED
        else black if game.winning_side is Player.BLACK else None
    )
    return replace(
        game,
        red_agent=red,
        black_agent=black,
        winner=winner,
        **changes,
    )


def test_result_rejects_unknown_agents_before_summarizing() -> None:
    valid = run_benchmark(
        {"alpha": RandomAgent, "beta": RandomAgent}, config=_config()
    )
    games = list(valid.games)
    games[0] = _replace_assignment(games[0], "intruder", "beta")

    with pytest.raises(ValueError, match="only configured agents"):
        BenchmarkResult(valid.config, tuple(games))


def test_result_rejects_repeated_pair_in_incomplete_round_robin() -> None:
    agents = (
        AgentConfig("alpha", "1"),
        AgentConfig("beta", "1"),
        AgentConfig("gamma", "1"),
    )
    config = _config(agents=agents)
    valid = run_benchmark(
        {agent.name: RandomAgent for agent in agents}, config=config
    )
    repeated = tuple(
        replace(valid.games[index % 2], index=index)
        for index in range(len(valid.games))
    )

    with pytest.raises(ValueError, match="every configured pair and round"):
        BenchmarkResult(config, repeated)


def test_result_rejects_missing_round_role_swap_or_paired_seed() -> None:
    valid = run_benchmark(
        {"alpha": RandomAgent, "beta": RandomAgent}, config=_config()
    )

    wrong_round = (replace(valid.games[0], pair_round=1), valid.games[1])
    with pytest.raises(ValueError, match="every configured pair and round"):
        BenchmarkResult(valid.config, wrong_round)

    duplicate_role = (
        valid.games[0],
        _replace_assignment(
            valid.games[1],
            valid.games[0].red_agent,
            valid.games[0].black_agent,
        ),
    )
    with pytest.raises(ValueError, match="both role assignments"):
        BenchmarkResult(valid.config, duplicate_role)

    different_seed = (
        valid.games[0],
        replace(valid.games[1], seed=valid.games[1].seed + 1),
    )
    with pytest.raises(ValueError, match="same seed"):
        BenchmarkResult(valid.config, different_seed)


def test_cli_writes_a_reproducible_machine_readable_artifact(tmp_path: Path) -> None:
    output = tmp_path / "benchmark.json"

    assert main([
        "--agent", "baseline=random",
        "--agent", "candidate=random",
        "--games-per-pair", "2",
        "--width", "4",
        "--height", "4",
        "--seed", "88",
        "--elo",
        "--output", str(output),
    ]) == 0

    artifact = json.loads(output.read_text())
    assert artifact["config"]["seed"] == 88
    assert [agent["name"] for agent in artifact["config"]["agents"]] == [
        "baseline", "candidate"
    ]
    assert artifact["summary"]["first_player"]["games"] == 2
    assert "elo" in artifact["summary"]
