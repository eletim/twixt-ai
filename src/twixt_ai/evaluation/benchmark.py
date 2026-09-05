"""Reproducible round-robin benchmarks for Twixt agents."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
import math
from random import Random
from statistics import NormalDist
from types import MappingProxyType

from twixt_ai.agents import Agent
from twixt_ai.game import BoardDimensions, Player

from .match import MatchConfig, run_match


BENCHMARK_FORMAT = "twixt-ai-benchmark"
BENCHMARK_FORMAT_VERSION = 1
AgentFactory = Callable[[], Agent]


def _positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _validate_json(value: object, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"{path} must contain only finite JSON numbers")
        return
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{path} must contain only string-keyed objects")
        for key, item in value.items():
            _validate_json(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json(item, f"{path}[{index}]")
        return
    raise TypeError(
        f"{path} contains non-JSON-compatible value of type {type(value).__name__}"
    )


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _json_mapping(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    _validate_json(value, name)
    restored = _freeze_json(value)
    assert isinstance(restored, Mapping)
    return restored


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Stable identity, version, and construction settings for one entrant."""

    name: str
    version: str
    configuration: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("version must be a non-empty string")
        object.__setattr__(
            self,
            "configuration",
            _json_mapping(self.configuration, "configuration"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "configuration": _thaw_json(self.configuration),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Serializable settings defining a complete paired round robin."""

    agents: tuple[AgentConfig, ...]
    games_per_pair: int = 2
    board: BoardDimensions = BoardDimensions()
    seed: int = 0
    confidence_level: float = 0.95
    include_elo: bool = False

    def __post_init__(self) -> None:
        agents = tuple(self.agents)
        if len(agents) < 2:
            raise ValueError("agents must contain at least two entrants")
        if any(not isinstance(agent, AgentConfig) for agent in agents):
            raise TypeError("agents must contain only AgentConfig values")
        names = [agent.name for agent in agents]
        if len(set(names)) != len(names):
            raise ValueError("agent names must be unique")
        _positive_integer(self.games_per_pair, "games_per_pair")
        if self.games_per_pair % 2:
            raise ValueError("games_per_pair must be even so player roles can be swapped")
        if not isinstance(self.board, BoardDimensions):
            raise TypeError("board must be BoardDimensions")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if (
            isinstance(self.confidence_level, bool)
            or not isinstance(self.confidence_level, (int, float))
            or not 0.0 < self.confidence_level < 1.0
        ):
            raise ValueError("confidence_level must be between zero and one")
        if not isinstance(self.include_elo, bool):
            raise TypeError("include_elo must be a boolean")
        object.__setattr__(self, "agents", agents)

    def to_dict(self) -> dict[str, object]:
        return {
            "agents": [agent.to_dict() for agent in self.agents],
            "games_per_pair": self.games_per_pair,
            "board": self.board.to_dict(),
            "seed": self.seed,
            "confidence_level": self.confidence_level,
            "include_elo": self.include_elo,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkGame:
    """Compact result for one scheduled game."""

    index: int
    pair_index: int
    pair_round: int
    seed: int
    red_agent: str
    black_agent: str
    winner: str | None
    winning_side: Player | None
    move_count: int

    def __post_init__(self) -> None:
        for name in ("index", "pair_index", "pair_round"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if (
            not isinstance(self.red_agent, str)
            or not self.red_agent
            or not isinstance(self.black_agent, str)
            or not self.black_agent
            or self.red_agent == self.black_agent
        ):
            raise ValueError("red_agent and black_agent must be distinct non-empty strings")
        if self.winner not in (None, self.red_agent, self.black_agent):
            raise ValueError("winner must be one of the assigned agents or None")
        if self.winning_side is not None and not isinstance(self.winning_side, Player):
            raise TypeError("winning_side must be a Player or None")
        expected_winner = (
            self.red_agent
            if self.winning_side is Player.RED
            else self.black_agent if self.winning_side is Player.BLACK else None
        )
        if self.winner != expected_winner:
            raise ValueError("winner must agree with winning_side")
        if (
            isinstance(self.move_count, bool)
            or not isinstance(self.move_count, int)
            or self.move_count < 0
        ):
            raise ValueError("move_count must be a non-negative integer")

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "pair_index": self.pair_index,
            "pair_round": self.pair_round,
            "seed": self.seed,
            "agents": {
                Player.RED.value: self.red_agent,
                Player.BLACK.value: self.black_agent,
            },
            "result": {
                "winner": self.winner,
                "winning_side": (
                    self.winning_side.value if self.winning_side is not None else None
                ),
                "move_count": self.move_count,
            },
        }


def _rate(wins: int, games: int, confidence_level: float) -> dict[str, object]:
    """Return a win rate and Wilson score interval for a binomial outcome."""

    if games == 0:
        return {"value": None, "lower": None, "upper": None}
    value = wins / games
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    denominator = 1.0 + z * z / games
    center = (value + z * z / (2.0 * games)) / denominator
    margin = z * math.sqrt(
        value * (1.0 - value) / games + z * z / (4.0 * games * games)
    ) / denominator
    return {
        "value": value,
        "lower": max(0.0, center - margin),
        "upper": min(1.0, center + margin),
    }


def _record(aggregate: dict[str, int], won: bool, lost: bool) -> None:
    aggregate["games"] += 1
    if won:
        aggregate["wins"] += 1
    elif lost:
        aggregate["losses"] += 1
    else:
        aggregate["draws"] += 1


def _empty_record() -> dict[str, int]:
    return {"games": 0, "wins": 0, "losses": 0, "draws": 0}


def _with_rates(record: Mapping[str, int], confidence_level: float) -> dict[str, object]:
    value: dict[str, object] = dict(record)
    games = record["games"]
    value["win_rate"] = _rate(record["wins"], games, confidence_level)
    value["draw_rate"] = record["draws"] / games if games else None
    return value


def _elo_ratings(games: tuple[BenchmarkGame, ...], names: tuple[str, ...]) -> dict[str, object]:
    """Calculate deterministic, schedule-ordered Elo ratings."""

    ratings = {name: 1500.0 for name in names}
    k_factor = 32.0
    for game in games:
        red_rating = ratings[game.red_agent]
        black_rating = ratings[game.black_agent]
        red_expected = 1.0 / (1.0 + 10.0 ** ((black_rating - red_rating) / 400.0))
        red_score = 0.5 if game.winner is None else float(game.winner == game.red_agent)
        change = k_factor * (red_score - red_expected)
        ratings[game.red_agent] += change
        ratings[game.black_agent] -= change
    return {
        "method": "sequential_elo",
        "initial_rating": 1500.0,
        "k_factor": k_factor,
        "ratings": {name: ratings[name] for name in names},
    }


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Completed benchmark schedule plus derived statistical summaries."""

    config: BenchmarkConfig
    games: tuple[BenchmarkGame, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.config, BenchmarkConfig):
            raise TypeError("config must be a BenchmarkConfig")
        games = tuple(self.games)
        names = tuple(agent.name for agent in self.config.agents)
        pairings = tuple(
            (left, right)
            for left_index, left in enumerate(names[:-1])
            for right in names[left_index + 1 :]
        )
        expected = len(pairings) * self.config.games_per_pair
        if len(games) != expected:
            raise ValueError("games must contain the complete round-robin schedule")
        if any(not isinstance(game, BenchmarkGame) for game in games):
            raise TypeError("games must contain only BenchmarkGame values")
        if tuple(game.index for game in games) != tuple(range(expected)):
            raise ValueError("game indices must be consecutive and ordered")
        configured_names = set(names)
        if any(
            game.red_agent not in configured_names
            or game.black_agent not in configured_names
            for game in games
        ):
            raise ValueError("games must contain only configured agents")

        scheduled: dict[tuple[int, int], list[BenchmarkGame]] = {}
        for game in games:
            scheduled.setdefault((game.pair_index, game.pair_round), []).append(game)
        expected_slots = {
            (pair_index, pair_round)
            for pair_index in range(len(pairings))
            for pair_round in range(self.config.games_per_pair // 2)
        }
        if set(scheduled) != expected_slots:
            raise ValueError("games must contain every configured pair and round")
        for (pair_index, _), paired_games in scheduled.items():
            left, right = pairings[pair_index]
            assignments = {
                (game.red_agent, game.black_agent) for game in paired_games
            }
            if len(paired_games) != 2 or assignments != {
                (left, right),
                (right, left),
            }:
                raise ValueError("each pair and round must contain both role assignments")
            if len({game.seed for game in paired_games}) != 1:
                raise ValueError("role-swapped games must use the same seed")
        object.__setattr__(self, "games", games)

    def _agent_summaries(self) -> dict[str, object]:
        records = {
            agent.name: {
                "overall": _empty_record(),
                Player.RED.value: _empty_record(),
                Player.BLACK.value: _empty_record(),
            }
            for agent in self.config.agents
        }
        for game in self.games:
            for name, side in (
                (game.red_agent, Player.RED),
                (game.black_agent, Player.BLACK),
            ):
                won = game.winner == name
                lost = game.winner is not None and not won
                _record(records[name]["overall"], won, lost)
                _record(records[name][side.value], won, lost)
        return {
            name: {
                key: _with_rates(record, self.config.confidence_level)
                for key, record in sections.items()
            }
            for name, sections in records.items()
        }

    def _pair_summaries(self) -> list[dict[str, object]]:
        output: list[dict[str, object]] = []
        names = [agent.name for agent in self.config.agents]
        for left_index, left in enumerate(names[:-1]):
            for right in names[left_index + 1 :]:
                pair_games = tuple(
                    game
                    for game in self.games
                    if {game.red_agent, game.black_agent} == {left, right}
                )
                left_wins = sum(game.winner == left for game in pair_games)
                right_wins = sum(game.winner == right for game in pair_games)
                output.append(
                    {
                        "agents": [left, right],
                        "games": len(pair_games),
                        "wins": {left: left_wins, right: right_wins},
                        "draws": len(pair_games) - left_wins - right_wins,
                        "win_rate": {
                            left: _rate(
                                left_wins,
                                len(pair_games),
                                self.config.confidence_level,
                            ),
                            right: _rate(
                                right_wins,
                                len(pair_games),
                                self.config.confidence_level,
                            ),
                        },
                    }
                )
        return output

    def to_dict(self) -> dict[str, object]:
        first_wins = sum(game.winning_side is Player.RED for game in self.games)
        second_wins = sum(game.winning_side is Player.BLACK for game in self.games)
        first_player = {
            "side": Player.RED.value,
            "games": len(self.games),
            "wins": first_wins,
            "losses": second_wins,
            "draws": len(self.games) - first_wins - second_wins,
            "win_rate": _rate(
                first_wins, len(self.games), self.config.confidence_level
            ),
        }
        summary: dict[str, object] = {
            "agents": self._agent_summaries(),
            "pairs": self._pair_summaries(),
            "first_player": first_player,
        }
        if self.config.include_elo:
            summary["elo"] = _elo_ratings(
                self.games, tuple(agent.name for agent in self.config.agents)
            )
        return {
            "format": BENCHMARK_FORMAT,
            "version": BENCHMARK_FORMAT_VERSION,
            "config": self.config.to_dict(),
            "summary": summary,
            "games": [game.to_dict() for game in self.games],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        separators = None if indent is not None else (",", ":")
        return json.dumps(self.to_dict(), sort_keys=True, separators=separators, indent=indent)


def run_benchmark(
    agent_factories: Mapping[str, AgentFactory], *, config: BenchmarkConfig
) -> BenchmarkResult:
    """Run a deterministic paired round robin.

    Every pairing uses each generated seed for two games, swapping the red
    (first-player) and black roles. A new agent instance is constructed for
    each game so stateful implementations cannot leak information between
    trials.
    """

    if not isinstance(agent_factories, Mapping):
        raise TypeError("agent_factories must be a mapping")
    if not isinstance(config, BenchmarkConfig):
        raise TypeError("config must be a BenchmarkConfig")
    names = tuple(agent.name for agent in config.agents)
    if set(agent_factories) != set(names):
        raise ValueError("agent_factories keys must exactly match configured agent names")
    if any(not callable(agent_factories[name]) for name in names):
        raise TypeError("agent factories must be callable")

    random_source = Random(config.seed)
    games: list[BenchmarkGame] = []
    pair_index = 0
    for left_index, left in enumerate(names[:-1]):
        for right in names[left_index + 1 :]:
            for pair_round in range(config.games_per_pair // 2):
                seed = random_source.getrandbits(64)
                for red, black in ((left, right), (right, left)):
                    match = run_match(
                        agent_factories[red](),
                        agent_factories[black](),
                        config=MatchConfig(config.board, seed, red, black),
                    )
                    winning_side = match.winner
                    winner = (
                        red
                        if winning_side is Player.RED
                        else black if winning_side is Player.BLACK else None
                    )
                    games.append(
                        BenchmarkGame(
                            index=len(games),
                            pair_index=pair_index,
                            pair_round=pair_round,
                            seed=seed,
                            red_agent=red,
                            black_agent=black,
                            winner=winner,
                            winning_side=winning_side,
                            move_count=len(match.moves),
                        )
                    )
            pair_index += 1
    return BenchmarkResult(config, tuple(games))


__all__ = [
    "BENCHMARK_FORMAT",
    "BENCHMARK_FORMAT_VERSION",
    "AgentConfig",
    "AgentFactory",
    "BenchmarkConfig",
    "BenchmarkGame",
    "BenchmarkResult",
    "run_benchmark",
]
