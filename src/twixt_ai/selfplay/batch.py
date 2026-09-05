"""Parallel, reproducible generation of headless self-play games."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
import json
from pathlib import Path
from random import Random, SystemRandom
from typing import Any

from twixt_ai.agents import Agent
from twixt_ai.evaluation import MatchConfig, run_match
from twixt_ai.game import BoardDimensions, Player


BATCH_FORMAT = "twixt-ai-selfplay-batch"
BATCH_FORMAT_VERSION = 1
GAME_FAILURE_FORMAT = "twixt-ai-selfplay-failure"
AgentFactory = Callable[[], Agent]


def _positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _require_seed(seed: int | None) -> None:
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None")


@dataclass(frozen=True, slots=True)
class BatchConfig:
    """Serializable settings for one self-play generation run."""

    games: int
    workers: int = 1
    seed: int | None = None
    board: BoardDimensions = BoardDimensions()
    red_agent: str = "red"
    black_agent: str = "black"

    def __post_init__(self) -> None:
        _positive_integer(self.games, "games")
        _positive_integer(self.workers, "workers")
        _require_seed(self.seed)
        if not isinstance(self.board, BoardDimensions):
            raise TypeError("board must be BoardDimensions")
        for name in ("red_agent", "black_agent"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")

    def to_dict(self) -> dict[str, object]:
        return {
            "games": self.games,
            "workers": self.workers,
            "seed": self.seed,
            "board": self.board.to_dict(),
            "agents": {
                Player.RED.value: self.red_agent,
                Player.BLACK.value: self.black_agent,
            },
        }


@dataclass(frozen=True, slots=True)
class GameReport:
    """Manifest entry for one completed or failed game."""

    index: int
    seed: int
    artifact: str
    status: str
    winner: Player | None = None
    move_count: int | None = None
    error_type: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("index must be a non-negative integer")
        _require_seed(self.seed)
        if self.status not in ("completed", "failed"):
            raise ValueError("status must be completed or failed")
        if not isinstance(self.artifact, str) or not self.artifact:
            raise ValueError("artifact must be a non-empty string")
        if self.status == "completed":
            if self.move_count is None or self.error_type is not None or self.error is not None:
                raise ValueError("completed reports require move_count and no error")
        elif self.error_type is None or self.error is None:
            raise ValueError("failed reports require error details")

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "index": self.index,
            "seed": self.seed,
            "artifact": self.artifact,
            "status": self.status,
        }
        if self.status == "completed":
            value.update(
                winner=self.winner.value if self.winner is not None else None,
                move_count=self.move_count,
            )
        else:
            value["error"] = {"type": self.error_type, "message": self.error}
        return value


@dataclass(frozen=True, slots=True)
class BatchSummary:
    """Aggregate output and artifact manifest for a self-play batch."""

    config: BatchConfig
    games: tuple[GameReport, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.config, BatchConfig):
            raise TypeError("config must be a BatchConfig")
        reports = tuple(self.games)
        if len(reports) != self.config.games:
            raise ValueError("games must contain one report per configured game")
        if tuple(report.index for report in reports) != tuple(range(self.config.games)):
            raise ValueError("game reports must be ordered by consecutive index")
        object.__setattr__(self, "games", reports)

    @property
    def completed(self) -> int:
        return sum(report.status == "completed" for report in self.games)

    @property
    def failed(self) -> int:
        return len(self.games) - self.completed

    def to_dict(self) -> dict[str, object]:
        wins = {
            Player.RED.value: sum(report.winner is Player.RED for report in self.games),
            Player.BLACK.value: sum(report.winner is Player.BLACK for report in self.games),
            "draw": sum(
                report.status == "completed" and report.winner is None
                for report in self.games
            ),
        }
        return {
            "format": BATCH_FORMAT,
            "version": BATCH_FORMAT_VERSION,
            "config": self.config.to_dict(),
            "aggregate": {
                "completed": self.completed,
                "failed": self.failed,
                "wins": wins,
                "total_moves": sum(report.move_count or 0 for report in self.games),
            },
            "games": [report.to_dict() for report in self.games],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        separators = None if indent is not None else (",", ":")
        return json.dumps(self.to_dict(), sort_keys=True, separators=separators, indent=indent)


def _play_game(
    red_factory: AgentFactory,
    black_factory: AgentFactory,
    board: BoardDimensions,
    seed: int,
    red_name: str,
    black_name: str,
) -> dict[str, object]:
    """Process worker returning only pickle-friendly JSON data."""

    result = run_match(
        red_factory(),
        black_factory(),
        config=MatchConfig(board, seed, red_name, black_name),
    )
    return result.to_dict()


def _error_payload(index: int, seed: int, exc: BaseException) -> dict[str, object]:
    return {
        "format": GAME_FAILURE_FORMAT,
        "version": BATCH_FORMAT_VERSION,
        "index": index,
        "seed": seed,
        "error": {"type": type(exc).__name__, "message": str(exc)},
    }


def _write_json(path: Path, value: Mapping[str, Any], *, indent: int | None) -> None:
    separators = None if indent is not None else (",", ":")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=separators, indent=indent) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_batch(
    red_factory: AgentFactory,
    black_factory: AgentFactory,
    *,
    config: BatchConfig,
    output_dir: str | Path,
    pretty: bool = False,
) -> BatchSummary:
    """Generate games, persist every outcome, and return an aggregate manifest.

    Agent factories must be pickleable when more than one worker is requested.
    A failure in an agent, match, or worker becomes that game's failure artifact;
    other scheduled games continue. Seeded batches produce the same per-game
    seeds and ordered manifest regardless of worker count.
    """

    if not callable(red_factory) or not callable(black_factory):
        raise TypeError("agent factories must be callable")
    if not isinstance(config, BatchConfig):
        raise TypeError("config must be a BatchConfig")
    if not isinstance(output_dir, (str, Path)):
        raise TypeError("output_dir must be a path")

    root = Path(output_dir)
    games_dir = root / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    random_source = Random(config.seed) if config.seed is not None else SystemRandom()
    seeds = [random_source.getrandbits(64) for _ in range(config.games)]
    reports: dict[int, GameReport] = {}
    indent = 2 if pretty else None

    def capture(index: int, seed: int, payload: dict[str, object]) -> None:
        relative_path = f"games/game-{index:06d}.json"
        _write_json(root / relative_path, payload, indent=indent)
        result = payload["result"]
        assert isinstance(result, dict)
        winner_value = result["winner"]
        reports[index] = GameReport(
            index=index,
            seed=seed,
            artifact=relative_path,
            status="completed",
            winner=Player(winner_value) if winner_value is not None else None,
            move_count=int(result["move_count"]),
        )

    def capture_failure(index: int, seed: int, exc: BaseException) -> None:
        relative_path = f"games/game-{index:06d}.json"
        payload = _error_payload(index, seed, exc)
        _write_json(root / relative_path, payload, indent=indent)
        error = payload["error"]
        assert isinstance(error, dict)
        reports[index] = GameReport(
            index=index,
            seed=seed,
            artifact=relative_path,
            status="failed",
            error_type=str(error["type"]),
            error=str(error["message"]),
        )

    if config.workers == 1:
        for index, seed in enumerate(seeds):
            try:
                capture(
                    index,
                    seed,
                    _play_game(
                        red_factory,
                        black_factory,
                        config.board,
                        seed,
                        config.red_agent,
                        config.black_agent,
                    ),
                )
            except Exception as exc:  # one broken game must not stop the batch
                capture_failure(index, seed, exc)
    else:
        max_workers = min(config.workers, config.games)
        next_index = 0
        while next_index < config.games:
            try:
                pool = ProcessPoolExecutor(max_workers=max_workers)
            except Exception as exc:
                # Executor startup is an infrastructure failure, but the batch
                # must still leave a complete machine-readable manifest.
                for index in range(next_index, config.games):
                    capture_failure(index, seeds[index], exc)
                break

            pending: dict[Future[dict[str, object]], tuple[int, int]] = {}
            pool_broken = False
            try:
                while next_index < config.games or pending:
                    while (
                        not pool_broken
                        and next_index < config.games
                        and len(pending) < max_workers
                    ):
                        index = next_index
                        seed = seeds[index]
                        try:
                            future = pool.submit(
                                _play_game,
                                red_factory,
                                black_factory,
                                config.board,
                                seed,
                                config.red_agent,
                                config.black_agent,
                            )
                        except BrokenProcessPool as exc:
                            # This game never entered the pool, so report the
                            # infrastructure failure and resume after restart.
                            capture_failure(index, seed, exc)
                            next_index += 1
                            pool_broken = True
                        except Exception as exc:
                            capture_failure(index, seed, exc)
                            next_index += 1
                        else:
                            pending[future] = (index, seed)
                            next_index += 1

                    if not pending:
                        break

                    completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in completed:
                        index, seed = pending.pop(future)
                        try:
                            capture(index, seed, future.result())
                        except BrokenProcessPool as exc:
                            capture_failure(index, seed, exc)
                            pool_broken = True
                        except Exception as exc:
                            capture_failure(index, seed, exc)

                    if pool_broken:
                        # A terminated worker poisons every job still assigned
                        # to that executor. They settle as BrokenProcessPool;
                        # record each before starting a fresh bounded pool.
                        for future, (index, seed) in tuple(pending.items()):
                            try:
                                capture(index, seed, future.result())
                            except Exception as exc:
                                capture_failure(index, seed, exc)
                            finally:
                                del pending[future]
                        break
            finally:
                # A broken executor can still be shut down normally. Do not let
                # a secondary shutdown error suppress the batch manifest.
                try:
                    pool.shutdown(wait=True, cancel_futures=True)
                except Exception:
                    pass

    summary = BatchSummary(config, tuple(reports[index] for index in range(config.games)))
    _write_json(root / "summary.json", summary.to_dict(), indent=indent)
    return summary


__all__ = [
    "BATCH_FORMAT",
    "BATCH_FORMAT_VERSION",
    "GAME_FAILURE_FORMAT",
    "AgentFactory",
    "BatchConfig",
    "BatchSummary",
    "GameReport",
    "run_batch",
]
