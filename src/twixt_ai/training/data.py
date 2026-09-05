"""Reproducible conversion of headless match artifacts into training datasets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from twixt_ai.evaluation import MATCH_FORMAT, MATCH_FORMAT_VERSION, MatchConfig
from twixt_ai.game import (
    BoardDimensions,
    Coordinate,
    GameRecord,
    GameState,
    PegPlacement,
    Player,
    apply_move,
    legal_peg_placements,
)


DATASET_FORMAT = "twixt-ai-training-dataset"
DATASET_VERSION = 1
EXAMPLE_FORMAT = "twixt-ai-training-example"
EXAMPLE_VERSION = 1


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _copy_json(value: object, name: str) -> object:
    try:
        encoded = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain only JSON-compatible values") from exc
    return json.loads(encoded)


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Stable sharding, splitting, and caller metadata for a dataset build."""

    shard_size: int = 10_000
    validation_fraction: float = 0.1
    split_seed: str = "0"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.shard_size, bool)
            or not isinstance(self.shard_size, int)
            or self.shard_size < 1
        ):
            raise ValueError("shard_size must be a positive integer")
        if (
            isinstance(self.validation_fraction, bool)
            or not isinstance(self.validation_fraction, (int, float))
            or not math.isfinite(self.validation_fraction)
            or not 0 <= self.validation_fraction <= 1
        ):
            raise ValueError("validation_fraction must be in [0, 1]")
        if not isinstance(self.split_seed, str):
            raise TypeError("split_seed must be a string")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        copied = _copy_json(self.metadata, "metadata")
        assert isinstance(copied, dict)
        object.__setattr__(self, "validation_fraction", float(self.validation_fraction))
        object.__setattr__(self, "metadata", copied)

    def to_dict(self) -> dict[str, object]:
        return {
            "shard_size": self.shard_size,
            "validation_fraction": self.validation_fraction,
            "split_seed": self.split_seed,
            "metadata": _copy_json(self.metadata, "metadata"),
        }


@dataclass(frozen=True, slots=True)
class Shard:
    """One JSON Lines shard recorded in the dataset manifest."""

    path: str
    examples: int
    sha256: str


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    """Manifest returned after a complete dataset build."""

    config: DatasetConfig
    source_games: int
    train_examples: int
    validation_examples: int
    train_shards: tuple[Shard, ...]
    validation_shards: tuple[Shard, ...]

    @property
    def examples(self) -> int:
        return self.train_examples + self.validation_examples

    def to_dict(self) -> dict[str, object]:
        def split(examples: int, shards: tuple[Shard, ...]) -> dict[str, object]:
            return {
                "examples": examples,
                "shards": [asdict(shard) for shard in shards],
            }

        return {
            "format": DATASET_FORMAT,
            "version": DATASET_VERSION,
            "example_format": {
                "format": EXAMPLE_FORMAT,
                "version": EXAMPLE_VERSION,
            },
            "config": self.config.to_dict(),
            "source_games": self.source_games,
            "examples": self.examples,
            "splits": {
                "train": split(self.train_examples, self.train_shards),
                "validation": split(
                    self.validation_examples, self.validation_shards
                ),
            },
        }

    def to_json(self, *, indent: int | None = None) -> str:
        separators = None if indent is not None else (",", ":")
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=separators, indent=indent
        )


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"artifact {path} must contain a JSON object")
    return value


def _source_paths(source: str | Path | Iterable[str | Path]) -> tuple[Path, ...]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            summary_path = path / "summary.json"
            if summary_path.is_file():
                summary = _load_object(summary_path)
                if summary.get("format") != "twixt-ai-selfplay-batch":
                    raise ValueError(f"unsupported batch manifest {summary_path}")
                games = summary.get("games")
                if not isinstance(games, list):
                    raise ValueError("batch manifest games must be an array")
                values = []
                for index, game in enumerate(games):
                    if not isinstance(game, dict):
                        raise ValueError(
                            f"batch manifest games[{index}] must be an object"
                        )
                    if game.get("status") == "completed":
                        artifact = game.get("artifact")
                        if not isinstance(artifact, str):
                            raise ValueError(
                                f"batch manifest games[{index}].artifact must be a string"
                            )
                        artifact_path = path / artifact
                        if not artifact_path.resolve().is_relative_to(path.resolve()):
                            raise ValueError(
                                f"batch manifest games[{index}].artifact must stay "
                                "inside the batch directory"
                            )
                        values.append(artifact_path)
                return tuple(values)
            return tuple(sorted(path.glob("*.json")))
        return (path,)
    try:
        return tuple(
            sorted((Path(item) for item in source), key=lambda item: str(item))
        )
    except TypeError as exc:
        raise TypeError("source must be a path or iterable of paths") from exc


def _validated_match(
    value: Mapping[str, Any], path: Path
) -> tuple[GameRecord, list[dict[str, Any]]]:
    if value.get("format") != MATCH_FORMAT:
        raise ValueError(f"unsupported match format in {path}: {value.get('format')!r}")
    if value.get("version") != MATCH_FORMAT_VERSION:
        raise ValueError(f"unsupported match version in {path}: {value.get('version')!r}")
    config = value.get("config")
    decisions = value.get("decisions")
    record_value = value.get("record")
    result = value.get("result")
    if not isinstance(config, dict):
        raise ValueError(f"match config in {path} must be an object")
    if set(config) != {"board", "seed", "agents"}:
        raise ValueError(
            f"match config in {path} must contain exactly agents, board, and seed"
        )
    board = config["board"]
    agents = config["agents"]
    if not isinstance(board, dict) or set(board) != {"width", "height"}:
        raise ValueError(
            f"match config board in {path} must contain exactly height and width"
        )
    if not isinstance(agents, dict) or set(agents) != {"red", "black"}:
        raise ValueError(
            f"match config agents in {path} must contain exactly black and red"
        )
    try:
        match_config = MatchConfig(
            board=BoardDimensions(width=board["width"], height=board["height"]),
            seed=config["seed"],
            red_agent=agents["red"],
            black_agent=agents["black"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid match config in {path}: {exc}") from exc
    if not isinstance(decisions, list) or any(not isinstance(item, dict) for item in decisions):
        raise ValueError(f"match decisions in {path} must be an array of objects")
    if not isinstance(record_value, dict):
        raise ValueError(f"match record in {path} must be an object")
    record = GameRecord.from_dict(record_value)
    if not record.final_state.is_terminal:
        raise ValueError(f"match record in {path} is not terminal")
    if match_config.board != record.initial_state.board:
        raise ValueError(f"match config board in {path} does not match record")
    expected_result = {
        "status": record.final_state.result.value,
        "winner": (
            record.final_state.winner.value
            if record.final_state.winner is not None
            else None
        ),
        "move_count": len(record.moves),
    }
    if result != expected_result:
        raise ValueError(f"match result in {path} does not match record")
    if len(decisions) != len(record.moves):
        raise ValueError(f"match decisions in {path} do not align with moves")
    for index, (decision, move) in enumerate(zip(decisions, record.moves)):
        if (
            decision.get("player") != move.player.value
            or decision.get("coordinate") != move.coordinate.to_dict()
        ):
            raise ValueError(f"match decision {index} in {path} does not match record")
        if not isinstance(decision.get("metadata"), dict):
            raise ValueError(f"match decision {index} metadata in {path} must be an object")
        if "seed" not in decision:
            raise ValueError(
                f"match decision {index} seed in {path} must be an integer or null"
            )
        decision_seed = decision["seed"]
        if decision_seed is not None and (
            isinstance(decision_seed, bool) or not isinstance(decision_seed, int)
        ):
            raise ValueError(
                f"match decision {index} seed in {path} must be an integer or null"
            )
    return record, decisions  # type: ignore[return-value]


def _policy_target(
    metadata: Mapping[str, Any], state: GameState
) -> list[dict[str, object]] | None:
    root_moves = metadata.get("root_moves")
    if root_moves is None:
        return None
    if not isinstance(root_moves, list):
        raise ValueError("root_moves metadata must be an array")
    simulations = metadata.get("simulations")
    if (
        isinstance(simulations, bool)
        or not isinstance(simulations, int)
        or simulations < 1
    ):
        raise ValueError(
            "root_moves metadata requires a positive integer simulations count"
        )
    visits: list[tuple[int, int, int]] = []
    legal = set(legal_peg_placements(state))
    seen: set[Coordinate] = set()
    for index, item in enumerate(root_moves):
        if not isinstance(item, dict):
            raise ValueError(f"root_moves[{index}] must be an object")
        x, y, count = item.get("x"), item.get("y"), item.get("visits")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (x, y, count)
        ):
            raise ValueError(
                f"root_moves[{index}] coordinates and visits must be integers"
            )
        coordinate = Coordinate(x, y)
        if (
            count < 0
            or coordinate in seen
            or PegPlacement(state.side_to_move, coordinate) not in legal
        ):
            raise ValueError(
                f"root_moves[{index}] contains an invalid move or visit count"
            )
        seen.add(coordinate)
        visits.append((x, y, count))
    total = sum(item[2] for item in visits)
    if total != simulations:
        raise ValueError("root move visits must sum to metadata.simulations")
    return [
        {"coordinate": {"x": x, "y": y}, "probability": count / total}
        for x, y, count in visits
        if count
    ]


def _examples(value: dict[str, Any], path: Path) -> tuple[str, list[dict[str, object]]]:
    record, decisions = _validated_match(value, path)
    game_id = _digest(value)
    winner = record.final_state.winner
    state = record.initial_state
    examples: list[dict[str, object]] = []
    for ply, (move, decision) in enumerate(zip(record.moves, decisions)):
        outcome = 0 if winner is None else (1 if winner is state.side_to_move else -1)
        source = {
            "game_id": game_id,
            "ply": ply,
            "config": _copy_json(value["config"], "match config"),
            "decision": {
                "seed": decision.get("seed"),
                "metadata": _copy_json(decision["metadata"], "decision metadata"),
            },
        }
        example: dict[str, object] = {
            "format": EXAMPLE_FORMAT,
            "version": EXAMPLE_VERSION,
            "id": f"{game_id}:{ply}",
            "position": state.to_dict(),
            "action": move.coordinate.to_dict(),
            "outcome": outcome,
            "source": source,
        }
        policy = _policy_target(decision["metadata"], state)
        if policy is not None:
            example["policy"] = policy
        examples.append(example)
        state = apply_move(state, move)
    return game_id, examples


def _split(game_id: str, config: DatasetConfig) -> str:
    digest = hashlib.sha256(f"{config.split_seed}:{game_id}".encode()).digest()
    fraction = int.from_bytes(digest, "big") / (1 << (8 * len(digest)))
    return "validation" if fraction < config.validation_fraction else "train"


def _write_shards(
    root: Path,
    split: str,
    examples: list[dict[str, object]],
    size: int,
) -> tuple[Shard, ...]:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for stale in split_dir.glob("shard-*.jsonl"):
        stale.unlink()
    for stale in split_dir.glob("shard-*.jsonl.tmp"):
        stale.unlink()
    shards: list[Shard] = []
    for number, start in enumerate(range(0, len(examples), size)):
        path = split_dir / f"shard-{number:05d}.jsonl"
        chunk = examples[start : start + size]
        content = b"".join(_json_bytes(item) + b"\n" for item in chunk)
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
        shards.append(
            Shard(
                str(path.relative_to(root)),
                len(chunk),
                hashlib.sha256(content).hexdigest(),
            )
        )
    return tuple(shards)


def build_dataset(
    source: str | Path | Iterable[str | Path],
    output_dir: str | Path,
    *,
    config: DatasetConfig | None = None,
) -> DatasetSummary:
    """Convert match artifacts into deterministic, game-level split shards.

    ``source`` may be a self-play run directory, a match artifact, or an
    iterable of match artifacts. Existing output files are replaced atomically;
    callers should use a dedicated output directory.
    """

    dataset_config = config or DatasetConfig()
    if not isinstance(dataset_config, DatasetConfig):
        raise TypeError("config must be a DatasetConfig or None")
    if not isinstance(output_dir, (str, Path)):
        raise TypeError("output_dir must be a path")
    paths = _source_paths(source)
    if not paths:
        raise ValueError("source contains no completed match artifacts")

    by_id: dict[str, tuple[str, list[dict[str, object]]]] = {}
    for path in paths:
        game_id, examples = _examples(_load_object(path), path)
        if game_id in by_id:
            raise ValueError(f"duplicate source game: {path}")
        by_id[game_id] = (_split(game_id, dataset_config), examples)

    split_examples = {"train": [], "validation": []}
    for game_id in sorted(by_id):
        split, examples = by_id[game_id]
        split_examples[split].extend(examples)

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    train_shards = _write_shards(
        root, "train", split_examples["train"], dataset_config.shard_size
    )
    validation_shards = _write_shards(
        root,
        "validation",
        split_examples["validation"],
        dataset_config.shard_size,
    )
    summary = DatasetSummary(
        dataset_config,
        len(by_id),
        len(split_examples["train"]),
        len(split_examples["validation"]),
        train_shards,
        validation_shards,
    )
    temporary = root / "manifest.json.tmp"
    temporary.write_text(summary.to_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(root / "manifest.json")
    return summary


__all__ = [
    "DATASET_FORMAT",
    "DATASET_VERSION",
    "EXAMPLE_FORMAT",
    "EXAMPLE_VERSION",
    "DatasetConfig",
    "DatasetSummary",
    "Shard",
    "build_dataset",
]
