"""Tests for reproducible self-play dataset preparation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from twixt_ai.agents import AgentRequest, AgentResult, RandomAgent
from twixt_ai.evaluation import MatchConfig, run_match
from twixt_ai.game import BoardDimensions
from twixt_ai.selfplay import BatchConfig, run_batch
from twixt_ai.training import DatasetConfig, build_dataset
from twixt_ai.training.cli import main


class FirstWithPolicy:
    def choose_move(self, request: AgentRequest) -> AgentResult:
        moves = request.legal_moves
        visits = (4,) if len(moves) == 1 else (3, 1)
        return AgentResult(
            moves[0],
            {
                "root_moves": [
                    {
                        "x": move.coordinate.x,
                        "y": move.coordinate.y,
                        "visits": visits[index],
                    }
                    for index, move in enumerate(moves[:2])
                ],
                "simulations": 4,
                "agent": "test-search",
            },
        )


def _lines(root: Path, summary: object, split: str) -> list[dict[str, object]]:
    manifest = summary.to_dict()  # type: ignore[attr-defined]
    result = []
    for shard in manifest["splits"][split]["shards"]:  # type: ignore[index]
        result.extend(
            json.loads(line)
            for line in (root / shard["path"]).read_text().splitlines()
        )
    return result


def test_build_dataset_retains_targets_and_provenance(tmp_path: Path) -> None:
    match = run_match(
        FirstWithPolicy(),
        FirstWithPolicy(),
        config=MatchConfig(
            BoardDimensions(4, 4), 17, "policy-red", "policy-black"
        ),
    )
    source = tmp_path / "match.json"
    source.write_text(match.to_json())
    output = tmp_path / "dataset"

    summary = build_dataset(
        source,
        output,
        config=DatasetConfig(
            shard_size=2,
            validation_fraction=0,
            split_seed="experiment-a",
            metadata={"run": 4},
        ),
    )

    examples = _lines(output, summary, "train")
    assert len(examples) == len(match.moves)
    first = examples[0]
    assert first["position"] == match.record.initial_state.to_dict()
    assert first["action"] == match.moves[0].coordinate.to_dict()
    expected_outcome = (
        0 if match.winner is None
        else 1 if match.winner is match.moves[0].player
        else -1
    )
    assert first["outcome"] == expected_outcome
    assert sum(item["probability"] for item in first["policy"]) == pytest.approx(1)
    assert first["policy"][0]["probability"] == pytest.approx(0.75)
    assert first["source"]["config"] == match.config.to_dict()
    assert first["source"]["decision"]["metadata"]["agent"] == "test-search"
    assert json.loads((output / "manifest.json").read_text()) == summary.to_dict()
    assert summary.config.metadata == {"run": 4}


def test_split_is_by_game_and_reproducible_across_input_order(tmp_path: Path) -> None:
    source = tmp_path / "selfplay"
    run_batch(
        RandomAgent,
        RandomAgent,
        config=BatchConfig(
            games=4,
            workers=1,
            seed=99,
            board=BoardDimensions(4, 4),
        ),
        output_dir=source,
    )
    paths = sorted((source / "games").glob("*.json"))
    config = DatasetConfig(shard_size=100, validation_fraction=0.5, split_seed="x")
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    first = build_dataset(paths, first_root, config=config)
    second = build_dataset(reversed(paths), second_root, config=config)

    assert first.to_dict() == second.to_dict()
    for split in ("train", "validation"):
        assert _lines(first_root, first, split) == _lines(second_root, second, split)
        game_splits = {
            example["source"]["game_id"] for example in _lines(first_root, first, split)
        }
        other = {
            example["source"]["game_id"]
            for example in _lines(
                first_root, first, "validation" if split == "train" else "train"
            )
        }
        assert game_splits.isdisjoint(other)


def test_batch_directory_skips_failed_games_and_cli_emits_manifest(
    tmp_path: Path, capsys: object
) -> None:
    source = tmp_path / "selfplay"
    run_batch(
        RandomAgent,
        RandomAgent,
        config=BatchConfig(
            games=2, workers=1, seed=5, board=BoardDimensions(4, 4)
        ),
        output_dir=source,
    )
    output = tmp_path / "dataset"

    assert main([
        "--input", str(source), "--output-dir", str(output),
        "--validation-fraction", "0", "--shard-size", "3",
    ]) == 0

    emitted = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert emitted == json.loads((output / "manifest.json").read_text())
    assert emitted["source_games"] == 2
    assert emitted["examples"] > 0


def test_rejects_tampered_match_history(tmp_path: Path) -> None:
    match = run_match(
        RandomAgent(), RandomAgent(),
        config=MatchConfig(BoardDimensions(4, 4), 1),
    ).to_dict()
    match["decisions"][0]["coordinate"] = {"x": 99, "y": 99}  # type: ignore[index]
    source = tmp_path / "bad.json"
    source.write_text(json.dumps(match))

    with pytest.raises(ValueError, match="does not match record"):
        build_dataset(source, tmp_path / "dataset")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda match: match["config"].update(seed={}), "invalid match config"),
        (
            lambda match: match["config"].update(agents=[]),
            "match config agents",
        ),
        (
            lambda match: match["decisions"][0].update(seed="not-an-integer"),
            "decision 0 seed",
        ),
        (
            lambda match: match["decisions"][0].update(seed=True),
            "decision 0 seed",
        ),
        (
            lambda match: match["decisions"][0].pop("seed"),
            "decision 0 seed",
        ),
    ],
)
def test_rejects_invalid_match_provenance(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], object],
    message: str,
) -> None:
    match = run_match(
        RandomAgent(),
        RandomAgent(),
        config=MatchConfig(BoardDimensions(4, 4), 1),
    ).to_dict()
    mutation(match)
    source = tmp_path / "bad-provenance.json"
    source.write_text(json.dumps(match))

    with pytest.raises(ValueError, match=message):
        build_dataset(source, tmp_path / "dataset")


def test_rejects_inconsistent_mcts_visit_total(tmp_path: Path) -> None:
    match = run_match(
        FirstWithPolicy(),
        FirstWithPolicy(),
        config=MatchConfig(BoardDimensions(4, 4), 2),
    ).to_dict()
    metadata = match["decisions"][0]["metadata"]  # type: ignore[index]
    metadata["simulations"] = 5  # type: ignore[index]
    source = tmp_path / "bad-statistics.json"
    source.write_text(json.dumps(match))

    with pytest.raises(ValueError, match="visits must sum"):
        build_dataset(source, tmp_path / "dataset")


@pytest.mark.parametrize("simulations", [None, True, 0, "4"])
def test_rejects_invalid_mcts_simulation_count(
    tmp_path: Path, simulations: object
) -> None:
    match = run_match(
        FirstWithPolicy(),
        FirstWithPolicy(),
        config=MatchConfig(BoardDimensions(4, 4), 2),
    ).to_dict()
    metadata = match["decisions"][0]["metadata"]  # type: ignore[index]
    metadata["simulations"] = simulations  # type: ignore[index]
    source = tmp_path / "bad-simulations.json"
    source.write_text(json.dumps(match))

    with pytest.raises(ValueError, match="positive integer simulations"):
        build_dataset(source, tmp_path / "dataset")
