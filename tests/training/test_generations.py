"""End-to-end coverage for iterative Mini training generations."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from twixt_ai.models import (
    MINI_POLICY_VALUE_CONFIG,
    PolicyValueNetwork,
    save_policy_value_checkpoint,
)
from twixt_ai.training import generations
from twixt_ai.training.generations import (
    MiniGenerationConfig,
    run_mini_training_generations,
)
from twixt_ai.training.generations_cli import main as generations_main


def test_runs_two_generations_with_explicit_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    champion = tmp_path / "champion.pt"
    save_policy_value_checkpoint(
        champion, PolicyValueNetwork(MINI_POLICY_VALUE_CONFIG),
        metadata={"name": "fixture champion"},
    )
    output = tmp_path / "run"
    config = MiniGenerationConfig(
        generations=2,
        games_per_generation=2,
        dataset_window=2,
        selfplay_simulations=1,
        evaluation_games=2,
        evaluation_simulations=1,
        workers=1,
        epochs=1,
        batch_size=256,
        validation_fraction=0,
        promotion_win_rate=0,
        seed=59,
    )

    report = run_mini_training_generations(champion, output, config=config)

    assert report["status"] == "completed"
    assert len(report["generations"]) == 2
    assert [item["status"] for item in report["generations"]] == [
        "completed", "completed"
    ]
    assert [item["decision"] for item in report["generations"]] == [
        "promoted", "promoted"
    ]
    assert report["generations"][0]["dataset"]["source_generations"] == [1]
    assert report["generations"][1]["dataset"]["source_generations"] == [1, 2]
    assert report["lineage"][1]["parent_sha256"] == report["lineage"][0][
        "candidate_sha256"
    ]
    assert (output / "generation-0001" / "candidate" / "best.pt").is_file()
    assert (output / "generation-0002" / "evaluation.json").is_file()
    assert json.loads((output / "report.json").read_text()) == report


@pytest.mark.parametrize(
    "kwargs",
    [
        {"generations": 0},
        {"evaluation_games": 3},
        {"promotion_win_rate": 1.1},
        {"validation_fraction": 1},
    ],
)
def test_generation_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        MiniGenerationConfig(**kwargs)  # type: ignore[arg-type]


def test_generation_cli_rejects_all_validation_split(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        generations_main([
            "--initial-champion",
            "champion.pt",
            "--output-dir",
            "output",
            "--validation-fraction",
            "1",
        ])

    assert raised.value.code == 2
    assert "validation_fraction must be in [0, 1)" in capsys.readouterr().err


def test_generation_rejects_empty_training_split_after_dataset_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    champion = tmp_path / "champion.pt"
    save_policy_value_checkpoint(
        champion, PolicyValueNetwork(MINI_POLICY_VALUE_CONFIG)
    )
    monkeypatch.setattr(
        generations,
        "run_batch",
        lambda *args, **kwargs: SimpleNamespace(
            failed=0, to_dict=lambda: {"aggregate": {"completed": 1}}
        ),
    )
    monkeypatch.setattr(
        generations, "_game_paths", lambda roots: (tmp_path / "game.json",)
    )
    monkeypatch.setattr(
        generations,
        "build_dataset",
        lambda *args, **kwargs: SimpleNamespace(
            train_examples=0,
            to_dict=lambda: {
                "splits": {
                    "train": {"examples": 0},
                    "validation": {"examples": 1},
                }
            },
        ),
    )
    monkeypatch.setattr(
        generations,
        "train_model",
        lambda *args, **kwargs: pytest.fail("training should not start"),
    )
    config = MiniGenerationConfig(
        generations=1,
        games_per_generation=1,
        selfplay_simulations=1,
        evaluation_games=2,
        evaluation_simulations=1,
        workers=1,
        epochs=1,
        validation_fraction=0.5,
    )
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="training split must contain"):
        run_mini_training_generations(champion, output, config=config)

    report = json.loads((output / "report.json").read_text())
    assert report["generations"][0]["failed_stage"] == "dataset"


def test_refuses_nonempty_output_and_requires_reproducible_hash_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    champion = tmp_path / "champion.pt"
    save_policy_value_checkpoint(champion, PolicyValueNetwork(MINI_POLICY_VALUE_CONFIG))
    output = tmp_path / "output"
    output.mkdir()
    (output / "keep").write_text("existing")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    with pytest.raises(ValueError, match="must be empty"):
        run_mini_training_generations(champion, output)

    monkeypatch.delenv("PYTHONHASHSEED")
    with pytest.raises(ValueError, match="PYTHONHASHSEED"):
        run_mini_training_generations(champion, tmp_path / "new-output")
