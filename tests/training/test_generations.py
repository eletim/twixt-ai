"""End-to-end coverage for iterative Mini training generations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from twixt_ai.models import (
    MINI_POLICY_VALUE_CONFIG,
    PolicyValueNetwork,
    save_policy_value_checkpoint,
)
from twixt_ai.training.generations import (
    MiniGenerationConfig,
    run_mini_training_generations,
)


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
    ],
)
def test_generation_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        MiniGenerationConfig(**kwargs)  # type: ignore[arg-type]


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
