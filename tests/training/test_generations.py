"""End-to-end coverage for iterative Mini training generations."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from twixt_ai.device import DeviceSelection
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
    assert report["environment"]["device"]["requested_device"] == "auto"
    assert report["environment"]["device"]["resolved_device"] in {"cpu", "cuda"}
    expected_worker_mode = (
        "thread"
        if report["environment"]["device"]["resolved_device"] == "cuda"
        else "process"
    )
    assert report["generations"][0]["resolved_config"]["worker_mode"] == expected_worker_mode
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
        {"inference_batch_size": 0},
        {"inference_max_wait_seconds": -0.1},
    ],
)
def test_generation_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        MiniGenerationConfig(**kwargs)  # type: ignore[arg-type]


def test_cuda_selfplay_loads_one_shared_model_and_records_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    champion = tmp_path / "champion.pt"
    save_policy_value_checkpoint(
        champion, PolicyValueNetwork(MINI_POLICY_VALUE_CONFIG)
    )
    original_load = generations.load_policy_value_checkpoint
    loads: list[object] = []

    def load_once(path: object, **kwargs: object) -> object:
        loads.append(path)
        # Exercise shared-path orchestration on CPU-only CI while presenting
        # the same DeviceSelection contract as a CUDA host.
        return original_load(path, map_location="cpu")

    monkeypatch.setattr(generations, "load_policy_value_checkpoint", load_once)
    device = DeviceSelection("cuda", "cuda", True, "fixture GPU", "12.1", "2")
    config = MiniGenerationConfig(
        generations=1,
        games_per_generation=2,
        selfplay_simulations=1,
        evaluation_games=2,
        evaluation_simulations=1,
        workers=2,
        inference_batch_size=2,
        inference_max_wait_seconds=0.05,
        epochs=1,
    )

    batch, inference = generations._run_selfplay(
        champion, tmp_path / "selfplay", config, 86, device
    )

    assert batch.completed == 2
    assert loads == [champion]
    assert inference["mode"] == "shared-batched"
    assert inference["model_instances"] == 1
    assert inference["device"]["resolved_device"] == "cuda"
    statistics = inference["statistics"]
    assert statistics["requests"] > 0
    assert statistics["maximum_batch_size"] == 2


def test_cuda_selfplay_snapshots_statistics_after_batcher_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DelayedStatistics:
        def __init__(self, batcher: DelayedStatisticsBatcher) -> None:
            self.batcher = batcher

        def to_dict(self) -> dict[str, int]:
            return {"requests": int(self.batcher.closed)}

    class DelayedStatisticsBatcher:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.closed = False

        @property
        def statistics(self) -> DelayedStatistics:
            return DelayedStatistics(self)

        def __enter__(self) -> DelayedStatisticsBatcher:
            return self

        def __exit__(self, *args: object) -> None:
            self.closed = True

    champion = tmp_path / "champion.pt"
    model = PolicyValueNetwork(MINI_POLICY_VALUE_CONFIG)
    monkeypatch.setattr(
        generations,
        "load_policy_value_checkpoint",
        lambda *args, **kwargs: SimpleNamespace(model=model),
    )
    monkeypatch.setattr(
        generations, "NeuralInferenceBatcher", DelayedStatisticsBatcher
    )
    monkeypatch.setattr(
        generations,
        "run_batch",
        lambda *args, **kwargs: SimpleNamespace(completed=1, failed=0),
    )
    device = DeviceSelection("cuda", "cuda", True, "fixture GPU", "12.1", "2")
    config = MiniGenerationConfig(
        generations=1,
        games_per_generation=1,
        selfplay_simulations=1,
        evaluation_games=2,
        evaluation_simulations=1,
        workers=1,
        epochs=1,
    )

    _, inference = generations._run_selfplay(
        champion, tmp_path / "selfplay", config, 86, device
    )

    assert inference["statistics"] == {"requests": 1}


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
