from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from twixt_ai.models import (
    MINI_POLICY_VALUE_CONFIG,
    PolicyValueNetwork,
    save_policy_value_checkpoint,
)
from twixt_ai.training import selfplay_dataset_cli
from twixt_ai.training.selfplay_dataset import (
    SELFPLAY_DATASET_FORMAT,
    MiniSelfplayDatasetConfig,
    run_mini_selfplay_dataset,
)


def test_generates_dataset_only_with_generation_search_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    champion = tmp_path / "champion.pt"
    save_policy_value_checkpoint(
        champion, PolicyValueNetwork(MINI_POLICY_VALUE_CONFIG)
    )
    output = tmp_path / "output"
    config = MiniSelfplayDatasetConfig(
        games=2,
        simulations=1,
        workers=1,
        validation_fraction=0,
        seed=143_101,
        split_seed="test-v2-split",
        workflow_label="test-v2-workflow",
        device="cpu",
    )

    report = run_mini_selfplay_dataset(champion, output, config=config)

    manifest_path = output / "dataset" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert report["format"] == SELFPLAY_DATASET_FORMAT
    assert report["status"] == "completed"
    assert report["selfplay"]["summary"]["aggregate"]["completed"] == 2
    assert report["selfplay"]["summary"]["aggregate"]["failed"] == 0
    assert manifest["source_games"] == 2
    assert manifest["examples"] > 0
    assert report["dataset"]["manifest_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert manifest["config"]["split_seed"] == "test-v2-split"
    assert manifest["config"]["metadata"] == {
        "champion_sha256": hashlib.sha256(champion.read_bytes()).hexdigest(),
        "mcts": report["resolved_config"]["search"],
        "workflow": "test-v2-workflow",
    }
    assert not (output / "candidate").exists()
    assert not (output / "evaluation.json").exists()


def test_refuses_nonempty_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "historical.json"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="empty or not exist"):
        run_mini_selfplay_dataset(tmp_path / "missing.pt", output)

    assert marker.read_text(encoding="utf-8") == "keep"


def test_cli_resolves_reproduction_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}

    def run_stub(
        champion: Path,
        output_dir: Path,
        *,
        config: MiniSelfplayDatasetConfig,
    ) -> dict[str, object]:
        captured.update(
            champion=champion, output_dir=output_dir, config=config
        )
        return {"status": "completed"}

    monkeypatch.setattr(selfplay_dataset_cli, "run_mini_selfplay_dataset", run_stub)
    champion = tmp_path / "champion.pt"
    output = tmp_path / "dataset"

    assert selfplay_dataset_cli.main([
        "--champion", str(champion),
        "--output-dir", str(output),
        "--seed", "143100",
        "--split-seed", "issue-143-v2-143100",
        "--workflow-label", "issue-143-v2-selfplay-dataset",
        "--device", "cuda",
    ]) == 0

    config = captured["config"]
    assert isinstance(config, MiniSelfplayDatasetConfig)
    assert captured["champion"] == champion
    assert captured["output_dir"] == output
    assert config.seed == 143_100
    assert config.resolved_split_seed == "issue-143-v2-143100"
    assert config.workflow_label == "issue-143-v2-selfplay-dataset"
    assert config.device == "cuda"
    assert json.loads(capsys.readouterr().out) == {"status": "completed"}


def test_committed_v2_artifact_uses_runner_defaults() -> None:
    root = (
        Path(__file__).resolve().parents[2]
        / "experiments"
        / "issue-143"
        / "v2-selfplay-dataset"
    )
    config_record = json.loads((root / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (root / "dataset" / "manifest.json").read_text(encoding="utf-8")
    )
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    defaults = MiniSelfplayDatasetConfig()
    search = config_record["resolved_config"]["search"]

    assert report["format"] == SELFPLAY_DATASET_FORMAT
    assert report["resolved_config"] == config_record["resolved_config"]
    assert report["dataset"]["manifest"] == manifest
    assert report["dataset"]["manifest_sha256"] == hashlib.sha256(
        (root / "dataset" / "manifest.json").read_bytes()
    ).hexdigest()
    assert search == {
        "simulations": defaults.simulations,
        "exploration": defaults.exploration,
        "rollout_limit": defaults.rollout_limit,
        "progressive_widening_constant": defaults.progressive_widening_constant,
        "progressive_widening_exponent": defaults.progressive_widening_exponent,
        "guidance": "policy-value",
    }
    assert config_record["seeds"] == {
        "selfplay": 143_100,
        "dataset_split": "issue-143-v2-143100",
    }
    assert manifest["config"]["metadata"]["workflow"] == (
        "issue-143-v2-selfplay-dataset"
    )
