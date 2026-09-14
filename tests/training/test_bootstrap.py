from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from twixt_ai.models import (
    ARCHITECTURE_NAME,
    ARCHITECTURE_VERSION,
    MINI_POLICY_VALUE_CONFIG,
    load_policy_value_checkpoint,
)
from twixt_ai.training.bootstrap import (
    DEFAULT_BOOTSTRAP_SEED,
    bootstrap_mini_champion,
)
from twixt_ai.training.bootstrap_cli import main


def test_bootstrap_creates_fresh_v2_mini_champion_with_full_manifest(tmp_path) -> None:
    output = tmp_path / "architecture-v2"

    manifest = bootstrap_mini_champion(output, seed=143_001)

    checkpoint = output / "champion.pt"
    loaded = load_policy_value_checkpoint(checkpoint)
    initialization = manifest["initialization"]
    assert manifest["checkpoint"]["sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    assert initialization["seed"] == 143_001
    assert initialization["warm_started"] is False
    assert initialization["parent_checkpoint"] is None
    assert initialization["architecture"] == {
        "name": ARCHITECTURE_NAME,
        "version": ARCHITECTURE_VERSION,
    }
    assert initialization["model_config"] == MINI_POLICY_VALUE_CONFIG.to_dict()
    assert initialization["input_shape"] == [22, 10, 10]
    assert initialization["trunk_output_shape"] == [8, 10, 10]
    assert initialization["policy_head_linear_shapes"] == [
        [800, 256], [256, 256], [256, 100]
    ]
    assert initialization["value_head_linear_shapes"] == [
        [800, 256], [256, 256], [256, 1]
    ]
    assert initialization["state_dict_shapes"] == {
        name: list(tensor.shape) for name, tensor in loaded.model.state_dict().items()
    }
    assert loaded.metadata["bootstrap"] == initialization


def test_bootstrap_is_reproducible_and_preserves_caller_rng(tmp_path) -> None:
    torch.manual_seed(7)
    state_before = torch.random.get_rng_state()

    first = bootstrap_mini_champion(tmp_path / "first", seed=143_002)

    assert torch.equal(torch.random.get_rng_state(), state_before)
    second = bootstrap_mini_champion(tmp_path / "second", seed=143_002)
    assert first["checkpoint"]["sha256"] == second["checkpoint"]["sha256"]


def test_committed_bootstrap_reproduces_from_recorded_seed(tmp_path: Path) -> None:
    root = (
        Path(__file__).resolve().parents[2]
        / "experiments"
        / "issue-143"
        / "architecture-v2-bootstrap"
    )
    committed = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    reproduced = bootstrap_mini_champion(tmp_path / "reproduced")

    assert committed["initialization"]["seed"] == DEFAULT_BOOTSTRAP_SEED
    assert committed["checkpoint"]["sha256"] == hashlib.sha256(
        (root / "champion.pt").read_bytes()
    ).hexdigest()
    assert reproduced["checkpoint"]["sha256"] == committed["checkpoint"]["sha256"]


def test_bootstrap_refuses_to_overwrite_an_existing_experiment(tmp_path) -> None:
    output = tmp_path / "architecture-v2"
    output.mkdir()
    (output / "historical.pt").write_bytes(b"do not overwrite")

    with pytest.raises(ValueError, match="empty or not exist"):
        bootstrap_mini_champion(output)

    assert (output / "historical.pt").read_bytes() == b"do not overwrite"


def test_bootstrap_cli_reports_manifest(tmp_path, capsys) -> None:
    output = tmp_path / "architecture-v2"

    assert main(["--output-dir", str(output), "--seed", "143003"]) == 0

    assert '"seed": 143003' in capsys.readouterr().out
    assert (output / "manifest.json").is_file()
