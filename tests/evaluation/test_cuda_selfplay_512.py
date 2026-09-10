"""Tests for the canonical 512-game CUDA self-play benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from twixt_ai.evaluation.cuda_selfplay_512 import (
    CONTRACT_FORMAT,
    BenchmarkOptions,
    run_cuda_selfplay_512_benchmark,
)
from twixt_ai.models import (
    MINI_POLICY_VALUE_CONFIG,
    PolicyValueNetwork,
    save_policy_value_checkpoint,
)


def _tiny_contract(games: int = 2) -> dict[str, object]:
    return {
        "format": CONTRACT_FORMAT,
        "version": 1,
        "board": {"preset": "mini", "width": 10, "height": 10},
        "checkpoint": {
            "path": "model.pt",
            "sha256": "placeholder",
            "encoding_version": MINI_POLICY_VALUE_CONFIG.encoding_version,
        },
        "config": {
            "games": games,
            "device": "cuda",
            "agents": {
                "red": "checkpoint-mcts",
                "black": "checkpoint-mcts",
                "guidance": "policy-value",
            },
            "mcts": {"simulations": 2, "rollout_limit": 2},
            "seeds": {"python_hash_seed": 0, "batch_seed": 123456},
            "workers": {"count": 1, "mode": "thread"},
            "shared_inference": {
                "model_instances": 1,
                "batch_size": 2,
                "max_wait_seconds": 0.0005,
            },
        },
        "output": {"timing_scope": "test scope"},
    }


def _write_contract_and_checkpoint(
    root: Path, games: int = 2
) -> tuple[Path, Path]:
    checkpoint = root / "model.pt"
    save_policy_value_checkpoint(
        checkpoint, PolicyValueNetwork(MINI_POLICY_VALUE_CONFIG)
    )
    contract = _tiny_contract(games)
    import hashlib

    contract["checkpoint"]["sha256"] = hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    contract_path = root / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    return contract_path, checkpoint


def test_rejects_unexpected_contract_format(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps({"format": "not-it"}), encoding="utf-8")
    with pytest.raises(ValueError, match="format"):
        run_cuda_selfplay_512_benchmark(contract_path, tmp_path / "out")


def test_rejects_checkpoint_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    contract_path, checkpoint = _write_contract_and_checkpoint(tmp_path)
    contract = json.loads(contract_path.read_text())
    contract["checkpoint"]["sha256"] = "0" * 64
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint"):
        run_cuda_selfplay_512_benchmark(
            contract_path, tmp_path / "out", repo_root=tmp_path
        )


def test_rejects_wrong_pythonhashseed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "1")
    contract_path, _ = _write_contract_and_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="PYTHONHASHSEED"):
        run_cuda_selfplay_512_benchmark(
            contract_path, tmp_path / "out", repo_root=tmp_path
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_end_to_end_tiny_workload_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    contract_path, _ = _write_contract_and_checkpoint(tmp_path, games=2)
    output_dir = tmp_path / "out"
    report = run_cuda_selfplay_512_benchmark(
        contract_path,
        output_dir,
        repo_root=tmp_path,
        options=BenchmarkOptions(
            gpu_sample_interval_seconds=0.05,
            phase_sample_interval_seconds=0.005,
            implementation_label="test",
        ),
    )
    assert report["workload"]["games"] == 2
    assert report["workload"]["completed"] == 2
    assert report["workload"]["failed"] == 0
    assert report["validation"]["all_policy_root_visit_sums_valid"] is True
    assert report["validation"]["policy_root_visit_sum_expected"] == 2
    assert report["timing"]["end_to_end_wall_seconds"] > 0
    assert report["rates"]["games_per_hour"] > 0
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "games" / "game-000000.json").exists()
    assert (output_dir / "games" / "game-000001.json").exists()
