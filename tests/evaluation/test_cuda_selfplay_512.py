"""Tests for the canonical 512-game CUDA self-play benchmark runner."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import pytest

from twixt_ai.evaluation.cuda_selfplay_512 import (
    CONTRACT_FORMAT,
    CONTRACT_VERSION,
    BenchmarkTuning,
    _resolved_config,
    _validate_outputs,
    run_cuda_selfplay_512_benchmark,
)
from twixt_ai.game import experiment_board
from twixt_ai.models import (
    MINI_POLICY_VALUE_CONFIG,
    PolicyValueNetwork,
    save_policy_value_checkpoint,
)
from twixt_ai.search import MCTSAgent
from twixt_ai.selfplay.batch import BatchConfig, run_batch


def _tiny_contract(games: int = 2) -> dict[str, object]:
    return {
        "format": CONTRACT_FORMAT,
        "version": CONTRACT_VERSION,
        "board": {
            "preset": "mini",
            "width": 10,
            "height": 10,
            "rules": {
                "version": "v0.0.1",
                "move": "peg-placement-only",
                "links": "automatic",
                "pie_or_swap": False,
            },
        },
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
            "mcts": {
                "simulations": 2,
                "exploration": 1.4142135623730951,
                "rollout_limit": 2,
                "rollout_evaluator": "heuristic_rollout_value",
                "progressive_widening": {"constant": 1.5, "exponent": 0.5},
            },
            "seeds": {"python_hash_seed": 0, "batch_seed": 123456},
            "workers": {"mode": "thread"},
            "shared_inference": {"model_instances": 1},
        },
        "optimization_variables": {
            "worker_concurrency": {"default": 1, "minimum": 1},
            "inference_batch_size": {"default": 2, "minimum": 1},
            "queue_flush_max_wait_seconds": {
                "default": 0.0005,
                "minimum": 0.0,
            },
        },
        "output": {"timing_scope": "test scope"},
    }


def test_rejects_unexpected_contract_format(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps({"format": "not-it"}), encoding="utf-8")
    with pytest.raises(ValueError, match="format"):
        run_cuda_selfplay_512_benchmark(contract_path, tmp_path / "out")


def test_rejects_checkpoint_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    checkpoint = tmp_path / "model.pt"
    save_policy_value_checkpoint(
        checkpoint, PolicyValueNetwork(MINI_POLICY_VALUE_CONFIG)
    )
    with pytest.raises(ValueError, match="checkpoint"):
        run_cuda_selfplay_512_benchmark(
            "benchmarks/mini-cuda-selfplay-512-v006-contract.json",
            tmp_path / "out",
            checkpoint_path=checkpoint,
        )


def test_rejects_wrong_pythonhashseed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "1")
    with pytest.raises(ValueError, match="PYTHONHASHSEED"):
        run_cuda_selfplay_512_benchmark(
            "benchmarks/mini-cuda-selfplay-512-v006-contract.json",
            tmp_path / "out",
        )


def test_runner_rejects_legacy_v1_contract(tmp_path: Path) -> None:
    contract = _tiny_contract()
    contract["version"] = 1
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ValueError, match="requires contract version 2"):
        run_cuda_selfplay_512_benchmark(legacy, tmp_path / "out")


def test_v006_contract_exposes_only_semantics_neutral_tuning() -> None:
    contract = _tiny_contract()
    config, values = _resolved_config(
        contract,
        BenchmarkTuning(
            worker_concurrency=7,
            inference_batch_size=16,
            queue_flush_max_wait_seconds=0.002,
        ),
    )

    assert values == {
        "worker_concurrency": 7,
        "inference_batch_size": 16,
        "queue_flush_max_wait_seconds": 0.002,
    }
    assert config["workers"] == {"mode": "thread", "count": 7}
    assert config["shared_inference"] == {
        "model_instances": 1,
        "batch_size": 16,
        "max_wait_seconds": 0.002,
    }
    assert config["games"] == 2
    assert config["mcts"] == contract["config"]["mcts"]
    assert config["seeds"] == contract["config"]["seeds"]


def test_v006_contract_rejects_an_extra_optimization_variable() -> None:
    contract = _tiny_contract()
    contract["optimization_variables"]["simulations"] = {"default": 1}

    with pytest.raises(ValueError, match="exactly"):
        _resolved_config(contract, BenchmarkTuning())


@pytest.mark.parametrize("fixed_field", ["games", "checkpoint", "seeds", "mcts"])
def test_runner_rejects_modified_fixed_v006_contract(
    tmp_path: Path, fixed_field: str
) -> None:
    canonical_path = Path("benchmarks/mini-cuda-selfplay-512-v006-contract.json")
    contract = json.loads(canonical_path.read_text(encoding="utf-8"))
    if fixed_field == "games":
        contract["config"]["games"] = 1
    elif fixed_field == "checkpoint":
        contract["checkpoint"]["sha256"] = "0" * 64
    elif fixed_field == "seeds":
        contract["config"]["seeds"]["batch_seed"] += 1
    else:
        contract["config"]["mcts"]["simulations"] -= 1
    modified = tmp_path / "modified-contract.json"
    modified.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical contract"):
        run_cuda_selfplay_512_benchmark(modified, tmp_path / "out")


def test_canonical_v006_contract_fixes_semantics() -> None:
    path = Path("benchmarks/mini-cuda-selfplay-512-v006-contract.json")
    contract = json.loads(path.read_text(encoding="utf-8"))

    assert contract["version"] == CONTRACT_VERSION
    assert contract["release"] == "v0.0.6"
    assert contract["config"]["games"] == 512
    assert contract["config"]["device"] == "cuda"
    assert contract["config"]["mcts"]["simulations"] == 4
    assert set(contract["optimization_variables"]) == {
        "worker_concurrency",
        "inference_batch_size",
        "queue_flush_max_wait_seconds",
    }
    assert contract["invariants"]["allowed_optimization_variables"] == [
        "worker_concurrency",
        "inference_batch_size",
        "queue_flush_max_wait_seconds",
    ]


def test_output_validation_checks_complete_reproducible_match_artifacts(
    tmp_path: Path,
) -> None:
    contract = _tiny_contract(games=1)
    config, _ = _resolved_config(contract, BenchmarkTuning())
    batch_config = BatchConfig(
        games=1,
        workers=1,
        seed=config["seeds"]["batch_seed"],
        board=experiment_board("mini"),
        red_agent=config["agents"]["red"],
        black_agent=config["agents"]["black"],
        worker_mode="thread",
    )
    factory = partial(MCTSAgent, simulations=2, rollout_limit=2)
    run_batch(factory, factory, config=batch_config, output_dir=tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))

    validation = _validate_outputs(
        tmp_path, summary, config, batch_config.to_dict()
    )

    assert validation["all_required_artifacts_valid"] is True
    assert validation["all_match_records_replay_valid"] is True
    assert validation["all_decision_seeds_match_contract_derivation"] is True
    assert validation["all_search_parameters_match_contract"] is True
    assert validation["all_value_targets_valid"] is True
