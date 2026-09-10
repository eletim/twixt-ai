"""Tests for the canonical 512-game CUDA self-play benchmark runner."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import pytest
import torch

from twixt_ai.evaluation.cuda_selfplay_512 import (
    BenchmarkOptions,
    CONTRACT_FORMAT,
    CONTRACT_VERSION,
    BenchmarkTuning,
    _resolved_config,
    _validate_outputs,
    run_cuda_selfplay_512_benchmark,
)
from twixt_ai.evaluation.cuda_inference_profile import (
    CudaInferencePhaseProfile,
)
from twixt_ai.game import GameState, experiment_board, legal_peg_placements
from twixt_ai.models import (
    MINI_POLICY_VALUE_CONFIG,
    PolicyValueNetwork,
    save_policy_value_checkpoint,
)
from twixt_ai.search import MCTSAgent
from twixt_ai.search.neural import NeuralPolicyValue
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


def test_detailed_profile_is_an_explicit_non_workload_option() -> None:
    options = BenchmarkOptions(detailed_inference_profile=True)

    assert options.detailed_inference_profile is True


def test_detailed_profile_ranks_host_phases_and_keeps_cuda_separate() -> None:
    profile = CudaInferencePhaseProfile()
    profile.batches = 2
    profile.positions = 16
    profile.total_evaluate_batch_seconds = 1.0
    profile.host_seconds["cpu_encoding_and_stack"] = 0.6
    profile.host_seconds["python_result_extraction"] = 0.2
    profile.cuda_seconds["model_execution"] = 0.05

    result = profile.to_dict()

    assert result["ranked_host_phases"][0] == {
        "rank": 1,
        "phase": "cpu_encoding_and_stack",
        "seconds": 0.6,
        "percent_of_evaluate_batch": 60.0,
    }
    assert result["host_milliseconds_per_batch"]["cpu_encoding_and_stack"] == 300
    assert result["cuda_milliseconds_per_batch"]["model_execution"] == 25
    assert result["interpretation"]["host_and_cuda_times_are_not_additive"]


def test_detailed_profile_reports_batching_separately_from_contention() -> None:
    profile = CudaInferencePhaseProfile()
    profile.queue_submission(0.000001, 0.000002)
    profile.batch_dispatch(
        "full_batch", 0.0005, 0.000003, 0.000004, 0.000007
    )
    profile.batch_completion(0.000005, 0.000006)

    result = profile.contention_to_dict()

    assert result["producer_queue_condition_lock_acquisition"]["average"] == 1
    assert result["producer_queue_condition_critical_section"]["average"] == 2
    assert result["worker_dispatch_condition_lock_acquisition"]["average"] == 3
    assert result["batch_formation_delay"]["all"]["average"] == 0.5
    assert result["worker_condition_wait_deadline_overshoot"]["average"] == 0.007
    assert result["batch_formation_delay"]["by_flush_reason"]["full_batch"][
        "samples"
    ] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_detailed_profile_preserves_policy_value_results() -> None:
    model = PolicyValueNetwork(MINI_POLICY_VALUE_CONFIG).cuda()
    state = GameState(board=experiment_board("mini"))
    moves = legal_peg_placements(state)
    requests = ((state, moves),) * 8

    expected = NeuralPolicyValue(model).evaluate_batch(requests)
    profile = CudaInferencePhaseProfile()
    actual = NeuralPolicyValue(model, observer=profile).evaluate_batch(requests)

    assert actual == expected
    assert profile.batches == 1
    assert profile.positions == 8
    assert profile.cuda_seconds["model_execution"] > 0


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


def _write_tiny_output(
    root: Path,
) -> tuple[dict[str, object], BatchConfig, dict[str, object]]:
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
    run_batch(factory, factory, config=batch_config, output_dir=root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    return config, batch_config, summary


def test_output_validation_checks_complete_reproducible_match_artifacts(
    tmp_path: Path,
) -> None:
    config, batch_config, summary = _write_tiny_output(tmp_path)

    validation = _validate_outputs(
        tmp_path, summary, config, batch_config.to_dict()
    )

    assert validation["all_required_artifacts_valid"] is True
    assert validation["all_match_records_replay_valid"] is True
    assert validation["all_decision_seeds_match_contract_derivation"] is True
    assert validation["all_search_parameters_match_contract"] is True
    assert validation["all_value_targets_valid"] is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("simulations", 3, "search budget"),
        ("exploration", 0.5, "exploration"),
        ("rollout_limit", 3, "rollout limit"),
        ("rollout_evaluator", "other_evaluator", "rollout evaluator"),
        (
            "progressive_widening",
            {"constant": 2.0, "exponent": 0.5},
            "progressive widening",
        ),
        (
            "progressive_widening",
            {"constant": 1.5, "exponent": 0.75},
            "progressive widening",
        ),
    ],
)
def test_output_validation_rejects_changed_search_parameter(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    config, batch_config, summary = _write_tiny_output(tmp_path)
    artifact = tmp_path / "games/game-000000.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    metadata = payload["decisions"][0]["metadata"]
    metadata[field] = value
    if field == "simulations":
        metadata["root_moves"][0]["visits"] += 1
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _validate_outputs(tmp_path, summary, config, batch_config.to_dict())
