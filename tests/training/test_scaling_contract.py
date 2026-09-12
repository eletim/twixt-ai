"""Regression checks for the immutable Issue 128 scaling protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from twixt_ai.training import generations_cli
from twixt_ai.training.generations import MiniGenerationConfig


def test_issue_128_contract_fixes_matched_scaling_semantics() -> None:
    contract = json.loads(
        Path("experiments/issue-128/scaling-contract.json").read_text(
            encoding="utf-8"
        )
    )
    champion = contract["starting_champion"]
    champion_path = Path(champion["path"])

    assert hashlib.sha256(champion_path.read_bytes()).hexdigest() == champion["sha256"]
    assert champion["sha256"] == contract["selfplay"]["teacher_sha256"]
    assert champion["sha256"] == contract["training"]["initial_checkpoint_sha256"]
    assert contract["board"]["width"] == contract["board"]["height"] == 10
    assert contract["encoding"] == {
        "version": 1,
        "channels": 22,
        "perspective": "side-to-move",
    }
    assert contract["architecture"]["encoding_version"] == 1
    assert contract["architecture"]["input_channels"] == 22
    assert contract["selfplay"]["mcts"] == {
        "guidance": "policy-value",
        "simulations": 64,
        "exploration": 0.7,
        "rollout_limit": 4,
        "rollout_evaluator": "heuristic_rollout_value",
        "progressive_widening": {"constant": 3.0, "exponent": 0.5},
        "move_selection": "maximum root visits, then mean value, then row-major coordinate",
        "temperature": 0.0,
        "root_noise": False,
    }
    assert contract["training"]["optimizer"] == "adamw"
    assert contract["training"]["scheduler"] == "none"
    assert contract["evaluation"]["paired_role_swaps"] is True
    assert contract["evaluation"]["games_per_opponent"] == 40
    assert contract["evaluation"]["seed"] == 1_289_000
    assert contract["evaluation"]["generation_cli_option"] == (
        "--evaluation-seed 1289000"
    )
    assert contract["seeds"]["promotion_and_fixed_opponent_evaluation"] == 1_289_000
    assert "generation_gate" not in contract["seeds"]


def test_issue_128_requires_fresh_1k_and_explicit_stop_gates() -> None:
    contract = json.loads(
        Path("experiments/issue-128/scaling-contract.json").read_text(
            encoding="utf-8"
        )
    )
    stages = contract["stages"]

    assert [(stage["name"], stage["games"]) for stage in stages] == [
        ("matched-1k", 1000),
        ("5k", 5000),
        ("10k", 10000),
        ("25k", 25000),
        ("50k", 50000),
    ]
    assert stages[0]["required"] is True
    assert "generation-2" in stages[0]["reason"]
    assert contract["invariants"]["historical_generation_3_dataset_is_excluded"] is True
    assert contract["gates"]["promotion"]["minimum_wins"] == 22
    assert contract["gates"]["meaningful_scaling_gain"]["minimum_wins"] == 22
    assert contract["gates"]["saturation"]["loss_and_calibration_override_strength"] is False
    assert contract["gates"]["stretch_50k"]["maximum_projected_storage_bytes"] == 25 * 1024**3
    assert "dataset JSONL shards" in contract["retention"]["external_durable_storage"]
    manifest = contract["retention"]["manifest"]
    assert manifest["format"] == "twixt-ai-artifact-retention-manifest"
    assert manifest["categories"] == [
        "selfplay", "dataset", "training", "evaluation"
    ]
    assert manifest["object_fields"] == ["path", "sha256", "bytes"]
    assert "canonical SHA-256" in manifest["inventory_complete"]
    assert "separate process" in manifest["storage_attestation"]["timing"]
    assert "both" in manifest["pruning_ready"]
    assert "All four categories" in manifest["validation"]
    assert "every fixed-opponent evaluation" in contract["reporting"][
        "evaluation_artifacts"
    ]


def test_documented_matched_1k_command_resolves_the_complete_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = json.loads(
        Path("experiments/issue-128/scaling-contract.json").read_text(
            encoding="utf-8"
        )
    )
    command = contract["generation_cli"]
    fixed = command["fixed_arguments"]
    stage = contract["stages"][0]
    captured: dict[str, object] = {}

    def run_stub(
        initial_champion: Path, output_dir: Path, *, config: MiniGenerationConfig
    ) -> dict[str, object]:
        captured.update({
            "initial_champion": initial_champion,
            "output_dir": output_dir,
            "config": config,
        })
        return {"status": "fixture"}

    monkeypatch.setattr(generations_cli, "run_mini_training_generations", run_stub)
    output_dir = Path("/tmp/issue-128/matched-1k")
    artifact_uri = "s3://bucket/issue-128/matched-1k"
    argv = [
        "--initial-champion", command["initial_champion"],
        "--output-dir", str(output_dir),
        "--artifact-uri", artifact_uri,
        "--games-per-generation", str(stage["games"]),
        "--seed", str(stage["root_seed"]),
    ]
    for name, value in fixed.items():
        argv.extend((f"--{name.replace('_', '-')}", str(value)))

    assert generations_cli.main(argv) == 0

    resolved = captured["config"]
    assert isinstance(resolved, MiniGenerationConfig)
    assert set(fixed) | {"games_per_generation", "seed", "artifact_uri"} == set(
        MiniGenerationConfig.__dataclass_fields__
    )
    for name, value in fixed.items():
        assert getattr(resolved, name) == value
    assert resolved.games_per_generation == stage["games"]
    assert resolved.seed == stage["root_seed"]
    assert resolved.artifact_uri == artifact_uri
    assert captured["initial_champion"] == Path(command["initial_champion"])
    assert captured["output_dir"] == output_dir

    selfplay = contract["selfplay"]
    search = selfplay["mcts"]
    dataset = contract["dataset"]
    training = contract["training"]
    evaluation = contract["evaluation"]
    evaluation_search = evaluation["learned_and_non_neural_mcts"]
    assert fixed == {
        "generations": 1,
        "dataset_window": dataset["window_generations"],
        "selfplay_simulations": search["simulations"],
        "selfplay_exploration": search["exploration"],
        "selfplay_progressive_widening_constant": search[
            "progressive_widening"
        ]["constant"],
        "selfplay_progressive_widening_exponent": search[
            "progressive_widening"
        ]["exponent"],
        "evaluation_games": evaluation["games_per_opponent"],
        "evaluation_simulations": evaluation_search["simulations"],
        "rollout_limit": search["rollout_limit"],
        "workers": selfplay["workers"],
        "inference_batch_size": selfplay["inference_batch_size"],
        "inference_max_wait_seconds": selfplay["inference_max_wait_seconds"],
        "epochs": training["epochs"],
        "batch_size": training["batch_size"],
        "learning_rate": training["learning_rate"],
        "weight_decay": training["weight_decay"],
        "selection_metric": training["selection_metric"],
        "validation_fraction": dataset["validation_fraction"],
        "shard_size": dataset["shard_size"],
        "promotion_win_rate": (
            contract["gates"]["promotion"]["minimum_wins"]
            / evaluation["games_per_opponent"]
        ),
        "evaluation_seed": evaluation["seed"],
        "device": selfplay["device"],
    }
    assert search["rollout_limit"] == evaluation_search["rollout_limit"]
    assert selfplay["device"] == training["device"]

    documentation = Path("docs/issue-128-strength-scaling-contract.md").read_text(
        encoding="utf-8"
    )
    for name, value in fixed.items():
        assert f"--{name.replace('_', '-')} {value}" in documentation
    assert f"--games-per-generation {stage['games']}" in documentation
    assert f"--seed {stage['root_seed']}" in documentation
    assert command["initial_champion"] in documentation
