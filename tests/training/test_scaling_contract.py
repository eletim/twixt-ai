"""Regression checks for the immutable Issue 128 scaling protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlparse

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


def test_issue_128_5k_stage_preserves_protocol_and_negative_results() -> None:
    one_k = json.loads(
        Path("experiments/issue-128/matched-1k/report.json").read_text(
            encoding="utf-8"
        )
    )
    five_k = json.loads(
        Path("experiments/issue-128/5k/report.json").read_text(encoding="utf-8")
    )
    one_k_config = one_k["config"]
    five_k_config = five_k["config"]

    differing = {
        key for key in one_k_config if one_k_config[key] != five_k_config[key]
    }
    assert differing == {"artifact_uri", "games_per_generation", "seed"}
    assert five_k_config["games_per_generation"] == 5_000
    assert five_k_config["seed"] == 1_285_000

    generation = five_k["generations"][0]
    teacher_sha256 = "aee1036dbda115eeec0e245909d30e1f8330454a82099e852e1b6a9c26c0dab9"
    assert generation["champion_before"]["sha256"] == teacher_sha256
    assert generation["training"]["initialized_from_sha256"] == teacher_sha256
    assert generation["dataset"]["manifest"]["source_games"] == 5_000
    assert generation["evaluation"]["promotion"]["candidate_wins"] == 29
    assert generation["evaluation"]["promotion"]["promoted"] is True

    previous_stage = generation["fixed_opponent_evaluations"][0]
    assert previous_stage["opponent"] == "previous retained stage candidate"
    assert previous_stage["candidate_wins"] == 19
    assert previous_stage["minimum_wins"] == 22
    assert previous_stage["meaningful_scaling_gain"] is False
    assert generation["scaling_decision"]["decision"] == "saturated"
    assert generation["scaling_decision"]["next_optional_stage"] == "not run"


def test_issue_128_final_report_preserves_saturation_and_provenance() -> None:
    report = Path("docs/issue-128-strength-scaling.md").read_text(encoding="utf-8")

    required_evidence = (
        "aee1036dbda115eeec0e245909d30e1f8330454a82099e852e1b6a9c26c0dab9",
        "9d79b587e041f044084d8f14933b98baca31b9fbc0c799b358322e6160d5d6f7",
        "d0ab6252d29945b1e25735615460cc42f9528191ce46e5e4e2c25f29d2360af8",
        "5320f9ca4c146b1c925054ac4bd1e5e605d5f228c2dfb1d8c2a8a490d0981af2",
        "60087bdb8c11fdd04c665775b6dc0e5595b206a8cac26859f4330ad8bc14db19",
        "9895fbb545029311942fe2b124b3543ee2e4904dbe55522d238c165d1b459a2a",
        "5447d23e68d0df76348c4077d502a8e5fd227f55f236349544e9d1abdbbc03e1",
    )
    assert all(value in report for value in required_evidence)
    assert "19-21-0" in report
    assert "32.9%-62.5%" in report
    assert "fails 22-win scaling gate; saturated" in report
    assert "The 10k stage was conditional" in report
    assert "25k was also ineligible" in report
    assert "50k was neither\neligible nor run" in report
    assert "does not\nreverse the saturation decision" in report
    assert "quality of the fixed teacher/search targets" in report


@pytest.mark.parametrize("stage", ("matched-1k", "5k"))
def test_issue_128_retained_stage_has_matching_storage_attestation(
    stage: str,
) -> None:
    report = json.loads(
        Path(f"experiments/issue-128/{stage}/report.json").read_text(
            encoding="utf-8"
        )
    )
    retention = report["generations"][0]["retention_manifest"]
    payload = {
        "categories": retention["categories"],
        "files": retention["files"],
        "bytes": retention["bytes"],
    }
    inventory_sha256 = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    attestation = retention["storage_attestation"]

    assert retention["inventory_complete"] is True
    assert inventory_sha256 == retention["inventory_sha256"]
    assert attestation["verified"] is True
    assert attestation["external_uri"] == retention["external_uri"]
    assert attestation["inventory_sha256"] == inventory_sha256
    assert attestation["verified_at"]
    assert attestation["verifier"]


@pytest.mark.parametrize("stage", ("matched-1k", "5k"))
def test_issue_128_file_retention_contains_every_inventoried_object(
    stage: str,
) -> None:
    report = json.loads(
        Path(f"experiments/issue-128/{stage}/report.json").read_text(
            encoding="utf-8"
        )
    )
    retention = report["generations"][0]["retention_manifest"]
    parsed_uri = urlparse(retention["external_uri"])

    assert parsed_uri.scheme == "file"
    assert not parsed_uri.netloc
    root = Path(unquote(parsed_uri.path))
    assert root.name == "generation-0001"
    if not root.is_dir():
        pytest.skip(f"external retention storage is not mounted: {root}")

    checked_files = 0
    checked_bytes = 0
    for category in retention["categories"].values():
        for item in category["objects"]:
            retained_path = root / item["path"]
            assert retained_path.is_file(), retained_path
            assert retained_path.stat().st_size == item["bytes"]
            assert hashlib.sha256(retained_path.read_bytes()).hexdigest() == (
                item["sha256"]
            )
            checked_files += 1
            checked_bytes += item["bytes"]

    assert checked_files == retention["files"]
    assert checked_bytes == retention["bytes"]
