"""Regression checks for the immutable Issue 128 scaling protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


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
