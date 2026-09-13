"""Regression checks for the retained Issue 151 1k stage evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "experiments" / "issue-151" / "1k"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(matchup: dict[str, object]) -> list[int]:
    strength = matchup["strength"]
    assert isinstance(strength, dict)
    wins = strength["wins"]
    assert isinstance(wins, dict)
    candidate = matchup["candidate"]
    baseline = matchup["baseline"]
    assert isinstance(candidate, str)
    assert isinstance(baseline, str)
    return [wins[candidate], strength["draws"], wins[baseline]]


def test_stage_controls_match_issue_143_except_game_count() -> None:
    config = _json(STAGE / "config.json")
    baseline = _json(
        ROOT / "experiments/issue-143/v2-selfplay-dataset/report.json"
    )
    generated = _json(STAGE / "selfplay-report.json")

    assert config["stage"] == {"games": 1000, "name": "1k"}
    current = generated["resolved_config"]
    previous = baseline["resolved_config"]
    assert isinstance(current, dict)
    assert isinstance(previous, dict)
    assert current["games"] == 1000
    assert previous["games"] == 100
    for key in ("board", "dataset", "execution", "search"):
        assert current[key] == previous[key]
    assert generated["seeds"] == baseline["seeds"]

    model = config["model"]
    assert isinstance(model, dict)
    assert model["encoding"] == {"planes": 22, "version": 1}
    assert model["trunk_channels"] == 8
    assert model["residual_blocks"] == 1
    assert model["policy_head"] == [800, 256, 256, 100]
    assert model["value_head"] == [800, 256, 256, 1, "tanh"]


def test_stage_hashes_and_measurements_match_source_artifacts() -> None:
    report = _json(STAGE / "report.json")
    artifacts = report["artifacts"]
    assert isinstance(artifacts, dict)
    for artifact in artifacts.values():
        assert isinstance(artifact, dict)
        path = ROOT / artifact["path"]
        assert path.is_file()
        assert artifact["sha256"] == _sha256(path)

    manifest = _json(STAGE / "dataset/manifest.json")
    dataset = report["dataset"]
    assert isinstance(dataset, dict)
    assert dataset["games"] == manifest["source_games"] == 1000
    assert dataset["positions"] == manifest["examples"] == 47561
    assert dataset["train_positions"] == manifest["splits"]["train"]["examples"]
    assert dataset["validation_positions"] == manifest["splits"]["validation"][
        "examples"
    ]

    summary = _json(STAGE / "candidate/summary.json")
    training = report["training"]
    assert isinstance(training, dict)
    assert summary["dataset_sha256"] == artifacts["dataset_manifest"]["sha256"]
    assert summary["best_epoch"] == training["best_epoch"]
    best = summary["history"][summary["best_epoch"] - 1]
    final = summary["history"][-1]
    assert best["validation_policy_loss"] == training[
        "best_validation_policy_loss"
    ]
    assert best["validation_value_loss"] == training["best_validation_value_loss"]
    for split in ("train", "validation"):
        for head in ("policy", "value"):
            key = f"{split}_{head}_loss"
            assert final[key] == training[f"final_{key}"]

    diagnostics = _json(STAGE / "value-head.json")
    quality = report["value_quality"]
    assert isinstance(quality, dict)
    assert diagnostics["checkpoint"]["sha256"] == artifacts["best_checkpoint"][
        "sha256"
    ]
    assert diagnostics["checkpoint"]["dataset_sha256_matches"] is True
    for split in ("train", "validation"):
        metrics = diagnostics["splits"][split]["metrics"]
        for metric in ("mse", "mae"):
            assert quality[split][metric] == metrics[metric]
    assert quality["validation"]["calibration_error"] == diagnostics["splits"][
        "validation"
    ]["metrics"]["calibration_error"]
    assert quality["validation_minus_train_mse"] == diagnostics["generalization"][
        "mse_gap"
    ]


def test_paired_strength_and_baseline_comparison_are_source_backed() -> None:
    report = _json(STAGE / "report.json")
    evaluation = _json(STAGE / "evaluation.json")
    config = _json(STAGE / "config.json")

    assert evaluation["config"]["games_per_matchup"] == 40
    assert evaluation["config"]["seed"] == 1289000
    assert evaluation["methodology"]["paired_role_swaps"] is True
    assert evaluation["methodology"]["shared_pair_seed_schedule"] is True
    assert evaluation["config"]["settings"][0] == config["evaluation"]["search"]
    assert evaluation["config"]["heuristic_search"] == config["evaluation"][
        "heuristic_search"
    ]

    for matchup in evaluation["matchups"]:
        assert report["strength"][matchup["guidance"]][matchup["baseline"]] == (
            _record(matchup)
        )
        assert len(matchup["games"]) == 40

    baseline = _json(ROOT / "experiments/issue-143/final-report.json")
    assert report["baseline_100_games"][
        "strength"
    ] == baseline["architecture_v2_strength_wins_draws_losses"]
    old_value = baseline["architecture_v2_value_quality"]
    assert report["baseline_100_games"]["held_out_value"] == {
        "mae": old_value["validation_mae"],
        "mse": old_value["validation_mse"],
        "mse_gap": old_value["validation_minus_training_mse"],
    }


def test_readme_cites_headline_stage_results() -> None:
    report = _json(STAGE / "report.json")
    readme = (STAGE / "README.md").read_text(encoding="utf-8")

    assert report["artifacts"]["dataset_manifest"]["sha256"] in readme
    assert report["artifacts"]["best_checkpoint"]["sha256"] in readme
    for baselines in report["strength"].values():
        for wins, draws, losses in baselines.values():
            assert f"{wins}-{draws}-{losses}" in readme
    validation = report["value_quality"]["validation"]
    assert f"{validation['mse']:.6f}" in readme
    assert f"{validation['mae']:.6f}" in readme
    assert "This candidate is not promoted." in readme
