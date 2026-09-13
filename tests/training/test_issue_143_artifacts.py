"""Regression checks for the evidence cited by the Issue 143 report."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ISSUE_143 = ROOT / "experiments" / "issue-143"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _v2_strength_record(matchup: dict[str, object]) -> list[int]:
    strength = matchup["strength"]
    assert isinstance(strength, dict)
    wins = strength["wins"]
    assert isinstance(wins, dict)
    candidate = matchup["candidate"]
    baseline = matchup["baseline"]
    assert isinstance(candidate, str)
    assert isinstance(baseline, str)
    return [wins[candidate], strength["draws"], wins[baseline]]


def test_final_report_citations_match_source_artifacts() -> None:
    report = _json(ISSUE_143 / "final-report.json")
    source_paths = report["source_artifacts"]
    assert isinstance(source_paths, dict)
    assert all((ROOT / path).is_file() for path in source_paths.values())

    strength = _json(ROOT / source_paths["architecture_v2_strength"])
    protocol = report["matched_strength_protocol"]
    assert isinstance(protocol, dict)
    assert protocol["games_per_matchup"] == strength["config"]["games_per_matchup"]
    assert protocol["seed"] == strength["config"]["seed"]
    assert protocol["setting"] == strength["config"]["settings"][0]["name"]
    assert protocol["opponents"] == strength["config"]["baselines"]
    assert protocol["paired_role_swaps"] is True
    setting = strength["config"]["settings"][0]
    for key in (
        "simulations",
        "exploration",
        "progressive_widening_constant",
        "progressive_widening_exponent",
    ):
        assert protocol[key] == setting[key]
    assert protocol["rollout_limit"] == strength["config"]["rollout_limit"]

    v2_records = report["architecture_v2_strength_wins_draws_losses"]
    for matchup in strength["matchups"]:
        assert v2_records[matchup["guidance"]][matchup["baseline"]] == (
            _v2_strength_record(matchup)
        )

    v1_records = report["architecture_v1_policy_value_strength_wins_draws_losses"]
    opponent_keys = {
        "matched non-neural MCTS": "matched-non-neural-mcts",
        "heuristic search": "heuristic-search",
    }
    for stage, source_key in (
        ("matched-1k", "architecture_v1_matched_1k"),
        ("5k", "architecture_v1_5k"),
    ):
        source = _json(ROOT / source_paths[source_key])
        for evaluation in source["generations"][0]["fixed_opponent_evaluations"]:
            if evaluation["opponent"] not in opponent_keys:
                continue
            key = opponent_keys[evaluation["opponent"]]
            assert v1_records[stage][key] == [
                evaluation["candidate_wins"],
                evaluation["candidate_draws"],
                evaluation["candidate_losses"],
            ]

    diagnostics = _json(ROOT / source_paths["architecture_v2_value_quality"])
    manifest = _json(
        ISSUE_143 / "v2-selfplay-dataset" / "dataset" / "manifest.json"
    )
    cited = report["architecture_v2_value_quality"]
    train = diagnostics["splits"]["train"]["metrics"]
    validation = diagnostics["splits"]["validation"]["metrics"]
    assert cited["training_positions"] == train["examples"]
    assert cited["training_mse"] == train["mse"]
    assert cited["training_mae"] == train["mae"]
    assert cited["validation_positions"] == validation["examples"]
    assert cited["validation_mse"] == validation["mse"]
    assert cited["validation_mae"] == validation["mae"]
    assert cited["validation_calibration_error"] == validation["calibration_error"]
    assert cited["validation_draw_targets"] == validation["target_balance"][
        "counts"
    ]["0"]
    assert cited["validation_minus_training_mse"] == diagnostics[
        "generalization"
    ]["mse_gap"]

    validation_games: set[str] = set()
    for shard in manifest["splits"]["validation"]["shards"]:
        with (ISSUE_143 / "v2-selfplay-dataset" / "dataset" / shard["path"]).open(
            encoding="utf-8"
        ) as stream:
            validation_games.update(
                json.loads(line)["id"].split(":", 1)[0] for line in stream
            )
    assert cited["validation_source_games"] == len(validation_games)

    negative_bin = diagnostics["splits"]["validation"]["calibration"][0]
    positive_bin = diagnostics["splits"]["validation"]["calibration"][-1]
    confident = cited["most_confident_signed_bins"]
    assert confident["negative_calibration_error"] == round(
        negative_bin["calibration_error"], 4
    )
    assert confident["positive_calibration_error"] == round(
        positive_bin["calibration_error"], 4
    )
    assert confident["predictions"] == (
        negative_bin["examples"] + positive_bin["examples"]
    )
    assert confident["wrong_sign_predictions"] == (
        negative_bin["target_balance"]["counts"]["1"]
        + positive_bin["target_balance"]["counts"]["-1"]
    )

    v1_value = _json(ROOT / source_paths["architecture_v1_value_context"])
    assert report["architecture_v1_value_context"] == {
        "validation_positions": v1_value["splits"]["validation"]["metrics"]["examples"],
        "validation_mse": v1_value["splits"]["validation"]["metrics"]["mse"],
    }


def test_final_report_records_selfplay_search_mismatch() -> None:
    report = _json(ISSUE_143 / "final-report.json")
    comparison = report["selfplay_search_protocols"]
    field_names = {
        "simulations": "selfplay_simulations",
        "exploration": "selfplay_exploration",
        "rollout_limit": "rollout_limit",
        "progressive_widening_constant": "selfplay_progressive_widening_constant",
        "progressive_widening_exponent": "selfplay_progressive_widening_exponent",
    }
    for stage, source_key in (
        ("matched-1k", "architecture_v1_matched_1k"),
        ("5k", "architecture_v1_5k"),
    ):
        source = _json(ROOT / report["source_artifacts"][source_key])
        expected = {
            report_key: source["config"][source_key_name]
            for report_key, source_key_name in field_names.items()
        }
        assert comparison["architecture_v1_matched"] == expected, stage

    v2 = _json(ROOT / report["source_artifacts"]["architecture_v2_selfplay_dataset"])
    search = v2["resolved_config"]["search"]
    expected_v2 = {key: search[key] for key in field_names}
    assert comparison["architecture_v2"] == expected_v2
    assert comparison["matched"] is False
    assert any(
        "training datasets were generated" in limit.lower()
        for limit in report["comparison_limits"]
    )


def test_issue_143_readme_retains_reported_figures() -> None:
    report = _json(ISSUE_143 / "final-report.json")
    readme = (ISSUE_143 / "README.md").read_text(encoding="utf-8")

    for baselines in report["architecture_v2_strength_wins_draws_losses"].values():
        for wins, draws, losses in baselines.values():
            assert f"{wins}-{draws}-{losses}" in readme
    v1_strength = report[
        "architecture_v1_policy_value_strength_wins_draws_losses"
    ]
    for baselines in v1_strength.values():
        for wins, draws, losses in baselines.values():
            assert f"{wins}-{draws}-{losses}" in readme

    value = report["architecture_v2_value_quality"]
    for key in ("training_mse", "training_mae", "validation_mse", "validation_mae"):
        assert f"{value[key]:.4f}" in readme
    assert f"{value['training_positions']:,}" in readme
    assert f"{value['validation_positions']:,}" in readme
    assert f"{value['validation_minus_training_mse']:.4f}" in readme
    assert f"{value['validation_calibration_error']:+.4f}" in readme
    confident = value["most_confident_signed_bins"]
    assert f"{confident['negative_calibration_error']:+.4f}" in readme
    assert f"{confident['positive_calibration_error']:+.4f}" in readme
    assert str(confident["wrong_sign_predictions"]) in readme
    assert str(confident["predictions"]) in readme
    v1_value = report["architecture_v1_value_context"]
    assert f"{v1_value['validation_mse']:.4f}" in readme
    assert f"{v1_value['validation_positions']:,}" in readme
    assert "64 simulations" in readme
    assert "100 simulations" in readme
    assert "training-data search mismatch remains a confound" in readme
