"""Tests for generated Mini Twixt inspection reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from twixt_ai.models import (
    MINI_POLICY_VALUE_CONFIG,
    PolicyValueNetwork,
    save_policy_value_checkpoint,
)
from twixt_ai.training import inspection_cli
from twixt_ai.training.inspection import (
    PROBE_SET,
    build_mini_inspection_report,
    render_mini_inspection_report,
)


def _checkpoint(path: Path) -> dict[str, object]:
    save_policy_value_checkpoint(path, PolicyValueNetwork(MINI_POLICY_VALUE_CONFIG))
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "model_config": MINI_POLICY_VALUE_CONFIG.to_dict(),
    }


def _run(tmp_path: Path) -> Path:
    initial = _checkpoint(tmp_path / "initial.pt")
    candidate = _checkpoint(tmp_path / "candidate.pt")
    history = [
        {
            "epoch": 1,
            "train_loss": 5.0,
            "train_policy_loss": 4.0,
            "train_value_loss": 1.0,
            "validation_loss": 5.2,
            "validation_policy_loss": 4.1,
            "validation_value_loss": 1.1,
        },
        {
            "epoch": 2,
            "train_loss": 4.0,
            "train_policy_loss": 3.5,
            "train_value_loss": 0.5,
            "validation_loss": 4.3,
            "validation_policy_loss": 3.7,
            "validation_value_loss": 0.6,
        },
    ]
    run = tmp_path / "run"
    run.mkdir()
    report = {
        "format": "twixt-ai-mini-training-generations",
        "version": 1,
        "status": "completed",
        "config": {"seed": 60, "board": {"width": 10, "height": 10}},
        "initial_champion": initial,
        "final_champion": candidate,
        "lineage": [{
            "generation": 1,
            "parent_sha256": initial["sha256"],
            "candidate_sha256": candidate["sha256"],
            "decision": "promoted",
            "champion_sha256": candidate["sha256"],
        }],
        "generations": [{
            "generation": 1,
            "status": "completed",
            "decision": "promoted",
            "runtime_seconds": 3.0,
            "resolved_config": {
                "selfplay_simulations": 100,
                "evaluation_simulations": 20,
                "rollout_limit": 4,
            },
            "selfplay": {
                "runtime_seconds": 2.0,
                "summary": {"aggregate": {
                    "completed": 2, "failed": 0, "total_moves": 40
                }},
            },
            "dataset": {
                "source_generations": [1],
                "manifest": {"source_games": 2, "examples": 40},
            },
            "training": {"summary": {"history": history}, "candidate": candidate},
            "evaluation": {"promotion": {
                "candidate_wins": 3,
                "games": 4,
                "win_rate": 0.75,
                "required_win_rate": 0.55,
                "promoted": True,
            }},
        }],
    }
    (run / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return run


def test_builds_summary_and_fixed_checkpoint_probes(tmp_path: Path) -> None:
    report = build_mini_inspection_report(_run(tmp_path), top_moves=3)

    assert report["probe_set"] == PROBE_SET
    assert len(report["checkpoints"]) == 2
    assert all(item["verification"] == "verified" for item in report["checkpoints"])
    assert all(len(item["probes"]) == 3 for item in report["checkpoints"])
    assert all(
        len(probe["top_policy"]) == 3
        for item in report["checkpoints"]
        for probe in item["probes"]
    )
    generation = report["generations"][0]
    assert generation["selfplay"]["games_per_hour"] == pytest.approx(3600)
    assert generation["dataset"]["examples"] == 40
    assert generation["losses"]["first"]["train_loss"] == 5.0
    assert generation["losses"]["last"]["validation_loss"] == 4.3
    assert generation["evaluation"]["win_rate"] == 0.75
    assert generation["evaluation"]["comparison"] == "candidate vs parent champion"
    assert generation["champion_change"] == "updated to candidate"
    assert "strength_change" not in generation


def test_render_and_cli_include_exact_inputs(tmp_path: Path) -> None:
    run = _run(tmp_path)
    structured = build_mini_inspection_report(run)
    rendered = render_mini_inspection_report(structured)

    assert "# Mini Twixt training inspection" in rendered
    assert structured["source"]["sha256"] in rendered
    assert structured["checkpoints"][0]["sha256"] in rendered
    assert "75.0%" in rendered
    assert "contested-midgame" in rendered

    output = tmp_path / "reports" / "inspection.md"
    assert inspection_cli.main([str(run), "--output", str(output)]) == 0
    assert output.read_text(encoding="utf-8").startswith(
        "# Mini Twixt training inspection"
    )


def test_hash_mismatch_is_flagged_without_running_probes(tmp_path: Path) -> None:
    run = _run(tmp_path)
    source = json.loads((run / "report.json").read_text(encoding="utf-8"))
    source["initial_champion"]["sha256"] = "0" * 64
    (run / "report.json").write_text(json.dumps(source), encoding="utf-8")

    report = build_mini_inspection_report(run)

    initial = report["checkpoints"][0]
    assert initial["verification"] == "sha256 mismatch"
    assert initial["mismatched_candidates"][0]["path"] == str(tmp_path / "initial.pt")
    assert initial["probes"] == []


def test_resolves_by_hash_after_stale_working_directory_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _run(tmp_path)
    source_path = run / "report.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    correct = run / "shared" / "initial.pt"
    correct.parent.mkdir()
    shutil.copyfile(tmp_path / "initial.pt", correct)
    source["initial_champion"]["path"] = "shared/initial.pt"
    source_path.write_text(json.dumps(source), encoding="utf-8")

    working = tmp_path / "working"
    stale = working / "shared" / "initial.pt"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale checkpoint with the same recorded path")
    monkeypatch.chdir(working)

    report = build_mini_inspection_report(run)

    initial = report["checkpoints"][0]
    assert initial["verification"] == "verified"
    assert initial["resolved_path"] == str(correct)
    assert initial["ignored_mismatched_candidates"] == [{
        "path": str(stale),
        "sha256": hashlib.sha256(stale.read_bytes()).hexdigest(),
    }]
    assert len(initial["probes"]) == 3


def test_distinguishes_missing_checkpoint_from_hash_mismatch(tmp_path: Path) -> None:
    run = _run(tmp_path)
    source_path = run / "report.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["initial_champion"]["path"] = "not-present/initial.pt"
    source_path.write_text(json.dumps(source), encoding="utf-8")

    report = build_mini_inspection_report(run)

    initial = report["checkpoints"][0]
    assert initial["verification"] == "matching checkpoint missing"
    assert "mismatched_candidates" not in initial


def test_does_not_compare_win_rates_against_different_parents(tmp_path: Path) -> None:
    run = _run(tmp_path)
    source_path = run / "report.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    second = json.loads(json.dumps(source["generations"][0]))
    second["generation"] = 2
    second["decision"] = "rejected"
    second["evaluation"]["promotion"].update({
        "candidate_wins": 3,
        "games": 5,
        "win_rate": 0.6,
        "promoted": False,
    })
    source["generations"].append(second)
    source_path.write_text(json.dumps(source), encoding="utf-8")

    report = build_mini_inspection_report(run)

    first, second = report["generations"]
    assert first["evaluation"]["win_rate"] == 0.75
    assert second["evaluation"]["win_rate"] == 0.6
    assert "strength_change" not in second
    assert second["evaluation"]["comparison"] == "candidate vs parent champion"
    assert second["champion_change"] == "unchanged"
    rendered = render_mini_inspection_report(report)
    assert "Δ vs prior" not in rendered
    assert "Candidate vs parent" in rendered


@pytest.mark.parametrize("top_moves", [0, True])
def test_rejects_invalid_top_move_counts(tmp_path: Path, top_moves: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        build_mini_inspection_report(_run(tmp_path), top_moves=top_moves)  # type: ignore[arg-type]
