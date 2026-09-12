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
            "champion_before": initial,
            "runtime_seconds": 3.0,
            "resolved_config": {
                "selfplay_simulations": 100,
                "evaluation_simulations": 20,
                "rollout_limit": 4,
            },
            "selfplay": {
                "runtime_seconds": 2.0,
                "summary_sha256": "1" * 64,
                "summary": {"aggregate": {
                    "completed": 2, "failed": 0, "total_moves": 40
                }},
            },
            "dataset": {
                "runtime_seconds": 0.25,
                "source_generations": [1],
                "manifest": {
                    "source_games": 2,
                    "examples": 40,
                    "splits": {
                        "train": {"shards": [{
                            "path": "train/shard-00000.jsonl",
                            "examples": 36,
                            "sha256": "4" * 64,
                        }]},
                        "validation": {"shards": [{
                            "path": "validation/shard-00000.jsonl",
                            "examples": 4,
                            "sha256": "5" * 64,
                        }]},
                    },
                },
                "manifest_sha256": "2" * 64,
                "target_distributions": {
                    "policy": {
                        "examples": 40,
                        "support": {"mean": 4.5, "minimum": 1, "maximum": 9},
                        "entropy_mean_nats": 1.25,
                        "maximum_probability_mean": 0.45,
                    },
                    "value": {
                        "examples": 40,
                        "counts": {"-1": 18, "0": 4, "1": 18},
                        "fractions": {"-1": 0.45, "0": 0.1, "1": 0.45},
                    },
                },
            },
            "training": {
                "runtime_seconds": 0.5,
                "summary": {
                    "history": history,
                    "performance": {"examples_per_second": 1600.0},
                },
                "candidate": candidate,
            },
            "evaluation": {"promotion": {
                "candidate_wins": 3,
                "games": 4,
                "win_rate": 0.75,
                "required_win_rate": 0.55,
                "promoted": True,
            }, "runtime_seconds": 0.25},
            "evaluation_artifacts": [
                {"opponent": "starting champion", "path": "start.json", "sha256": "6" * 64, "bytes": 256},
                {"opponent": "previous stage", "path": "previous.json", "sha256": "7" * 64, "bytes": 256},
                {"opponent": "matched non-neural MCTS", "path": "mcts.json", "sha256": "8" * 64, "bytes": 256},
                {"opponent": "heuristic search", "path": "heuristic.json", "sha256": "9" * 64, "bytes": 256},
            ],
            "artifact_storage": {"files": 8, "bytes": 4096},
            "retention_manifest": {
                "format": "twixt-ai-artifact-retention-manifest",
                "version": 1,
                "external_uri": "s3://twixt-ai/issue-128/matched-1k",
                "pruning_ready": True,
                "files": 7,
                "bytes": 2560 + candidate["bytes"],
                "categories": {
                    "selfplay": {
                        "files": 1,
                        "bytes": 1024,
                        "objects": [{
                            "path": "selfplay/games/game-000000.json",
                            "sha256": "a" * 64,
                            "bytes": 1024,
                        }],
                    },
                    "dataset": {
                        "files": 1,
                        "bytes": 512,
                        "objects": [{
                            "path": "dataset/train/shard-00000.jsonl",
                            "sha256": "4" * 64,
                            "bytes": 512,
                        }],
                    },
                    "training": {
                        "files": 1,
                        "bytes": candidate["bytes"],
                        "objects": [{
                            "path": "candidate/best.pt",
                            "sha256": candidate["sha256"],
                            "bytes": candidate["bytes"],
                        }],
                    },
                    "evaluation": {
                        "files": 4,
                        "bytes": 1024,
                        "objects": [
                            {"path": "start.json", "sha256": "6" * 64, "bytes": 256},
                            {"path": "previous.json", "sha256": "7" * 64, "bytes": 256},
                            {"path": "mcts.json", "sha256": "8" * 64, "bytes": 256},
                            {"path": "heuristic.json", "sha256": "9" * 64, "bytes": 256},
                        ],
                    },
                },
            },
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
    assert generation["dataset"]["manifest_sha256"] == "2" * 64
    assert [item["sha256"] for item in generation["dataset"]["shards"]] == [
        "4" * 64, "5" * 64
    ]
    assert generation["dataset"]["target_distributions"]["value"]["counts"] == {
        "-1": 18, "0": 4, "1": 18
    }
    assert generation["timing_seconds"] == {
        "total": 3.0,
        "selfplay": 2.0,
        "dataset": 0.25,
        "training": 0.5,
        "evaluation": 0.25,
    }
    assert generation["throughput"]["training_examples_per_second"] == 1600
    candidate_sha = report["checkpoints"][1]["sha256"]
    assert generation["hashes"]["candidate_checkpoint_sha256"] == candidate_sha
    assert generation["hashes"]["teacher"]["sha256"] == report["checkpoints"][0][
        "sha256"
    ]
    assert len(generation["hashes"]["evaluation_artifacts"]) == 4
    assert generation["artifact_storage"] == {"files": 8, "bytes": 4096}
    assert generation["retention_manifest"]["external_uri"] == (
        "s3://twixt-ai/issue-128/matched-1k"
    )
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
    assert "## Scaling evidence" in rendered
    assert "## Training target distributions" in rendered
    assert "18 / 4 / 18" in rendered
    assert "## Artifact identities" in rendered
    assert (
        f"| 1 | teacher | `{structured['generations'][0]['hashes']['teacher']['path']}` | "
        f"`{structured['generations'][0]['hashes']['teacher']['sha256']}` |"
    ) in rendered
    assert "dataset/train/shard-00000.jsonl" in rendered
    assert "`" + "4" * 64 + "`" in rendered
    assert "`" + "5" * 64 + "`" in rendered
    assert "evaluation: heuristic search" in rendered
    for character in "6789":
        assert "`" + character * 64 + "`" in rendered
    assert "s3://twixt-ai/issue-128/matched-1k" in rendered
    assert (
        "| 1 | `s3://twixt-ai/issue-128/matched-1k` | yes | selfplay | 1 | 1024 |"
        in rendered
    )
    assert "selfplay/games/game-000000.json" in rendered
    assert "`aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`" in rendered
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


def test_preserves_legacy_single_evaluation_artifact(tmp_path: Path) -> None:
    run = _run(tmp_path)
    source_path = run / "report.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    generation = source["generations"][0]
    generation["evaluation_artifact"] = generation.pop("evaluation_artifacts")[0]
    source_path.write_text(json.dumps(source), encoding="utf-8")

    report = build_mini_inspection_report(run)

    assert report["generations"][0]["hashes"]["evaluation_artifacts"] == [
        generation["evaluation_artifact"]
    ]


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
