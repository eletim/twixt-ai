"""Human-readable inspection reports for Mini Twixt generation artifacts."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from twixt_ai.game import (
    Coordinate,
    GameState,
    PegPlacement,
    apply_move,
    experiment_board,
    legal_peg_placements,
)
from twixt_ai.models import load_policy_value_checkpoint
from twixt_ai.search.neural import NeuralPolicyValue

from .generations import GENERATIONS_FORMAT, GENERATIONS_VERSION


INSPECTION_FORMAT = "twixt-ai-mini-inspection-report"
INSPECTION_VERSION = 1
PROBE_SET = "mini-fixed-positions-v1"

# Coordinates are deliberately data, not generated games: changing these positions
# would make reports from different runs incomparable and requires a new probe set.
_PROBES: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    ("opening", ()),
    ("linked-opening", ((4, 0), (0, 4), (5, 2), (2, 5))),
    (
        "contested-midgame",
        ((4, 0), (0, 4), (5, 2), (2, 5), (3, 4), (4, 3),
         (6, 6), (6, 5), (5, 8), (8, 4)),
    ),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_source(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path)
    if source.is_dir():
        source = source / "report.json"
    decoded = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise TypeError("generation report must contain a JSON object")
    if decoded.get("format") != GENERATIONS_FORMAT:
        raise ValueError(f"unsupported generation report format: {decoded.get('format')!r}")
    if decoded.get("version") != GENERATIONS_VERSION:
        raise ValueError(f"unsupported generation report version: {decoded.get('version')!r}")
    if not isinstance(decoded.get("config"), dict):
        raise ValueError("generation report config must be an object")
    if not isinstance(decoded.get("generations"), list):
        raise ValueError("generation report generations must be an array")
    return source, decoded


def _checkpoint_candidates(written_path: object, source: Path) -> tuple[Path, ...]:
    if not isinstance(written_path, str) or not written_path:
        return ()
    path = Path(written_path)
    if path.is_absolute():
        return (path,)
    candidates = (Path.cwd() / path, *(parent / path for parent in source.parents))
    # ``source`` and the current directory can share ancestors. Preserve search
    # order while avoiding repeated reads and hashes of the same candidate.
    return tuple(dict.fromkeys(candidate.resolve() for candidate in candidates))


def _resolve_checkpoint(
    written_path: object, source: Path, expected_sha256: object
) -> tuple[Path | None, list[dict[str, str]]]:
    mismatches: list[dict[str, str]] = []
    for candidate in _checkpoint_candidates(written_path, source):
        if not candidate.is_file():
            continue
        actual = _sha256(candidate)
        if isinstance(expected_sha256, str) and actual == expected_sha256:
            return candidate, mismatches
        mismatches.append({"path": str(candidate), "sha256": actual})
    return None, mismatches


def _probe_states() -> tuple[tuple[str, GameState], ...]:
    states: list[tuple[str, GameState]] = []
    for name, coordinates in _PROBES:
        state = GameState.initial(experiment_board("mini"))
        for x, y in coordinates:
            state = apply_move(
                state, PegPlacement(state.side_to_move, Coordinate(x, y))
            )
        states.append((name, state))
    return tuple(states)


def _probe_checkpoint(path: Path, top_moves: int) -> list[dict[str, Any]]:
    loaded = load_policy_value_checkpoint(path)
    evaluator = NeuralPolicyValue(loaded.model)
    probes: list[dict[str, Any]] = []
    for name, state in _probe_states():
        moves = legal_peg_placements(state)
        estimate = evaluator(state, moves)
        ranked = sorted(
            estimate.priors.items(),
            key=lambda item: (-item[1], item[0].coordinate.y, item[0].coordinate.x),
        )[:top_moves]
        probes.append({
            "name": name,
            "position": state.to_dict(),
            "legal_moves": len(moves),
            "value": estimate.value,
            "top_policy": [
                {"move": move.coordinate.to_dict(), "probability": probability}
                for move, probability in ranked
            ],
        })
    return probes


def _checkpoint_records(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[tuple[str, object]] = [("initial champion", report.get("initial_champion"))]
    for generation in report["generations"]:
        if not isinstance(generation, Mapping):
            continue
        number = generation.get("generation", "?")
        training = generation.get("training")
        if isinstance(training, Mapping):
            records.append((f"generation {number} candidate", training.get("candidate")))
    records.append(("final champion", report.get("final_champion")))

    by_identity: dict[str, dict[str, Any]] = {}
    for role, value in records:
        if not isinstance(value, Mapping):
            continue
        expected = value.get("sha256")
        written_path = value.get("path")
        identity = expected if isinstance(expected, str) else f"path:{written_path}"
        if identity in by_identity:
            by_identity[identity]["roles"].append(role)
            continue
        by_identity[identity] = {
            "roles": [role],
            "path": written_path,
            "sha256": expected,
            "bytes": value.get("bytes"),
            "model_config": value.get("model_config"),
        }
    return list(by_identity.values())


def _loss_summary(training: object) -> dict[str, Any] | None:
    if not isinstance(training, Mapping):
        return None
    summary = training.get("summary")
    history = summary.get("history") if isinstance(summary, Mapping) else None
    if not isinstance(history, list) or not history:
        return None
    rows = [row for row in history if isinstance(row, Mapping)]
    if not rows:
        return None
    best = min(
        rows,
        key=lambda row: float("inf")
        if row.get("validation_loss") is None else row["validation_loss"],
    )
    return {
        "epochs": len(rows),
        "first": dict(rows[0]),
        "last": dict(rows[-1]),
        "best_validation": dict(best) if best.get("validation_loss") is not None else None,
    }


def _dataset_shards(manifest: object) -> list[dict[str, Any]]:
    if not isinstance(manifest, Mapping):
        return []
    records: list[dict[str, Any]] = []
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        return records
    for split in ("train", "validation"):
        value = splits.get(split)
        shards = value.get("shards") if isinstance(value, Mapping) else None
        if not isinstance(shards, list):
            continue
        for shard in shards:
            if isinstance(shard, Mapping):
                records.append({"split": split, **dict(shard)})
    return records


def _valid_retention_inventory(
    categories: object, files: object, bytes_: object
) -> bool:
    required_categories = {"selfplay", "dataset", "training", "evaluation"}
    if not isinstance(categories, Mapping) or set(categories) != required_categories:
        return False
    if (
        isinstance(files, bool)
        or not isinstance(files, int)
        or isinstance(bytes_, bool)
        or not isinstance(bytes_, int)
    ):
        return False

    seen_paths: set[str] = set()
    counted_files = 0
    counted_bytes = 0
    for category in required_categories:
        details = categories[category]
        if not isinstance(details, Mapping):
            return False
        category_files = details.get("files")
        category_bytes = details.get("bytes")
        objects = details.get("objects")
        if (
            isinstance(category_files, bool)
            or not isinstance(category_files, int)
            or category_files < 1
            or isinstance(category_bytes, bool)
            or not isinstance(category_bytes, int)
            or category_bytes < 0
            or not isinstance(objects, list)
            or category_files != len(objects)
        ):
            return False
        object_bytes = 0
        for item in objects:
            if not isinstance(item, Mapping):
                return False
            path = item.get("path")
            sha256 = item.get("sha256")
            size = item.get("bytes")
            if not isinstance(path, str) or not path or "\\" in path:
                return False
            parsed = PurePosixPath(path)
            if parsed.is_absolute() or str(parsed) != path or ".." in parsed.parts:
                return False
            if path in seen_paths:
                return False
            if (
                not isinstance(sha256, str)
                or len(sha256) != 64
                or any(character not in "0123456789abcdef" for character in sha256)
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
            ):
                return False
            seen_paths.add(path)
            object_bytes += size
        if object_bytes != category_bytes:
            return False
        counted_files += category_files
        counted_bytes += category_bytes
    return counted_files == files and counted_bytes == bytes_


def _retention_status(retention: Mapping[str, Any]) -> dict[str, bool]:
    categories = retention.get("categories")
    files = retention.get("files")
    bytes_ = retention.get("bytes")
    if not _valid_retention_inventory(categories, files, bytes_):
        inventory_verified = False
    else:
        payload = {"categories": categories, "files": files, "bytes": bytes_}
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode()
        inventory_verified = hashlib.sha256(encoded).hexdigest() == retention.get(
            "inventory_sha256"
        )
    inventory_complete = (
        retention.get("inventory_complete") is True and inventory_verified
    )
    external_uri = retention.get("external_uri")
    attestation = retention.get("storage_attestation")
    storage_attested = (
        isinstance(external_uri, str)
        and bool(external_uri.strip())
        and isinstance(attestation, Mapping)
        and attestation.get("verified") is True
        and attestation.get("external_uri") == external_uri
        and attestation.get("inventory_sha256") == retention.get("inventory_sha256")
        and isinstance(attestation.get("verified_at"), str)
        and bool(attestation.get("verified_at"))
        and isinstance(attestation.get("verifier"), str)
        and bool(attestation.get("verifier"))
    )
    return {
        "inventory_complete": inventory_complete,
        "storage_attested": storage_attested,
        "pruning_ready": inventory_complete and storage_attested,
    }


def _generation_summary(generation: object) -> dict[str, Any]:
    if not isinstance(generation, Mapping):
        raise TypeError("generation entries must be objects")
    selfplay = generation.get("selfplay")
    batch = selfplay.get("summary") if isinstance(selfplay, Mapping) else None
    aggregate = batch.get("aggregate") if isinstance(batch, Mapping) else None
    completed = aggregate.get("completed") if isinstance(aggregate, Mapping) else None
    seconds = selfplay.get("runtime_seconds") if isinstance(selfplay, Mapping) else None
    throughput = None
    if isinstance(completed, int) and isinstance(seconds, (int, float)) and seconds > 0:
        throughput = completed * 3600 / seconds

    dataset = generation.get("dataset")
    manifest = dataset.get("manifest") if isinstance(dataset, Mapping) else None
    training = generation.get("training")
    training_summary = (
        training.get("summary") if isinstance(training, Mapping) else None
    )
    performance = (
        training_summary.get("performance")
        if isinstance(training_summary, Mapping) else None
    )
    candidate = training.get("candidate") if isinstance(training, Mapping) else None
    teacher = generation.get("champion_before")
    evaluation = generation.get("evaluation")
    promotion = evaluation.get("promotion") if isinstance(evaluation, Mapping) else None
    raw_evaluations = generation.get("evaluation_artifacts")
    if isinstance(raw_evaluations, list):
        evaluation_artifacts = [
            dict(item) for item in raw_evaluations if isinstance(item, Mapping)
        ]
    else:
        legacy_evaluation = generation.get("evaluation_artifact")
        evaluation_artifacts = (
            [dict(legacy_evaluation)]
            if isinstance(legacy_evaluation, Mapping) else []
        )
    raw_fixed_opponents = generation.get("fixed_opponent_evaluations")
    fixed_opponent_evaluations = (
        [
            dict(item)
            for item in raw_fixed_opponents
            if isinstance(item, Mapping)
        ]
        if isinstance(raw_fixed_opponents, list) else []
    )
    target_distributions = (
        dataset.get("target_distributions") if isinstance(dataset, Mapping) else None
    )
    # Version-1 generation reports before the scaling contract recorded only
    # the policy half. Keep them inspectable without pretending a value
    # distribution was measured.
    if not isinstance(target_distributions, Mapping) and isinstance(dataset, Mapping):
        policy = dataset.get("policy_target_quality")
        if isinstance(policy, Mapping):
            target_distributions = {"policy": dict(policy), "value": None}
    storage = generation.get("artifact_storage")
    retention = generation.get("retention_manifest")
    retention_summary = dict(retention) if isinstance(retention, Mapping) else None
    if retention_summary is not None:
        retention_summary["status"] = _retention_status(retention_summary)
    return {
        "generation": generation.get("generation"),
        "status": generation.get("status"),
        "decision": generation.get("decision"),
        "runtime_seconds": generation.get("runtime_seconds"),
        "search_budgets": {
            key: generation.get("resolved_config", {}).get(key)
            if isinstance(generation.get("resolved_config"), Mapping) else None
            for key in ("selfplay_simulations", "evaluation_simulations", "rollout_limit")
        },
        "selfplay": {
            "completed_games": completed,
            "failed_games": aggregate.get("failed") if isinstance(aggregate, Mapping) else None,
            "moves": aggregate.get("total_moves") if isinstance(aggregate, Mapping) else None,
            "runtime_seconds": seconds,
            "games_per_hour": throughput,
        },
        "dataset": {
            "source_generations": dataset.get("source_generations")
            if isinstance(dataset, Mapping) else None,
            "source_games": manifest.get("source_games")
            if isinstance(manifest, Mapping) else None,
            "examples": manifest.get("examples") if isinstance(manifest, Mapping) else None,
            "manifest_sha256": dataset.get("manifest_sha256")
            if isinstance(dataset, Mapping) else None,
            "target_distributions": dict(target_distributions)
            if isinstance(target_distributions, Mapping) else None,
            "shards": _dataset_shards(manifest),
        },
        "losses": _loss_summary(training),
        "timing_seconds": {
            "total": generation.get("runtime_seconds"),
            "selfplay": seconds,
            "dataset": dataset.get("runtime_seconds")
            if isinstance(dataset, Mapping) else None,
            "training": training.get("runtime_seconds")
            if isinstance(training, Mapping) else None,
            "evaluation": evaluation.get("runtime_seconds")
            if isinstance(evaluation, Mapping) else None,
        },
        "throughput": {
            "selfplay_games_per_hour": throughput,
            "training_examples_per_second": performance.get("examples_per_second")
            if isinstance(performance, Mapping) else None,
        },
        "hashes": {
            "teacher": dict(teacher) if isinstance(teacher, Mapping) else None,
            "selfplay_summary_sha256": selfplay.get("summary_sha256")
            if isinstance(selfplay, Mapping) else None,
            "dataset_manifest_sha256": dataset.get("manifest_sha256")
            if isinstance(dataset, Mapping) else None,
            "candidate_checkpoint_sha256": candidate.get("sha256")
            if isinstance(candidate, Mapping) else None,
            "dataset_shards": _dataset_shards(manifest),
            "evaluation_artifacts": evaluation_artifacts,
            "evaluation_sha256": evaluation_artifacts[0].get("sha256")
            if evaluation_artifacts else None,
        },
        "artifact_storage": dict(storage) if isinstance(storage, Mapping) else None,
        "retention_manifest": retention_summary,
        "evaluation": (
            {"comparison": "candidate vs parent champion", **dict(promotion)}
            if isinstance(promotion, Mapping) else None
        ),
        "fixed_opponent_evaluations": fixed_opponent_evaluations,
        "champion_change": (
            "updated to candidate"
            if isinstance(promotion, Mapping) and promotion.get("promoted") is True
            else "unchanged" if isinstance(promotion, Mapping)
            else None
        ),
    }


def build_mini_inspection_report(
    generation_report: str | Path, *, top_moves: int = 5
) -> dict[str, Any]:
    """Read stored generation artifacts and return a self-contained summary."""

    if isinstance(top_moves, bool) or not isinstance(top_moves, int) or top_moves < 1:
        raise ValueError("top_moves must be a positive integer")
    source, raw = _load_source(generation_report)
    generations = [_generation_summary(item) for item in raw["generations"]]

    checkpoints = _checkpoint_records(raw)
    for checkpoint in checkpoints:
        resolved, mismatches = _resolve_checkpoint(
            checkpoint["path"], source, checkpoint["sha256"]
        )
        checkpoint["resolved_path"] = str(resolved) if resolved else None
        checkpoint["available"] = resolved is not None
        if resolved is None:
            checkpoint["verification"] = (
                "sha256 mismatch" if mismatches else "matching checkpoint missing"
            )
            if mismatches:
                checkpoint["mismatched_candidates"] = mismatches
            checkpoint["probes"] = []
            continue
        checkpoint["verification"] = "verified"
        checkpoint["actual_sha256"] = checkpoint["sha256"]
        if mismatches:
            checkpoint["ignored_mismatched_candidates"] = mismatches
        try:
            checkpoint["probes"] = _probe_checkpoint(resolved, top_moves)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            checkpoint["probe_error"] = f"{type(exc).__name__}: {exc}"
            checkpoint["probes"] = []

    return {
        "format": INSPECTION_FORMAT,
        "version": INSPECTION_VERSION,
        "source": {
            "path": str(source),
            "sha256": _sha256(source),
            "format": raw["format"],
            "version": raw["version"],
            "status": raw.get("status"),
        },
        "config": raw["config"],
        "probe_set": PROBE_SET,
        "checkpoints": checkpoints,
        "lineage": raw.get("lineage", []),
        "generations": generations,
    }


def _number(value: object, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _change(first: Mapping[str, Any], last: Mapping[str, Any], key: str) -> str:
    return f"{_number(first.get(key))}→{_number(last.get(key))}"


def render_mini_inspection_report(report: Mapping[str, Any]) -> str:
    """Render an inspection report as stable, human-readable Markdown."""

    source = report["source"]
    lines = [
        "# Mini Twixt training inspection",
        "",
        f"Source: `{source['path']}`",
        f"Source SHA-256: `{source['sha256']}`",
        f"Run status: **{source.get('status', 'unknown')}**",
        f"Probe set: `{report['probe_set']}`",
        "",
        "## Exact run configuration",
        "",
        "```json",
        json.dumps(report["config"], indent=2, sort_keys=True),
        "```",
        "",
        "## Checkpoint lineage",
        "",
        "| Role(s) | Recorded path | SHA-256 | Verification |",
        "| --- | --- | --- | --- |",
    ]
    for checkpoint in report["checkpoints"]:
        lines.append(
            f"| {', '.join(checkpoint['roles'])} | `{checkpoint['path']}` | "
            f"`{checkpoint['sha256']}` | {checkpoint['verification']} |"
        )
    if report.get("lineage"):
        lines.extend([
            "",
            "```json",
            json.dumps(report["lineage"], indent=2, sort_keys=True),
            "```",
        ])

    lines.extend([
        "", "## Generation overview", "",
        "| Gen | Status | Self-play games | Games/hour | Dataset examples | "
        "Train loss (first→last) | Validation loss (first→last) | "
        "Candidate vs parent | Champion change | Decision | Search budgets |",
        "| ---: | --- | ---: | ---: | ---: | --- | --- | ---: | --- | --- | --- |",
    ])
    for generation in report["generations"]:
        losses = generation["losses"]
        train = validation = "—"
        if losses:
            train = _change(losses["first"], losses["last"], "train_loss")
            validation = _change(
                losses["first"], losses["last"], "validation_loss"
            )
        evaluation = generation["evaluation"] or {}
        rate = evaluation.get("win_rate")
        budgets = generation["search_budgets"]
        budget_text = (
            f"self-play {budgets['selfplay_simulations']} sims; evaluation "
            f"{budgets['evaluation_simulations']} sims; rollout {budgets['rollout_limit']}"
        )
        lines.append(
            f"| {generation['generation']} | {generation['status']} | "
            f"{_number(generation['selfplay']['completed_games'])} | "
            f"{_number(generation['selfplay']['games_per_hour'], 1)} | "
            f"{_number(generation['dataset']['examples'])} | {train} | {validation} | "
            f"{_number(None if rate is None else 100 * rate, 1)}% | "
            f"{generation['champion_change'] or '—'} | "
            f"{generation['decision'] or '—'} | {budget_text} |"
        )

    fixed_opponents = [
        (generation["generation"], evaluation)
        for generation in report["generations"]
        for evaluation in generation["fixed_opponent_evaluations"]
    ]
    if fixed_opponents:
        lines.extend([
            "", "## Fixed-opponent evaluation results", "",
            "| Gen | Opponent | Candidate W-L-D | Candidate win rate | "
            "Games | Paired role swaps | Seed |",
            "| ---: | --- | ---: | ---: | ---: | --- | ---: |",
        ])
        for generation, evaluation in fixed_opponents:
            win_rate = evaluation.get("candidate_win_rate")
            lines.append(
                f"| {generation} | {evaluation.get('opponent', '—')} | "
                f"{_number(evaluation.get('candidate_wins'))}-"
                f"{_number(evaluation.get('candidate_losses'))}-"
                f"{_number(evaluation.get('candidate_draws'))} | "
                f"{_number(None if win_rate is None else 100 * win_rate, 1)}% | "
                f"{_number(evaluation.get('games'))} | "
                f"{'yes' if evaluation.get('paired_role_swaps') is True else 'no'} | "
                f"{_number(evaluation.get('seed'))} |"
            )

    lines.extend([
        "", "## Training loss components", "",
        "| Gen | Train total | Train policy | Train value | Validation total | "
        "Validation policy | Validation value | Best validation epoch/loss |",
        "| ---: | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for generation in report["generations"]:
        losses = generation["losses"]
        if not losses:
            lines.append(
                f"| {generation['generation']} | — | — | — | — | — | — | — |"
            )
            continue
        first, last = losses["first"], losses["last"]
        best = losses["best_validation"]
        best_text = (
            "—" if best is None
            else f"{best.get('epoch', '—')} / {_number(best.get('validation_loss'))}"
        )
        lines.append(
            f"| {generation['generation']} | {_change(first, last, 'train_loss')} | "
            f"{_change(first, last, 'train_policy_loss')} | "
            f"{_change(first, last, 'train_value_loss')} | "
            f"{_change(first, last, 'validation_loss')} | "
            f"{_change(first, last, 'validation_policy_loss')} | "
            f"{_change(first, last, 'validation_value_loss')} | {best_text} |"
        )

    lines.extend([
        "", "## Scaling evidence", "",
        "| Gen | Total / self-play / dataset / training / evaluation seconds | "
        "Training examples/s | Retained bytes (self-play / dataset / training / "
        "evaluation) | Self-play / dataset / candidate / evaluation SHA-256 |",
        "| ---: | --- | ---: | --- | --- |",
    ])
    for generation in report["generations"]:
        timing = generation["timing_seconds"]
        storage = generation["artifact_storage"] or {}
        hashes = generation["hashes"]
        lines.append(
            f"| {generation['generation']} | "
            f"{_number(timing['total'])} / {_number(timing['selfplay'])} / "
            f"{_number(timing['dataset'])} / {_number(timing['training'])} / "
            f"{_number(timing['evaluation'])} | "
            f"{_number(generation['throughput']['training_examples_per_second'], 1)} | "
            f"{_number(storage.get('bytes'))} "
            f"({_number((storage.get('selfplay') or {}).get('bytes'))} / "
            f"{_number((storage.get('dataset') or {}).get('bytes'))} / "
            f"{_number((storage.get('training') or {}).get('bytes'))} / "
            f"{_number((storage.get('evaluation') or {}).get('bytes'))}) | "
            f"`{hashes['selfplay_summary_sha256'] or '—'}` / "
            f"`{hashes['dataset_manifest_sha256'] or '—'}` / "
            f"`{hashes['candidate_checkpoint_sha256'] or '—'}` / "
            f"`{hashes['evaluation_sha256'] or '—'}` |"
        )

    lines.extend([
        "", "## Artifact identities", "",
        "| Gen | Role | Path | SHA-256 |",
        "| ---: | --- | --- | --- |",
    ])
    for generation in report["generations"]:
        hashes = generation["hashes"]
        teacher = hashes["teacher"] or {}
        lines.append(
            f"| {generation['generation']} | teacher | `{teacher.get('path', '—')}` | "
            f"`{teacher.get('sha256', '—')}` |"
        )
        lines.append(
            f"| {generation['generation']} | self-play summary | `selfplay/summary.json` | "
            f"`{hashes['selfplay_summary_sha256'] or '—'}` |"
        )
        lines.append(
            f"| {generation['generation']} | dataset manifest | `dataset/manifest.json` | "
            f"`{hashes['dataset_manifest_sha256'] or '—'}` |"
        )
        for shard in hashes["dataset_shards"]:
            lines.append(
                f"| {generation['generation']} | dataset shard ({shard['split']}) | "
                f"`dataset/{shard.get('path', '—')}` | "
                f"`{shard.get('sha256', '—')}` |"
            )
        for artifact in hashes["evaluation_artifacts"]:
            lines.append(
                f"| {generation['generation']} | evaluation: "
                f"{artifact.get('opponent', 'unspecified opponent')} | "
                f"`{artifact.get('path', '—')}` | `{artifact.get('sha256', '—')}` |"
            )

    lines.extend([
        "", "## Retention manifests", "",
        "| Gen | External URI | Inventory complete | Storage attested | "
        "Pruning ready | Category | Files | Bytes |",
        "| ---: | --- | --- | --- | --- | --- | ---: | ---: |",
    ])
    for generation in report["generations"]:
        retention = generation["retention_manifest"]
        if not retention:
            lines.append(
                f"| {generation['generation']} | — | no | no | no | "
                "unavailable | — | — |"
            )
            continue
        categories = retention.get("categories") or {}
        status = retention["status"]
        for category, details in categories.items():
            lines.append(
                f"| {generation['generation']} | "
                f"`{retention.get('external_uri') or '—'}` | "
                f"{'yes' if status['inventory_complete'] else 'no'} | "
                f"{'yes' if status['storage_attested'] else 'no'} | "
                f"{'yes' if status['pruning_ready'] else 'no'} | {category} | "
                f"{_number(details.get('files'))} | {_number(details.get('bytes'))} |"
            )

    lines.extend([
        "", "### Retained object inventory", "",
        "| Gen | Category | Path | Bytes | SHA-256 |",
        "| ---: | --- | --- | ---: | --- |",
    ])
    for generation in report["generations"]:
        retention = generation["retention_manifest"] or {}
        for category, details in (retention.get("categories") or {}).items():
            for item in details.get("objects") or []:
                lines.append(
                    f"| {generation['generation']} | {category} | "
                    f"`{item.get('path', '—')}` | {_number(item.get('bytes'))} | "
                    f"`{item.get('sha256', '—')}` |"
                )

    lines.extend([
        "", "## Training target distributions", "",
        "| Gen | Policy support mean (min–max) | Mean max probability | "
        "Mean entropy (nats) | Value counts (-1 / 0 / +1) |",
        "| ---: | --- | ---: | ---: | --- |",
    ])
    for generation in report["generations"]:
        targets = generation["dataset"]["target_distributions"] or {}
        policy = targets.get("policy") or {}
        support = policy.get("support") or {}
        value = targets.get("value") or {}
        counts = value.get("counts") or {}
        support_text = "—"
        if support:
            support_text = (
                f"{_number(support.get('mean'))} "
                f"({_number(support.get('minimum'))}–{_number(support.get('maximum'))})"
            )
        value_text = "—" if not value else (
            f"{_number(counts.get('-1'))} / {_number(counts.get('0'))} / "
            f"{_number(counts.get('1'))}"
        )
        lines.append(
            f"| {generation['generation']} | {support_text} | "
            f"{_number(policy.get('maximum_probability_mean'))} | "
            f"{_number(policy.get('entropy_mean_nats'))} | {value_text} |"
        )

    lines.extend(["", "## Fixed policy/value probes", ""])
    for checkpoint in report["checkpoints"]:
        lines.extend([f"### {', '.join(checkpoint['roles'])}", ""])
        if checkpoint.get("probe_error"):
            lines.extend([f"Probe unavailable: {checkpoint['probe_error']}", ""])
        elif not checkpoint["probes"]:
            lines.extend([f"Probe unavailable: checkpoint {checkpoint['verification']}.", ""])
        else:
            lines.extend([
                "| Position | Side | Value | Top legal policy moves |",
                "| --- | --- | ---: | --- |",
            ])
            for probe in checkpoint["probes"]:
                policy = ", ".join(
                    f"({item['move']['x']},{item['move']['y']}) {item['probability']:.3f}"
                    for item in probe["top_policy"]
                )
                lines.append(
                    f"| {probe['name']} | {probe['position']['side_to_move']} | "
                    f"{_number(probe['value'])} | {policy} |"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "INSPECTION_FORMAT",
    "INSPECTION_VERSION",
    "PROBE_SET",
    "build_mini_inspection_report",
    "render_mini_inspection_report",
]
