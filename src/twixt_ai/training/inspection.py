"""Human-readable inspection reports for Mini Twixt generation artifacts."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
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
    evaluation = generation.get("evaluation")
    promotion = evaluation.get("promotion") if isinstance(evaluation, Mapping) else None
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
        },
        "losses": _loss_summary(generation.get("training")),
        "evaluation": (
            {"comparison": "candidate vs parent champion", **dict(promotion)}
            if isinstance(promotion, Mapping) else None
        ),
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
        f"Source: `{source['path']}`  ",
        f"Source SHA-256: `{source['sha256']}`  ",
        f"Run status: **{source.get('status', 'unknown')}**  ",
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
