"""Validated persisted matches and canonical self-play training targets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from twixt_ai.evaluation.match import MatchConfig, MatchResult
from twixt_ai.game import (
    Coordinate,
    GameState,
    PegPlacement,
    apply_move,
    legal_peg_placements,
)


@dataclass(frozen=True, slots=True)
class PolicyTarget:
    coordinate: Coordinate
    probability: float

    def to_dict(self) -> dict[str, object]:
        return {
            "coordinate": self.coordinate.to_dict(),
            "probability": self.probability,
        }


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    ply: int
    position: GameState
    action: Coordinate
    outcome: int
    decision_seed: int | None
    metadata: Mapping[str, object]
    policy: tuple[PolicyTarget, ...] | None


@dataclass(frozen=True, slots=True)
class MatchTrajectory:
    game_id: str
    config: MatchConfig
    steps: tuple[TrajectoryStep, ...]


def _game_id(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _policy_target(
    metadata: Mapping[str, Any], state: GameState
) -> tuple[PolicyTarget, ...] | None:
    root_moves = metadata.get("root_moves")
    if root_moves is None:
        return None
    if not isinstance(root_moves, list):
        raise ValueError("root_moves metadata must be an array")
    simulations = metadata.get("simulations")
    if (
        isinstance(simulations, bool)
        or not isinstance(simulations, int)
        or simulations < 1
    ):
        raise ValueError(
            "root_moves metadata requires a positive integer simulations count"
        )
    visits: list[tuple[Coordinate, int]] = []
    legal = set(legal_peg_placements(state))
    seen: set[Coordinate] = set()
    for index, item in enumerate(root_moves):
        if not isinstance(item, dict):
            raise ValueError(f"root_moves[{index}] must be an object")
        x, y, count = item.get("x"), item.get("y"), item.get("visits")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (x, y, count)
        ):
            raise ValueError(
                f"root_moves[{index}] coordinates and visits must be integers"
            )
        coordinate = Coordinate(x, y)
        if (
            count < 0
            or coordinate in seen
            or PegPlacement(state.side_to_move, coordinate) not in legal
        ):
            raise ValueError(
                f"root_moves[{index}] contains an invalid move or visit count"
            )
        seen.add(coordinate)
        visits.append((coordinate, count))
    total = sum(count for _, count in visits)
    if total != simulations:
        raise ValueError("root move visits must sum to metadata.simulations")
    return tuple(
        PolicyTarget(coordinate, count / total)
        for coordinate, count in visits
        if count
    )


def trajectory_from_match(
    value: dict[str, Any], source: str | Path
) -> MatchTrajectory:
    """Validate a persisted match and derive canonical per-position targets."""

    try:
        match = MatchResult.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid match artifact {Path(source)}: {exc}") from exc
    winner = match.winner
    state = match.record.initial_state
    steps: list[TrajectoryStep] = []
    for ply, (move, decision) in enumerate(zip(match.moves, match.decisions)):
        outcome = 0 if winner is None else (1 if winner is state.side_to_move else -1)
        metadata = _thaw_json(decision.metadata)
        assert isinstance(metadata, dict)
        steps.append(
            TrajectoryStep(
                ply=ply,
                position=state,
                action=move.coordinate,
                outcome=outcome,
                decision_seed=decision.seed,
                metadata=metadata,
                policy=_policy_target(metadata, state),
            )
        )
        state = apply_move(state, move)
    return MatchTrajectory(_game_id(value), match.config, tuple(steps))


__all__ = [
    "MatchTrajectory",
    "PolicyTarget",
    "TrajectoryStep",
    "trajectory_from_match",
]
