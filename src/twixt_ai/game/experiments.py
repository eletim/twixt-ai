"""Named board configurations shared by interactive and headless experiments."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from .state import BoardDimensions


STANDARD_EXPERIMENT = "standard"
MINI_EXPERIMENT = "mini"

EXPERIMENT_PRESETS: Mapping[str, BoardDimensions] = MappingProxyType(
    {
        STANDARD_EXPERIMENT: BoardDimensions(24, 24),
        MINI_EXPERIMENT: BoardDimensions(10, 10),
    }
)


def experiment_board(name: str) -> BoardDimensions:
    """Return the immutable dimensions for a named experiment preset."""

    if not isinstance(name, str):
        raise TypeError("experiment preset must be a string")
    try:
        return EXPERIMENT_PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"unknown experiment preset: {name}") from exc


def resolve_experiment_board(
    preset: str = STANDARD_EXPERIMENT,
    *,
    width: int | None = None,
    height: int | None = None,
) -> BoardDimensions:
    """Resolve a preset with optional per-dimension overrides.

    Supplying only one dimension retains the other dimension from the preset.
    This keeps existing custom rectangular-board workflows available while
    giving reproducible experiments a concise named configuration.
    """

    board = experiment_board(preset)
    return BoardDimensions(
        board.width if width is None else width,
        board.height if height is None else height,
    )


__all__ = [
    "EXPERIMENT_PRESETS",
    "MINI_EXPERIMENT",
    "STANDARD_EXPERIMENT",
    "experiment_board",
    "resolve_experiment_board",
]
