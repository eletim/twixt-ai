from __future__ import annotations

from twixt_ai.game import (
    EXPERIMENT_PRESETS,
    BoardDimensions,
    experiment_board,
    resolve_experiment_board,
)


def test_named_experiment_boards_are_stable() -> None:
    assert dict(EXPERIMENT_PRESETS) == {
        "standard": BoardDimensions(24, 24),
        "mini": BoardDimensions(10, 10),
    }
    assert experiment_board("mini") == BoardDimensions(10, 10)


def test_preset_dimensions_can_be_overridden() -> None:
    assert resolve_experiment_board("mini", width=12) == BoardDimensions(12, 10)
