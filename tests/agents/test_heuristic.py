"""Tests for the interpretable position-evaluation baseline."""

from __future__ import annotations

from dataclasses import replace

import pytest

from twixt_ai.agents import (
    TERMINAL_SCORE,
    EvaluationBreakdown,
    HeuristicWeights,
    evaluate_position,
    position_features,
)
from twixt_ai.game import (
    BoardDimensions,
    Coordinate,
    GameResult,
    GameState,
    Link,
    Peg,
    Player,
)


BOARD = BoardDimensions(8, 8)


def test_empty_position_is_neutral_and_perspective_is_antisymmetric() -> None:
    assert evaluate_position(GameState.initial(BOARD), Player.RED) == 0.0
    state = GameState(board=BOARD, pegs=(Peg(Player.RED, Coordinate(2, 1)),))
    red_score = evaluate_position(state, Player.RED)
    assert red_score == -evaluate_position(state, Player.BLACK)


def test_connected_component_scores_above_same_scattered_pegs() -> None:
    points = (Coordinate(1, 0), Coordinate(2, 2), Coordinate(3, 4))
    connected = GameState(
        board=BOARD,
        pegs=tuple(Peg(Player.RED, point) for point in points),
        links=(
            Link(Player.RED, points[0], points[1]),
            Link(Player.RED, points[1], points[2]),
        ),
    )
    scattered = GameState(
        board=BOARD,
        pegs=tuple(
            Peg(Player.RED, point)
            for point in (Coordinate(1, 0), Coordinate(4, 2), Coordinate(6, 4))
        ),
    )
    connected_features = position_features(connected, Player.RED)
    scattered_features = position_features(scattered, Player.RED)
    assert connected_features.progress > scattered_features.progress
    assert connected_features.connectivity > scattered_features.connectivity
    assert evaluate_position(connected, Player.RED) > evaluate_position(
        scattered, Player.RED
    )


def test_open_and_opponent_blocked_link_opportunities_are_counted() -> None:
    open_state = GameState(board=BOARD, pegs=(Peg(Player.RED, Coordinate(1, 1)),))
    blocked_state = GameState(
        board=BOARD,
        pegs=(
            Peg(Player.RED, Coordinate(1, 1)),
            Peg(Player.BLACK, Coordinate(1, 3)),
            Peg(Player.BLACK, Coordinate(3, 2)),
        ),
        links=(Link(Player.BLACK, Coordinate(1, 3), Coordinate(3, 2)),),
    )
    open_features = position_features(open_state, Player.RED)
    blocked_features = position_features(blocked_state, Player.RED)
    assert open_features.threats > 0
    assert blocked_features.blocked > 0
    assert blocked_features.threats < open_features.threats


def test_debug_mode_exposes_terms_that_sum_to_the_scalar() -> None:
    state = GameState(
        board=BOARD,
        pegs=(
            Peg(Player.RED, Coordinate(2, 1)),
            Peg(Player.BLACK, Coordinate(4, 4)),
        ),
    )
    breakdown = evaluate_position(state, Player.RED, debug=True)
    assert isinstance(breakdown, EvaluationBreakdown)
    assert set(breakdown.contributions) == {
        "progress",
        "connectivity",
        "threats",
        "blocking",
        "terminal",
    }
    assert breakdown.score == sum(breakdown.contributions.values())
    assert breakdown.score == evaluate_position(state, Player.RED)
    with pytest.raises(TypeError):
        breakdown.contributions["progress"] = 0.0  # type: ignore[index]


@pytest.mark.parametrize(
    "result, player, expected",
    [
        (GameResult.RED_WINS, Player.RED, TERMINAL_SCORE),
        (GameResult.RED_WINS, Player.BLACK, -TERMINAL_SCORE),
        (GameResult.BLACK_WINS, Player.BLACK, TERMINAL_SCORE),
        (GameResult.DRAW, Player.RED, 0.0),
    ],
)
def test_terminal_result_dominates_the_score(
    result: GameResult, player: Player, expected: float
) -> None:
    state = GameState(
        board=BOARD,
        pegs=(Peg(Player.RED, Coordinate(1, 0)),),
        result=result,
    )
    assert evaluate_position(state, player) == expected


def test_custom_weights_make_feature_contributions_explicit() -> None:
    state = GameState(
        board=BOARD,
        pegs=(Peg(Player.RED, Coordinate(1, 0)), Peg(Player.RED, Coordinate(2, 2))),
        links=(Link(Player.RED, Coordinate(1, 0), Coordinate(2, 2)),),
    )
    weights = HeuristicWeights(progress=2, connectivity=0, threats=0, blocking=0)
    breakdown = evaluate_position(state, Player.RED, debug=True, weights=weights)
    assert breakdown.score == breakdown.contributions["progress"]
    assert breakdown.contributions["connectivity"] == 0.0


@pytest.mark.parametrize("value", [None, "red", 1])
def test_evaluation_rejects_invalid_players(value: object) -> None:
    with pytest.raises(TypeError, match="player"):
        evaluate_position(GameState.initial(BOARD), value)  # type: ignore[arg-type]


def test_weights_reject_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        replace(HeuristicWeights(), progress=float("inf"))
