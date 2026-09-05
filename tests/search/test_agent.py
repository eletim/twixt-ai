from __future__ import annotations

import pytest

from agents.contract import AgentContract
from twixt_ai.agents import TERMINAL_SCORE, AgentRequest, evaluate_position
from twixt_ai.game import (
    BoardDimensions,
    Coordinate,
    GameState,
    Link,
    Peg,
    PegPlacement,
    Player,
    apply_move,
    legal_peg_placements,
)
from twixt_ai.search import HeuristicSearchAgent, SearchAgent


class TestSearchAgentContract(AgentContract):
    agent_factory = SearchAgent


def test_search_reports_bounded_thinking_metadata() -> None:
    agent = SearchAgent(depth=2, node_budget=7)

    result = agent.choose_move(AgentRequest(GameState.initial()))

    assert result.metadata["depth"] == 2
    assert result.metadata["nodes"] == 7
    inspection = result.metadata["inspection"]
    assert inspection["value"] == result.metadata["score"]
    assert inspection["statistics"] == {
        "depth": 2,
        "nodes": 7,
        "node_budget": 7,
    }
    assert inspection["candidates"]
    assert set(inspection["candidates"][0]) == {"x", "y", "score"}


def test_inspection_reports_full_window_candidate_scores() -> None:
    state = GameState.initial(BoardDimensions(4, 4))
    result = SearchAgent(depth=2).choose_move(AgentRequest(state))
    inspected = {
        (item["x"], item["y"]): item["score"]
        for item in result.metadata["inspection"]["candidates"]
    }
    candidate = PegPlacement(Player.RED, Coordinate(2, 2))
    child = apply_move(state, candidate)
    expected = min(
        evaluate_position(apply_move(child, response), Player.RED)
        for response in legal_peg_placements(child)
    )

    assert inspected[(2, 2)] == expected


def test_search_takes_an_immediate_win() -> None:
    state = GameState(
        board=BoardDimensions(5, 4),
        pegs=(
            Peg(Player.RED, Coordinate(1, 0)),
            Peg(Player.RED, Coordinate(3, 1)),
        ),
        links=(
            Link(Player.RED, Coordinate(1, 0), Coordinate(3, 1)),
        ),
    )

    result = SearchAgent().choose_move(AgentRequest(state))

    assert result.move.coordinate == Coordinate(2, 3)


def test_terminal_win_dominates_an_above_terminal_custom_score() -> None:
    state = GameState(
        board=BoardDimensions(5, 4),
        pegs=(
            Peg(Player.RED, Coordinate(1, 0)),
            Peg(Player.RED, Coordinate(3, 1)),
        ),
        links=(Link(Player.RED, Coordinate(1, 0), Coordinate(3, 1)),),
    )
    evaluated: list[GameState] = []

    def nonterminal_evaluator(position: GameState, player: Player) -> float:
        assert not position.is_terminal
        evaluated.append(position)
        return TERMINAL_SCORE + 1.0

    result = SearchAgent(evaluator=nonterminal_evaluator).choose_move(
        AgentRequest(state)
    )

    assert result.move.coordinate == Coordinate(2, 3)
    assert result.metadata["score"] == TERMINAL_SCORE
    assert evaluated


def test_descriptive_and_compatibility_names_refer_to_the_same_agent() -> None:
    assert SearchAgent is HeuristicSearchAgent


def test_search_uses_a_configurable_evaluator_at_leaf_positions() -> None:
    visited: list[GameState] = []

    def prefer_larger_x(state: GameState, player: Player) -> float:
        visited.append(state)
        own_peg = next(peg for peg in state.pegs if peg.owner is player)
        return float(own_peg.coordinate.x)

    state = GameState.initial(BoardDimensions(5, 3))
    result = SearchAgent(evaluator=prefer_larger_x).choose_move(AgentRequest(state))

    assert result.move.coordinate.x == 3
    assert len(visited) == len(AgentRequest(state).legal_moves)


def test_move_orderer_controls_search_under_a_tight_budget() -> None:
    state = GameState.initial(BoardDimensions(5, 3))
    preferred = PegPlacement(Player.RED, Coordinate(3, 2))
    calls: list[GameState] = []

    def preferred_first(
        position: GameState, moves: tuple[PegPlacement, ...]
    ) -> tuple[PegPlacement, ...]:
        calls.append(position)
        return (preferred, *(move for move in moves if move != preferred))

    result = SearchAgent(node_budget=1, move_orderer=preferred_first).choose_move(
        AgentRequest(state)
    )

    assert result.move == preferred
    assert calls == [state]


def test_move_orderer_must_preserve_the_legal_move_set() -> None:
    agent = SearchAgent(move_orderer=lambda state, moves: moves[:-1])

    with pytest.raises(ValueError, match="each legal move exactly once"):
        agent.choose_move(AgentRequest(GameState.initial(BoardDimensions(4, 4))))
