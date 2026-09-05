from __future__ import annotations

from agents.contract import AgentContract
from twixt_ai.agents import AgentRequest
from twixt_ai.game import BoardDimensions, Coordinate, GameState, Link, Peg, Player
from twixt_ai.search import SearchAgent


class TestSearchAgentContract(AgentContract):
    agent_factory = SearchAgent


def test_search_reports_bounded_thinking_metadata() -> None:
    agent = SearchAgent(depth=2, node_budget=7)

    result = agent.choose_move(AgentRequest(GameState.initial()))

    assert result.metadata["depth"] == 2
    assert result.metadata["nodes"] == 7


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
