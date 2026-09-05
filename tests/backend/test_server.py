"""Integration tests for the minimal browser HTTP boundary."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

import pytest

from twixt_ai.agents import AgentRequest, AgentResult
from twixt_ai.backend import GameApplication, GameSession
from twixt_ai.game import (
    BoardDimensions,
    Coordinate,
    GameResult,
    GameState,
    Peg,
    Player,
    legal_peg_placements,
)


class RecordingAgent:
    def __init__(self) -> None:
        self.requests: list[AgentRequest] = []

    def choose_move(self, agent_request: AgentRequest) -> AgentResult:
        self.requests.append(agent_request)
        return AgentResult(agent_request.legal_moves[0], {"depth": 2})


class NonFiniteMetadataAgent:
    def __init__(self, value: float) -> None:
        self.value = value

    def choose_move(self, agent_request: AgentRequest) -> AgentResult:
        return AgentResult(agent_request.legal_moves[0], {"score": self.value})


def request(
    application: GameApplication,
    path: str,
    method: str = "GET",
    payload: object | None = None,
) -> tuple[str, dict[str, str], bytes]:
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    result: dict[str, object] = {}

    def start_response(status: str, headers: list[tuple[str, str]]) -> None:
        result["status"] = status
        result["headers"] = dict(headers)

    response = b"".join(
        application(
            {
                "REQUEST_METHOD": method,
                "PATH_INFO": path,
                "CONTENT_LENGTH": str(len(body)),
                "wsgi.input": BytesIO(body),
            },
            start_response,
        )
    )
    return (  # type: ignore[return-value]
        str(result["status"]),
        result["headers"],
        response,
    )


def test_move_uses_engine_and_returns_updated_canonical_state(tmp_path: Path) -> None:
    application = GameApplication(
        GameSession(GameState.initial(BoardDimensions(6, 6))), tmp_path
    )

    status, _, body = request(application, "/api/game/moves", "POST", {"x": 2, "y": 2})
    state = json.loads(body)

    assert status == "200 OK"
    assert state["side_to_move"] == "black"
    assert state["pegs"] == [
        {"owner": "red", "coordinate": {"x": 2, "y": 2}}
    ]
    assert application.session.snapshot().pegs == (
        Peg(Player.RED, Coordinate(2, 2)),
    )


def test_illegal_move_returns_conflict_without_mutating_state(tmp_path: Path) -> None:
    initial = GameState.initial(BoardDimensions(6, 6))
    application = GameApplication(GameSession(initial), tmp_path)

    status, _, body = request(application, "/api/game/moves", "POST", {"x": 0, "y": 2})
    error = json.loads(body)

    assert status == "409 Conflict"
    assert error["error"] == "illegal_move"
    assert error["reason"] == "forbidden_border"
    assert error["state"] == initial.to_dict()
    assert application.session.snapshot() == initial


def test_reset_and_read_game(tmp_path: Path) -> None:
    session = GameSession(GameState.initial(BoardDimensions(6, 5)))
    application = GameApplication(session, tmp_path)
    session.place(2, 2)

    reset_status, _, reset_body = request(application, "/api/game/reset", "POST")
    read_status, headers, read_body = request(application, "/api/game")

    assert reset_status == read_status == "200 OK"
    expected = GameState.initial(BoardDimensions(6, 5)).to_dict()
    assert json.loads(reset_body) == json.loads(read_body) == expected
    assert headers["Cache-Control"] == "no-store"


def test_invalid_payload_is_rejected_without_mutating_state(tmp_path: Path) -> None:
    application = GameApplication(ui_root=tmp_path)
    initial = application.session.snapshot()

    status, _, body = request(
        application, "/api/game/moves", "POST", {"x": True, "y": 2}
    )

    assert status == "400 Bad Request"
    assert json.loads(body)["error"] == "invalid_request"
    assert application.session.snapshot() == initial


def test_application_serves_replaceable_ui_files(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<h1>Twixt</h1>", encoding="utf-8")
    application = GameApplication(ui_root=tmp_path)

    status, headers, body = request(application, "/")

    assert status == "200 OK"
    assert headers["Content-Type"] == "text/html; charset=utf-8"
    assert body == b"<h1>Twixt</h1>"


def test_packaged_ui_disables_setup_until_session_loads() -> None:
    status, _, body = request(GameApplication(), "/")

    assert status == "200 OK"
    assert b'<button id="reset" type="button" disabled>' in body
    assert b'<select id="human-side" disabled>' in body
    assert b'<select id="agent" aria-label="AI opponent" disabled>' in body
    assert b'<input id="inspection-toggle" type="checkbox" disabled>' in body


def test_packaged_inspection_overlays_do_not_block_board_input() -> None:
    status, _, body = request(GameApplication(), "/styles.css")

    assert status == "200 OK"
    assert (
        b".candidate-overlays, .selected-move-overlay { pointer-events: none; }"
        in body
    )


def test_packaged_ui_summarizes_only_scalar_agent_metadata() -> None:
    status, _, body = request(GameApplication(), "/app.js")

    assert status == "200 OK"
    assert b'scalarTypes = ["string", "number", "boolean"]' in body
    assert b"scalarTypes.includes(typeof value)" in body


def test_session_selects_side_and_runs_registered_agent_through_contract(
    tmp_path: Path,
) -> None:
    agent = RecordingAgent()
    session = GameSession(
        GameState.initial(BoardDimensions(6, 6)), agents={"search": agent}
    )
    application = GameApplication(session, tmp_path)

    reset_status, _, reset_body = request(
        application,
        "/api/session/reset",
        "POST",
        {"human_side": "black", "agent": "search"},
    )
    reset_view = json.loads(reset_body)
    move_status, _, move_body = request(
        application,
        "/api/session/agent-moves",
        "POST",
        {"revision": reset_view["revision"]},
    )
    move_view = json.loads(move_body)

    assert reset_status == move_status == "200 OK"
    assert reset_view["available_agents"] == ["search"]
    assert move_view["human_side"] == "black"
    assert move_view["state"]["side_to_move"] == "black"
    assert move_view["thinking"]["metadata"] == {"depth": 2}
    assert move_view["thinking"]["move"]["player"] == "red"
    assert session.view()["thinking"] == move_view["thinking"]
    assert len(agent.requests) == 1
    assert agent.requests[0].state.side_to_move is Player.RED


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_session_rejects_non_finite_agent_metadata_without_advancing(
    tmp_path: Path, value: float
) -> None:
    session = GameSession(
        GameState.initial(BoardDimensions(6, 6)),
        agents={"invalid": NonFiniteMetadataAgent(value)},
        human_side=Player.BLACK,
    )
    application = GameApplication(session, tmp_path)
    initial_view = session.view()

    status, _, body = request(
        application,
        "/api/session/agent-moves",
        "POST",
        {"revision": initial_view["revision"]},
    )

    assert status == "500 Internal Server Error"
    assert json.loads(body) == {
        "error": "agent_error",
        "detail": "agent metadata must be JSON serializable",
        "session": initial_view,
    }
    assert session.view() == initial_view


def test_session_rejects_human_input_during_agent_turn(tmp_path: Path) -> None:
    session = GameSession(
        GameState.initial(BoardDimensions(6, 6)), human_side=Player.BLACK
    )
    application = GameApplication(session, tmp_path)

    status, _, body = request(
        application,
        "/api/session/human-moves",
        "POST",
        {"x": 1, "y": 0, "revision": 0},
    )

    assert status == "409 Conflict"
    assert json.loads(body)["error"] == "out_of_turn"
    assert session.snapshot().pegs == ()


def test_session_revision_prevents_stale_click_from_mutating_state(
    tmp_path: Path,
) -> None:
    session = GameSession(GameState.initial(BoardDimensions(6, 6)))
    application = GameApplication(session, tmp_path)
    payload = {"x": 1, "y": 0, "revision": 0}

    first_status, _, _ = request(
        application, "/api/session/human-moves", "POST", payload
    )
    stale_status, _, stale_body = request(
        application, "/api/session/human-moves", "POST", payload
    )

    assert first_status == "200 OK"
    assert stale_status == "409 Conflict"
    assert json.loads(stale_body)["error"] == "stale_state"
    assert len(session.snapshot().pegs) == 1


def test_session_configuration_rejects_unknown_agent(tmp_path: Path) -> None:
    application = GameApplication(ui_root=tmp_path)

    status, _, body = request(
        application,
        "/api/session/reset",
        "POST",
        {"human_side": "red", "agent": "missing"},
    )

    assert status == "400 Bad Request"
    assert json.loads(body) == {
        "error": "invalid_request",
        "detail": "unknown agent",
    }


@pytest.mark.parametrize("agent_name", ["random", "search", "mcts"])
def test_human_can_complete_match_against_default_agents_via_session_api(
    tmp_path: Path,
    agent_name: str,
) -> None:
    session = GameSession(GameState.initial(BoardDimensions(4, 4)))
    application = GameApplication(session, tmp_path)
    reset_status, _, reset_body = request(
        application,
        "/api/session/reset",
        "POST",
        {"human_side": "red", "agent": agent_name},
    )

    assert reset_status == "200 OK"
    assert agent_name in json.loads(reset_body)["available_agents"]

    while session.snapshot().result is GameResult.IN_PROGRESS:
        view = session.view()
        if session.snapshot().side_to_move is Player.RED:
            move = legal_peg_placements(session.snapshot())[0]
            payload = {
                "x": move.coordinate.x,
                "y": move.coordinate.y,
                "revision": view["revision"],
            }
            status, _, _ = request(
                application, "/api/session/human-moves", "POST", payload
            )
        else:
            status, _, _ = request(
                application,
                "/api/session/agent-moves",
                "POST",
                {"revision": view["revision"]},
            )
        assert status == "200 OK"

    assert session.snapshot().result.is_terminal
