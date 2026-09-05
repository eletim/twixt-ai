"""Integration tests for the minimal browser HTTP boundary."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

from twixt_ai.backend import GameApplication, GameSession
from twixt_ai.game import BoardDimensions, Coordinate, GameState, Peg, Player


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
