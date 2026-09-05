"""Small standard-library web server for the human-vs-human client."""

from __future__ import annotations

import argparse
from importlib import resources
import json
from pathlib import Path
from threading import Lock
from typing import Callable, Iterable, Protocol
from wsgiref.simple_server import make_server

from twixt_ai.game import (
    Coordinate,
    GameState,
    IllegalMoveError,
    PegPlacement,
    apply_move,
    create_game,
    reset_game,
)


StartResponse = Callable[[str, list[tuple[str, str]]], object]
Response = Iterable[bytes]


class ResourceRoot(Protocol):
    """Subset shared by filesystem paths and package resource trees."""

    def joinpath(self, *descendants: str) -> ResourceRoot: ...

    def read_bytes(self) -> bytes: ...


DEFAULT_UI_ROOT = resources.files("twixt_ai.ui")
_STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


class GameSession:
    """Own one in-memory game and serialize its canonical transitions."""

    def __init__(self, state: GameState | None = None) -> None:
        self._state = state or create_game()
        self._lock = Lock()

    def snapshot(self) -> GameState:
        with self._lock:
            return self._state

    def place(self, x: int, y: int) -> GameState:
        """Place for the current player exclusively through the game engine."""

        if isinstance(x, bool) or not isinstance(x, int):
            raise TypeError("x must be an integer")
        if isinstance(y, bool) or not isinstance(y, int):
            raise TypeError("y must be an integer")
        with self._lock:
            move = PegPlacement(self._state.side_to_move, Coordinate(x, y))
            self._state = apply_move(self._state, move)
            return self._state

    def reset(self) -> GameState:
        with self._lock:
            self._state = reset_game(self._state)
            return self._state


class GameApplication:
    """WSGI application exposing one game plus the replaceable static UI."""

    def __init__(
        self,
        session: GameSession | None = None,
        ui_root: Path | ResourceRoot | None = None,
    ) -> None:
        self.session = session or GameSession()
        self.ui_root = ui_root if ui_root is not None else DEFAULT_UI_ROOT

    @staticmethod
    def _json(
        start_response: StartResponse, status: str, value: object
    ) -> list[bytes]:
        body = json.dumps(value, separators=(",", ":")).encode("utf-8")
        start_response(
            status,
            [
                ("Content-Type", "application/json; charset=utf-8"),
                ("Content-Length", str(len(body))),
                ("Cache-Control", "no-store"),
            ],
        )
        return [body]

    def _read_json(self, environ: dict[str, object]) -> object:
        raw_length = str(environ.get("CONTENT_LENGTH") or "0")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length < 0 or length > 16_384:
            raise ValueError("request body is too large")
        stream = environ["wsgi.input"]
        body = stream.read(length)  # type: ignore[union-attr]
        try:
            return json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("request body must be valid JSON") from exc

    def _serve_static(self, path: str, start_response: StartResponse) -> Response:
        static = _STATIC_FILES.get(path)
        if static is None:
            return self._json(start_response, "404 Not Found", {"error": "not_found"})
        filename, content_type = static
        try:
            body = self.ui_root.joinpath(filename).read_bytes()
        except FileNotFoundError:
            return self._json(
                start_response, "500 Internal Server Error", {"error": "ui_unavailable"}
            )
        start_response(
            "200 OK",
            [
                ("Content-Type", content_type),
                ("Content-Length", str(len(body))),
                ("Cache-Control", "no-cache"),
            ],
        )
        return [body]

    def __call__(
        self, environ: dict[str, object], start_response: StartResponse
    ) -> Response:
        method = str(environ.get("REQUEST_METHOD", "GET")).upper()
        path = str(environ.get("PATH_INFO", "/"))

        if method == "GET" and path == "/api/game":
            state = self.session.snapshot().to_dict()
            return self._json(start_response, "200 OK", state)
        if method == "POST" and path == "/api/game/reset":
            return self._json(start_response, "200 OK", self.session.reset().to_dict())
        if method == "POST" and path == "/api/game/moves":
            try:
                payload = self._read_json(environ)
                if not isinstance(payload, dict) or set(payload) != {"x", "y"}:
                    raise ValueError("move must contain exactly x and y")
                state = self.session.place(payload["x"], payload["y"])
            except IllegalMoveError as exc:
                return self._json(
                    start_response,
                    "409 Conflict",
                    {
                        "error": "illegal_move",
                        "reason": exc.reason.value,
                        "state": self.session.snapshot().to_dict(),
                    },
                )
            except (KeyError, TypeError, ValueError) as exc:
                return self._json(
                    start_response,
                    "400 Bad Request",
                    {"error": "invalid_request", "detail": str(exc)},
                )
            return self._json(start_response, "200 OK", state.to_dict())
        if method == "GET":
            return self._serve_static(path, start_response)
        return self._json(
            start_response, "405 Method Not Allowed", {"error": "method_not_allowed"}
        )


def create_application() -> GameApplication:
    """Create a fresh application; useful to servers and integration tests."""

    return GameApplication()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Twixt browser UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    application = create_application()
    with make_server(args.host, args.port, application) as server:
        print(f"Twixt UI available at http://{args.host}:{args.port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


__all__ = ["GameApplication", "GameSession", "create_application", "main"]
