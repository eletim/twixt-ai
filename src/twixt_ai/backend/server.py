"""Small standard-library web server for the browser play client."""

from __future__ import annotations

import argparse
from importlib import resources
import json
from pathlib import Path
from threading import Lock
from typing import Callable, Iterable, Mapping, Protocol
from wsgiref.simple_server import make_server

from twixt_ai.agents import Agent, AgentContractError, RandomAgent, select_agent_move
from twixt_ai.game import (
    Coordinate,
    GameState,
    IllegalMoveError,
    PegPlacement,
    Player,
    apply_move,
    create_game,
    reset_game,
)
from twixt_ai.search import MCTSAgent, SearchAgent


StartResponse = Callable[[str, list[tuple[str, str]]], object]
Response = Iterable[bytes]


class ResourceRoot(Protocol):
    """Subset shared by filesystem paths and package resource trees."""

    def joinpath(self, *descendants: str) -> ResourceRoot: ...

    def read_bytes(self) -> bytes: ...


class SessionConflictError(ValueError):
    """Raised when a browser action no longer matches the live session."""

    def __init__(self, error: str, detail: str) -> None:
        self.error = error
        super().__init__(detail)


DEFAULT_UI_ROOT = resources.files("twixt_ai.ui")
_STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


class GameSession:
    """Own one in-memory human-vs-agent game and serialize its transitions."""

    def __init__(
        self,
        state: GameState | None = None,
        *,
        agents: Mapping[str, Agent] | None = None,
        agent_name: str | None = None,
        human_side: Player = Player.RED,
    ) -> None:
        self._state = state or create_game()
        self._agents = (
            {"random": RandomAgent(), "search": SearchAgent(), "mcts": MCTSAgent()}
            if agents is None
            else dict(agents)
        )
        if not self._agents:
            raise ValueError("at least one agent must be available")
        if any(not isinstance(name, str) or not name for name in self._agents):
            raise ValueError("agent names must be non-empty strings")
        if any(not isinstance(agent, Agent) for agent in self._agents.values()):
            raise TypeError("agents must implement choose_move(request)")
        selected_agent = agent_name if agent_name is not None else next(iter(self._agents))
        if selected_agent not in self._agents:
            raise ValueError(f"unknown agent: {agent_name}")
        if not isinstance(human_side, Player):
            raise TypeError("human_side must be a Player")
        self._agent_name = selected_agent
        self._human_side = human_side
        self._revision = 0
        self._thinking: dict[str, object] = {}
        self._lock = Lock()

    def _view_unlocked(
        self, thinking: Mapping[str, object] | None = None
    ) -> dict[str, object]:
        return {
            "state": self._state.to_dict(),
            "revision": self._revision,
            "human_side": self._human_side.value,
            "agent": self._agent_name,
            "available_agents": list(self._agents),
            "thinking": dict(self._thinking if thinking is None else thinking),
        }

    def view(self) -> dict[str, object]:
        """Return browser-facing state and configuration atomically."""

        with self._lock:
            return self._view_unlocked()

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
            self._revision += 1
            self._thinking = {}
            return self._state

    def reset(self) -> GameState:
        with self._lock:
            self._state = reset_game(self._state)
            self._revision += 1
            self._thinking = {}
            return self._state

    @staticmethod
    def _require_revision(revision: object) -> int:
        if isinstance(revision, bool) or not isinstance(revision, int):
            raise TypeError("revision must be an integer")
        return revision

    def _check_revision_unlocked(self, revision: object) -> None:
        expected = self._require_revision(revision)
        if expected != self._revision:
            raise SessionConflictError("stale_state", "the game has changed")

    def configure(self, human_side: object, agent_name: object) -> dict[str, object]:
        """Start a new game with validated human and agent choices."""

        try:
            side = Player(human_side)
        except (TypeError, ValueError) as exc:
            raise ValueError("human_side must be red or black") from exc
        if not isinstance(agent_name, str) or agent_name not in self._agents:
            raise ValueError("unknown agent")
        with self._lock:
            self._human_side = side
            self._agent_name = agent_name
            self._state = reset_game(self._state)
            self._revision += 1
            self._thinking = {}
            return self._view_unlocked()

    def place_human(self, x: object, y: object, revision: object) -> dict[str, object]:
        """Apply a current human move, rejecting stale and out-of-turn input."""

        if isinstance(x, bool) or not isinstance(x, int):
            raise TypeError("x must be an integer")
        if isinstance(y, bool) or not isinstance(y, int):
            raise TypeError("y must be an integer")
        with self._lock:
            self._check_revision_unlocked(revision)
            if self._state.side_to_move is not self._human_side:
                raise SessionConflictError("out_of_turn", "it is the agent's turn")
            move = PegPlacement(self._human_side, Coordinate(x, y))
            self._state = apply_move(self._state, move)
            self._revision += 1
            self._thinking = {}
            return self._view_unlocked()

    def play_agent(self, revision: object) -> dict[str, object]:
        """Select and apply one AI move through the common Agent interface."""

        with self._lock:
            self._check_revision_unlocked(revision)
            if self._state.side_to_move is self._human_side:
                raise SessionConflictError("out_of_turn", "it is the human's turn")
            result = select_agent_move(self._agents[self._agent_name], self._state)
            metadata = dict(result.metadata)
            try:
                json.dumps(metadata, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise AgentContractError(
                    "agent metadata must be JSON serializable"
                ) from exc
            self._thinking = {
                "move": {
                    "player": result.move.player.value,
                    "coordinate": result.move.coordinate.to_dict(),
                },
                "metadata": metadata,
            }
            self._state = apply_move(self._state, result.move)
            self._revision += 1
            return self._view_unlocked()


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
        if method == "GET" and path == "/api/session":
            return self._json(start_response, "200 OK", self.session.view())
        if method == "POST" and path == "/api/session/reset":
            try:
                payload = self._read_json(environ)
                if not isinstance(payload, dict) or set(payload) != {"human_side", "agent"}:
                    raise ValueError("reset must contain exactly human_side and agent")
                view = self.session.configure(payload["human_side"], payload["agent"])
            except (KeyError, TypeError, ValueError) as exc:
                return self._json(
                    start_response,
                    "400 Bad Request",
                    {"error": "invalid_request", "detail": str(exc)},
                )
            return self._json(start_response, "200 OK", view)
        if method == "POST" and path in {
            "/api/session/human-moves",
            "/api/session/agent-moves",
        }:
            try:
                payload = self._read_json(environ)
                required = (
                    {"x", "y", "revision"}
                    if path.endswith("human-moves")
                    else {"revision"}
                )
                if not isinstance(payload, dict) or set(payload) != required:
                    fields = ", ".join(sorted(required))
                    raise ValueError(f"request must contain exactly {fields}")
                if path.endswith("human-moves"):
                    view = self.session.place_human(
                        payload["x"], payload["y"], payload["revision"]
                    )
                else:
                    view = self.session.play_agent(payload["revision"])
            except SessionConflictError as exc:
                return self._json(
                    start_response,
                    "409 Conflict",
                    {
                        "error": exc.error,
                        "detail": str(exc),
                        "session": self.session.view(),
                    },
                )
            except IllegalMoveError as exc:
                return self._json(
                    start_response,
                    "409 Conflict",
                    {
                        "error": "illegal_move",
                        "reason": exc.reason.value,
                        "session": self.session.view(),
                    },
                )
            except AgentContractError as exc:
                return self._json(
                    start_response,
                    "500 Internal Server Error",
                    {
                        "error": "agent_error",
                        "detail": str(exc),
                        "session": self.session.view(),
                    },
                )
            except (KeyError, TypeError, ValueError) as exc:
                return self._json(
                    start_response,
                    "400 Bad Request",
                    {"error": "invalid_request", "detail": str(exc)},
                )
            return self._json(start_response, "200 OK", view)
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
