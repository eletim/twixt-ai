"""Versioned, deterministic records for saving and replaying games."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable, Mapping

from .rules import PegPlacement
from .state import GameState, Player, _coordinate_from_dict, _require_int
from .transitions import apply_move


GAME_RECORD_FORMAT = "twixt-ai-game-record"
GAME_RECORD_VERSION = 1


def _replay(initial_state: GameState, moves: tuple[PegPlacement, ...]) -> GameState:
    state = initial_state
    for move in moves:
        state = apply_move(state, move)
    return state


@dataclass(frozen=True, slots=True)
class GameRecord:
    """A self-contained move history with an independently checkable result.

    Records may begin at any canonical position. Construction and loading replay
    the moves so an inconsistent final state can never enter the application.
    """

    initial_state: GameState
    moves: tuple[PegPlacement, ...]
    final_state: GameState

    def __post_init__(self) -> None:
        if not isinstance(self.initial_state, GameState):
            raise TypeError("initial_state must be a GameState")
        if not isinstance(self.final_state, GameState):
            raise TypeError("final_state must be a GameState")
        try:
            moves = tuple(self.moves)
        except TypeError as exc:
            raise TypeError("moves must be iterable") from exc
        if any(not isinstance(move, PegPlacement) for move in moves):
            raise TypeError("moves must contain only PegPlacement values")
        object.__setattr__(self, "moves", moves)
        if _replay(self.initial_state, moves) != self.final_state:
            raise ValueError("recorded final_state does not match replayed moves")

    @classmethod
    def from_moves(
        cls, initial_state: GameState, moves: Iterable[PegPlacement]
    ) -> GameRecord:
        """Build a record and derive its final state by replaying *moves*."""

        if not isinstance(initial_state, GameState):
            raise TypeError("initial_state must be a GameState")
        try:
            move_values = tuple(moves)
        except TypeError as exc:
            raise TypeError("moves must be iterable") from exc
        if any(not isinstance(move, PegPlacement) for move in move_values):
            raise TypeError("moves must contain only PegPlacement values")
        return cls(initial_state, move_values, _replay(initial_state, move_values))

    def replay(self) -> GameState:
        """Replay the history and return the resulting canonical state."""

        return _replay(self.initial_state, self.moves)

    def to_dict(self) -> dict[str, object]:
        """Return the deterministic, versioned persisted representation."""

        return {
            "format": GAME_RECORD_FORMAT,
            "version": GAME_RECORD_VERSION,
            "initial_state": self.initial_state.to_dict(),
            "moves": [
                {
                    "player": move.player.value,
                    "coordinate": move.coordinate.to_dict(),
                }
                for move in self.moves
            ],
            "final_state": self.final_state.to_dict(),
        }

    def to_json(self) -> str:
        """Serialize the record to compact JSON with stable key ordering."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GameRecord:
        """Validate, replay, and deserialize a persisted game record."""

        if not isinstance(value, Mapping):
            raise TypeError("game record must be an object")
        expected = {"format", "version", "initial_state", "moves", "final_state"}
        if set(value) != expected:
            raise ValueError(
                f"game record must contain exactly {', '.join(sorted(expected))}"
            )
        if value["format"] != GAME_RECORD_FORMAT:
            raise ValueError(f"unsupported game record format: {value['format']!r}")
        version = _require_int(value["version"], "version")
        if version != GAME_RECORD_VERSION:
            raise ValueError(f"unsupported game record version: {version}")

        move_values = value["moves"]
        if not isinstance(move_values, list):
            raise TypeError("moves must be an array")
        moves: list[PegPlacement] = []
        for index, move_value in enumerate(move_values):
            if not isinstance(move_value, Mapping) or set(move_value) != {
                "player",
                "coordinate",
            }:
                raise ValueError(
                    f"moves[{index}] must contain exactly player and coordinate"
                )
            moves.append(
                PegPlacement(
                    player=Player(move_value["player"]),
                    coordinate=_coordinate_from_dict(
                        move_value["coordinate"], f"moves[{index}].coordinate"
                    ),
                )
            )

        initial_value = value["initial_state"]
        final_value = value["final_state"]
        if not isinstance(initial_value, Mapping):
            raise TypeError("initial_state must be an object")
        if not isinstance(final_value, Mapping):
            raise TypeError("final_state must be an object")
        return cls(
            initial_state=GameState.from_dict(initial_value),
            moves=tuple(moves),
            final_state=GameState.from_dict(final_value),
        )

    @classmethod
    def from_json(cls, value: str) -> GameRecord:
        """Validate, replay, and deserialize a JSON game record."""

        if not isinstance(value, str):
            raise TypeError("JSON game record must be a string")
        decoded = json.loads(value)
        if not isinstance(decoded, Mapping):
            raise TypeError("game record JSON must describe an object")
        return cls.from_dict(decoded)


__all__ = ["GAME_RECORD_FORMAT", "GAME_RECORD_VERSION", "GameRecord"]
