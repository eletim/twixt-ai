"""Immutable, canonical values describing a Twixt position.

This module deliberately contains no move-generation or presentation concerns.
It describes the state on which those layers operate and ensures that every
constructed state is structurally valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from typing import Any, Iterable, Mapping


def _require_int(value: object, name: str) -> int:
    """Return *value* as an int, rejecting bools and non-integers."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


class Player(str, Enum):
    """A player and the pair of goal edges they connect.

    Red connects north to south and moves first in standard Twixt. Black
    connects west to east. The directional names are aliases useful to code
    that cares about geometry more than the traditional colours.
    """

    RED = "red"
    BLACK = "black"

    VERTICAL = "red"
    HORIZONTAL = "black"

    @property
    def opponent(self) -> Player:
        return Player.BLACK if self is Player.RED else Player.RED


class GameResult(str, Enum):
    """The terminal status of a position."""

    IN_PROGRESS = "in_progress"
    RED_WINS = "red_wins"
    BLACK_WINS = "black_wins"
    DRAW = "draw"

    @property
    def is_terminal(self) -> bool:
        return self is not GameResult.IN_PROGRESS

    @property
    def winner(self) -> Player | None:
        if self is GameResult.RED_WINS:
            return Player.RED
        if self is GameResult.BLACK_WINS:
            return Player.BLACK
        return None


@dataclass(frozen=True, order=True, slots=True)
class Coordinate:
    """A zero-based coordinate, with ``x`` increasing left-to-right."""

    x: int
    y: int

    def __post_init__(self) -> None:
        _require_int(self.x, "x")
        _require_int(self.y, "y")
        if self.x < 0 or self.y < 0:
            raise ValueError("coordinates cannot be negative")

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y}


@dataclass(frozen=True, slots=True)
class BoardDimensions:
    """Dimensions of a rectangular board; standard Twixt uses 24 by 24."""

    width: int = 24
    height: int = 24

    def __post_init__(self) -> None:
        _require_int(self.width, "width")
        _require_int(self.height, "height")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("board dimensions must be positive")

    def contains(self, coordinate: Coordinate) -> bool:
        if not isinstance(coordinate, Coordinate):
            raise TypeError("coordinate must be a Coordinate")
        return coordinate.x < self.width and coordinate.y < self.height

    def to_dict(self) -> dict[str, int]:
        return {"width": self.width, "height": self.height}


@dataclass(frozen=True, slots=True)
class Peg:
    """A peg owned by a player at one board coordinate."""

    owner: Player
    coordinate: Coordinate

    def __post_init__(self) -> None:
        if not isinstance(self.owner, Player):
            raise TypeError("peg owner must be a Player")
        if not isinstance(self.coordinate, Coordinate):
            raise TypeError("peg coordinate must be a Coordinate")

    def to_dict(self) -> dict[str, object]:
        return {"owner": self.owner.value, "coordinate": self.coordinate.to_dict()}


@dataclass(frozen=True, slots=True)
class Link:
    """An undirected, knight-move connection between two same-owner pegs."""

    owner: Player
    start: Coordinate
    end: Coordinate

    def __post_init__(self) -> None:
        if not isinstance(self.owner, Player):
            raise TypeError("link owner must be a Player")
        if not isinstance(self.start, Coordinate) or not isinstance(self.end, Coordinate):
            raise TypeError("link endpoints must be Coordinates")

        # An undirected link has exactly one in-memory representation.
        if self.end < self.start:
            original_start = self.start
            object.__setattr__(self, "start", self.end)
            object.__setattr__(self, "end", original_start)

        dx = abs(self.start.x - self.end.x)
        dy = abs(self.start.y - self.end.y)
        if (dx, dy) not in {(1, 2), (2, 1)}:
            raise ValueError("link endpoints must be a knight's move apart")

    def to_dict(self) -> dict[str, object]:
        return {
            "owner": self.owner.value,
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
        }


def _coordinate_from_dict(value: object, name: str) -> Coordinate:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    if set(value) != {"x", "y"}:
        raise ValueError(f"{name} must contain exactly x and y")
    return Coordinate(x=_require_int(value["x"], f"{name}.x"), y=_require_int(value["y"], f"{name}.y"))


def _sorted_pegs(pegs: Iterable[Peg]) -> tuple[Peg, ...]:
    values = tuple(pegs)
    if any(not isinstance(peg, Peg) for peg in values):
        raise TypeError("pegs must contain only Peg values")
    return tuple(sorted(values, key=lambda peg: (peg.coordinate.x, peg.coordinate.y, peg.owner.value)))


def _sorted_links(links: Iterable[Link]) -> tuple[Link, ...]:
    values = tuple(links)
    if any(not isinstance(link, Link) for link in values):
        raise TypeError("links must contain only Link values")
    return tuple(
        sorted(
            values,
            key=lambda link: (
                link.start.x,
                link.start.y,
                link.end.x,
                link.end.y,
                link.owner.value,
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class GameState:
    """The canonical, immutable representation of a Twixt position.

    Iterables supplied for pegs and links are converted to deterministically
    ordered tuples. A link is valid only when both endpoint pegs exist and have
    the link's owner. Rule-level questions such as whether a position is
    reachable through legal play belong to the transition layer.
    """

    board: BoardDimensions = BoardDimensions()
    pegs: tuple[Peg, ...] = ()
    links: tuple[Link, ...] = ()
    side_to_move: Player = Player.RED
    result: GameResult = GameResult.IN_PROGRESS

    def __post_init__(self) -> None:
        if not isinstance(self.board, BoardDimensions):
            raise TypeError("board must be BoardDimensions")
        if not isinstance(self.side_to_move, Player):
            raise TypeError("side_to_move must be a Player")
        if not isinstance(self.result, GameResult):
            raise TypeError("result must be a GameResult")

        try:
            pegs = _sorted_pegs(self.pegs)
            links = _sorted_links(self.links)
        except TypeError as exc:
            if str(exc).endswith("is not iterable"):
                raise TypeError("pegs and links must be iterable") from exc
            raise
        object.__setattr__(self, "pegs", pegs)
        object.__setattr__(self, "links", links)

        occupied: dict[Coordinate, Player] = {}
        for peg in pegs:
            if not self.board.contains(peg.coordinate):
                raise ValueError(f"peg at {peg.coordinate!r} is outside the board")
            if peg.coordinate in occupied:
                raise ValueError(f"multiple pegs occupy {peg.coordinate!r}")
            occupied[peg.coordinate] = peg.owner

        if len(set(links)) != len(links):
            raise ValueError("duplicate links are not allowed")
        for link in links:
            if not self.board.contains(link.start) or not self.board.contains(link.end):
                raise ValueError("link endpoint is outside the board")
            if occupied.get(link.start) is not link.owner or occupied.get(link.end) is not link.owner:
                raise ValueError("link endpoints must contain pegs owned by the link owner")

    @classmethod
    def initial(cls, board: BoardDimensions | None = None) -> GameState:
        """Return an empty position with Red to move."""

        return cls(board=board or BoardDimensions())

    @property
    def is_terminal(self) -> bool:
        return self.result.is_terminal

    @property
    def winner(self) -> Player | None:
        return self.result.winner

    def peg_owner_at(self, coordinate: Coordinate) -> Player | None:
        """Return the owner at *coordinate*, or ``None`` for an empty point."""

        if not isinstance(coordinate, Coordinate):
            raise TypeError("coordinate must be a Coordinate")
        if not self.board.contains(coordinate):
            raise ValueError("coordinate is outside the board")
        return next((peg.owner for peg in self.pegs if peg.coordinate == coordinate), None)

    def copy(self) -> GameState:
        """Return an independent, equal snapshot of this immutable state."""

        return GameState(self.board, self.pegs, self.links, self.side_to_move, self.result)

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic dictionary made only of JSON values."""

        return {
            "board": self.board.to_dict(),
            "pegs": [peg.to_dict() for peg in self.pegs],
            "links": [link.to_dict() for link in self.links],
            "side_to_move": self.side_to_move.value,
            "result": self.result.value,
        }

    def to_json(self) -> str:
        """Serialize the state to compact, deterministic JSON."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GameState:
        """Validate and deserialize a dictionary produced by :meth:`to_dict`."""

        if not isinstance(value, Mapping):
            raise TypeError("state must be an object")
        expected = {"board", "pegs", "links", "side_to_move", "result"}
        if set(value) != expected:
            raise ValueError(f"state must contain exactly {', '.join(sorted(expected))}")

        board_value = value["board"]
        if not isinstance(board_value, Mapping) or set(board_value) != {"width", "height"}:
            raise ValueError("board must contain exactly width and height")
        board = BoardDimensions(
            width=_require_int(board_value["width"], "board.width"),
            height=_require_int(board_value["height"], "board.height"),
        )

        peg_values = value["pegs"]
        link_values = value["links"]
        if not isinstance(peg_values, list):
            raise TypeError("pegs must be an array")
        if not isinstance(link_values, list):
            raise TypeError("links must be an array")

        pegs: list[Peg] = []
        for index, peg_value in enumerate(peg_values):
            if not isinstance(peg_value, Mapping) or set(peg_value) != {"owner", "coordinate"}:
                raise ValueError(f"pegs[{index}] must contain exactly owner and coordinate")
            pegs.append(
                Peg(
                    owner=Player(peg_value["owner"]),
                    coordinate=_coordinate_from_dict(peg_value["coordinate"], f"pegs[{index}].coordinate"),
                )
            )

        links: list[Link] = []
        for index, link_value in enumerate(link_values):
            if not isinstance(link_value, Mapping) or set(link_value) != {"owner", "start", "end"}:
                raise ValueError(f"links[{index}] must contain exactly owner, start, and end")
            links.append(
                Link(
                    owner=Player(link_value["owner"]),
                    start=_coordinate_from_dict(link_value["start"], f"links[{index}].start"),
                    end=_coordinate_from_dict(link_value["end"], f"links[{index}].end"),
                )
            )

        return cls(
            board=board,
            pegs=tuple(pegs),
            links=tuple(links),
            side_to_move=Player(value["side_to_move"]),
            result=GameResult(value["result"]),
        )

    @classmethod
    def from_json(cls, value: str) -> GameState:
        """Validate and deserialize a JSON state."""

        if not isinstance(value, str):
            raise TypeError("JSON state must be a string")
        decoded = json.loads(value)
        if not isinstance(decoded, Mapping):
            raise TypeError("state JSON must describe an object")
        return cls.from_dict(decoded)


# A concise alternative name for callers that use "position" terminology.
Position = GameState


__all__ = [
    "BoardDimensions",
    "Coordinate",
    "GameResult",
    "GameState",
    "Link",
    "Peg",
    "Player",
    "Position",
]
