"""Side-to-move normalized, compact neural encoding for Mini Twixt.

This is encoding version 2.  Version 1 remains in :mod:`.encoding`; keeping
the entry points and version constants separate prevents an old checkpoint
from silently receiving tensors with different channel semantics.
"""

from __future__ import annotations

import torch
from torch import Tensor

from twixt_ai.game import BoardDimensions, Coordinate, GameState, Link, Peg, Player


MINI_BOARD_SIZE = 10
MINI_ENCODING_VERSION = 2

# Links are stored once, at their left-most endpoint.  A knight edge always
# has a non-zero x displacement, so these four positive-x directions are a
# complete and unambiguous partition of undirected links.
MINI_LINK_DIRECTIONS = (
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
)

MINI_CHANNEL_NAMES = (
    "self_pegs",
    "opponent_pegs",
    *(f"self_links_dx{dx:+d}_dy{dy:+d}" for dx, dy in MINI_LINK_DIRECTIONS),
    *(f"opponent_links_dx{dx:+d}_dy{dy:+d}" for dx, dy in MINI_LINK_DIRECTIONS),
)
MINI_NUM_CHANNELS = len(MINI_CHANNEL_NAMES)
MINI_INPUT_SHAPE = (MINI_NUM_CHANNELS, MINI_BOARD_SIZE, MINI_BOARD_SIZE)

_SELF_PEG_CHANNEL = 0
_OPPONENT_PEG_CHANNEL = 1
_SELF_LINK_BASE = 2
_OPPONENT_LINK_BASE = 2 + len(MINI_LINK_DIRECTIONS)
_DIRECTION_INDEX = {
    direction: index for index, direction in enumerate(MINI_LINK_DIRECTIONS)
}


def _require_player(side_to_move: Player) -> None:
    if not isinstance(side_to_move, Player):
        raise TypeError("side_to_move must be a Player")


def _board(board_width: int, board_height: int) -> BoardDimensions:
    return BoardDimensions(board_width, board_height)


def normalized_board_dimensions(
    side_to_move: Player,
    *,
    board_width: int = MINI_BOARD_SIZE,
    board_height: int = MINI_BOARD_SIZE,
) -> BoardDimensions:
    """Return dimensions after the canonical side-to-move transform.

    Red already has the canonical north/south goal orientation.  For Black,
    x and y are transposed so Black's west/east goals become north/south.
    """

    _require_player(side_to_move)
    board = _board(board_width, board_height)
    if side_to_move is Player.BLACK:
        return BoardDimensions(board.height, board.width)
    return board


def game_to_normalized_coordinate(
    coordinate: Coordinate,
    side_to_move: Player,
    *,
    board_width: int = MINI_BOARD_SIZE,
    board_height: int = MINI_BOARD_SIZE,
) -> Coordinate:
    """Map a game coordinate into the side-to-move canonical frame."""

    if not isinstance(coordinate, Coordinate):
        raise TypeError("coordinate must be a Coordinate")
    _require_player(side_to_move)
    board = _board(board_width, board_height)
    if not board.contains(coordinate):
        raise ValueError("coordinate is outside the board")
    if side_to_move is Player.BLACK:
        return Coordinate(coordinate.y, coordinate.x)
    return coordinate


def normalized_to_game_coordinate(
    coordinate: Coordinate,
    side_to_move: Player,
    *,
    board_width: int = MINI_BOARD_SIZE,
    board_height: int = MINI_BOARD_SIZE,
) -> Coordinate:
    """Invert :func:`game_to_normalized_coordinate`."""

    if not isinstance(coordinate, Coordinate):
        raise TypeError("coordinate must be a Coordinate")
    _require_player(side_to_move)
    normalized_board = normalized_board_dimensions(
        side_to_move, board_width=board_width, board_height=board_height
    )
    if not normalized_board.contains(coordinate):
        raise ValueError("coordinate is outside the normalized board")
    if side_to_move is Player.BLACK:
        return Coordinate(coordinate.y, coordinate.x)
    return coordinate


def game_coordinate_to_normalized_action_index(
    coordinate: Coordinate,
    side_to_move: Player,
    *,
    board_width: int = MINI_BOARD_SIZE,
    board_height: int = MINI_BOARD_SIZE,
) -> int:
    """Map a game coordinate to a row-major normalized policy index."""

    normalized = game_to_normalized_coordinate(
        coordinate,
        side_to_move,
        board_width=board_width,
        board_height=board_height,
    )
    dimensions = normalized_board_dimensions(
        side_to_move, board_width=board_width, board_height=board_height
    )
    return normalized.y * dimensions.width + normalized.x


def normalized_action_index_to_game_coordinate(
    index: int,
    side_to_move: Player,
    *,
    board_width: int = MINI_BOARD_SIZE,
    board_height: int = MINI_BOARD_SIZE,
) -> Coordinate:
    """Invert :func:`game_coordinate_to_normalized_action_index`."""

    _require_player(side_to_move)
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("action index must be an integer")
    dimensions = normalized_board_dimensions(
        side_to_move, board_width=board_width, board_height=board_height
    )
    action_count = dimensions.width * dimensions.height
    if not 0 <= index < action_count:
        raise ValueError(f"action index must be in [0, {action_count})")
    y, x = divmod(index, dimensions.width)
    return normalized_to_game_coordinate(
        Coordinate(x, y),
        side_to_move,
        board_width=board_width,
        board_height=board_height,
    )


def _normalized_endpoints(
    link: Link, state: GameState
) -> tuple[Coordinate, Coordinate]:
    first = game_to_normalized_coordinate(
        link.start,
        state.side_to_move,
        board_width=state.board.width,
        board_height=state.board.height,
    )
    second = game_to_normalized_coordinate(
        link.end,
        state.side_to_move,
        board_width=state.board.width,
        board_height=state.board.height,
    )
    return (second, first) if second.x < first.x else (first, second)


def encode_mini_position(
    state: GameState, *, device: torch.device | str | None = None
) -> Tensor:
    """Encode a position as ten binary planes in the current-player frame.

    The implementation accepts configurable rectangular dimensions.  The
    validated Mini preset is 10x10 and therefore has shape ``[10, 10, 10]``.
    """

    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState")
    dimensions = normalized_board_dimensions(
        state.side_to_move,
        board_width=state.board.width,
        board_height=state.board.height,
    )
    encoded = torch.zeros(
        (MINI_NUM_CHANNELS, dimensions.height, dimensions.width),
        dtype=torch.float32,
        device=device,
    )

    for peg in state.pegs:
        coordinate = game_to_normalized_coordinate(
            peg.coordinate,
            state.side_to_move,
            board_width=state.board.width,
            board_height=state.board.height,
        )
        channel = (
            _SELF_PEG_CHANNEL
            if peg.owner is state.side_to_move
            else _OPPONENT_PEG_CHANNEL
        )
        encoded[channel, coordinate.y, coordinate.x] = 1.0

    for link in state.links:
        start, end = _normalized_endpoints(link, state)
        direction = (end.x - start.x, end.y - start.y)
        base = (
            _SELF_LINK_BASE
            if link.owner is state.side_to_move
            else _OPPONENT_LINK_BASE
        )
        encoded[base + _DIRECTION_INDEX[direction], start.y, start.x] = 1.0
    return encoded


def decode_mini_position(
    encoded: Tensor,
    side_to_move: Player,
    *,
    board_width: int = MINI_BOARD_SIZE,
    board_height: int = MINI_BOARD_SIZE,
) -> GameState:
    """Losslessly reconstruct pegs and links for a known game perspective."""

    if not isinstance(encoded, Tensor):
        raise TypeError("encoded must be a torch.Tensor")
    _require_player(side_to_move)
    dimensions = normalized_board_dimensions(
        side_to_move, board_width=board_width, board_height=board_height
    )
    expected = (MINI_NUM_CHANNELS, dimensions.height, dimensions.width)
    if tuple(encoded.shape) != expected:
        raise ValueError(f"encoded tensor must have shape {expected}")
    if not torch.all((encoded == 0) | (encoded == 1)):
        raise ValueError("encoded tensor must contain only binary values")

    def game_coordinate(x: int, y: int) -> Coordinate:
        return normalized_to_game_coordinate(
            Coordinate(x, y),
            side_to_move,
            board_width=board_width,
            board_height=board_height,
        )

    pegs: list[Peg] = []
    for channel, owner in (
        (_SELF_PEG_CHANNEL, side_to_move),
        (_OPPONENT_PEG_CHANNEL, side_to_move.opponent),
    ):
        for y, x in encoded[channel].nonzero(as_tuple=False).tolist():
            pegs.append(Peg(owner, game_coordinate(x, y)))

    links: list[Link] = []
    for base, owner in (
        (_SELF_LINK_BASE, side_to_move),
        (_OPPONENT_LINK_BASE, side_to_move.opponent),
    ):
        for offset, (dx, dy) in enumerate(MINI_LINK_DIRECTIONS):
            for y, x in encoded[base + offset].nonzero(as_tuple=False).tolist():
                end_x, end_y = x + dx, y + dy
                if not (0 <= end_x < dimensions.width and 0 <= end_y < dimensions.height):
                    raise ValueError("encoded link endpoint is outside the board")
                links.append(
                    Link(owner, game_coordinate(x, y), game_coordinate(end_x, end_y))
                )
    try:
        return GameState(
            board=BoardDimensions(board_width, board_height),
            pegs=tuple(pegs),
            links=tuple(links),
            side_to_move=side_to_move,
        )
    except ValueError as exc:
        raise ValueError(f"invalid compact encoding: {exc}") from exc


__all__ = [
    "MINI_BOARD_SIZE",
    "MINI_CHANNEL_NAMES",
    "MINI_ENCODING_VERSION",
    "MINI_INPUT_SHAPE",
    "MINI_LINK_DIRECTIONS",
    "MINI_NUM_CHANNELS",
    "decode_mini_position",
    "encode_mini_position",
    "game_coordinate_to_normalized_action_index",
    "game_to_normalized_coordinate",
    "normalized_action_index_to_game_coordinate",
    "normalized_board_dimensions",
    "normalized_to_game_coordinate",
]
