"""Canonical neural-network input encoding for Twixt positions.

The public channel ordering in this module is checkpoint compatibility data.
Changing it requires a new ``ENCODING_VERSION``.
"""

from __future__ import annotations

from enum import Enum

import torch
from torch import Tensor

from twixt_ai.game import Coordinate, GameResult, GameState, Link, Peg, Player


BOARD_SIZE = 24
ENCODING_VERSION = 1

# A link is marked at each endpoint in the plane pointing toward its other
# endpoint. Keeping all eight directions makes geometric augmentation a pure
# spatial/channel permutation.
LINK_DIRECTIONS = (
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
)

CHANNEL_NAMES = (
    "red_pegs",
    "black_pegs",
    *(f"red_links_dx{dx:+d}_dy{dy:+d}" for dx, dy in LINK_DIRECTIONS),
    *(f"black_links_dx{dx:+d}_dy{dy:+d}" for dx, dy in LINK_DIRECTIONS),
    "red_to_move",
    "black_to_move",
    "red_goal_borders",
    "black_goal_borders",
)
NUM_CHANNELS = len(CHANNEL_NAMES)
INPUT_SHAPE = (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)

_PEG_CHANNEL = {Player.RED: 0, Player.BLACK: 1}
_LINK_BASE = {Player.RED: 2, Player.BLACK: 2 + len(LINK_DIRECTIONS)}
_DIRECTION_INDEX = {direction: index for index, direction in enumerate(LINK_DIRECTIONS)}
_TURN_CHANNEL = {Player.RED: 18, Player.BLACK: 19}
_GOAL_CHANNEL = {Player.RED: 20, Player.BLACK: 21}


class BoardSymmetry(str, Enum):
    """The dihedral symmetries of a square board.

    Symmetries that exchange the x and y axes also exchange Red and Black, so
    each transformed player retains the corresponding goal-edge orientation.
    """

    IDENTITY = "identity"
    FLIP_X = "flip_x"
    FLIP_Y = "flip_y"
    ROTATE_180 = "rotate_180"
    TRANSPOSE = "transpose"
    ANTI_TRANSPOSE = "anti_transpose"
    ROTATE_90 = "rotate_90"
    ROTATE_270 = "rotate_270"


SYMMETRIES = tuple(BoardSymmetry)
_AXIS_SWAPPING = {
    BoardSymmetry.TRANSPOSE,
    BoardSymmetry.ANTI_TRANSPOSE,
    BoardSymmetry.ROTATE_90,
    BoardSymmetry.ROTATE_270,
}


def _require_game_state(state: GameState) -> None:
    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState")


def encode_position(state: GameState, *, device: torch.device | str | None = None) -> Tensor:
    """Encode *state* as a deterministic ``float32`` channels-first tensor."""

    _require_game_state(state)
    shape = (NUM_CHANNELS, state.board.height, state.board.width)
    encoded = torch.zeros(shape, dtype=torch.float32, device=device)

    for peg in state.pegs:
        encoded[_PEG_CHANNEL[peg.owner], peg.coordinate.y, peg.coordinate.x] = 1.0

    for link in state.links:
        dx = link.end.x - link.start.x
        dy = link.end.y - link.start.y
        base = _LINK_BASE[link.owner]
        encoded[base + _DIRECTION_INDEX[(dx, dy)], link.start.y, link.start.x] = 1.0
        encoded[base + _DIRECTION_INDEX[(-dx, -dy)], link.end.y, link.end.x] = 1.0

    encoded[_TURN_CHANNEL[state.side_to_move]].fill_(1.0)

    # Corners are forbidden to both players and therefore belong to neither
    # goal-border plane.
    encoded[_GOAL_CHANNEL[Player.RED], 0, 1:-1] = 1.0
    encoded[_GOAL_CHANNEL[Player.RED], -1, 1:-1] = 1.0
    encoded[_GOAL_CHANNEL[Player.BLACK], 1:-1, 0] = 1.0
    encoded[_GOAL_CHANNEL[Player.BLACK], 1:-1, -1] = 1.0
    return encoded


def _coerce_symmetry(symmetry: BoardSymmetry | str) -> BoardSymmetry:
    try:
        return BoardSymmetry(symmetry)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown board symmetry: {symmetry!r}") from exc


def _transform_xy(x: int, y: int, size: int, symmetry: BoardSymmetry) -> tuple[int, int]:
    last = size - 1
    transforms = {
        BoardSymmetry.IDENTITY: (x, y),
        BoardSymmetry.FLIP_X: (last - x, y),
        BoardSymmetry.FLIP_Y: (x, last - y),
        BoardSymmetry.ROTATE_180: (last - x, last - y),
        BoardSymmetry.TRANSPOSE: (y, x),
        BoardSymmetry.ANTI_TRANSPOSE: (last - y, last - x),
        BoardSymmetry.ROTATE_90: (last - y, x),
        BoardSymmetry.ROTATE_270: (y, last - x),
    }
    return transforms[symmetry]


def transform_coordinate(
    coordinate: Coordinate,
    symmetry: BoardSymmetry | str,
    *,
    board_size: int = BOARD_SIZE,
) -> Coordinate:
    """Map a coordinate through one square-board symmetry."""

    if not isinstance(coordinate, Coordinate):
        raise TypeError("coordinate must be a Coordinate")
    if isinstance(board_size, bool) or not isinstance(board_size, int) or board_size <= 0:
        raise ValueError("board_size must be a positive integer")
    if coordinate.x >= board_size or coordinate.y >= board_size:
        raise ValueError("coordinate is outside the board")
    transformed = _transform_xy(coordinate.x, coordinate.y, board_size, _coerce_symmetry(symmetry))
    return Coordinate(*transformed)


def _transform_player(player: Player, symmetry: BoardSymmetry) -> Player:
    return player.opponent if symmetry in _AXIS_SWAPPING else player


def transform_state(state: GameState, symmetry: BoardSymmetry | str) -> GameState:
    """Return the game state produced by a square-board symmetry."""

    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState")
    if state.board.width != state.board.height:
        raise ValueError("board symmetries require a square board")
    symmetry = _coerce_symmetry(symmetry)
    size = state.board.width

    def coordinate(value: Coordinate) -> Coordinate:
        return transform_coordinate(value, symmetry, board_size=size)

    pegs = tuple(
        Peg(_transform_player(peg.owner, symmetry), coordinate(peg.coordinate))
        for peg in state.pegs
    )
    links = tuple(
        Link(_transform_player(link.owner, symmetry), coordinate(link.start), coordinate(link.end))
        for link in state.links
    )
    result = state.result
    if symmetry in _AXIS_SWAPPING:
        if result is GameResult.RED_WINS:
            result = GameResult.BLACK_WINS
        elif result is GameResult.BLACK_WINS:
            result = GameResult.RED_WINS
    return GameState(
        board=state.board,
        pegs=pegs,
        links=links,
        side_to_move=_transform_player(state.side_to_move, symmetry),
        result=result,
    )


def _spatial_transform(encoded: Tensor, symmetry: BoardSymmetry) -> Tensor:
    if symmetry is BoardSymmetry.IDENTITY:
        return encoded.clone()
    if symmetry is BoardSymmetry.FLIP_X:
        return encoded.flip(-1)
    if symmetry is BoardSymmetry.FLIP_Y:
        return encoded.flip(-2)
    if symmetry is BoardSymmetry.ROTATE_180:
        return encoded.flip((-2, -1))
    if symmetry is BoardSymmetry.TRANSPOSE:
        return encoded.transpose(-2, -1)
    if symmetry is BoardSymmetry.ANTI_TRANSPOSE:
        return encoded.transpose(-2, -1).flip((-2, -1))
    if symmetry is BoardSymmetry.ROTATE_90:
        return torch.rot90(encoded, -1, (-2, -1))
    return torch.rot90(encoded, 1, (-2, -1))


def transform_encoding(encoded: Tensor, symmetry: BoardSymmetry | str) -> Tensor:
    """Transform one encoded position without reconstructing a ``GameState``.

    Leading batch dimensions are allowed; the final dimensions must be the
    canonical ``[channels, height, width]`` shape.
    """

    if not isinstance(encoded, Tensor):
        raise TypeError("encoded must be a torch.Tensor")
    if encoded.ndim < 3 or encoded.shape[-3] != NUM_CHANNELS:
        raise ValueError(f"encoded tensor must have {NUM_CHANNELS} channels")
    if encoded.shape[-2] != encoded.shape[-1]:
        raise ValueError("encoding symmetries require a square board")
    symmetry = _coerce_symmetry(symmetry)
    spatial = _spatial_transform(encoded, symmetry)
    transformed = torch.empty_like(spatial)

    def destination_player(player: Player) -> Player:
        return _transform_player(player, symmetry)

    for player in Player:
        transformed[..., _PEG_CHANNEL[destination_player(player)], :, :] = spatial[
            ..., _PEG_CHANNEL[player], :, :
        ]
        transformed[..., _TURN_CHANNEL[destination_player(player)], :, :] = spatial[
            ..., _TURN_CHANNEL[player], :, :
        ]
        transformed[..., _GOAL_CHANNEL[destination_player(player)], :, :] = spatial[
            ..., _GOAL_CHANNEL[player], :, :
        ]
        for index, (dx, dy) in enumerate(LINK_DIRECTIONS):
            origin = Coordinate(BOARD_SIZE // 2, BOARD_SIZE // 2)
            endpoint = Coordinate(origin.x + dx, origin.y + dy)
            mapped_origin = transform_coordinate(origin, symmetry)
            mapped_endpoint = transform_coordinate(endpoint, symmetry)
            mapped_direction = (
                mapped_endpoint.x - mapped_origin.x,
                mapped_endpoint.y - mapped_origin.y,
            )
            source_channel = _LINK_BASE[player] + index
            destination_channel = (
                _LINK_BASE[destination_player(player)]
                + _DIRECTION_INDEX[mapped_direction]
            )
            transformed[..., destination_channel, :, :] = spatial[..., source_channel, :, :]
    return transformed


__all__ = [
    "BOARD_SIZE",
    "CHANNEL_NAMES",
    "ENCODING_VERSION",
    "INPUT_SHAPE",
    "LINK_DIRECTIONS",
    "NUM_CHANNELS",
    "SYMMETRIES",
    "BoardSymmetry",
    "encode_position",
    "transform_coordinate",
    "transform_encoding",
    "transform_state",
]
