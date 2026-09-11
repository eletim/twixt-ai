"""Dispatch neural inputs and policy coordinates by encoding version."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import Tensor

from twixt_ai.game import BoardDimensions, Coordinate, GameState, PegPlacement, Player

from .encoding import ENCODING_VERSION, encode_position, encode_positions
from .mini_encoding import (
    MINI_ENCODING_VERSION,
    encode_mini_position,
    game_coordinate_to_normalized_action_index,
)


def _row_major_action_strides(board_width: int) -> tuple[int, int]:
    return 1, board_width


def _normalized_action_strides(
    side_to_move: Player,
    *,
    board_width: int,
    board_height: int,
) -> tuple[int, int]:
    if side_to_move is Player.BLACK:
        return board_height, 1
    return _row_major_action_strides(board_width)


def _action_index(coordinate: Coordinate, strides: tuple[int, int]) -> int:
    x_stride, y_stride = strides
    return coordinate.x * x_stride + coordinate.y * y_stride


def encode_position_for_version(
    state: GameState,
    encoding_version: int,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Encode *state* with the explicitly selected checkpoint encoding."""

    if encoding_version == ENCODING_VERSION:
        return encode_position(state, device=device)
    if encoding_version == MINI_ENCODING_VERSION:
        return encode_mini_position(state, device=device)
    raise ValueError(f"unsupported encoding version: {encoding_version}")


def encode_positions_for_version(
    states: Sequence[GameState],
    encoding_version: int,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Encode a batch with the explicitly selected checkpoint encoding."""

    if encoding_version == ENCODING_VERSION:
        return encode_positions(states, device=device)
    if encoding_version == MINI_ENCODING_VERSION:
        return torch.stack(
            [encode_mini_position(state, device=device) for state in states]
        )
    raise ValueError(f"unsupported encoding version: {encoding_version}")


def coordinate_to_action_index_for_version(
    coordinate: Coordinate,
    side_to_move: Player,
    encoding_version: int,
    *,
    board_width: int,
    board_height: int,
) -> int:
    """Map a game coordinate into the selected encoding's policy frame."""

    if encoding_version == ENCODING_VERSION:
        if not isinstance(coordinate, Coordinate):
            raise TypeError("coordinate must be a Coordinate")
        board = BoardDimensions(board_width, board_height)
        if not board.contains(coordinate):
            raise ValueError(
                f"coordinate must lie on a {board_width}x{board_height} board"
            )
        return _action_index(coordinate, _row_major_action_strides(board_width))
    if encoding_version == MINI_ENCODING_VERSION:
        # Retain the public transform's validation contract while sharing the
        # actual mapping primitive with batched inference below.
        game_coordinate_to_normalized_action_index(
            coordinate,
            side_to_move,
            board_width=board_width,
            board_height=board_height,
        )
        return _action_index(
            coordinate,
            _normalized_action_strides(
                side_to_move,
                board_width=board_width,
                board_height=board_height,
            ),
        )
    raise ValueError(f"unsupported encoding version: {encoding_version}")


def move_to_action_index_for_version(
    move: PegPlacement,
    encoding_version: int,
    *,
    board_width: int,
    board_height: int,
) -> int:
    """Map a peg placement into the selected encoding's policy frame."""

    if not isinstance(move, PegPlacement):
        raise TypeError("move must be a PegPlacement")
    return coordinate_to_action_index_for_version(
        move.coordinate,
        move.player,
        encoding_version,
        board_width=board_width,
        board_height=board_height,
    )


def legal_move_mask_for_version(
    moves: Iterable[PegPlacement],
    encoding_version: int,
    *,
    board_width: int,
    board_height: int,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return a policy mask in the selected encoding's coordinate frame."""

    action_count = BoardDimensions(board_width, board_height).width * board_height
    mask = torch.zeros(action_count, dtype=torch.bool, device=device)
    try:
        for move in moves:
            mask[
                move_to_action_index_for_version(
                    move,
                    encoding_version,
                    board_width=board_width,
                    board_height=board_height,
                )
            ] = True
    except TypeError as exc:
        if str(exc).endswith("is not iterable"):
            raise TypeError("moves must be an iterable of PegPlacement values") from exc
        raise
    return mask


def batched_action_indices_for_version(
    move_batches: Sequence[Sequence[PegPlacement]],
    encoding_version: int,
    *,
    board_width: int,
    board_height: int,
) -> list[list[int]]:
    """Map batches of legal moves with one encoding-version dispatch."""

    board = BoardDimensions(board_width, board_height)
    if encoding_version == ENCODING_VERSION:
        _, row_stride = _row_major_action_strides(board.width)
        action_indices: list[list[int]] = []
        for moves in move_batches:
            indices: list[int] = []
            for move in moves:
                if not isinstance(move, PegPlacement):
                    raise TypeError("move must be a PegPlacement")
                if not board.contains(move.coordinate):
                    raise ValueError(
                        f"coordinate must lie on a {board_width}x{board_height} board"
                    )
                indices.append(move.coordinate.y * row_stride + move.coordinate.x)
            action_indices.append(indices)
        return action_indices
    if encoding_version == MINI_ENCODING_VERSION:
        action_indices: list[list[int]] = []
        for moves in move_batches:
            indices = []
            for move in moves:
                if not isinstance(move, PegPlacement):
                    raise TypeError("move must be a PegPlacement")
                if not board.contains(move.coordinate):
                    raise ValueError("coordinate is outside the board")
                indices.append(
                    _action_index(
                        move.coordinate,
                        _normalized_action_strides(
                            move.player,
                            board_width=board.width,
                            board_height=board.height,
                        ),
                    )
                )
            action_indices.append(indices)
        return action_indices
    raise ValueError(f"unsupported encoding version: {encoding_version}")


def batched_legal_move_mask(
    action_indices: Sequence[Sequence[int]], action_count: int
) -> Tensor:
    """Build all legal-mask rows with one indexed tensor update."""

    masks = torch.zeros((len(action_indices), action_count), dtype=torch.bool)
    flat_indices = [
        row_index * action_count + action_index
        for row_index, indices in enumerate(action_indices)
        for action_index in indices
    ]
    masks.view(-1)[flat_indices] = True
    return masks


__all__ = [
    "batched_action_indices_for_version",
    "batched_legal_move_mask",
    "coordinate_to_action_index_for_version",
    "encode_position_for_version",
    "encode_positions_for_version",
    "legal_move_mask_for_version",
    "move_to_action_index_for_version",
]
