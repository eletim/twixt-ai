"""Dispatch neural inputs and policy coordinates by encoding version."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor

from twixt_ai.game import BoardDimensions, Coordinate, GameState, PegPlacement, Player

from .encoding import ENCODING_VERSION, encode_position
from .mini_encoding import (
    MINI_ENCODING_VERSION,
    encode_mini_position,
    game_coordinate_to_normalized_action_index,
)


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
        return coordinate.y * board_width + coordinate.x
    if encoding_version == MINI_ENCODING_VERSION:
        return game_coordinate_to_normalized_action_index(
            coordinate,
            side_to_move,
            board_width=board_width,
            board_height=board_height,
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


__all__ = [
    "coordinate_to_action_index_for_version",
    "encode_position_for_version",
    "legal_move_mask_for_version",
    "move_to_action_index_for_version",
]
