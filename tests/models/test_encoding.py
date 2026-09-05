from __future__ import annotations

import pytest
import torch

from twixt_ai.game import BoardDimensions, Coordinate, GameState, Link, Peg, Player
from twixt_ai.models import (
    CHANNEL_NAMES,
    INPUT_SHAPE,
    LINK_DIRECTIONS,
    SYMMETRIES,
    BoardSymmetry,
    encode_position,
    transform_coordinate,
    transform_encoding,
    transform_state,
)


def sample_state() -> GameState:
    return GameState(
        pegs=(
            Peg(Player.RED, Coordinate(2, 3)),
            Peg(Player.RED, Coordinate(4, 4)),
            Peg(Player.BLACK, Coordinate(11, 9)),
            Peg(Player.BLACK, Coordinate(12, 11)),
        ),
        links=(
            Link(Player.RED, Coordinate(2, 3), Coordinate(4, 4)),
            Link(Player.BLACK, Coordinate(11, 9), Coordinate(12, 11)),
        ),
        side_to_move=Player.BLACK,
    )


def test_channel_contract_and_position_features() -> None:
    encoded = encode_position(sample_state())

    assert tuple(encoded.shape) == INPUT_SHAPE == (22, 24, 24)
    assert encoded.dtype is torch.float32
    assert len(CHANNEL_NAMES) == len(set(CHANNEL_NAMES)) == 22
    assert encoded[0, 3, 2] == 1
    assert encoded[1, 9, 11] == 1
    assert encoded[2 + LINK_DIRECTIONS.index((2, 1)), 3, 2] == 1
    assert encoded[2 + LINK_DIRECTIONS.index((-2, -1)), 4, 4] == 1
    assert encoded[18].count_nonzero() == 0
    assert encoded[19].count_nonzero() == 24 * 24
    assert encoded[20].count_nonzero() == 44
    assert encoded[21].count_nonzero() == 44
    assert encoded[20, 0, 0] == encoded[21, 0, 0] == 0


def test_encoding_is_deterministic_and_does_not_alias() -> None:
    first = encode_position(sample_state())
    second = encode_position(sample_state())
    assert torch.equal(first, second)
    first.zero_()
    assert second.count_nonzero() > 0


@pytest.mark.parametrize("symmetry", SYMMETRIES)
def test_encoded_and_state_symmetry_transforms_agree(symmetry: BoardSymmetry) -> None:
    state = sample_state()
    expected = encode_position(transform_state(state, symmetry))
    actual = transform_encoding(encode_position(state), symmetry)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("symmetry", SYMMETRIES)
def test_symmetry_supports_leading_batch_dimensions(symmetry: BoardSymmetry) -> None:
    encoded = encode_position(sample_state())
    batch = torch.stack((encoded, encoded))
    transformed = transform_encoding(batch, symmetry)
    assert torch.equal(transformed[0], transform_encoding(encoded, symmetry))


def test_axis_swapping_symmetry_swaps_players_and_turn() -> None:
    transformed = transform_state(sample_state(), BoardSymmetry.ROTATE_90)
    assert transformed.side_to_move is Player.RED
    assert Peg(Player.BLACK, Coordinate(20, 2)) in transformed.pegs
    assert Peg(Player.RED, Coordinate(14, 11)) in transformed.pegs


def test_coordinate_transform_rejects_out_of_board_coordinate() -> None:
    with pytest.raises(ValueError, match="outside"):
        transform_coordinate(Coordinate(24, 0), BoardSymmetry.IDENTITY)


def test_encoder_supports_mini_board() -> None:
    encoded = encode_position(GameState.initial(BoardDimensions(10, 10)))

    assert encoded.shape == (22, 10, 10)
    assert encoded[18].count_nonzero() == 10 * 10
    assert encoded[20].count_nonzero() == 16
    assert encoded[21].count_nonzero() == 16


def test_transform_supports_other_square_board_sizes() -> None:
    encoded = torch.zeros((22, 10, 10))
    assert transform_encoding(encoded, BoardSymmetry.IDENTITY).shape == encoded.shape


def test_transform_rejects_rectangular_tensor() -> None:
    with pytest.raises(ValueError, match="square"):
        transform_encoding(torch.zeros((22, 10, 12)), BoardSymmetry.IDENTITY)
