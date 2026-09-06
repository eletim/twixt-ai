from __future__ import annotations

import pytest
import torch

from twixt_ai.game import BoardDimensions, Coordinate, GameState, Link, Peg, Player
from twixt_ai.models import (
    ENCODING_VERSION,
    MINI_CHANNEL_NAMES,
    MINI_ENCODING_VERSION,
    MINI_INPUT_SHAPE,
    MINI_LINK_DIRECTIONS,
    BoardSymmetry,
    decode_mini_position,
    encode_mini_position,
    game_coordinate_to_normalized_action_index,
    game_to_normalized_coordinate,
    normalized_action_index_to_game_coordinate,
    normalized_board_dimensions,
    normalized_to_game_coordinate,
    transform_state,
)


def sample_state(*, side_to_move: Player = Player.RED) -> GameState:
    return GameState(
        board=BoardDimensions(10, 10),
        pegs=(
            Peg(Player.RED, Coordinate(2, 3)),
            Peg(Player.RED, Coordinate(4, 4)),
            Peg(Player.BLACK, Coordinate(6, 5)),
            Peg(Player.BLACK, Coordinate(7, 7)),
        ),
        links=(
            Link(Player.RED, Coordinate(2, 3), Coordinate(4, 4)),
            Link(Player.BLACK, Coordinate(6, 5), Coordinate(7, 7)),
        ),
        side_to_move=side_to_move,
    )


def test_compact_encoding_has_a_distinct_version_and_ten_planes() -> None:
    encoded = encode_mini_position(sample_state())

    assert MINI_ENCODING_VERSION != ENCODING_VERSION
    assert encoded.shape == MINI_INPUT_SHAPE == (10, 10, 10)
    assert encoded.dtype is torch.float32
    assert len(MINI_CHANNEL_NAMES) == len(set(MINI_CHANNEL_NAMES)) == 10
    assert not any("turn" in name or "history" in name for name in MINI_CHANNEL_NAMES)


def test_black_perspective_transposes_board_and_exchanges_semantic_owners() -> None:
    state = sample_state(side_to_move=Player.BLACK)
    encoded = encode_mini_position(state)

    assert encoded[0, 6, 5] == 1  # Black (6,5) becomes normalized (5,6).
    assert encoded[1, 2, 3] == 1  # Red (2,3) becomes normalized (3,2).
    assert encoded[2:].count_nonzero() == len(state.links)


def test_equivalent_opposite_player_positions_normalize_identically() -> None:
    red_position = sample_state(side_to_move=Player.RED)
    black_position = transform_state(red_position, BoardSymmetry.TRANSPOSE)

    assert black_position.side_to_move is Player.BLACK
    assert torch.equal(
        encode_mini_position(red_position), encode_mini_position(black_position)
    )


def test_each_undirected_link_orientation_is_encoded_exactly_once() -> None:
    starts_and_ends = (
        (Coordinate(2, 4), Coordinate(3, 2)),
        (Coordinate(2, 4), Coordinate(3, 6)),
        (Coordinate(5, 4), Coordinate(7, 3)),
        (Coordinate(5, 4), Coordinate(7, 5)),
    )
    coordinates = sorted({value for pair in starts_and_ends for value in pair})
    state = GameState(
        board=BoardDimensions(10, 10),
        pegs=tuple(Peg(Player.RED, coordinate) for coordinate in coordinates),
        links=tuple(Link(Player.RED, *pair) for pair in starts_and_ends),
    )

    encoded = encode_mini_position(state)

    assert encoded[2:6].count_nonzero() == 4
    for offset, (start, direction) in enumerate(
        zip((pair[0] for pair in starts_and_ends), MINI_LINK_DIRECTIONS)
    ):
        assert encoded[2 + offset, start.y, start.x] == 1
        assert (
            encoded[2 + offset].count_nonzero() == 1
        ), f"orientation {direction} was not unique"


@pytest.mark.parametrize("side_to_move", tuple(Player))
def test_compact_encoding_round_trips_complete_position(side_to_move: Player) -> None:
    state = sample_state(side_to_move=side_to_move)

    reconstructed = decode_mini_position(
        encode_mini_position(state), side_to_move
    )

    assert reconstructed == state


@pytest.mark.parametrize("side_to_move", tuple(Player))
def test_policy_actions_round_trip_for_every_mini_coordinate(
    side_to_move: Player,
) -> None:
    for y in range(10):
        for x in range(10):
            coordinate = Coordinate(x, y)
            index = game_coordinate_to_normalized_action_index(
                coordinate, side_to_move
            )
            assert normalized_action_index_to_game_coordinate(
                index, side_to_move
            ) == coordinate


def test_coordinate_transform_supports_configurable_rectangular_boards() -> None:
    coordinate = Coordinate(6, 3)

    normalized = game_to_normalized_coordinate(
        coordinate, Player.BLACK, board_width=8, board_height=5
    )

    assert normalized == Coordinate(3, 6)
    assert normalized_board_dimensions(
        Player.BLACK, board_width=8, board_height=5
    ) == BoardDimensions(5, 8)
    assert normalized_to_game_coordinate(
        normalized, Player.BLACK, board_width=8, board_height=5
    ) == coordinate
    assert encode_mini_position(
        GameState(board=BoardDimensions(8, 5), side_to_move=Player.BLACK)
    ).shape == (10, 8, 5)


def test_decoder_rejects_links_without_endpoint_pegs() -> None:
    encoded = torch.zeros(MINI_INPUT_SHAPE)
    encoded[2, 4, 2] = 1

    with pytest.raises(ValueError, match="link endpoints"):
        decode_mini_position(encoded, Player.RED)
