from __future__ import annotations

import torch

from twixt_ai.game import (
    BoardDimensions,
    Coordinate,
    GameState,
    PegPlacement,
    Player,
    legal_peg_placements,
)
from twixt_ai.models import (
    ACTION_COUNT,
    PolicyValueConfig,
    PolicyValueNetwork,
    action_index_to_coordinate,
    coordinate_to_action_index,
    encode_position,
    legal_move_mask,
    load_policy_value_checkpoint,
    mask_policy_logits,
    move_to_action_index,
    save_policy_value_checkpoint,
)


def small_model() -> PolicyValueNetwork:
    return PolicyValueNetwork(PolicyValueConfig(channels=4, residual_blocks=1, value_hidden=8))


def test_forward_returns_training_logits_and_side_to_move_value() -> None:
    model = small_model()
    inputs = torch.stack((encode_position(GameState.initial()),) * 2)

    logits, values = model(inputs)

    assert logits.shape == (2, ACTION_COUNT)
    assert values.shape == (2,)
    assert torch.all(values >= -1)
    assert torch.all(values <= 1)
    (logits.sum() + values.sum()).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_mini_model_uses_matching_input_and_action_dimensions() -> None:
    config = PolicyValueConfig(
        channels=2,
        residual_blocks=1,
        value_hidden=4,
        board_width=10,
        board_height=10,
    )
    model = PolicyValueNetwork(config)

    logits, values = model(encode_position(
        GameState.initial(BoardDimensions(10, 10))
    ).unsqueeze(0))

    assert logits.shape == (1, 100)
    assert values.shape == (1,)
    assert action_index_to_coordinate(
        99, board_width=10, board_height=10
    ) == Coordinate(9, 9)


def test_action_mapping_is_row_major_and_invertible() -> None:
    for index in range(ACTION_COUNT):
        coordinate = action_index_to_coordinate(index)
        assert coordinate_to_action_index(coordinate) == index
    move = PegPlacement(Player.RED, Coordinate(7, 11))
    assert move_to_action_index(move) == 11 * 24 + 7


def test_legal_mask_excludes_borders_and_occupied_points() -> None:
    state = GameState.initial()
    moves = legal_peg_placements(state)
    mask = legal_move_mask(moves)
    logits = torch.zeros((2, ACTION_COUNT))

    masked = mask_policy_logits(logits, mask)

    assert mask.dtype is torch.bool
    assert mask.sum().item() == len(moves)
    assert torch.isneginf(masked[:, coordinate_to_action_index(Coordinate(0, 3))]).all()
    assert masked[:, move_to_action_index(moves[0])].eq(0).all()
    assert logits.eq(0).all()


def test_checkpoint_round_trip_preserves_config_weights_and_metadata(tmp_path) -> None:
    model = small_model()
    checkpoint_path = tmp_path / "model.pt"
    save_policy_value_checkpoint(checkpoint_path, model, metadata={"step": 12})

    loaded = load_policy_value_checkpoint(checkpoint_path)

    assert loaded.model.config == model.config
    assert loaded.metadata == {"step": 12}
    for expected, actual in zip(model.state_dict().values(), loaded.model.state_dict().values()):
        assert torch.equal(expected, actual)


def test_forward_runs_on_cuda_when_available() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    model = small_model().to(device)
    inputs = encode_position(GameState.initial(), device=device).unsqueeze(0)
    logits, values = model(inputs)
    assert logits.device.type == values.device.type == "cuda"
