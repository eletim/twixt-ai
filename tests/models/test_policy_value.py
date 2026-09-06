from __future__ import annotations

from dataclasses import replace

import pytest
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
    ENCODING_VERSION,
    MINI_ENCODING_VERSION,
    MINI_NORMALIZED_POLICY_VALUE_CONFIG,
    MINI_NUM_CHANNELS,
    MINI_POLICY_VALUE_CONFIG,
    NUM_CHANNELS,
    PolicyValueConfig,
    PolicyValueNetwork,
    action_index_to_coordinate,
    coordinate_to_action_index,
    encode_mini_position,
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


def test_mini_baseline_is_compact_and_preserves_the_model_contract() -> None:
    model = PolicyValueNetwork(MINI_POLICY_VALUE_CONFIG)
    state = GameState.initial(BoardDimensions(10, 10))
    inputs = encode_position(state).unsqueeze(0)

    logits, values = model(inputs)
    mask = legal_move_mask(
        legal_peg_placements(state), board_width=10, board_height=10
    )
    masked = mask_policy_logits(logits, mask)

    assert MINI_POLICY_VALUE_CONFIG == PolicyValueConfig(
        channels=8,
        residual_blocks=1,
        value_hidden=16,
        board_width=10,
        board_height=10,
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 24_547
    assert inputs.shape == (1, 22, 10, 10)
    assert logits.shape == (1, 100)
    assert values.shape == (1,)
    assert -1 <= values.item() <= 1
    assert torch.isneginf(masked[0, ~mask]).all()
    assert torch.equal(masked[0, mask], logits[0, mask])


def test_normalized_mini_model_accepts_the_ten_plane_encoding() -> None:
    model = PolicyValueNetwork(MINI_NORMALIZED_POLICY_VALUE_CONFIG)
    inputs = encode_mini_position(
        GameState.initial(BoardDimensions(10, 10))
    ).unsqueeze(0)

    logits, values = model(inputs)

    assert model.input_shape == (10, 10, 10)
    assert MINI_NORMALIZED_POLICY_VALUE_CONFIG.input_channels == MINI_NUM_CHANNELS
    assert (
        MINI_NORMALIZED_POLICY_VALUE_CONFIG.encoding_version
        == MINI_ENCODING_VERSION
    )
    assert logits.shape == (1, 100)
    assert values.shape == (1,)


def test_encoding_version_and_channel_count_must_match() -> None:
    with pytest.raises(ValueError, match="requires 10 input channels"):
        PolicyValueConfig(
            input_channels=NUM_CHANNELS,
            encoding_version=MINI_ENCODING_VERSION,
        )
    with pytest.raises(ValueError, match="unsupported encoding version"):
        PolicyValueConfig(input_channels=10, encoding_version=99)


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


@pytest.mark.parametrize(
    ("config", "encoding_version", "input_channels"),
    (
        (MINI_POLICY_VALUE_CONFIG, ENCODING_VERSION, NUM_CHANNELS),
        (
            MINI_NORMALIZED_POLICY_VALUE_CONFIG,
            MINI_ENCODING_VERSION,
            MINI_NUM_CHANNELS,
        ),
    ),
)
def test_checkpoint_round_trip_preserves_selected_encoding(
    tmp_path,
    config: PolicyValueConfig,
    encoding_version: int,
    input_channels: int,
) -> None:
    checkpoint_path = tmp_path / f"encoding-v{encoding_version}.pt"
    model = PolicyValueNetwork(config)

    save_policy_value_checkpoint(checkpoint_path, model)
    payload = torch.load(checkpoint_path, weights_only=True)
    loaded = load_policy_value_checkpoint(checkpoint_path)

    assert payload["encoding_version"] == encoding_version
    assert payload["config"]["encoding_version"] == encoding_version
    assert payload["config"]["input_channels"] == input_channels
    assert loaded.model.config == config
    assert loaded.model.input_shape == (input_channels, 10, 10)


def test_loader_accepts_legacy_v1_config_without_encoding_fields(tmp_path) -> None:
    checkpoint_path = tmp_path / "legacy-v1.pt"
    model = PolicyValueNetwork(MINI_POLICY_VALUE_CONFIG)
    save_policy_value_checkpoint(checkpoint_path, model)
    payload = torch.load(checkpoint_path, weights_only=True)
    payload["config"].pop("input_channels")
    payload["config"].pop("encoding_version")
    torch.save(payload, checkpoint_path)

    loaded = load_policy_value_checkpoint(checkpoint_path)

    assert loaded.model.config.input_channels == NUM_CHANNELS
    assert loaded.model.config.encoding_version == ENCODING_VERSION


def test_loader_rejects_disagreeing_checkpoint_encoding_metadata(tmp_path) -> None:
    checkpoint_path = tmp_path / "mismatched.pt"
    save_policy_value_checkpoint(
        checkpoint_path, PolicyValueNetwork(MINI_NORMALIZED_POLICY_VALUE_CONFIG)
    )
    payload = torch.load(checkpoint_path, weights_only=True)
    payload["encoding_version"] = ENCODING_VERSION
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match="encoding_version"):
        load_policy_value_checkpoint(checkpoint_path)


def test_mini_checkpoint_records_and_enforces_complete_model_config(tmp_path) -> None:
    checkpoint_path = tmp_path / "mini.pt"
    save_policy_value_checkpoint(
        checkpoint_path,
        PolicyValueNetwork(MINI_POLICY_VALUE_CONFIG),
        metadata={"baseline": "mini"},
    )

    payload = torch.load(checkpoint_path, weights_only=True)
    assert payload["config"] == MINI_POLICY_VALUE_CONFIG.to_dict()
    assert payload["metadata"] == {"baseline": "mini"}

    payload["config"] = replace(MINI_POLICY_VALUE_CONFIG, channels=16).to_dict()
    torch.save(payload, checkpoint_path)
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_policy_value_checkpoint(checkpoint_path)


def test_mini_baseline_forward_runs_on_cuda_when_available() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    model = PolicyValueNetwork(MINI_POLICY_VALUE_CONFIG).to(device)
    inputs = encode_position(
        GameState.initial(BoardDimensions(10, 10)), device=device
    ).unsqueeze(0)
    logits, values = model(inputs)
    assert logits.device.type == values.device.type == "cuda"
