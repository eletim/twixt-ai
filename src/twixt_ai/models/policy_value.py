"""Small residual policy/value network for configured Twixt board dimensions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from twixt_ai.game import BoardDimensions, Coordinate, PegPlacement

from .encoding import BOARD_SIZE, ENCODING_VERSION, NUM_CHANNELS
from .mini_encoding import MINI_ENCODING_VERSION, MINI_NUM_CHANNELS


ACTION_COUNT = BOARD_SIZE * BOARD_SIZE
ARCHITECTURE_NAME = "twixt-resnet-policy-value"
ARCHITECTURE_VERSION = 1
CHECKPOINT_FORMAT = "twixt-ai-policy-value"
CHECKPOINT_VERSION = 1

_ENCODING_CHANNELS = {
    ENCODING_VERSION: NUM_CHANNELS,
    MINI_ENCODING_VERSION: MINI_NUM_CHANNELS,
}


@dataclass(frozen=True, slots=True)
class PolicyValueConfig:
    """Checkpoint-stable architecture settings for :class:`PolicyValueNetwork`."""

    channels: int = 32
    residual_blocks: int = 3
    value_hidden: int = 64
    board_width: int = BOARD_SIZE
    board_height: int = BOARD_SIZE
    input_channels: int = NUM_CHANNELS
    encoding_version: int = ENCODING_VERSION

    def __post_init__(self) -> None:
        for name in (
            "channels",
            "residual_blocks",
            "value_hidden",
            "board_width",
            "board_height",
            "input_channels",
            "encoding_version",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        expected_channels = _ENCODING_CHANNELS.get(self.encoding_version)
        if expected_channels is None:
            raise ValueError(f"unsupported encoding version: {self.encoding_version}")
        if self.input_channels != expected_channels:
            raise ValueError(
                f"encoding version {self.encoding_version} requires "
                f"{expected_channels} input channels"
            )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PolicyValueConfig:
        if not isinstance(value, Mapping):
            raise TypeError("model config must be a mapping")
        legacy = {"channels", "residual_blocks", "value_hidden"}
        with_board = {*legacy, "board_width", "board_height"}
        with_encoding = {*legacy, "input_channels", "encoding_version"}
        expected = {*with_board, "input_channels", "encoding_version"}
        if set(value) not in (legacy, with_board, with_encoding, expected):
            raise ValueError(
                f"model config must contain {sorted(legacy)}, with optional legacy "
                "board dimensions, or the complete versioned encoding fields"
            )
        return cls(**dict(value))  # type: ignore[arg-type]


# Issue 77 retained the version 1 encoding as the explicit Mini default because
# the v0.0.3 strength comparison did not establish that version 2 preserved
# playing strength. Keep this separate from the 24x24 defaults and from the
# loadable version 2 comparison preset below.
MINI_POLICY_VALUE_CONFIG = PolicyValueConfig(
    channels=8,
    residual_blocks=1,
    value_hidden=16,
    board_width=10,
    board_height=10,
)

# Non-default v2 comparison preset. Its distinct encoding metadata keeps v2
# checkpoints loadable without reinterpreting them as the v1 default.
MINI_NORMALIZED_POLICY_VALUE_CONFIG = PolicyValueConfig(
    channels=8,
    residual_blocks=1,
    value_hidden=16,
    board_width=10,
    board_height=10,
    input_channels=MINI_NUM_CHANNELS,
    encoding_version=MINI_ENCODING_VERSION,
)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(inputs + self.layers(inputs))


class PolicyValueNetwork(nn.Module):
    """Shared residual trunk with policy logits and a bounded value head.

    ``forward`` is the training interface. It accepts a batch shaped
    ``[N, input_channels, height, width]`` and returns unmasked policy logits shaped
    ``[N, height * width]`` plus values shaped ``[N]``. Values are in
    ``[-1, 1]`` and always describe the encoded position from its side-to-move
    perspective.
    """

    def __init__(self, config: PolicyValueConfig | None = None) -> None:
        super().__init__()
        self.config = config or PolicyValueConfig()
        self.input_shape = (
            self.config.input_channels,
            self.config.board_height,
            self.config.board_width,
        )
        self.action_count = self.config.board_width * self.config.board_height
        channels = self.config.channels
        self.trunk = nn.Sequential(
            nn.Conv2d(
                self.config.input_channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            *(_ResidualBlock(channels) for _ in range(self.config.residual_blocks)),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * self.action_count, self.action_count),
        )
        self.value_features = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.action_count, self.config.value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.value_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        if not isinstance(inputs, Tensor):
            raise TypeError("inputs must be a torch.Tensor")
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != self.input_shape:
            raise ValueError(f"inputs must have shape [N, {', '.join(map(str, self.input_shape))}]")
        shared = self.trunk(inputs)
        logits = self.policy_head(shared)
        values = self.value_head(self.value_features(shared)).squeeze(-1)
        return logits, values


def coordinate_to_action_index(
    coordinate: Coordinate,
    *,
    board_width: int = BOARD_SIZE,
    board_height: int = BOARD_SIZE,
) -> int:
    """Map an in-bounds coordinate to its row-major policy index."""

    if not isinstance(coordinate, Coordinate):
        raise TypeError("coordinate must be a Coordinate")
    board = BoardDimensions(board_width, board_height)
    if not board.contains(coordinate):
        raise ValueError(f"coordinate must lie on a {board_width}x{board_height} board")
    return coordinate.y * board_width + coordinate.x


def action_index_to_coordinate(
    index: int,
    *,
    board_width: int = BOARD_SIZE,
    board_height: int = BOARD_SIZE,
) -> Coordinate:
    """Invert :func:`coordinate_to_action_index`."""

    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("action index must be an integer")
    action_count = BoardDimensions(board_width, board_height).width * board_height
    if not 0 <= index < action_count:
        raise ValueError(f"action index must be in [0, {action_count})")
    y, x = divmod(index, board_width)
    return Coordinate(x, y)


def move_to_action_index(
    move: PegPlacement,
    *,
    board_width: int = BOARD_SIZE,
    board_height: int = BOARD_SIZE,
) -> int:
    if not isinstance(move, PegPlacement):
        raise TypeError("move must be a PegPlacement")
    return coordinate_to_action_index(
        move.coordinate, board_width=board_width, board_height=board_height
    )


def legal_move_mask(
    moves: Iterable[PegPlacement], *,
    board_width: int = BOARD_SIZE,
    board_height: int = BOARD_SIZE,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return a Boolean action mask using the configured row-major mapping."""

    action_count = BoardDimensions(board_width, board_height).width * board_height
    mask = torch.zeros(action_count, dtype=torch.bool, device=device)
    try:
        for move in moves:
            mask[move_to_action_index(
                move, board_width=board_width, board_height=board_height
            )] = True
    except TypeError as exc:
        if str(exc).endswith("is not iterable"):
            raise TypeError("moves must be an iterable of PegPlacement values") from exc
        raise
    return mask


def mask_policy_logits(logits: Tensor, mask: Tensor) -> Tensor:
    """Return logits with illegal actions set to negative infinity.

    ``mask`` may be one-dimensional and broadcast across a batch, or have the
    same shape as ``logits``. The input tensor is never modified.
    """

    if not isinstance(logits, Tensor) or not isinstance(mask, Tensor):
        raise TypeError("logits and mask must be torch.Tensor values")
    if logits.ndim < 1:
        raise ValueError("logits must have at least one dimension")
    if not logits.is_floating_point():
        raise TypeError("logits must have a floating-point dtype")
    if mask.dtype is not torch.bool:
        raise TypeError("mask must have Boolean dtype")
    if mask.device != logits.device:
        raise ValueError("mask and logits must be on the same device")
    if tuple(mask.shape) not in {(logits.shape[-1],), tuple(logits.shape)}:
        raise ValueError("mask must be one-dimensional or match logits")
    return logits.masked_fill(~mask, -torch.inf)


@dataclass(frozen=True, slots=True)
class LoadedPolicyValueCheckpoint:
    model: PolicyValueNetwork
    metadata: Mapping[str, object]


def save_policy_value_checkpoint(
    path: str | PathLike[str],
    model: PolicyValueNetwork,
    *,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Save weights together with all architecture compatibility metadata."""

    if not isinstance(model, PolicyValueNetwork):
        raise TypeError("model must be a PolicyValueNetwork")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping or None")
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE_NAME,
        "architecture_version": ARCHITECTURE_VERSION,
        "encoding_version": model.config.encoding_version,
        "config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, Path(path))


def load_policy_value_checkpoint(
    path: str | PathLike[str],
    *,
    map_location: torch.device | str | None = "cpu",
) -> LoadedPolicyValueCheckpoint:
    """Load a checkpoint, rejecting incompatible formats before its weights."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    compatibility = {
        "format": CHECKPOINT_FORMAT,
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE_NAME,
        "architecture_version": ARCHITECTURE_VERSION,
    }
    for key, expected in compatibility.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"incompatible checkpoint {key}: expected {expected!r}, "
                f"got {payload.get(key)!r}"
            )
    config = PolicyValueConfig.from_dict(payload.get("config"))  # type: ignore[arg-type]
    if payload.get("encoding_version") != config.encoding_version:
        raise ValueError(
            "incompatible checkpoint encoding_version: expected "
            f"{config.encoding_version!r} from model config, "
            f"got {payload.get('encoding_version')!r}"
        )
    state_dict = payload.get("state_dict")
    metadata = payload.get("metadata")
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint state_dict must be a mapping")
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint metadata must be a mapping")
    model = PolicyValueNetwork(config)
    if map_location is not None:
        model.to(map_location)
    model.load_state_dict(state_dict)
    return LoadedPolicyValueCheckpoint(model, dict(metadata))


__all__ = [
    "ACTION_COUNT",
    "ARCHITECTURE_NAME",
    "ARCHITECTURE_VERSION",
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "LoadedPolicyValueCheckpoint",
    "MINI_NORMALIZED_POLICY_VALUE_CONFIG",
    "MINI_POLICY_VALUE_CONFIG",
    "PolicyValueConfig",
    "PolicyValueNetwork",
    "action_index_to_coordinate",
    "coordinate_to_action_index",
    "legal_move_mask",
    "load_policy_value_checkpoint",
    "mask_policy_logits",
    "move_to_action_index",
    "save_policy_value_checkpoint",
]
