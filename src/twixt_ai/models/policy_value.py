"""Small residual policy/value network for standard Twixt positions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from twixt_ai.game import Coordinate, PegPlacement

from .encoding import BOARD_SIZE, ENCODING_VERSION, INPUT_SHAPE, NUM_CHANNELS


ACTION_COUNT = BOARD_SIZE * BOARD_SIZE
ARCHITECTURE_NAME = "twixt-resnet-policy-value"
ARCHITECTURE_VERSION = 1
CHECKPOINT_FORMAT = "twixt-ai-policy-value"
CHECKPOINT_VERSION = 1


@dataclass(frozen=True, slots=True)
class PolicyValueConfig:
    """Checkpoint-stable architecture settings for :class:`PolicyValueNetwork`."""

    channels: int = 32
    residual_blocks: int = 3
    value_hidden: int = 64

    def __post_init__(self) -> None:
        for name in ("channels", "residual_blocks", "value_hidden"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PolicyValueConfig:
        if not isinstance(value, Mapping):
            raise TypeError("model config must be a mapping")
        expected = {"channels", "residual_blocks", "value_hidden"}
        if set(value) != expected:
            raise ValueError(f"model config must contain exactly {sorted(expected)}")
        return cls(**{name: value[name] for name in expected})  # type: ignore[arg-type]


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
    ``[N, 22, 24, 24]`` and returns unmasked policy logits shaped ``[N, 576]``
    plus values shaped ``[N]``. Values are in ``[-1, 1]`` and always describe
    the encoded position from its side-to-move perspective.
    """

    def __init__(self, config: PolicyValueConfig | None = None) -> None:
        super().__init__()
        self.config = config or PolicyValueConfig()
        channels = self.config.channels
        self.trunk = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            *(_ResidualBlock(channels) for _ in range(self.config.residual_blocks)),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * ACTION_COUNT, ACTION_COUNT),
        )
        self.value_features = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(ACTION_COUNT, self.config.value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.value_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        if not isinstance(inputs, Tensor):
            raise TypeError("inputs must be a torch.Tensor")
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != INPUT_SHAPE:
            raise ValueError(f"inputs must have shape [N, {', '.join(map(str, INPUT_SHAPE))}]")
        shared = self.trunk(inputs)
        logits = self.policy_head(shared)
        values = self.value_head(self.value_features(shared)).squeeze(-1)
        return logits, values


def coordinate_to_action_index(coordinate: Coordinate) -> int:
    """Map a standard-board coordinate to its row-major policy index."""

    if not isinstance(coordinate, Coordinate):
        raise TypeError("coordinate must be a Coordinate")
    if coordinate.x >= BOARD_SIZE or coordinate.y >= BOARD_SIZE:
        raise ValueError(f"coordinate must lie on a {BOARD_SIZE}x{BOARD_SIZE} board")
    return coordinate.y * BOARD_SIZE + coordinate.x


def action_index_to_coordinate(index: int) -> Coordinate:
    """Invert :func:`coordinate_to_action_index`."""

    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("action index must be an integer")
    if not 0 <= index < ACTION_COUNT:
        raise ValueError(f"action index must be in [0, {ACTION_COUNT})")
    y, x = divmod(index, BOARD_SIZE)
    return Coordinate(x, y)


def move_to_action_index(move: PegPlacement) -> int:
    if not isinstance(move, PegPlacement):
        raise TypeError("move must be a PegPlacement")
    return coordinate_to_action_index(move.coordinate)


def legal_move_mask(
    moves: Iterable[PegPlacement], *, device: torch.device | str | None = None
) -> Tensor:
    """Return a Boolean ``[576]`` mask using the canonical action mapping."""

    mask = torch.zeros(ACTION_COUNT, dtype=torch.bool, device=device)
    try:
        for move in moves:
            mask[move_to_action_index(move)] = True
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
    if logits.ndim < 1 or logits.shape[-1] != ACTION_COUNT:
        raise ValueError(f"logits must end with {ACTION_COUNT} actions")
    if not logits.is_floating_point():
        raise TypeError("logits must have a floating-point dtype")
    if mask.dtype is not torch.bool:
        raise TypeError("mask must have Boolean dtype")
    if mask.device != logits.device:
        raise ValueError("mask and logits must be on the same device")
    if tuple(mask.shape) not in {(ACTION_COUNT,), tuple(logits.shape)}:
        raise ValueError("mask must have shape [576] or match logits")
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
        "encoding_version": ENCODING_VERSION,
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
        "encoding_version": ENCODING_VERSION,
    }
    for key, expected in compatibility.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"incompatible checkpoint {key}: expected {expected!r}, "
                f"got {payload.get(key)!r}"
            )
    config = PolicyValueConfig.from_dict(payload.get("config"))  # type: ignore[arg-type]
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
