"""Fresh, reproducible Mini architecture-v2 champion initialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from twixt_ai.models import (
    ARCHITECTURE_NAME,
    ARCHITECTURE_VERSION,
    MINI_POLICY_VALUE_CONFIG,
    PolicyValueNetwork,
    save_policy_value_checkpoint,
)


BOOTSTRAP_FORMAT = "twixt-ai-mini-champion-bootstrap"
BOOTSTRAP_VERSION = 1
DEFAULT_BOOTSTRAP_SEED = 143_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_shapes(model: PolicyValueNetwork) -> dict[str, list[int]]:
    return {
        name: list(tensor.shape)
        for name, tensor in model.state_dict().items()
    }


def _linear_shapes(module: nn.Module) -> list[list[int]]:
    return [
        [layer.in_features, layer.out_features]
        for layer in module.modules()
        if isinstance(layer, nn.Linear)
    ]


def bootstrap_mini_champion(
    output_dir: str | Path,
    *,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Create a fresh v2 Mini champion and its reproducibility manifest.

    There is deliberately no initial-checkpoint argument: architecture-v1
    history cannot enter this bootstrap path through warm-starting or conversion.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed < 2**63:
        raise ValueError("seed must be in [0, 2**63)")

    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError("output directory must be empty or not exist")
    root.mkdir(parents=True, exist_ok=True)

    # Isolate initialization from the caller's RNG state and construct on CPU so
    # that the recorded seed has one unambiguous initialization path.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = PolicyValueNetwork(MINI_POLICY_VALUE_CONFIG)

    initialization = {
        "kind": "fresh-random-pytorch-defaults",
        "seed": seed,
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "warm_started": False,
        "parent_checkpoint": None,
        "architecture": {
            "name": ARCHITECTURE_NAME,
            "version": ARCHITECTURE_VERSION,
        },
        "model_config": MINI_POLICY_VALUE_CONFIG.to_dict(),
        "input_shape": list(model.input_shape),
        "trunk_output_shape": [
            MINI_POLICY_VALUE_CONFIG.channels,
            MINI_POLICY_VALUE_CONFIG.board_height,
            MINI_POLICY_VALUE_CONFIG.board_width,
        ],
        "policy_head_linear_shapes": _linear_shapes(model.policy_head),
        "value_head_linear_shapes": _linear_shapes(model.value_head),
        "state_dict_shapes": _tensor_shapes(model),
    }
    checkpoint = root / "champion.pt"
    save_policy_value_checkpoint(
        checkpoint,
        model,
        metadata={
            "role": "initial-champion",
            "bootstrap": initialization,
        },
    )
    manifest = {
        "format": BOOTSTRAP_FORMAT,
        "version": BOOTSTRAP_VERSION,
        "checkpoint": {
            "path": checkpoint.name,
            "sha256": _sha256(checkpoint),
            "bytes": checkpoint.stat().st_size,
        },
        "initialization": initialization,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "BOOTSTRAP_FORMAT",
    "BOOTSTRAP_VERSION",
    "DEFAULT_BOOTSTRAP_SEED",
    "bootstrap_mini_champion",
]
