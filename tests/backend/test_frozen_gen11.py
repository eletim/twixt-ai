"""The browser and evaluation baseline use the committed Gen11 checkpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from twixt_ai.backend.viewer import GEN11_CHECKPOINT, HUMAN_AI_SIMULATIONS, ViewerService
from twixt_ai.evaluation.frozen_gen11_cli import FROZEN_SHA256, FROZEN_GEN11
from twixt_ai.models import load_policy_value_checkpoint


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_checkpoint_digest_load_and_browser_agent() -> None:
    assert GEN11_CHECKPOINT == "models/frozen/gen11/best.pt"
    assert FROZEN_GEN11 == ROOT / GEN11_CHECKPOINT
    assert FROZEN_GEN11.is_file()
    assert hashlib.sha256(FROZEN_GEN11.read_bytes()).hexdigest() == FROZEN_SHA256
    loaded = load_policy_value_checkpoint(FROZEN_GEN11, map_location="cpu")
    assert (loaded.model.config.board_width, loaded.model.config.board_height) == (10, 10)
    agent = ViewerService(ROOT).human_agent()
    assert agent.simulations == HUMAN_AI_SIMULATIONS


def test_browser_has_no_missing_checkpoint_fallback(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown checkpoint"):
        ViewerService(tmp_path).human_agent()
