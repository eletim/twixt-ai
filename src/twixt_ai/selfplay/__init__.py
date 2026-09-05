"""Headless self-play generation and persisted batch manifests."""

from .batch import (
    BATCH_FORMAT,
    BATCH_FORMAT_VERSION,
    GAME_FAILURE_FORMAT,
    AgentFactory,
    BatchConfig,
    BatchSummary,
    GameReport,
    run_batch,
)

__all__ = [
    "BATCH_FORMAT",
    "BATCH_FORMAT_VERSION",
    "GAME_FAILURE_FORMAT",
    "AgentFactory",
    "BatchConfig",
    "BatchSummary",
    "GameReport",
    "run_batch",
]
