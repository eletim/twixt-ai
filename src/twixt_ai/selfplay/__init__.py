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
from .trajectory import (
    MatchTrajectory,
    PolicyTarget,
    TrajectoryStep,
    trajectory_from_match,
)

__all__ = [
    "BATCH_FORMAT",
    "BATCH_FORMAT_VERSION",
    "GAME_FAILURE_FORMAT",
    "AgentFactory",
    "BatchConfig",
    "BatchSummary",
    "GameReport",
    "MatchTrajectory",
    "PolicyTarget",
    "TrajectoryStep",
    "run_batch",
    "trajectory_from_match",
]
