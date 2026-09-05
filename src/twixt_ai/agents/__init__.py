"""Agent protocols and baseline move-selection strategies."""

from .interface import (
    Agent,
    AgentContractError,
    AgentRequest,
    AgentResult,
    ThinkingMetadata,
    select_agent_move,
)
from .random import RandomAgent

__all__ = [
    "Agent",
    "AgentContractError",
    "AgentRequest",
    "AgentResult",
    "RandomAgent",
    "ThinkingMetadata",
    "select_agent_move",
]
