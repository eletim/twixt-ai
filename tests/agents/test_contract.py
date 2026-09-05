"""Validate the reusable agent contract against a minimal reference agent."""

from __future__ import annotations

from twixt_ai.game import GameState, PegPlacement, legal_peg_placements

from .contract import AgentContract


class FirstLegalAgent:
    """Small test double proving that the shared contract is executable."""

    def choose_move(self, state: GameState) -> PegPlacement:
        return legal_peg_placements(state)[0]


class TestFirstLegalAgentContract(AgentContract):
    agent_factory = FirstLegalAgent
