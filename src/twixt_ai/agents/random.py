"""Uniform-random baseline agent."""

from __future__ import annotations

from random import Random

from .interface import AgentContractError, AgentRequest, AgentResult


class RandomAgent:
    """Select uniformly from the legal moves supplied by the engine.

    A request seed initializes a decision-local random number generator.  This
    keeps seeded choices reproducible without sharing or modifying process-wide
    random state.
    """

    def choose_move(self, request: AgentRequest) -> AgentResult:
        """Return one uniformly sampled legal move for *request*."""

        if not isinstance(request, AgentRequest):
            raise TypeError("request must be an AgentRequest")
        if not request.legal_moves:
            raise AgentContractError("cannot select a move when no legal moves exist")

        move = Random(request.seed).choice(request.legal_moves)
        return AgentResult(move)


__all__ = ["RandomAgent"]
