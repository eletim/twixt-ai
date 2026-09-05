"""Inference adapter connecting a policy/value network to MCTS."""

from __future__ import annotations

import torch

from twixt_ai.game import GameState, PegPlacement
from twixt_ai.models import (
    PolicyValueNetwork,
    encode_position,
    legal_move_mask,
    mask_policy_logits,
    move_to_action_index,
)

from .mcts import PolicyValueEstimate


class NeuralPolicyValue:
    """Callable inference hook for :class:`~twixt_ai.search.MCTSAgent`.

    Inference is performed without gradients and with the model in evaluation
    mode. The model's device determines where inputs and masks are allocated.
    """

    def __init__(self, model: PolicyValueNetwork) -> None:
        if not isinstance(model, PolicyValueNetwork):
            raise TypeError("model must be a PolicyValueNetwork")
        self.model = model

    def __call__(
        self, state: GameState, moves: tuple[PegPlacement, ...]
    ) -> PolicyValueEstimate:
        if not isinstance(state, GameState):
            raise TypeError("state must be a GameState")
        if any(not isinstance(move, PegPlacement) for move in moves):
            raise TypeError("moves must contain only PegPlacement values")
        parameter = next(self.model.parameters())
        device = parameter.device
        inputs = encode_position(state, device=device).unsqueeze(0)
        mask = legal_move_mask(moves, device=device)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                logits, values = self.model(inputs)
                masked = mask_policy_logits(logits, mask)
                probabilities = torch.softmax(masked, dim=-1)[0]
        finally:
            self.model.train(was_training)
        priors = {
            move: float(probabilities[move_to_action_index(move)].item())
            for move in moves
        }
        return PolicyValueEstimate(priors, float(values[0].item()))


__all__ = ["NeuralPolicyValue"]
