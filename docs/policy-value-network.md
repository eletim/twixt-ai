# Policy/value network v1

`PolicyValueNetwork` is the first learned Twixt baseline. It consumes batches
of canonical encoding-v1 positions shaped `[N, 22, 24, 24]`. A small shared
convolutional residual trunk feeds two heads:

- the policy head returns 576 unnormalized logits for training;
- the value head returns one `tanh`-bounded value per position in `[-1, 1]`,
  from the encoded position's side-to-move perspective.

Policy index `y * 24 + x` represents placing a peg at coordinate `(x, y)`.
This row-major mapping is independent of player because the input encodes the
side to move. `coordinate_to_action_index`, `action_index_to_coordinate`, and
`move_to_action_index` expose the mapping. `legal_move_mask` creates the
Boolean action mask, while `mask_policy_logits` replaces illegal logits with
negative infinity without modifying the training output.

Training code calls the network directly and applies its own policy/value
losses. Inference uses `twixt_ai.search.neural.NeuralPolicyValue`, which switches
the model to evaluation behavior for a gradient-free call, restores its prior
mode, masks illegal actions, and returns normalized priors and the value through
MCTS's explicit `PolicyValueEstimate` hook.

`save_policy_value_checkpoint` stores the state dictionary together with the
checkpoint format, encoding version, architecture name/version, complete
`PolicyValueConfig`, and caller metadata. `load_policy_value_checkpoint`
validates that compatibility metadata before constructing the model and
loading weights. A change to tensor semantics or model structure therefore
requires a version change rather than silently loading incompatible weights.
