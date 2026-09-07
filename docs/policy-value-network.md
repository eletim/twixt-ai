# Policy/value network v1

`PolicyValueNetwork` is the first learned Twixt baseline. It consumes batches
of canonical encoding-v1 positions shaped `[N, 22, height, width]`. A small shared
convolutional residual trunk feeds two heads:

- the policy head returns `width * height` unnormalized logits for training;
- the value head returns one `tanh`-bounded value per position in `[-1, 1]`,
  from the encoded position's side-to-move perspective.

Policy index `y * width + x` represents placing a peg at coordinate `(x, y)`.
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
`PolicyValueConfig` (including board width and height), and caller metadata.
`load_policy_value_checkpoint`
validates that compatibility metadata before constructing the model and
loading weights. A change to tensor semantics or model structure therefore
requires a version change rather than silently loading incompatible weights.

## Mini Twixt baseline

`MINI_POLICY_VALUE_CONFIG` is the deliberately small 10x10 baseline. It uses
8 trunk channels, 1 residual block, and 16 value-head hidden units for exactly
24,547 trainable parameters (98,188 bytes of float32 weights). The ordinary
`PolicyValueConfig` fields and the training CLI's `--channels`,
`--residual-blocks`, and `--value-hidden` options remain available for larger
comparison models.

Its input/output contract is `[N, 22, 10, 10]` float tensors to `[N, 100]`
unmasked row-major policy logits and `[N]` side-to-move values in `[-1, 1]`.
Legal moves are applied with `legal_move_mask` and `mask_policy_logits`, exactly
as for the standard model. Mini checkpoints include the complete preset config
and board dimensions plus the format, architecture, and encoding versions;
loading constructs that exact model before weights are accepted, and training
resume additionally rejects a requested config mismatch.

On an AMD Ryzen 9 7900X CPU with PyTorch 2.8.0, one thread, evaluation mode,
and inference mode, a representative batch of 64 positions took a median of
0.91 ms (14 microseconds per position) over 100 timed forwards after 20
warm-ups. This is an environment-specific reference rather than a performance
guarantee. Reproduce it with:

```python
import statistics
import time

import torch
from twixt_ai.models import MINI_POLICY_VALUE_CONFIG, PolicyValueNetwork

torch.set_num_threads(1)
model = PolicyValueNetwork(MINI_POLICY_VALUE_CONFIG).eval()
batch = torch.zeros((64, *model.input_shape))
with torch.inference_mode():
    for _ in range(20):
        model(batch)
    samples = []
    for _ in range(100):
        start = time.perf_counter()
        model(batch)
        samples.append(time.perf_counter() - start)
print(statistics.median(samples))
```
