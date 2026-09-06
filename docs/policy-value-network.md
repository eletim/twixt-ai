# Policy/value network v1

`PolicyValueNetwork` is the first learned Twixt baseline. Its checkpoint-stable
configuration selects encoding v1 with 22 input channels or encoding v2 with
10 input channels. A small shared convolutional residual trunk feeds two heads:

- the policy head returns `width * height` unnormalized logits for training;
- the value head returns one `tanh`-bounded value per position in `[-1, 1]`,
  from the encoded position's side-to-move perspective.

For v1, policy index `y * width + x` represents game coordinate `(x, y)`. For
v2, it represents the row-major coordinate after side-to-move normalization.
Training and `NeuralPolicyValue` dispatch input encoding, targets, masks, and
returned move priors from the configured encoding version. `mask_policy_logits`
replaces illegal logits with negative infinity without modifying the training
output.

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

`MINI_NORMALIZED_POLICY_VALUE_CONFIG` is the corresponding opt-in encoding-v2
preset with `[N, 10, 10, 10]` inputs. It has the same trunk and heads, but
23,683 parameters because its first convolution consumes 10 rather than 22
planes. The original Mini preset remains v1 until comparison work selects a
default.

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
