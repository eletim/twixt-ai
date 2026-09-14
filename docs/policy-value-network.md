# Policy/value network v2

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

Architecture version 2 is structurally incompatible with every version-1
checkpoint, including the committed Mini champion lineage through Issue 128.
Those checkpoints remain historical evidence, but current loaders reject them
and they cannot be passed to `--initial-checkpoint` or `--initial-champion`.
Further training under the widened heads must begin from a newly bootstrapped
version-2 champion rather than warm-starting that lineage.

## Mini Twixt baseline

`MINI_POLICY_VALUE_CONFIG` is the explicit default 10x10 baseline. Issue 77
retained its 22-plane encoding v1 after the v0.0.3 measurements did not show
that encoding v2 preserved learned playing strength. It uses
8 trunk channels and 1 residual block. Both heads flatten the 8x10x10 trunk
output directly, then use two 256-unit fully connected hidden layers. The
policy head emits 100 logits and the value head emits one tanh-bounded scalar.
The model has exactly 570,437 trainable parameters. The ordinary
`PolicyValueConfig` fields and the training CLI's `--channels`,
`--residual-blocks`, and `--value-hidden` options remain available for
comparison models; `value_hidden` controls the shared hidden width of both
heads.

Its input/output contract is `[N, 22, 10, 10]` float tensors to `[N, 100]`
unmasked row-major policy logits and `[N]` side-to-move values in `[-1, 1]`.
Legal moves are applied with `legal_move_mask` and `mask_policy_logits`, exactly
as for the standard model. Mini checkpoints include the complete preset config
and board dimensions plus the format, architecture, and encoding versions;
loading constructs that exact model before weights are accepted, and training
resume additionally rejects a requested config mismatch.

`MINI_NORMALIZED_POLICY_VALUE_CONFIG` is the corresponding opt-in encoding-v2
preset with `[N, 10, 10, 10]` inputs. It has the same trunk and heads, but
569,573 parameters because its first convolution consumes 10 rather than 22
planes. It remains supported for regression and comparison, but is not the
default. See the [Mini encoding decision](mini-encoding-decision.md) for the
measured correctness, cost, training, and playing-strength evidence.

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
