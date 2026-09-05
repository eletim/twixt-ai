# First learned Mini Twixt model

Issue 57 trained the compact 10×10 policy/value network on the first self-play
dataset from Issue 56. The complete measured report, loss history, resumable
checkpoints, and deterministic tiny overfit fixture are checked in under
[`experiments/issue-57`](../experiments/issue-57).

## Configuration

The run used the stable Mini architecture (8 trunk channels, one residual
block, and 16 value-head hidden units), AdamW, a batch size of 64, a learning
rate of 0.001, weight decay of 0.0001, and seed 570100. It trained for five
epochs, loaded `latest.pt`, and resumed through epoch 20. The resumed history
retained the exact five committed metric rows.

The experiment also derives a deterministic one-position, one-hot fixture from
the source dataset and trains it for 100 epochs. This exercises the same data
loader, policy loss, value loss, optimizer, and checkpoint path while providing
a strong overfit sanity signal.

## Results

| Metric | Initial | Final/best | Change |
| --- | ---: | ---: | ---: |
| Training policy loss | 4.5612 | 3.8055 | -16.6% |
| Training value loss | 0.9596 | 0.1433 | -85.1% |
| Training total loss | 5.5208 | 3.9488 | -28.5% |
| Validation total loss | 5.4745 | 5.2986 (epoch 2) | -3.2% |
| Tiny-fixture total loss | 4.7261 | 1.42e-14 | >99.99% |

The 20 real-data epochs took 2.891 seconds on the recorded 24-CPU Linux
environment, including validation and checkpoint writes, for 14,422.2 processed
examples/second. Runtime is machine-dependent; the full environment and exact
floating-point measurements are in `report.json`.

Both real-data policy and value losses decreased, and validation improved
through epoch 2 before worsening. This is positive evidence that optimization
and targets function, while also showing rapid overfitting on this small first
dataset. `best.pt` therefore identifies epoch 2; `latest.pt` retains epoch 20
and the complete resumable state. The report records each epoch's selection
loss and whether it advanced the best checkpoint, plus SHA-256 and byte size for
both final checkpoint files.

All recorded losses and checkpoint tensors were finite, total loss equaled the
policy/value components, and the tiny fixture overfit threshold passed. The
best checkpoint loaded through `NeuralPolicyValue`, produced a normalized
legal-move policy and finite value, and completed a learned two-simulation MCTS
decision.

## Reproduce

From an installed checkout:

```bash
twixt-ai-mini-training-experiment \
  --dataset experiments/issue-56/baseline/dataset \
  --output-dir mini-training-experiment
```

For a source checkout without an installed entry point:

```bash
PYTHONPATH=src python3 -m twixt_ai.training.mini_experiment_cli \
  --dataset experiments/issue-56/baseline/dataset \
  --output-dir mini-training-experiment
```

The command refuses a nonempty output directory, so previous artifacts cannot
be silently overwritten or mixed into a recovery run.
