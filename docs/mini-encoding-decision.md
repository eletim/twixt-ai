# Mini neural encoding decision

Issue 77 retains the 22-plane version 1 encoding as the default for Mini Twixt
in v0.0.3. The compact 10-plane version 2 encoding remains supported through
`MINI_NORMALIZED_POLICY_VALUE_CONFIG` for regression and further comparison,
but the measured evidence does not justify making it the default.

The default is therefore explicit:

| Role | Model config | Encoding | Input shape |
| --- | --- | --- | --- |
| Mini default | `MINI_POLICY_VALUE_CONFIG` | v1, 22-plane | `[N, 22, 10, 10]` |
| Comparison alternative | `MINI_NORMALIZED_POLICY_VALUE_CONFIG` | v2, 10-plane normalized | `[N, 10, 10, 10]` |

## Evidence

The decision combines the checked-in v0.0.3 artifacts rather than choosing on
one metric alone.

- **Correctness and information completeness.** The v2 encoder round-trips
  every peg and link when supplied the side-to-move perspective, and its
  normalized action mapping is invertible. Both versions retain the complete
  position information consumed by the current model. These contracts are
  covered by the encoder and model tests.
- **Encoding and model cost.** The
  [Issue 74 artifact](../benchmarks/mini-encoding-comparison.json) shows that v2
  reduces encoded tensor storage by 54.5%, encoding latency by 12.1%, and model
  parameters by 3.5%. Its paired CPU throughput improved by 5.8% for
  single-position inference, 23.8% for batch-32 inference, and 11.6% for
  training. CUDA differences were not material on the measured workload.
- **Training behavior.** In the matched
  [Issue 75 artifact](../experiments/issue-75/report.json), both encodings
  passed the one-position overfit check and produced finite, internally
  consistent losses. On the full dataset, v2 processed 15,800 examples/second
  versus 13,619 for v1 and reached a similar best validation loss (5.335 versus
  5.347). Both then overfit: their final validation losses were worse than
  their selected best checkpoints. Training therefore establishes that both
  pipelines work, not that v2 is stronger.
- **Learned playing strength.** In the primary paired policy+value matchup from
  the [Issue 76 artifact](../experiments/issue-76/report.json), v2 scored 6 wins,
  8 losses, and 6 draws against v1. Its 30% win-rate 95% Wilson interval was
  14.5–51.9%; v1's 40% interval was 21.9–61.3%. The sample is inconclusive and
  the observed result does not demonstrate preserved strength. Both encodings
  were also weak against equal-budget non-neural MCTS, while policy-only
  diagnostics showed that both value heads were harmful.

The cost advantage is real, but playing strength is the adoption gate. Because
that gate was not met, no default change is justified. More training data or
corrected value targets should precede another adoption decision.

## Checkpoint compatibility

Encoding selection continues to come from checkpoint metadata, never tensor
shape inference. Every new checkpoint records `encoding_version` both at the
top level and in its complete model config, and loading rejects disagreement,
unknown versions, and channel/version mismatches. Older v1 checkpoints that
predate the config's encoding fields continue to be interpreted only as v1
with 22 channels. Existing v2 checkpoints retain their explicit v2 metadata
and continue to load through the non-default preset. Thus retaining v1 as the
default does not silently reinterpret any existing checkpoint.
