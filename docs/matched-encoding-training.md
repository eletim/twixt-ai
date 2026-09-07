# Matched Mini encoding training

Issue 75 compares end-to-end training of the version 1 22-plane encoding and
the side-to-move-normalized version 2 10-plane encoding. Both runs consume the
same Issue 56 dataset manifest and train/validation shards. They use the same
8-channel, one-residual-block trunk, policy and value heads, AdamW settings,
batch size, epoch count, shuffle seed, and one-thread CPU environment. There is
no training-time data augmentation. The input encoding and resulting first
convolution width are the only intended differences.

The tiny fixture runs for 300 epochs with learning rate 0.01. A 100-epoch pilot
did not overfit the 10-plane path (its final/initial loss ratio was 0.483), so
the matched schedule was extended rather than weakening the 0.25 pass
threshold. This sensitivity is retained here instead of being hidden by the
final passing run.

The command writes each encoding beneath an unmistakably named directory. Its
checkpoints redundantly record the encoding version, input plane count, full
model config, and dataset digest. `report.json` includes every policy, value,
and total training/validation loss; selection history for `best.pt`; hashes and
sizes for `best.pt` and `latest.pt`; wall time; and examples per second. It also
trains both encodings on the exact same derived one-position tiny-overfit
fixture. Real-dataset loss improvement is deliberately not a pass condition,
so divergence or instability remains visible in the artifact.

Reproduce the checked-in experiment from an installed checkout:

```bash
PYTHONHASHSEED=0 twixt-ai-matched-encoding-training \
  --dataset experiments/issue-56/baseline/dataset \
  --output-dir experiments/issue-75
```

The output directory must be empty or absent, preventing results or
checkpoints from different runs from being mixed.

## Recorded result

The checked-in `experiments/issue-75/report.json` is the source of exact loss
curves, runtime measurements, environment details, and checkpoint history.
Both tiny-overfit checks passed and both full runs completed with finite,
component-consistent losses and weights.

| Metric | 22-plane v1 | 10-plane v2 |
| --- | ---: | ---: |
| Training policy loss, epoch 1 → 20 | 4.5417 → 3.7837 | 4.5292 → 3.6562 |
| Training value loss, epoch 1 → 20 | 0.9528 → 0.1340 | 0.9397 → 0.1505 |
| Training total loss, epoch 1 → 20 | 5.4945 → 3.9177 | 5.4690 → 3.8067 |
| Best validation loss (epoch) | 5.3475 (2) | 5.3349 (3) |
| Final validation loss | 5.7622 | 6.0690 |
| Throughput (examples/s) | 13,619 | 15,800 |
| Wall time (seconds) | 3.062 | 2.639 |
| Tiny final/initial loss ratio | 1.47e-11 | 0.0853 |

Both models overfit the small real dataset: validation loss worsened after its
early best epoch, and the 10-plane model ended with the larger validation
regression despite a slightly better best loss. On this host the compact run
was 16.0% faster by throughput and took 13.8% less wall time. Runtime is
machine-specific, and this experiment does not make a playing-strength claim.
