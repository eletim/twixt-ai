# Model training

`twixt-ai-train` trains `PolicyValueNetwork` from a version 1 dataset created
by `twixt-ai-dataset`. The trainer verifies every shard digest before training,
uses the sparse MCTS policy target when present (or the played move otherwise),
and minimizes policy cross-entropy plus value mean squared error.

```bash
twixt-ai-train --dataset dataset --output-dir training-run \
  --epochs 20 --batch-size 64 --learning-rate 0.001 --seed 1234
```

The output directory contains `latest.pt`, `best.pt`, `metrics.jsonl`, and
`summary.json`. The summary records the complete config, seed, dataset manifest
digest, checkpoint names, best epoch, and metric history. The best checkpoint
uses validation loss, or training loss when there are no validation examples.

Both checkpoints can be loaded for inference with
`load_policy_value_checkpoint`. They also contain optimizer and scheduler state
for recovery. Resume an interrupted run by supplying the same settings and a
new total epoch target if needed:

```bash
twixt-ai-train --dataset dataset --output-dir training-run \
  --epochs 40 --batch-size 64 --learning-rate 0.001 --seed 1234 --resume
```

The dataset, model shape, optimizer, scheduler, device, seed, and all other
settings must match. Only the total epoch target may increase. Available
optimizers are AdamW and SGD; `--scheduler step` enables a configurable StepLR.
