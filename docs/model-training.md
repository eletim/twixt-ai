# Model training

`twixt-ai-train` trains `PolicyValueNetwork` from a version 1 dataset created
by `twixt-ai-dataset`. The trainer verifies every shard digest before training,
uses the sparse MCTS policy target when present (or the played move otherwise),
and minimizes policy cross-entropy plus value mean squared error.

```bash
twixt-ai-train --dataset dataset --output-dir training-run \
  --epochs 20 --batch-size 64 --learning-rate 0.001 --seed 1234 \
  --device auto
```

The output directory contains `latest.pt`, `best.pt`, `metrics.jsonl`, and
`summary.json`. The summary records the complete config, seed, dataset manifest
digest, checkpoint names, best epoch, and metric history. The best checkpoint
uses validation loss, or training loss when there are no validation examples.
The CLI defaults to `auto`, which selects CUDA exactly when PyTorch reports it
available. Use `--device cpu` for a forced CPU run or `--device cuda` for a
strict GPU run that fails when CUDA is unavailable. The summary and checkpoints
record both the request and resolved device along with CUDA/PyTorch runtime
details. The summary's `performance` object records optimization throughput and
elapsed time, plus peak allocated CUDA memory for GPU runs. Run
`twixt-ai-device --device auto` for a lightweight local probe.

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
CUDA checkpoints are saved portably: the default checkpoint loader maps model
weights to CPU for inference, while CUDA resume restores model and optimizer
state to the selected GPU.

The Mini experiment command accepts the same device contract, for example
`twixt-ai-mini-training-experiment --dataset dataset --output-dir experiment
--device cuda`. Its environment record reflects the device used by training,
and its checkpoint validation still loads the resulting model on CPU.
