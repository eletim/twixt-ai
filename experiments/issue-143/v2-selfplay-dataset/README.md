# Architecture-v2 Mini self-play dataset

This directory contains the first fresh self-play dataset generated from the
architecture-v2 Mini champion bootstrapped for Issue 143. The source checkpoint
is `../architecture-v2-bootstrap/champion.pt`, with SHA-256
`fd5a9ee7dcd6cd12af8429ab0370b858c5271f2a71afd20956a36ea21db3bb87`.
No architecture-v1 checkpoint contributed games or weights.

The run used seed `143100` for 100 games and split seed
`issue-143-v2-143100`. It kept the current `MiniGenerationConfig` search
defaults unchanged: 100 simulations, exploration `sqrt(2)`, a four-move
rollout limit, progressive-widening constant `1.5` and exponent `0.5`, with
policy/value guidance on the 10x10 encoding-v1 Mini board. CUDA execution used
two game threads and shared inference batches of up to 16 positions with a
0.002-second maximum wait; these execution settings did not alter search
semantics.

All 100 games completed without failure and produced 4,702 positions: 4,371
training examples and 331 validation examples. The SHA-256 of
`dataset/manifest.json` is
`e79de1637a0ec0f852c150345ee0ea80b1fe8b15d9474b92c6f07aca29a402e4`.
The manifest records each shard digest, source game and position counts,
dataset split configuration, checkpoint identity, and full MCTS configuration.

`config.json` is the compact resolved configuration and seed record.
`report.json` additionally retains device/runtime information, the full
self-play summary, inference statistics, and policy/value target diagnostics.
The source games and dataset shards are checked in so the manifest and counts
can be independently verified.

Reproduce the artifact from a source checkout with PyTorch 2.8.0+cu128 and an
available CUDA device, targeting an absent or empty output directory:

```bash
PYTHONHASHSEED=0 PYTHONPATH="$PWD/src" \
  python3 -m twixt_ai.training.selfplay_dataset_cli \
  --champion experiments/issue-143/architecture-v2-bootstrap/champion.pt \
  --output-dir /tmp/issue-143-v2-selfplay-dataset \
  --seed 143100 \
  --split-seed issue-143-v2-143100 \
  --workflow-label issue-143-v2-selfplay-dataset \
  --device cuda
```
