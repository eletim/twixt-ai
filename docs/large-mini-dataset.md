# Larger measured Mini Twixt dataset

Issue 88 uses the CUDA self-play recommendation recorded by Issue 87: four
game threads share one checkpoint on the GPU through an inference batcher of
eight positions with a 0.5 ms flush latency. MCTS uses four simulations and a
four-move rollout limit. The fixed checkpoint is
`experiments/issue-57/baseline/best.pt`.

Generation is staged and resumable. The first stage contains 1,000 games. Its
artifacts and policy/value targets are validated before the 5,000-game stage is
started. The checked-in benchmark projects 0.155 and 0.771 hours respectively;
the latter is below the experiment's one-hour scale-up threshold. The 10,000
projection (1.541 hours) is not selected.

## Recorded result

The checked-in run used CPython 3.10.12, PyTorch 2.8.0 with CUDA 12.8, 24
available CPUs, and an NVIDIA GeForce RTX 4060. Both stages completed without a
failed game and passed the full dataset integrity check.

| Stage | Games | Positions | Self-play time | Games/hour | Dataset time |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1k | 1,000 | 60,984 | 500.448 s | 7,193.6 | 45.276 s |
| 5k | 5,000 | 302,299 | 2,527.796 s | 7,120.8 | 233.857 s |

The 1k and 5k manifest SHA-256 hashes are respectively
`9b438f4c3f674934bf51b5657abff4eb44b87cea6be36c7461b3a02bf12d70fc`
and
`a47eda5c2bb48fcd103a3380517a91470054170b24c176969d81c33df16290e6`.
All 363,283 positions retain normalized visit distributions and outcome
targets. The 5k run sustained 7,120.8 games/hour, close to the benchmark's
6,493.1 games/hour projection.

Run from a source checkout with the CUDA-enabled Python environment:

```bash
PYTHONHASHSEED=0 PYTHONPATH=src python3 -m twixt_ai.selfplay.large_experiment_cli \
  --checkpoint experiments/issue-57/baseline/best.pt \
  --benchmark benchmarks/mini-cuda-tuning.json \
  --output-dir experiments/issue-88
```

After every completed stage, `report.json` is atomically updated. A repeat of
the same command skips completed stages. A partial stage is never silently
overwritten. The report records exact seeds, checkpoint and benchmark hashes,
effective CPU/CUDA configuration, failure counts, runtime, games/hour, inference
batch statistics, peak CUDA allocation, dataset manifests and hashes, and the
observed value-target distribution. Dataset validation rechecks every shard
digest and requires every example to contain a normalized MCTS visit policy and
a valid final-outcome value target.
