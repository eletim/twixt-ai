# CUDA throughput tuning

Issue 87 adds one end-to-end benchmark for choosing Mini Twixt training and
neural self-play settings before increasing game counts. Reproduce the checked-in
RTX-class report with:

```bash
PYTHONHASHSEED=0 twixt-ai-cuda-tuning \
  --dataset experiments/issue-56/baseline/dataset \
  --checkpoint experiments/issue-57/baseline/best.pt \
  --output benchmarks/mini-cuda-tuning.json \
  --games 4 --warmup-games 4 --workers 1 4 \
  --inference-batch-sizes 1 4 8 \
  --flush-latencies-seconds 0.0005 0.002 \
  --simulations 4 20 --training-batch-sizes 32 64 128
```

The benchmark trains the fixed Mini network on the same dataset on CPU and CUDA
at every requested training batch size. It then runs identical seeded games on
the CPU fallback and CUDA paths while sweeping MCTS simulations, game workers,
inference batch sizes, and batch flush latency. Batch size one is the synchronous
CUDA control. Larger batches use the same shared-model path as generation runs.

Both paths run games in memory and exclude game-artifact I/O. Before timing,
each row starts its executor and model, runs enough warm-up games to exercise
every worker, and creates its inference batcher. The report separates this
setup/warm-up duration from the measured duration. Runtime projections add that
fixed cost once to the resulting steady-state game time, matching one large
generation rather than repeatedly charging startup for every small sample.

Every self-play row reports aggregate moves/s, games/hour, simulations/s,
per-game minimum/median/p95/maximum average move latency, inference queue and
batch statistics, peak CUDA allocation, and GPU utilization and memory samples
collected by `nvidia-smi` during the measured workload. Training rows report
examples/s and peak memory. The generated recommendation always selects the
highest measured throughput rather than encoding a hardware guess. Runtime
estimates for 1,000, 5,000, and 10,000 games are straight-line steady-state
capacity projections; longer runs can vary with thermals and machine contention.
The sampler resolves PyTorch's logical CUDA device to its physical GPU UUID, so
`CUDA_VISIBLE_DEVICES` remapping cannot redirect metrics to another GPU.

## Recorded result

[`benchmarks/mini-cuda-tuning.json`](../benchmarks/mini-cuda-tuning.json) records
the Issue 87 sweep on an RTX 4060 and its CPU fallback. The recommendation and
bottleneck section in that artifact are generated directly from its measurements.
GPU utilization below 70% is classified as a CPU game/MCTS submission bottleneck;
utilization at or above 70% is classified as GPU inference saturation. This
threshold and the observed average inference batch size make the classification
explicit and auditable. If `nvidia-smi` yields no valid utilization samples, the
bottleneck is reported as `unknown` rather than inferred without evidence.

The recorded RTX 4060 result recommends CUDA batch 128 for training: it measured
49,056 examples/s versus the best measured CPU rate of 34,884 examples/s. For
self-play at four simulations, the overall throughput default is one CPU worker
at 10,919 games/hour. Its projected runtimes are 0.092, 0.458, and 0.916 hours
for 1,000, 5,000, and 10,000 games.

When CUDA self-play is required, the best measured setting is four workers,
inference batch 8, and a 0.5 ms flush latency at 6,493 games/hour. Synchronous
CUDA measured 5,761 games/hour, so sharing improved end-to-end throughput by
12.7%, but the achieved average batch was only 2.86 and average GPU utilization
was 11.7%. The recommended CUDA setting therefore projects 0.154, 0.770, and
1.540 hours for 1,000, 5,000, and 10,000 games. CPU game/MCTS submission is the
explicit limiting resource; raising the configured batch above 4 did not fill
the larger batch and improved games/hour by only 1.2%.

Use the artifact's recommended training batch size, worker count, inference batch
size, flush latency, and simulation budget only for comparable RTX-class hosts.
Rerun the command after changing the model, board, CUDA stack, or hardware.
