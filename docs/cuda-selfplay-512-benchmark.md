# Fixed 512-game CUDA self-play benchmark

Issue 98 optimizes end-to-end wall-clock time for one fixed, reproducible
512-game Mini Twixt neural self-play workload without weakening it. The
workload — board/rules, checkpoint identity, encoding version, MCTS
simulation budget and search parameters, seeds, worker/batching settings, and
required outputs — is locked in
[`benchmarks/mini-cuda-selfplay-512-contract.json`](../benchmarks/mini-cuda-selfplay-512-contract.json).
Every baseline, profiling, and optimized measurement must run that exact
contract; only the self-play *implementation* may change between runs.

## Running the benchmark

```bash
PYTHONHASHSEED=0 python -m twixt_ai.evaluation.cuda_selfplay_512_cli \
  --contract benchmarks/mini-cuda-selfplay-512-contract.json \
  --output-dir /path/to/scratch/selfplay-out \
  --report /path/to/report.json \
  --implementation-label "pre-optimization"
```

`twixt_ai.evaluation.cuda_selfplay_512.run_cuda_selfplay_512_benchmark` is the
underlying implementation. It:

- loads the contract file unmodified and verifies the checkpoint's SHA-256
  and `PYTHONHASHSEED` against it before running anything;
- builds the shared-model inference path with the contract's exact
  `NeuralInferenceBatcher`/`MCTSAgent` settings and runs `selfplay.batch.run_batch`
  with the contract's worker count, seed, and board — the same primitives
  `selfplay.large_experiment` uses for staged dataset generation;
- measures end-to-end wall time over the same scope the contract defines
  (dispatch through durable artifact/summary writes; checkpoint load and
  sampler setup are excluded and reported separately);
- validates every required artifact: exactly the contract's game count
  completed with zero failures, every game's derived seed matches
  `Random(batch_seed).getrandbits(64)` in index order, and every recorded
  decision's root-move visit counts sum to the contract's simulation budget
  (so policy targets are a valid probability distribution);
- reports GPU utilization/memory (`nvidia-smi` sampling, reusing
  `cuda_tuning._GpuSampler`), effective inference batch-size distribution
  (`NeuralInferenceBatcher.statistics`), and an approximate phase breakdown.

### Phase breakdown method

The phase breakdown is a lightweight stack-sampling profiler, not
instrumentation added to search or self-play code: every 5 ms it inspects
every live thread's current frames and attributes that sample to the
highest-priority phase whose code is on any stack — in priority order,
serialization/I/O, GPU inference (`NeuralPolicyValue.evaluate_batch`),
CPU/MCTS (`search/mcts.py`), then batching/queueing wait; a sample matching
none of those is `idle`. It adds no locking or synchronization to the timed
path and can be read as an approximate, not exact, wall-time split.

## Recorded baseline

[`benchmarks/mini-cuda-selfplay-512-baseline.json`](../benchmarks/mini-cuda-selfplay-512-baseline.json)
is the pre-optimization measurement on an RTX 4060: 265.357 s wall time for
512 games (6,946.1 games/hour), 575.35 positions/s, 466.22 simulations/s,
11.20% average GPU utilization, and 473 MiB peak GPU memory. Its phase
breakdown attributes 82.3% of wall time to `gpu_inference` even though
average GPU utilization is only 11.2%: the inference batcher thread spends
almost all of that time inside `NeuralPolicyValue.evaluate_batch`, not
blocked on a saturated GPU kernel, while CPU/MCTS work is a negligible 0.01%
of sampled wall time. Issue #98's optimization work investigates and reports
on this gap between "time attributed to inference" and "GPU compute time"
rather than assuming the GPU itself is the bottleneck; see the follow-up
optimization report for the identified cause and any resulting change.

Do not regenerate this baseline file to reflect a new implementation; it is
the fixed pre-optimization reference every later measurement is compared
against.
