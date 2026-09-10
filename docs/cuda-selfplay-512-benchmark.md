# v0.0.6 512-game CUDA self-play performance contract

Issue 98 optimizes end-to-end wall-clock time for one fixed, reproducible
512-game Mini Twixt neural self-play workload without weakening it. The
workload — board/rules, checkpoint identity, encoding version, MCTS
simulation budget and every other search-quality parameter, seeds, CUDA model
path, and required outputs — is locked in
[`benchmarks/mini-cuda-selfplay-512-v006-contract.json`](../benchmarks/mini-cuda-selfplay-512-v006-contract.json).

Exactly three settings are optimization variables because they affect how
fixed requests are scheduled, not what any game or training target means:
`worker_concurrency`, `inference_batch_size`, and
`queue_flush_max_wait_seconds`. No other contract field may vary between
comparable v0.0.6 runs. `--contract` may select another path for portability,
but the runner compares its full JSON content with the committed canonical
contract and rejects any fixed-field or declaration change. The older
`mini-cuda-selfplay-512-contract.json` remains the immutable v0.0.5 contract
referenced by the recorded baseline and optimized result files below.

## Running the benchmark

```bash
PYTHONHASHSEED=0 python -m twixt_ai.evaluation.cuda_selfplay_512_cli \
  --output-dir /path/to/scratch/selfplay-out \
  --report /path/to/report.json \
  --implementation-label "v0.0.6-default"
```

The CLI defaults to the v0.0.6 contract. To compare scheduling choices, add
any combination of `--worker-concurrency`, `--inference-batch-size`, and
`--queue-flush-max-wait-seconds`. Those are the only workload overrides the
runner accepts; sampler intervals and the implementation label affect only
measurement/report metadata.

`twixt_ai.evaluation.cuda_selfplay_512.run_cuda_selfplay_512_benchmark` is the
underlying implementation. It:

- requires every field in the supplied v2 contract to equal the committed
  canonical contract, verifies the checkpoint's SHA-256 and
  `PYTHONHASHSEED`, and resolves only its three declared optimization variables
  from explicit runner inputs before running anything;
- builds the shared-model inference path with the contract's exact
  `NeuralInferenceBatcher`/`MCTSAgent` settings and runs `selfplay.batch.run_batch`
  with the contract's worker count, seed, and board — the same primitives
  `selfplay.large_experiment` uses for staged dataset generation;
- measures end-to-end wall time over the same scope the contract defines
  (dispatch through artifact/summary writes; checkpoint load and
  sampler setup are excluded and reported separately);
- validates every required artifact and its semantics: exact paths/counts,
  summary and match schema versions, resolved batch and per-game
  configurations, replay-valid terminal records, game and per-decision seed
  derivations, recorded decisions matching the replay, unchanged simulation
  and rollout budgets, a complete legal root-move set whose visits sum to the
  fixed budget, normalized policy targets, and terminal side-to-move value
  targets in `{-1, 0, 1}`. Match/replay/decision and policy/value validation
  uses `training.data.training_examples_from_match`, the same conversion used
  to build training datasets, rather than defining benchmark-local targets;
- reports GPU utilization/memory (`nvidia-smi` sampling, reusing
  `cuda_tuning._GpuSampler`), effective inference batch-size distribution
  (`NeuralInferenceBatcher.statistics`), throughput rates, an approximate
  phase breakdown, the complete fixed configuration, and resolved values for
  all three optimization variables.

### Phase breakdown method

The phase breakdown is a lightweight stack-sampling profiler, not
instrumentation added to search or self-play code: every 5 ms it inspects
every live thread's current frames and attributes that sample to the
highest-priority phase whose code is on any stack — in priority order,
serialization/I/O, GPU inference (`NeuralPolicyValue.evaluate_batch`),
CPU/MCTS (`search/mcts.py`), then batching/queueing wait; a sample matching
none of those is `idle`. It adds no locking or synchronization to the timed
path and can be read as an approximate, not exact, wall-time split.

## Historical v0.0.5 recorded baseline

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

## Optimization: batch tensors on the CPU, not the CUDA device

`NeuralPolicyValue.evaluate_batch` (`src/twixt_ai/search/neural.py`) built
every position's encoding and legal-move mask by passing `device=<cuda>`
into `encode_position_for_version`/`legal_move_mask_for_version`, which
populate those tensors with a Python loop of individual element writes (one
per peg, link, and legal move). On a CUDA tensor, each write is its own
host/device kernel launch. Extracting results then called `.item()` on the
softmax output once per legal move per position — another synchronization
each. With up to roughly 100 legal moves on the 10x10 Mini board and 152,674
total inference requests in the fixed benchmark, this was thousands of tiny
synchronous CUDA calls per shared batch — enough to dominate wall time while
using very little actual GPU compute, which is why GPU utilization measured
so low even though the phase breakdown attributed most of the wall time to
"GPU inference." CPU/MCTS game-tree work was not the bottleneck (0.01% of
sampled baseline wall time).

The fix builds those tensors on the CPU (the encoding/masking functions'
existing default) and moves each stacked batch to the model's device in one
transfer each way, then extracts the whole probabilities/values batch to
Python once with `.tolist()` instead of per-element `.item()`. A follow-up
fix in the same change removed a second, smaller redundancy: each legal
move's action index was computed once to build the mask and recomputed again
on extraction; it is now computed once per move and reused for both. No
model, encoding, search, worker-count, or batch-size setting changed.

[`benchmarks/mini-cuda-selfplay-512-optimized.json`](../benchmarks/mini-cuda-selfplay-512-optimized.json)
records the result of re-running the exact same committed contract after
these changes: 116.031 s wall time for 512 games — a **2.29x** speedup over
the 265.357 s baseline — at 15,885.4 games/hour, 1,315.8 positions/s, and
1,066.2 simulations/s. `output_summary_sha256` for this run is bit-identical
to the recorded baseline's, confirming every game's winner, move count, and
derived seed reproduce exactly; the per-decision policy training target
(root-move visit counts, validated to sum to the contract's 4-simulation
budget) also matched in every one of the 30,929 recorded decisions.

The phase breakdown shifted accordingly: `gpu_inference` dropped from 82.3%
of baseline wall time to 68.2% of (a much smaller) total, while `cpu_mcts`
rose from 0.01% (baseline) to 29.4% — its *absolute* time (~29-42 s across
these runs) did not grow with the optimization, so the shift is proportional
to the smaller total, not a regression. GPU utilization fell further, to
3.4% average, because each shared batch now completes so much faster that
`nvidia-smi`'s 100 ms sampling interval rarely catches a GPU active moment;
this is expected, not evidence of a new inefficiency, given the model's
total compute cost for this workload.

With the CUDA-call-overhead bottleneck resolved, GPU inference (now mostly
genuine batched model calls, still capped at an effective batch size of 4 by
the contract's 4-worker concurrency) is still the largest remaining cost,
with CPU-side MCTS/game-tree Python work next. Increasing effective GPU
batch size further would require more concurrent self-play workers than the
committed contract's 4, which this work deliberately left unchanged rather
than reinterpret the frozen contract; CPU-side MCTS hot-path optimization is
a separate, larger body of work touching code shared by every self-play use
of this engine, not just this benchmark, and was left for future work rather
than attempted without dedicated profiling. A related but out-of-scope
finding from code review: `twixt_ai.training.value_diagnostics` builds its
per-position batches directly on the CUDA device the same way
`evaluate_batch` used to, which would reproduce this same overhead pattern
when run on a CUDA-capable machine; it is a separate training-diagnostics
tool outside this self-play benchmark's scope and was left for a follow-up
issue rather than fixed here.
