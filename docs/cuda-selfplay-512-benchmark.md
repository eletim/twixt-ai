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
comparable v0.0.6 runs. Version 1 contracts are historical artifacts and are
not executable through this v0.0.6 runner. `--contract` may select another
path for portability, but the runner compares its full JSON content with the
committed canonical contract and rejects any fixed-field or declaration
change. The older
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
  `NeuralInferenceBatcher` settings and explicitly binds every `MCTSAgent`
  search parameter: simulations, exploration, rollout limit, rollout
  evaluator, and both progressive-widening values. It runs
  `selfplay.batch.run_batch` with the contract's worker count, seed, and board
  — the same primitives `selfplay.large_experiment` uses for staged dataset
  generation;
- measures end-to-end wall time over the same scope the contract defines
  (dispatch through artifact/summary writes; checkpoint load and
  sampler setup are excluded and reported separately);
- validates every required artifact and its semantics: exact paths/counts,
  summary and match schema versions, resolved batch and per-game
  configurations, replay-valid terminal records, game and per-decision seed
  derivations, recorded decisions matching the replay, unchanged simulation
  budget, exploration, rollout limit/evaluator, and progressive-widening
  values recorded by every decision, a complete legal root-move set whose
  visits sum to the fixed budget, normalized policy targets, and terminal
  side-to-move value targets in `{-1, 0, 1}`. Match/replay/decision and
  policy/value validation
  uses `selfplay.trajectory.trajectory_from_match`, the shared persisted-match
  and trajectory-target boundary also used to build training datasets, rather
  than defining benchmark-local targets or depending on the training layer;
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

## Concurrency, batch, and queue scaling on the RTX 4060

The v0.0.5 optimized configuration was reproduced on 2026-09-11 from the
v0.0.6 contract runner before varying any scheduling input. The reproduction
used 4 workers, batch size 8, and a 0.5 ms maximum queue wait and completed in
119.423 s, within 2.93% of the recorded 116.031 s result. It produced the same
summary SHA-256 as the recorded run and passed every artifact, replay, seed,
search-parameter, policy-target, and value-target check.

Each sweep row below is one complete 512-game run. Only the three declared
optimization variables changed; the checkpoint, seeds, board/rules, MCTS
parameters, thread worker mode, single shared CUDA model, required artifacts,
and target semantics remained fixed. No multiprocessing or implementation
optimization was introduced. `GPU idle` is the approximate complement of
average `nvidia-smi` utilization over the timed scope; it is different from
the stack sampler's `idle` phase because CPU or inference-host work can run
while the GPU is inactive. The complete machine-readable results, including
all ineffective and regressive trials, are in
[`benchmarks/mini-cuda-selfplay-512-concurrency-scaling.json`](../benchmarks/mini-cuda-selfplay-512-concurrency-scaling.json).

| Trial | Workers | Batch | Flush wait | Wall (s) | Games/hour | Effective batch | GPU util. | GPU idle | CPU/MCTS phase | Inference phase | Result vs reproduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reproduction | 4 | 8 | 0.5 ms | 119.423 | 15,434 | 3.87 | 2.96% | 97.04% | 29.80% | 67.86% | reference |
| Worker floor | 1 | 8 | 0.5 ms | 252.974 | 7,286 | 1.00 | 4.74% | 95.26% | 51.08% | 45.91% | 111.83% slower |
| Worker scale-up | 8 | 8 | 0.5 ms | 110.469 | 16,685 | 7.32 | 1.99% | 98.01% | 16.06% | 81.17% | 7.50% faster |
| Worker saturation | 16 | 8 | 0.5 ms | 110.700 | 16,650 | 7.94 | 1.93% | 98.07% | 1.63% | 95.46% | 7.30% faster |
| No batching | 8 | 1 | 0.5 ms | 178.516 | 10,325 | 1.00 | 6.42% | 93.58% | 1.16% | 97.32% | 49.48% slower |
| Smaller batch | 8 | 4 | 0.5 ms | 122.311 | 15,070 | 3.98 | 2.86% | 97.14% | 1.46% | 96.08% | 2.42% slower |
| Unreachable batch cap | 8 | 16 | 0.5 ms | 110.353 | 16,703 | 7.31 | 2.00% | 98.00% | 20.08% | 77.18% | 7.59% faster |
| Immediate flush | 8 | 8 | 0 ms | 121.342 | 15,190 | 3.97 | 2.90% | 97.10% | 1.00% | 96.55% | 1.61% slower |
| Longer coalescing | 8 | 8 | 2 ms | **108.075** | **17,055** | **7.83** | 1.95% | 98.05% | 22.23% | 75.01% | **9.50% faster** |

Worker scaling is substantial through eight threads, but saturates there:
sixteen workers made batches 99.2% full yet was 0.21% slower than eight
workers. At eight workers, batch size 1 was 61.60% slower than batch size 8,
and batch size 4 was 10.72% slower. Raising the cap to 16 was ineffective
because eight producers cannot make a batch larger than eight. Immediate
flush was also regressive: it reduced the realized batch from 7.83 to 3.97
and was 12.28% slower than the 2 ms endpoint. The 2 ms result was 2.22% faster
than the matching 0.5 ms trial, but this single-run profiling sweep is not
enough evidence to change the canonical default.

The best measured configuration completed in 108.075 s (17,054.8
games/hour), a 1.074x speedup over the recorded 116.031 s configuration and
1.105x over this run's reproduction. It still averaged only 1.95% sampled GPU
utilization (approximately 98.05% GPU idle) while realizing a 7.83-position
batch. The RTX 4060 is therefore not compute-saturated. The remaining primary
bottleneck is the host/launch side of batched inference—CPU encoding and
result extraction around small CUDA operations—with CPU game/MCTS work the
secondary bottleneck (22.23% of best-run wall-state samples). Implementation
changes for either path are intentionally outside this measurement task.

To reproduce any row, start with the documented benchmark command and pass
the row's values, for example:

```bash
PYTHONHASHSEED=0 python -m twixt_ai.evaluation.cuda_selfplay_512_cli \
  --output-dir /path/to/scratch/selfplay-out \
  --report /path/to/report.json \
  --implementation-label "v0.0.6-profile-w8-b8-f2ms" \
  --worker-concurrency 8 \
  --inference-batch-size 8 \
  --queue-flush-max-wait-seconds 0.002
```

## Detailed inference-host profile

The best measured scheduling configuration above was profiled again on the
same RTX 4060 with all 512 canonical games: eight workers, batch size eight,
and a 2 ms maximum flush wait. The run used the unchanged canonical contract,
completed all games, validated all 30,929 decisions and policy/value targets,
and reproduced output summary SHA-256
`f6dc7621b70a017cff91bf00825de0bd6e7f4483ca2d984a48201d4f07bdc52f`.
The full ranked evidence is in
[`benchmarks/mini-cuda-selfplay-512-inference-host-profile.json`](../benchmarks/mini-cuda-selfplay-512-inference-host-profile.json).

The opt-in `--detailed-inference-profile` observers measure the source-of-truth
`NeuralPolicyValue.evaluate_batch` and `NeuralInferenceBatcher` directly;
there is no benchmark evaluator or copied inference pipeline. Host phases use
`perf_counter`; CUDA transfers, model execution, mask application, and softmax
use CUDA events. At the point where production `Tensor.tolist()` must
synchronize and copy results, the observer explicitly separates that boundary
into synchronization, device-to-host copy, then CPU conversion. Its 112.690 s
end-to-end time includes event, observer, and clock overhead and is diagnostic,
not a new optimized throughput claim.

| Cost | Total (s) | Per batch (ms) | Share of profiled inference wall |
| --- | ---: | ---: | ---: |
| CPU encoding + stack | **38.230** | **1.968** | **63.73%** |
| CPU action-index construction | 4.935 | 0.254 | 8.23% |
| CPU legal-mask construction | 1.535 | 0.079 | 2.56% |
| Python result extraction | 1.742 | 0.090 | 2.90% |
| Explicit CUDA synchronization wait | 0.066 | 0.003 | 0.11% |

CUDA-event timings are a separate, overlapping view and must not be added to
the host wall spans. Across 19,425 inference batches, model execution used
6.315 device seconds (0.325 ms/batch), H2D copies 1.336 s, policy mask
application 0.631 s, D2H copies 0.568 s, and softmax 0.245 s. CPU encoding
alone therefore consumed 6.05 times the aggregate CUDA model-execution time.
The earlier inference label was too broad: it included this 38.230 s of host
encoding and other submission/extraction work around only 6.315 s of model
device work.

Batching and contention are measured independently. The average batch
contained 7.860 positions (98.25% of capacity), and 18,610 of 19,425 batches
were full. Batch formation from the first observed request to dispatch took
1.301 ms on average; full batches took 0.955 ms on average. The 815
latency-flushed batches took 9.210 ms on average and 6.691 ms at p50, showing
that the configured 2 ms timeout was not a hard observed dispatch bound.

Direct `Condition` measurements explain what can and cannot be called
contention:

| Condition measurement | p50 | p95 | p99 | Aggregate |
| --- | ---: | ---: | ---: | ---: |
| Producer lock acquisition | 0.551 us | 1.092 us | 1.763 us | 0.494 s across 152,674 calls |
| Producer lock-held section | 1.002 us | 3.156 us | 4.558 us | 0.220 s across 152,674 calls |
| Worker dispatch acquisition | 0.241 us | 0.340 us | 0.451 us | 0.005 s across 19,425 batches |
| Worker completion acquisition | 0.501 us | 0.852 us | 1.382 us | 0.011 s across 19,425 batches |
| Wait deadline overshoot | 0 ms | 0.526 ms | 6.978 ms | 7.299 s across 19,425 batches |

Typical lock acquisition and critical sections are therefore microsecond-scale,
although one producer acquisition reached 156 ms. Separately, wait calls
overshot their remaining timeout by 7.299 s in aggregate. That tail includes
delayed GIL/OS scheduling and `Condition` lock reacquisition; this profiler
cannot assign it to only one of those causes. The earlier interpretation of
queue wait as primarily coalescing was unsupported and is withdrawn. Aggregate
per-request queue wait likewise sums concurrent waits and is not additive wall
time.

The ranked next target remains CPU position encoding and stacking. The
condition wake/reacquisition scheduling tail is second, ahead of model
execution and action-index/mask construction. Result extraction and explicit
synchronization are smaller. `nvidia-smi` averaged 2.23%
utilization during this run, but that coarse sample is only corroboration: the
source-of-truth observer spans, direct Condition timings, CUDA events,
near-full batches, and validated fixed workload support the ranking. No
production optimization was attempted in this work item.

Reproduce the diagnostic profile with:

```bash
PYTHONHASHSEED=0 PYTHONPATH=src python3 -m \
  twixt_ai.evaluation.cuda_selfplay_512_cli \
  --output-dir /path/to/scratch/selfplay-profile-out \
  --report /path/to/scratch/inference-profile-report.json \
  --implementation-label "v0.0.6-profile-w8-b8-f2ms-source-observer" \
  --worker-concurrency 8 \
  --inference-batch-size 8 \
  --queue-flush-max-wait-seconds 0.002 \
  --detailed-inference-profile
```

## Batched version-1 position encoding

The next optimization replaces per-position version-1 tensor allocation plus
`torch.stack` with one batch allocation and bulk indexed writes for pegs and
links. The public single-position encoder and encoding version remain
unchanged. Version 2 continues to use its prior path.

Both unprofiled measurements used the exact 512-game contract with eight
workers, inference batch eight, and a 2 ms maximum flush wait:

| Measurement | Baseline | Batched encoder | Change |
| --- | ---: | ---: | ---: |
| End-to-end wall time | 108.409 s | **83.773 s** | **-22.72%** |
| Games/hour | 17,002.3 | **22,002.2** | **+29.41%** |
| Positions/second | 1,408.3 | **1,822.5** | **+29.41%** |
| Simulations/second | 1,141.2 | **1,476.8** | **+29.41%** |
| Inference positions/second | 2,595.1 | **4,365.6** | **+68.22%** |

The detailed source observer measured CPU encoding and stacking at 13.924 s,
or 0.716 ms per batch, down 63.58% from the earlier 38.230 s and 1.968 ms per
batch. Observer overhead is diagnostic and is not used for the throughput
claim.

Both throughput runs completed all 512 games and validated all 30,929
decisions. Their output summary SHA-256 values are byte-identical at
`f6dc7621b70a017cff91bf00825de0bd6e7f4483ca2d984a48201d4f07bdc52f`.
Differential regression tests also compare the complete batch byte buffer
against the single-position version-1 encoder across deterministic legal
trajectories on tiny, rectangular, Mini, and standard board dimensions.

An alternative that replaced `stack` with `unsqueeze` plus `torch.cat` was
rejected. It retained every per-position allocation and measured 1,316.9 us
per representative batch, 0.57% slower than the 1,309.4 us legacy median. The
retained bulk encoder measured 200.4 us (6.53x faster) in the same nine-sample
microbenchmark. Full machine-readable evidence is in
[`benchmarks/mini-batched-position-encoding.json`](../benchmarks/mini-batched-position-encoding.json).
