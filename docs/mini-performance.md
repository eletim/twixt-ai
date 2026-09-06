# Mini Twixt performance

The NN-free 10×10 baseline is reproduced with:

```bash
PYTHONHASHSEED=0 twixt-ai-selfplay-benchmark \
  --output benchmarks/mini-selfplay-performance.json
```

The command uses seed `2401`, a deterministic ply-8 fixture, the MCTS default
four-move rollout horizon, and 100, 400, and 1,600 simulations per decision.
It records the median of three fixed-position searches, one complete game at
each budget, and four identical seeded games at both one and two workers.
Machine, Python, CPU availability, workload settings, raw rates, latencies, CPU
time, and utilization are retained in the versioned JSON artifact.

CPU `aggregate_percent` is process CPU time divided by wall time and can exceed
100% for multi-process runs. `available_cpu_percent` normalizes that value by
the logical CPUs available to the process. The scaling comparison runs the
same game seeds at every worker count; its speedup therefore reflects parallel
execution rather than different games.

Full games use the normal headless match orchestration and retain the same
decision metadata as self-play generation. The benchmark returns compact
per-game counters between workers instead of writing each match artifact, so
the scaling measurement isolates generation throughput from filesystem speed.

The 1,000- and 10,000-game estimates are straight-line projections from each
measured single-worker full-game rate. They are capacity estimates, not elapsed
time guarantees: thermal limits, co-tenancy, filesystem output, and trained
network inference can change real experiment time. Neural inference is not
mixed into this artifact, and `policy_value_network: false` makes the NN-free
scope explicit.

## Recorded baseline

[`benchmarks/mini-selfplay-performance.json`](../benchmarks/mini-selfplay-performance.json)
is the baseline captured for Issue 52. Its `identified_bottlenecks` entries are
generated from that run: the slowest engine micro-operation, the largest
measured MCTS move latency, and the worker count with the highest measured
games/hour all cite the values that selected them. This keeps performance
conclusions tied to measurements on the recorded machine.

## Issue 53 optimization

The comparable post-optimization run is checked in as
[`benchmarks/mini-selfplay-performance-optimized.json`](../benchmarks/mini-selfplay-performance-optimized.json).
Both artifacts use the same workload, seed, Python version, platform, and CPU
availability. The optimization caches bounded, immutable placement templates
by board and player; stores an immutable occupancy set on each canonical state;
uses a private trusted construction path for states derived from an already
validated transition; and reuses the root request's legal moves in MCTS.
Public state construction remains fully validated, and canonical ordering,
serialization, equality, move ordering, and seeded choices are unchanged.

| Measurement | Issue 52 baseline | Optimized | Change |
| --- | ---: | ---: | ---: |
| 100-simulation move latency | 0.0735 s | 0.0443 s | 39.8% lower |
| 400-simulation move latency | 0.2900 s | 0.1668 s | 42.5% lower |
| 1,600-simulation move latency | 1.1901 s | 0.6832 s | 42.6% lower |
| 100-simulation full games/hour | 2,772.6 | 5,063.7 | 1.83× |
| 400-simulation full games/hour | 989.2 | 1,860.7 | 1.88× |
| 1,600-simulation full games/hour | 260.4 | 485.1 | 1.86× |
| Two-worker games/hour | 3,999.6 | 7,095.6 | 1.77× |

A deterministic 400-simulation `cProfile` run attributed 0.342 seconds to
legal-move generation and made 216,948 coordinate constructions before the
change. Afterwards, legal-move generation took 0.058 seconds and total profile
calls fell from 4.10 million to 2.10 million. The profile showed that link and
win recomputation were not leading costs; the optimized run's engine
microbenchmarks nevertheless reduced automatic-link latency by 56.3% and full
move-application latency by 42.2%. Heuristic cutoff evaluation is now the
largest measured MCTS cost and is intentionally left unchanged because it
defines search behavior.

The standard 24×24 engine benchmark remains separate and unchanged:

```bash
PYTHONHASHSEED=0 twixt-ai-engine-benchmark --output engine-benchmark.json
```
