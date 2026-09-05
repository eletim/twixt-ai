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

The standard 24×24 engine benchmark remains separate and unchanged:

```bash
PYTHONHASHSEED=0 twixt-ai-engine-benchmark --output engine-benchmark.json
```
