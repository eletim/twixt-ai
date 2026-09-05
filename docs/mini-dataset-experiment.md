# First Mini Twixt self-play dataset

Issue 56 generated the first nontrivial NN-free 10×10 MCTS training dataset.
The checked-in run is under [`experiments/issue-56`](../experiments/issue-56),
including source matches, training shards, manifests, and the measured
[`report.json`](../experiments/issue-56/report.json).

## Configuration choice

The optimized Issue 53 benchmark measured 100, 400, and 1,600 simulations per
move. The 100-simulation budget had the highest full-game throughput while
remaining a genuine MCTS search whose root visit counts produce policy targets.
The same benchmark measured its best throughput with two process workers. The
experiment therefore used 100 simulations, the default four-move rollout
horizon, two workers, and no policy/value network.

Both players used MCTS on the named `mini` 10×10 board. The two-game smoke seed
was `560001`; the 100-game baseline seed was `560100`. Dataset splitting used a
0.1 validation fraction, 10,000-example shards, and stage-specific split seeds
recorded in the report and manifests. `PYTHONHASHSEED=0` is required.

## Results

| Stage | Games | Failures | Positions | Wall time | Games/hour |
| --- | ---: | ---: | ---: | ---: | ---: |
| Smoke | 2 | 0 | 33 | 0.740 s | 9,728.9 |
| Baseline | 100 | 0 | 2,085 | 46.351 s | 7,766.9 |

The baseline contains 1,961 training examples and 124 validation examples.
Every example retains the final outcome as its side-to-move value target and a
normalized sparse policy target derived from the corresponding MCTS root visit
counts. Dataset construction replayed and validated all source matches before
writing the shards. The manifests include SHA-256 digests for integrity checks.

## Reproduce

From an installed checkout, generate both stages and their report into a fresh
directory:

```bash
PYTHONHASHSEED=0 twixt-ai-mini-dataset-experiment \
  --output-dir mini-dataset-experiment
```

For a source checkout without an installed entry point, use:

```bash
PYTHONHASHSEED=0 PYTHONPATH=src python3 -m twixt_ai.selfplay.experiment_cli \
  --output-dir mini-dataset-experiment
```

The command refuses a nonempty output directory so an earlier experiment cannot
be silently mixed with or overwritten by a recovery run. Game and dataset
content is deterministic for this configuration; runtime measurements naturally
depend on the machine and concurrent load.
