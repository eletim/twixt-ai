# Matched Mini encoding strength comparison

Issue 76 compares the Issue 75 best 10-plane and 22-plane checkpoints under
matched Mini Twixt learned-MCTS conditions. The machine-readable report stores
the checkpoint hashes and metadata, complete game schedule, role splits,
confidence intervals, environment, and runtime in
[`experiments/issue-76/report.json`](../experiments/issue-76/report.json).

## Method

Each matchup contains 20 games arranged as ten seed-matched pairs. Every pair
uses the same seed twice and swaps red/first and black/second roles. The direct
encoding comparison and both comparisons against the same non-neural MCTS
baseline all restart the same seed schedule at `760100`. Every entrant receives
20 MCTS simulations per decision and a four-move rollout limit.

The checkpoints share their dataset, training schedule, trunk and head capacity;
their encoding and first convolution input width are the intended differences.
The experiment refuses mismatched inputs. Policy+value is the primary playing
strength comparison. Policy-only and value-only schedules are retained as
diagnostics because the earlier Mini strength experiment found unstable value
guidance.

Win-rate intervals are 95% Wilson score intervals with draws in the denominator.
Runtime is reported independently as complete two-agent wall time per matchup,
game, and move. It is machine- and game-length-dependent and is not used to
declare playing strength.

## Results

| Guidance | Matchup | Listed model W-L-D | Listed model win rate (95% CI) | Seconds/game |
| --- | --- | ---: | ---: | ---: |
| Policy+value | 10-plane vs 22-plane | 6-8-6 | 30% (14.5–51.9%) | 1.123 |
| Policy+value | 10-plane vs non-neural MCTS | 1-17-2 | 5% (0.9–23.6%) | 0.655 |
| Policy+value | 22-plane vs non-neural MCTS | 2-15-3 | 10% (2.8–30.1%) | 0.717 |
| Policy-only | 10-plane vs 22-plane | 6-12-2 | 30% (14.5–51.9%) | 1.111 |
| Policy-only | 10-plane vs non-neural MCTS | 12-7-1 | 60% (38.7–78.1%) | 0.708 |
| Policy-only | 22-plane vs non-neural MCTS | 11-7-2 | 55% (34.2–74.2%) | 0.773 |
| Value-only | 10-plane vs 22-plane | 5-11-4 | 25% (11.2–46.9%) | 1.095 |
| Value-only | 10-plane vs non-neural MCTS | 2-18-0 | 10% (2.8–30.1%) | 0.568 |
| Value-only | 22-plane vs non-neural MCTS | 3-14-3 | 15% (5.2–36.0%) | 0.778 |

The listed model is 10-plane except where only 22-plane faces the baseline.
The exact symmetric rates, role splits, games, and unrounded timings remain in
the report.

The primary policy+value result is negative for both encodings: neither model
approaches the equal-budget non-neural MCTS baseline. The direct comparison is
statistically inconclusive at this sample size, but the 10-plane model also has
the less favorable observed record. Policy-only guidance reverses the baseline
result for both models, confirming that their value heads remain harmful; in
the direct policy-only diagnostic, 10-plane loses 6-12 with two draws. The
value-only records are poor throughout.

These results do not establish that the compact encoding preserves playing
strength, so 10-plane is **not safe to adopt as the default Mini encoding** on
this evidence. More training data or corrected value targets should be tested
before revisiting adoption. This conclusion preserves the negative/inconclusive
outcome rather than selecting the faster representation on cost alone.

The complete run took 150.6 seconds. Per-game matchup time is reported above,
separately from strength. It includes both players and depends on game length,
so it cannot isolate encoding inference speed; the dedicated encoding cost
benchmark remains the source for that question.

## Reproduce

From an installed checkout:

```bash
PYTHONHASHSEED=0 twixt-ai-encoding-strength \
  --ten-plane-checkpoint experiments/issue-75/10-plane-v2/training/best.pt \
  --twenty-two-plane-checkpoint experiments/issue-75/22-plane-v1/training/best.pt \
  --output encoding-strength-report.json
```

For a source checkout without an installed entry point:

```bash
PYTHONHASHSEED=0 PYTHONPATH=src python3 -m twixt_ai.evaluation.encoding_strength_cli \
  --ten-plane-checkpoint experiments/issue-75/10-plane-v2/training/best.pt \
  --twenty-two-plane-checkpoint experiments/issue-75/22-plane-v1/training/best.pt \
  --output encoding-strength-report.json
```

The command refuses to overwrite an existing report, preventing recovery runs
from silently mixing measurements.
