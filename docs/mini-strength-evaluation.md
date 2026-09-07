# Learned Mini MCTS strength evaluation

Issue 58 evaluates the Issue 57 best checkpoint against Random, depth-one
Heuristic Search, and non-neural MCTS on the 10×10 board. The complete
machine-readable report, individual games, seed assignments, role splits,
confidence intervals, environment, checkpoint identity, and runtime are stored
in [`experiments/issue-58/report.json`](../experiments/issue-58/report.json).

## Method

Each matchup contains 20 games arranged as ten pairs. A pair uses the same seed
twice and swaps the red/first and black/second roles. Every matchup reuses seed
`580100` and therefore the same generated pair-seed schedule. Learned and
non-neural MCTS both receive exactly 20 simulations per decision and use the
same four-move rollout limit. Heuristic Search uses depth one and a 10,000-node
budget; Random is the seeded baseline.

The three learned entrants use the same epoch-2 `best.pt` checkpoint (SHA-256
`ce20f05c8a3d687fce4860595d40d710177f5901298c0bea95afb488cadcc3c8`):

- policy+value uses both network outputs;
- policy-only uses network priors and ordinary MCTS rollout values;
- value-only uses uniform priors and network leaf values.

Win-rate intervals are 95% Wilson score intervals over all games, with draws
remaining in the denominator. Because game length varies with playing style,
the report retains wall time, seconds/game, total moves, and seconds/move for
each matchup. These timings measure the complete two-agent workload and should
not be interpreted as isolated per-agent latency.

## Results

| Learned guidance | Baseline | W-L-D | Win rate (95% CI) | Seconds/game |
| --- | --- | ---: | ---: | ---: |
| Policy+value | Random | 11-1-8 | 55% (34.2–74.2%) | 0.598 |
| Policy+value | Heuristic Search | 0-20-0 | 0% (0.0–16.1%) | 0.173 |
| Policy+value | Non-neural MCTS | 1-14-5 | 5% (0.9–23.6%) | 0.899 |
| Policy-only | Random | 18-0-2 | 90% (69.9–97.2%) | 0.578 |
| Policy-only | Heuristic Search | 0-20-0 | 0% (0.0–16.1%) | 0.193 |
| Policy-only | Non-neural MCTS | 11-6-3 | 55% (34.2–74.2%) | 0.880 |
| Value-only | Random | 3-6-11 | 15% (5.2–36.0%) | 0.739 |
| Value-only | Heuristic Search | 0-20-0 | 0% (0.0–16.1%) | 0.163 |
| Value-only | Non-neural MCTS | 0-18-2 | 0% (0.0–16.1%) | 0.613 |

The combined learned model does not improve on either search baseline at this
budget. Policy guidance alone is promising—it records more wins than losses
against equal-budget non-neural MCTS—but its confidence interval remains wide
and it still loses every game to Heuristic Search. Value-only guidance is worse
than Random and non-neural MCTS here, and adding the value output reverses most
of the policy-only gain. This identifies value-target quality or calibration as
the first follow-up to investigate; it is not evidence for promoting the full
checkpoint.

The measured run took 96.7 seconds overall on the environment recorded in the
report. Runtime varies by machine and by game length, so strength comparisons
come from the fixed search budgets and paired games, not elapsed time.

## Reproduce

From an installed checkout:

```bash
PYTHONHASHSEED=0 twixt-ai-mini-strength-evaluation \
  --checkpoint experiments/issue-57/baseline/best.pt \
  --output mini-strength-report.json
```

For a source checkout without an installed entry point:

```bash
PYTHONHASHSEED=0 PYTHONPATH=src python3 -m twixt_ai.evaluation.mini_strength_cli \
  --checkpoint experiments/issue-57/baseline/best.pt \
  --output mini-strength-report.json
```

The command refuses to overwrite an existing output file so recovery runs
cannot silently replace measurements.
