# Architecture-v2 widened-head outcome

The widened architecture-v2 policy/value heads did **not** materially improve
value quality or playing strength over the architecture-v1 Mini lineage in
this run. The first v2 candidate is not promoted, and the evidence does not
justify another generation from it. The machine-readable conclusion and its
source-artifact references are retained in
[`final-report.json`](final-report.json).

## Measured evidence

The v2 strength evaluation used the same seed-`1289000`, 40-game paired
role-swap schedule, fixed opponents, and `standard-20` search setting as the
recent architecture-v1 Issue 128 evaluations. Results below are candidate
wins-draws-losses.

| Candidate and guidance | Matched non-neural MCTS | Heuristic search |
| --- | ---: | ---: |
| Architecture-v2, policy only | 13-8-19 | 0-0-40 |
| Architecture-v2, value only | 5-6-29 | 0-0-40 |
| Architecture-v2, policy + value | **4-4-32** | **0-0-40** |
| Architecture-v1 matched-1k, policy + value | 32-4-4 | 6-0-34 |
| Architecture-v1 5k, policy + value | 34-3-3 | 14-0-26 |

The v2 policy+value mode won only 10.0% of its matched-MCTS games (95% Wilson
interval 4.0%-23.1%), versus 80.0% (65.2%-89.5%) for architecture-v1
matched-1k and 85.0% (70.9%-92.9%) for architecture-v1 5k. It won no
heuristic games (0%-8.8%), while the v1 candidates won 15.0% and 35.0%.
Even v2 policy-only, its strongest ablation, remained below the matched MCTS
at 13-8-19; adding its value head reduced that result to 4-4-32. These are
playing results, not conclusions drawn from training loss.

The independent value audit also found a large generalization failure. On
4,371 training positions the v2 checkpoint had MSE 0.1655 and MAE 0.2569; on
331 held-out positions from nine source games it had MSE 0.8039 and MAE
0.7125, a validation-minus-training MSE gap of 0.6383. Aggregate validation
calibration error was only +0.0039, but cancellation hid errors of -0.4146 and
+0.2924 in the two most-confident signed bins; 19 of their 82 predictions had
the wrong outcome sign. This independently measured behavior agrees with the
value-only and policy+value playing regressions.

For context, the architecture-v1 generation-2 audit reported validation MSE
0.4239 on 3,626 positions. That value is not treated as a controlled
head-width comparison because it came from a different checkpoint and probe
dataset. Likewise, the v1 candidates used 1,000- and 5,000-game datasets while
this first fresh v2 lineage used 100 games. The matched baseline games are the
decisive evidence that the realized widened-head pipeline did not improve
strength; the available data do not isolate head width as the sole cause.

## Decision and next bottleneck

Retain this as a negative result. Do not promote the v2 candidate or continue
self-play from it on the strength of its lower training loss. The clearest
next bottleneck is the quality and coverage of the learning signal produced by
fresh self-play from a randomly initialized bootstrap, especially value-target
generalization. The 100-game dataset supplied only 331 validation positions,
no validation draws, and just nine held-out games; the much wider heads then
fit training values while held-out value error stayed high. A next experiment
should improve or enlarge the bootstrap teacher/search data under a frozen
comparison contract and require held-out value diagnostics plus paired
strength gates before attributing any change to head capacity.

## Source artifacts and verification

- V2 value diagnostics: [`v2-value-quality/value-head.json`](v2-value-quality/value-head.json)
- V2 paired strength games: [`v2-strength/evaluation.json`](v2-strength/evaluation.json)
- Architecture-v1 matched evidence: [`../issue-128/matched-1k/report.json`](../issue-128/matched-1k/report.json)
  and [`../issue-128/5k/report.json`](../issue-128/5k/report.json)
- Architecture-v1 value context:
  [`../issue-125/diagnostics/generation-2-value-head.json`](../issue-125/diagnostics/generation-2-value-head.json)

The complete project test suite (498 tests), Python byte-compilation check,
JSON parsing, and a regression cross-check of the cited artifact values pass
on the final report commit.
