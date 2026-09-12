# Issue 128 Mini strength-scaling result

Issue 128 stops after the two required matched stages. Increasing the fresh
fixed-teacher dataset from 1,000 to 5,000 games did not clear the preregistered
meaningful-scaling gate: the 5k candidate scored **19-21-0** against the 1k
candidate, or 47.5% wins (95% Wilson interval 32.9%-62.5%), below the required
22 wins in 40 games. This result is retained as negative saturation evidence.
The optional 10k and 25k stages were therefore not run, and 50k was neither
eligible nor run.

The immutable protocol is in
[`experiments/issue-128/scaling-contract.json`](../experiments/issue-128/scaling-contract.json).
The complete stage reports, including every game, shard identity, metric, and
retained-object digest, are the matched-1k
[`report.json`](../experiments/issue-128/matched-1k/report.json) and 5k
[`report.json`](../experiments/issue-128/5k/report.json). The tables below are
a compact synthesis of those machine-readable sources.

## Fixed lineage and protocol

Both stages independently used the Issue 125 generation-3 champion as the
self-play teacher and training initialization:

- path: `experiments/issue-125/generation-3/generation-0001/candidate/best.pt`
- SHA-256: `aee1036dbda115eeec0e245909d30e1f8330454a82099e852e1b6a9c26c0dab9`
- model: encoding v1, 22 inputs, 8 trunk channels, one residual block, and 16
  value-hidden units on the 10x10 Mini board

The 1k candidate was only an evaluation opponent for 5k; it was not the 5k
teacher or initialization. Both stages used the same 64-simulation self-play
search, optimizer and 20-epoch schedule, and 40-game evaluations arranged as
20 identical-seed role-swapped pairs. The only primary-curve differences were
game count, the declared root seed (1281000 or 1285000), and artifact location.
All evaluations used seed 1289000. This preserves a data-volume comparison
rather than a changing-teacher training generation.

## Data, checkpoints, and measured cost

| Stage | Games / examples | Dataset manifest SHA-256 | Candidate SHA-256 | Self-play | Dataset build | Training wrapper (CUDA loop) |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| matched-1k | 1,000 / 25,535 | `9d79b587e041f044084d8f14933b98baca31b9fbc0c799b358322e6160d5d6f7` | `5320f9ca4c146b1c925054ac4bd1e5e605d5f228c2dfb1d8c2a8a490d0981af2` | 441.574 s; 8,152.7 games/h | 49.748 s | 18.631 s (6.090 s) |
| 5k | 5,000 / 130,825 | `d0ab6252d29945b1e25735615460cc42f9528191ce46e5e4e2c25f29d2360af8` | `60087bdb8c11fdd04c665775b6dc0e5595b206a8cac26859f4330ad8bc14db19` | 2,304.013 s; 7,812.5 games/h | 249.791 s | 94.183 s (32.148 s) |

“Training wrapper” includes loading, validation, checkpointing, and report
work; the parenthesized value is the measured CUDA epoch loop. The comparable
generation-through-training totals were 509.953 seconds (8m 30s) for 1k and
2,647.987 seconds (44m 08s) for 5k. Self-play consumed 86.6% and 87.0% of
those totals respectively, so generation—not training—was the dominant
compute cost. These are measured wall-clock costs on the recorded RTX 4060
environment, not cloud-price estimates.

## Targets and losses

| Stage | Policy support mean | Max target probability mean | Entropy mean | Value targets loss/draw/win | Selected epoch | Selected train policy/value | Selected validation policy/value (total) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| matched-1k | 23.550 | 0.3852 | 2.2801 nats | 11,172 / 2,853 / 11,510 | 9 | 2.8462 / 0.2889 | 2.9185 / 0.2994 (3.2179) |
| 5k | 23.458 | 0.3844 | 2.2764 nats | 55,857 / 17,408 / 57,560 | 12 | 2.8606 / 0.3020 | 2.8780 / 0.3215 (3.1995) |

The policy-target distributions are nearly unchanged at 5x scale. At their
value-selected epochs, 5k has slightly better validation policy and total
loss but worse validation value loss. No separately preregistered calibration
metric was recorded, so none is inferred. In any case, the contract explicitly
forbids loss or calibration from overriding measured playing strength.

## Paired strength and uncertainty

Results are candidate wins-losses-draws. Intervals are the reports' 95% Wilson
intervals for candidate win rate; draws remain in the denominator. Role swaps
control color within each shared-seed pair, but 40 games still give wide
sampling uncertainty and each scale was trained only once.

| Stage | Opponent | Result | Win rate (95% interval) | Contract consequence |
| --- | --- | ---: | ---: | --- |
| matched-1k | fixed starting champion | 30-8-2 | 75.0% (59.8%-85.8%) | passes promotion gate |
| matched-1k | matched non-neural MCTS | 32-4-4 | 80.0% (65.2%-89.5%) | diagnostic |
| matched-1k | unchanged heuristic search | 6-34-0 | 15.0% (7.1%-29.1%) | diagnostic |
| 5k | fixed starting champion | 29-10-1 | 72.5% (57.2%-83.9%) | passes promotion gate |
| 5k | matched-1k candidate | **19-21-0** | **47.5% (32.9%-62.5%)** | **fails 22-win scaling gate; saturated** |
| 5k | matched non-neural MCTS | 34-3-3 | 85.0% (70.9%-92.9%) | diagnostic |
| 5k | unchanged heuristic search | 14-26-0 | 35.0% (22.1%-50.5%) | diagnostic |

The heuristic result moved from 15% to 35%, an encouraging 8-game/20-point
change under the same opponent and seed schedule. It is not a preregistered
scale gate, its Wilson intervals overlap, and the candidates' direct paired
comparison favored 1k by 21-19. It therefore motivates follow-up but does not
reverse the saturation decision or establish a general 5k improvement.

## Retention

| Stage | Verified URI | Objects | Bytes | Inventory SHA-256 |
| --- | --- | ---: | ---: | --- |
| matched-1k | `file:///home/eletim/twixt-ai-artifacts/issue-128/matched-1k/generation-0001` | 1,015 | 544,426,771 | `9895fbb545029311942fe2b124b3543ee2e4904dbe55522d238c165d1b459a2a` |
| 5k | `file:///home/eletim/twixt-ai-artifacts/issue-128/5k/generation-0001` | 5,037 | 2,769,910,005 | `5447d23e68d0df76348c4077d502a8e5fd227f55f236349544e9d1abdbbc03e1` |

Both manifests contain all four required categories (self-play, dataset,
training, and evaluation), have exact category/global rollups, and carry an
independent `local-filesystem-sha256-audit` attestation of the same URI and
inventory digest. The checked-in reports consequently mark both inventories
complete, verified, and pruning-ready. The `file://` locations describe the
verified durable copies; the earlier 1k S3 value was only a placeholder and is
not claimed as uploaded.

## Stop decision and next bottleneck

The 10k stage was conditional on 5k reaching 22 wins against the previous
retained candidate. Because 5k reached only 19, the saturation rule stops the
curve before 10k. With 10k ineligible, 25k was also ineligible. The stretch
50k stage additionally required a passing 25k gain result plus runtime and
storage checks, so it could not be considered. This is a gate-based stop, not
a claim that larger datasets can never help.

The next bottleneck is the quality of the fixed teacher/search targets, with
value modeling the clearest learned diagnostic, rather than raw example count
under this teacher. Five times more data reproduced almost identical policy
targets and worsened selected validation value loss, while both candidates
still lost clearly to the unchanged heuristic. The next experiment should
improve teacher/search or value-target quality as a separately labelled
diagnostic, then establish a new frozen contract before resuming a data-scale
curve. Model capacity remains a secondary hypothesis; this two-point,
single-training-seed experiment does not separate it from target quality.

## Reproduction and verification

The exact matched-1k invocation is frozen in the
[contract documentation](issue-128-strength-scaling-contract.md#evidence-and-retention).
To reproduce 5k, change only game count to 5000, root seed to 1285000, output
directory, and artifact URI. Use a fresh output path because the commands
refuse to overwrite retained evidence. CUDA kernels may introduce small
floating-point differences; the hashes above identify the exact datasets and
checkpoints used for the reported evaluations.

From the repository root, validate the contract, checked-in hashes, stage
decisions, and retention attestations with:

```bash
PYTHONPATH=src python3 -m pytest tests/training/test_scaling_contract.py
```
