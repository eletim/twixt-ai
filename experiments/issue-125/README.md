# Issue 125 starting baseline

Issue 125 starts from the immutable Issue 118 generation-2 champion at
[`../issue-118/generation-2/generation-0001/candidate/best.pt`](../issue-118/generation-2/generation-0001/candidate/best.pt),
SHA-256 `742229c59caf251a07c7ecac6dc77ff75cbf09643a22cca16b92fe083df5a5ec`.
The checkpoint stays at its original path and must remain read-only.

On `dev/v0.0.8` commit `cd7f2714011231cc6a38fe99110e34b0c54bc627`, its hash
and deterministic paired evaluations were re-verified without training or modifying
either checkpoint. The results reproduced as 28-8-4 versus generation 1
([promotion report](../issue-118/generation-2/generation-0001/report.json)), 39-0-1
versus the [Issue 57 baseline](../issue-118/generation-2/baseline-comparison.json),
and 6-14-0 policy+value, 5-15-0 policy-only, and 0-19-1 value-only versus
heuristic search ([strength report](../issue-118/generation-2/strength.json)).

The immutable Issue 57 reference remains
[`../issue-57/baseline/best.pt`](../issue-57/baseline/best.pt), SHA-256
`ce20f05c8a3d687fce4860595d40d710177f5901298c0bea95afb488cadcc3c8`.
Both checkpoints are read-only reference points for every later Issue 125 comparison.

The deeper value-head audit is retained in
[`diagnostics/`](diagnostics/README.md). It uses the largest self-play dataset
whose manifest hash matches the generation-2 champion, and records phase,
search-ambiguity, and value-confidence breakdowns without changing either
artifact.

## Full-scale search-configuration confirmation

Before any candidate training, the unmodified generation-2 champion was run
against the unchanged depth-1, 10,000-node heuristic with seed 1188200. Each
setting used 40 games (20 identical-seed role-swapped pairs), policy+value
guidance, rollout limit 4, and CUDA inference. The complete games, role splits,
configuration, checkpoint hash, device metadata, and Wilson intervals are in
[`search-confirmation.json`](search-confirmation.json).

| Confirmed setting | Champion W-L-D | Win rate | 95% Wilson interval |
| --- | ---: | ---: | ---: |
| budget-128 (128, sqrt(2), 1.5/0.5) | 19-21-0 | 47.5% | 32.9%-62.5% |
| teacher-like-64 (64, 0.7, 3.0/0.5) | 24-16-0 | 60.0% | 44.6%-73.7% |

The 20-game screen's stronger configurations materially close the standard
6-14 heuristic gap through search configuration alone. Teacher-like-64 won
this fresh full-scale schedule, while budget-128 reached near parity. The
overlapping intervals do not establish that either configuration is
universally superior. The heuristic settings were identical in both matchups,
and the champion remained SHA-256
`742229c59caf251a07c7ecac6dc77ff75cbf09643a22cca16b92fe083df5a5ec`
before and after confirmation.

Reproduce the confirmation with:

```bash
PYTHONHASHSEED=0 PYTHONPATH=src python3 -m \
  twixt_ai.evaluation.matched_search_cli \
  --checkpoint experiments/issue-118/generation-2/generation-0001/candidate/best.pt \
  --output experiments/issue-125/search-confirmation.json \
  --games-per-matchup 40 --seed 1188200 --rollout-limit 4 \
  --setting budget-128 --setting teacher-like-64 \
  --guidance-mode policy-value --baseline heuristic-search --device cuda
```

## Value-guided candidate training

The value audit found validation MSE 0.4239, weak middle-confidence buckets,
and phase-specific generalization gaps. The matched screen found that removing
value guidance tied or improved the strongest 20-game results. Generation 2
had selected epoch 4 by combined loss even though epoch 1 had its lowest
validation value loss, then showed worsening validation value loss through
epoch 20. These results motivated warm-starting from the immutable champion,
lowering the learning rate, and selecting checkpoints by validation value loss.

Both CUDA candidates and all of their metrics are retained under
[`candidates/`](candidates/README.md). Neither improved convincingly: the
1e-4 run's best value loss was 0.423955 at epoch 2 (effectively tied with the
champion's 0.423941), and the follow-up 3e-4 run's best was 0.425781 at epoch
1. Both then regressed. They are rejected as replacements, and no playing-
strength claim is made from training metrics alone.
