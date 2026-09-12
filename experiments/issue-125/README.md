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
1. Both then regressed. No playing-strength or promotion claim is made from
those metrics alone.

## Fixed-protocol candidate evaluation

The predeclared promotion rule is generation 2's unchanged rule: promote a
candidate only if it wins at least 55% (22 of 40) of the games against the
generation-2 champion. Every matchup uses 20 MCTS simulations, rollout limit
4, policy+value guidance, 20 identical-seed role-swapped pairs, and seed
1251300. The Issue 57 and learned opponents use the same MCTS budget; the
non-neural MCTS is budget-matched; and the heuristic remains depth 1 with a
10,000-node budget. Supporting matchups do not override the champion gate.

The complete configuration, hashes, games, role splits, intervals, and explicit
decisions are in [`candidate-evaluation.json`](candidate-evaluation.json).

| Candidate | Generation-2 champion | Issue 57 | Matched non-neural MCTS | Unchanged heuristic | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| value-selected-lr-1e-4 | 19-18-3 | 40-0-0 | 34-5-1 | 4-36-0 | reject (47.5% champion wins) |
| value-selected-lr-3e-4 | 21-15-4 | 40-0-0 | 34-3-3 | 6-34-0 | reject (52.5% champion wins) |

Reproduce the complete matrix with:

```bash
PYTHONHASHSEED=0 PYTHONPATH=src python3 -m \
  twixt_ai.evaluation.value_candidates_cli \
  --champion experiments/issue-118/generation-2/generation-0001/candidate/best.pt \
  --issue-57 experiments/issue-57/baseline/best.pt \
  --candidate value-selected-lr-1e-4=experiments/issue-125/candidates/value-selected-lr-1e-4/best.pt \
  --candidate value-selected-lr-3e-4=experiments/issue-125/candidates/value-selected-lr-3e-4/best.pt \
  --output experiments/issue-125/candidate-evaluation.json \
  --games-per-matchup 40 --seed 1251300 --simulations 20 \
  --rollout-limit 4 --promotion-win-rate 0.55 --device cuda
```

## Search-configuration decision

Before the additional games, the confirmation rule was fixed as follows: add
160 independent-seed games per setting, combine them with the original 40, and
call a margin real only when the combined 95% Wilson interval is wholly above
50%. Only a setting that passed that gate would receive 40-game paired
follow-ups against Issue 57 and matched non-neural MCTS; adoption would then
require at least 55% wins in each follow-up as well.

The independent seed-1251400 games reversed the original point estimates. Full
games are in
[`search-confirmation-additional.json`](search-confirmation-additional.json),
and the aggregation and explicit decisions are in
[`search-configuration-decision.json`](search-configuration-decision.json).

| Setting | Initial W-L-D | Additional W-L-D | Combined W-L-D | Combined win rate (95% Wilson) | Adoption decision |
| --- | ---: | ---: | ---: | ---: | --- |
| budget-128 | 19-21-0 | 70-90-0 | 89-111-0 | 44.5% (37.8%-51.4%) | reject; inconclusive margin |
| teacher-like-64 | 24-16-0 | 73-87-0 | 97-103-0 | 48.5% (41.7%-55.4%) | reject; inconclusive margin |

Neither interval excludes 50%, so neither setting held up and the conditional
Issue 57/MCTS follow-ups were not triggered. The champion checkpoint and the
heuristic depth/node budget are identical in the original and additional
reports.

Reproduce the additional schedule with:

```bash
PYTHONHASHSEED=0 PYTHONPATH=src python3 -m \
  twixt_ai.evaluation.matched_search_cli \
  --checkpoint experiments/issue-118/generation-2/generation-0001/candidate/best.pt \
  --output experiments/issue-125/search-confirmation-additional.json \
  --games-per-matchup 160 --seed 1251400 --rollout-limit 4 \
  --setting budget-128 --setting teacher-like-64 \
  --guidance-mode policy-value --baseline heuristic-search --device cuda
```

## Generation-3 candidate

After the same-dataset value retrains and search-configuration changes failed
to establish an improvement, the generations pipeline produced 1,000 new
self-play games from the unchanged generation-2 champion. The teacher used the
screened generation 1-2 settings: 64 simulations, exploration 0.7, rollout
limit 4, and progressive widening 3.0/0.5. The complete games, dataset,
training outputs, paired gate, configuration, seeds, lineage, runtimes, and
CUDA metadata are retained under [`generation-3/`](generation-3/).

The dataset contains 20,770 positions and has manifest SHA-256
`9f00059cb9c6c2a3fc17ab3c847aa527158c3b9a9c2dbc42e521ab6318330619`.
Training warm-started only from generation 2 SHA-256
`742229c59caf251a07c7ecac6dc77ff75cbf09643a22cca16b92fe083df5a5ec`.
Following the value-head diagnostics, `best.pt` was selected by validation
value loss; no class reweighting or target reformulation was applied because
the diagnostics did not support either. Epoch 10 was selected at validation
value loss 0.107454. The retained candidate SHA-256 is
`aee1036dbda115eeec0e245909d30e1f8330454a82099e852e1b6a9c26c0dab9`.

The pipeline's unchanged 40-game promotion gate used 20 simulations and seed
1253012. The candidate scored 26-11-3 against generation 2 and passed the 55%
gate at 65% wins. This is a candidate-versus-parent result, not yet a claim
against the fixed heuristic baseline.

Reproduce the retained generation with:

```bash
PYTHONHASHSEED=0 PYTHONPATH=src python3 -m \
  twixt_ai.training.generations_cli \
  --initial-champion experiments/issue-118/generation-2/generation-0001/candidate/best.pt \
  --output-dir experiments/issue-125/generation-3 \
  --generations 1 --games-per-generation 1000 --dataset-window 1 \
  --selfplay-simulations 64 --selfplay-exploration 0.7 \
  --selfplay-progressive-widening-constant 3.0 \
  --selfplay-progressive-widening-exponent 0.5 \
  --evaluation-games 40 --evaluation-simulations 20 --rollout-limit 4 \
  --workers 8 --inference-batch-size 8 \
  --inference-max-wait-seconds 0.002 \
  --epochs 20 --batch-size 128 --learning-rate 0.001 \
  --weight-decay 0.0001 --selection-metric value \
  --validation-fraction 0.1 --shard-size 5000 \
  --promotion-win-rate 0.55 --seed 1253000 --device cuda
```
