# Architecture-v2 1,000-game scaling stage

This is the first retained data-scaling stage for Issue 151. It uses the
immutable Issue 143 architecture-v2 bootstrap and changes the self-play game
count from 100 to 1,000. The model, encoding, search semantics, optimizer,
training schedule, selection rule, paired evaluation seeds, and fixed
opponents are unchanged. [`config.json`](config.json) records the complete
contract and [`report.json`](report.json) records the measured outcome.

## Dataset and device

All 1,000 fresh games completed without failure and produced 47,561 positions:
43,492 train positions from 917 games and 4,069 validation positions from 83
held-out games. The held-out split includes 1,047 draw positions. Self-play
took 7,007.39 seconds (513.74 games/hour) on an NVIDIA GeForce RTX 4060 with
PyTorch 2.8.0+cu128 and CUDA runtime 12.8. The resolved two-thread, shared
inference-batch configuration and its full statistics are in
[`selfplay-report.json`](selfplay-report.json).

The dataset manifest SHA-256 is
`c3877371ed27b83188ca968ae829ff7c79b3704e1ad3a193e0be0e56b2e99cee`.
It records every shard digest, count, split seed, bootstrap digest, and search
setting. The raw source games and JSONL shards were consumed from scratch and
are intentionally not duplicated in Git: each full training shard is about
108 MB. They can be regenerated from the retained checkpoint and command
below; the committed manifest is the immutable dataset identity used by the
checkpoint and all downstream evidence.

## Training and value quality

The unchanged 20-epoch AdamW schedule selected epoch 1 by validation value
loss. The best checkpoint SHA-256 is
`0a5809b1123e77d92869c97d49df5536f752ca88ad1f657126a971b41c0a1716`;
the final resumable checkpoint SHA-256 is
`ab9e10cc619aa7b940996023c30b8196d5b039ba0007614157917fffe028702d`.
[`candidate/metrics.jsonl`](candidate/metrics.jsonl) retains all four loss
curves separately, and [`candidate/run.json`](candidate/run.json) retains CUDA
driver/device, runtime, input, and output hashes.

| Point | Train policy | Train value | Validation policy | Validation value |
| --- | ---: | ---: | ---: | ---: |
| Best, epoch 1 | 4.241112 | 0.539172 | 4.233018 | 0.736819 |
| Final, epoch 20 | 3.893392 | 0.056063 | 4.103909 | 0.978325 |

Fresh inference over every example independently confirmed the best-epoch
value result:

| Split | Positions | MSE | MAE | Calibration error |
| --- | ---: | ---: | ---: | ---: |
| Train | 43,492 | 0.394623 | 0.481364 | +0.058386 |
| Validation | 4,069 | 0.736819 | 0.704534 | +0.054576 |

The validation-minus-train MSE gap fell from 0.638338 at 100 games to 0.342196
at 1,000 games. Held-out MSE improved by 0.067056 (8.34%) and MAE by 0.007996.
This is measurable improvement, but not good calibration: the most-confident
negative and positive signed bins have calibration errors -0.428225 and
+0.465561, and 107 of their 456 predictions have the wrong outcome sign. The
full phase, entropy, signed-calibration, confidence, and target-balance audit
is retained in [`value-head.json`](value-head.json).

## Paired playing strength

Every matchup used the unchanged seed-`1289000`, 40-game/20-pair role-swap
schedule. Learned and matched non-neural MCTS used `standard-20`: 20
simulations, rollout limit 4, exploration sqrt(2), and progressive widening
1.5/0.5. The fixed heuristic opponent retained depth 1 and a 10,000-node
budget. Results are candidate wins-draws-losses:

| Guidance | Matched non-neural MCTS | Heuristic search |
| --- | ---: | ---: |
| Policy only | 13-5-22 | 1-0-39 |
| Value only | 13-6-21 | 1-0-39 |
| Policy + value | 11-8-21 | 0-0-40 |

Relative to the 100-game v2 candidate, value-only improved from 5-6-29 and
policy+value improved from 4-4-32 against matched MCTS. Value-only is no
longer clearly harmful relative to policy-only in this 40-game screen, but
policy+value did not beat policy-only and still failed to win against heuristic
search. The complete 240-game record and Wilson intervals are in
[`evaluation.json`](evaluation.json).

Architecture-v1 matched-1k remains far ahead at 32-4-4 against matched MCTS
and 6-0-34 against heuristic search. That context shares evaluation semantics,
but its self-play teacher used a different search contract, so it is not a
controlled architecture-only comparison. The 1k v2 evidence supports more
data helping value generalization and guidance strength, but data quantity
alone is not sufficient; conditional value miscalibration and weak baseline
strength remain the next bottleneck. This candidate is not promoted.

## Reproduction

Generate into an absent scratch directory from the repository root:

```bash
PYTHONHASHSEED=0 PYTHONPATH="$PWD/src" \
  python3 -m twixt_ai.training.selfplay_dataset_cli \
  --champion experiments/issue-143/architecture-v2-bootstrap/champion.pt \
  --output-dir /tmp/twixt-issue-151-1k --games 1000 \
  --seed 143100 --split-seed issue-143-v2-143100 \
  --workflow-label issue-151-v2-scale-1k --device cuda
```

Train with the unchanged schedule:

```bash
PYTHONHASHSEED=0 PYTHONPATH="$PWD/src" \
  python3 -m twixt_ai.training.train_cli \
  --dataset /tmp/twixt-issue-151-1k/dataset \
  --output-dir /tmp/twixt-issue-151-candidate \
  --epochs 20 --batch-size 128 --learning-rate 0.001 \
  --weight-decay 0.0001 --seed 143200 --device cuda \
  --selection-metric value --channels 8 --residual-blocks 1 \
  --value-hidden 256 --initial-checkpoint \
  experiments/issue-143/architecture-v2-bootstrap/champion.pt
```

Run the independent audit and paired evaluation:

```bash
PYTHONHASHSEED=0 PYTHONPATH="$PWD/src" python3 -m \
  twixt_ai.training.value_diagnostics_cli \
  --dataset /tmp/twixt-issue-151-1k/dataset \
  --checkpoint experiments/issue-151/1k/candidate/best.pt \
  --output /tmp/issue-151-value-head.json \
  --ply-bucket-size 16 --calibration-bins 10 --difficulty-bins 3 \
  --confidence-bins 5 --batch-size 512 --device cpu

PYTHONHASHSEED=0 PYTHONPATH="$PWD/src" python3 -m \
  twixt_ai.evaluation.matched_search_cli \
  --checkpoint experiments/issue-151/1k/candidate/best.pt \
  --output /tmp/issue-151-evaluation.json \
  --games-per-matchup 40 --seed 1289000 --rollout-limit 4 \
  --setting standard-20 \
  --guidance-mode policy-only --guidance-mode value-only \
  --guidance-mode policy-value \
  --baseline matched-non-neural-mcts --baseline heuristic-search \
  --device cuda
```
