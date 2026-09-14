# Architecture-v2 playing-strength evaluation

This experiment evaluates the first trained architecture-v2 candidate in
policy-only, value-only, and policy+value MCTS modes against the fixed matched
non-neural MCTS and heuristic-search baselines used by the Issue 128 Mini
experiments. [`evaluation.json`](evaluation.json) retains all 240 games, every
per-role split, Wilson intervals, runtime and environment details, and the full
checkpoint metadata and search configuration.

## Results

Candidate win/draw/loss counts under the shared 40-game paired schedule were:

| Candidate guidance | Matched non-neural MCTS | Heuristic search |
| --- | ---: | ---: |
| Policy only | 13-8-19 | 0-0-40 |
| Value only | 5-6-29 | 0-0-40 |
| Policy + value | 4-4-32 | 0-0-40 |

Policy-only guidance was the strongest v2 ablation against matched non-neural
MCTS, but it did not beat that baseline, and none of the modes won a game
against heuristic search. Combining the heads was weaker than using the policy
alone in this fixed screen. This is consistent with the separately retained
value-head audit, which found poor held-out generalization; the playing result
does not support promoting this v2 candidate.

## Matched protocol and provenance

Every matchup used 40 games grouped into 20 same-seed pairs. Candidate and
baseline exchanged red/black roles inside each pair, and seed `1289000`
generated the same pair-seed schedule for all six matchups. The 10x10 board,
95% Wilson intervals, game rules, and opponent implementations were unchanged.

Candidate and non-neural baseline MCTS both used the Issue 128 `standard-20`
setting: 20 simulations, rollout limit 4, exploration sqrt(2), and progressive
widening constant/exponent 1.5/0.5. The fixed heuristic baseline used depth 1
and a 10,000-node budget. Learned inference ran on CUDA; the report records the
resolved NVIDIA GeForce RTX 4060, driver/runtime context, PyTorch version, and
97.42-second total evaluation runtime.

- Candidate checkpoint: `../v2-candidate-cuda/best.pt`, SHA-256
  `5a4af18afee078a87d30e72d894f16617e2efde71375214abe4e68a518365461`.
- Training dataset manifest: `../v2-selfplay-dataset/dataset/manifest.json`,
  SHA-256
  `e79de1637a0ec0f852c150345ee0ea80b1fe8b15d9474b92c6f07aca29a402e4`.
- Evaluation report: `evaluation.json`, SHA-256
  `a3fbb60eb9ccd1e50a59628d6f47b14f86fd6775415f207e796ab635918547a4`.

The checkpoint embeds the same dataset-manifest digest, and both input hashes
were recomputed from the committed files before recording this evaluation.

## Reproduction

From the repository root, with the output path absent:

```bash
PYTHONHASHSEED=0 PYTHONPATH="$PWD/src" python3 -m \
  twixt_ai.evaluation.matched_search_cli \
  --checkpoint experiments/issue-143/v2-candidate-cuda/best.pt \
  --output experiments/issue-143/v2-strength/evaluation.json \
  --games-per-matchup 40 --seed 1289000 --rollout-limit 4 \
  --setting standard-20 \
  --guidance-mode policy-only --guidance-mode value-only \
  --guidance-mode policy-value \
  --baseline matched-non-neural-mcts --baseline heuristic-search \
  --device cuda
```
