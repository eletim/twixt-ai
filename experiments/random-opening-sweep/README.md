# Frozen Gen11 random opening sweep

This experiment changes only `random_opening_moves` across N=0, 2, 4, 6.
It uses the PR #164 implementation at commit
`98036e1a116554ffc2f5ca9d6a05b601fe0b9120`, available in the
adjacent `twixt-ai-frozen-opening` checkout. The common initial checkpoint is
`models/frozen/gen11/best.pt` in that checkout, SHA-256
`31832286c6493e1bbff77c3357b6787fd17d60e52a5a572483b70d2077fee536`.
No game rules, encoding, action mapping, or Pie rule are changed.

Each condition has 5,000 Mini 10x10 self-play games from the same checkpoint.
Policy+Value MCTS uses 64 simulations, exploration 0.7, progressive widening
constant 3.0 and exponent 0.5. Eight thread workers use shared CUDA inference,
batch size 8 and 2 ms maximum flush wait. Game seeds start from 164000 for all
conditions. Random opening moves are uniform over legal actions and excluded
from policy and value training examples. The game-held-out validation split
uses `random-opening-sweep-164000`.

Each candidate warm starts Frozen Gen11's 32 channel, four residual block
network and uses AdamW, batch 512, learning rate 3e-4, weight decay 1e-4,
step scheduler (30 epochs, factor 0.5), maximum 100 epochs and selection by
validation total loss. The common training seed is 164100. Both best and latest
checkpoints are retained locally.

Strength screens use 64-simulation P+V MCTS with the same search parameters.
Each matchup has 40 games, 20 seed pairs, with candidate Red and Black once
per seed. All matchups start from evaluation seed 164900. Four candidates play
Frozen Gen11 and all six candidate pairs play one another. Draws count half
for score rate. No promotion or champion update occurs.

Run `python3 experiments/random-opening-sweep/run.py` from this checkout with
the adjacent PR checkout present. `run.py` verifies the checkpoint hash and
source commit. It then generates, trains, diagnoses and evaluates every
condition. `summary/results.json` and `summary/comparison.md` are produced
from the recorded artifacts. Raw games, dataset shards and checkpoints stay
local; their hashes and derived metrics are tracked in Git.

The validation policy diagnostics mask illegal actions and compare the model
to visit distribution. This is distinct from the unmasked cross entropy in
the training curve. Calibration error is the weighted mean absolute gap of
predicted and observed values in ten prediction bins; decisive sign error
excludes drawn targets. Both are computed on held-out games only.

## Paired 4-ply evaluation contract

The follow-up evaluation uses the retained `training/best.pt` checkpoints;
there is no retraining. Every matchup uses the same seed sequence, with
`opening_seed=164904` and `decision_seed=164900`. Each pair plays the identical
uniformly sampled legal 4-ply opening twice, swapping Red and Black. The
opening moves and SHA-256 of the canonical state are checked for equality.
Only later moves use the normal P+V MCTS. Frozen matchups have 200 pairs;
candidate matchups have 100 pairs, using the same first 100 openings.

If an opening reaches a win, draw, or no-legal-move terminal state before four
plies, both games stop there. The pair is recorded as `opening_terminal` and
excluded from all strength W-D-L, paired score, move-count, and draw-rate
denominators. It remains in the opening diagnostics and terminal count.
The paired score is candidate game points divided by twice the number of
non-terminal pairs (win=1, draw=0.5). A `better_pair` gives the candidate
more than one point across both color assignments; `equal_pair` gives one;
`worse_pair` gives less than one. `both_win`, `split` (one win, one loss), and
`both_loss` are also reported. The full result for each color is retained.

Run `PYTHONPATH=src python3 experiments/random-opening-sweep/evaluate_paired.py
--left n0 --right frozen` from the PR checkout. Pair JSONL is written after
each pair to permit resume. Full JSON, including opening seeds, moves, side to
move, terminal status, position hashes, winners, and move counts, is stored
under `paired-4ply/`. Unique hash count and maximum multiplicity check opening
diversity. No opening advantage proxy is asserted because no calibrated
position evaluator is available independent of the tested checkpoints.
