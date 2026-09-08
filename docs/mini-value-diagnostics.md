# Mini value-head diagnostics

Issue 89 audits the value path and retrains the retained 22-plane Mini model on
the 5,000-game v0.0.4 dataset from Issue 88. The source manifest SHA-256 is
`a47eda5c2bb48fcd103a3380517a91470054170b24c176969d81c33df16290e6`;
it contains 271,890 training and 30,409 validation positions. The raw results
are in [`experiments/issue-89/report.json`](../experiments/issue-89/report.json).

## Perspective audit

The value contract is consistent end to end:

| Stage | Perspective behavior |
| --- | --- |
| Self-play artifact | Records the terminal winner in canonical Red/Black coordinates. |
| Dataset conversion | Emits +1 when the winner equals the position's side to move, -1 for the opponent, and 0 for a draw. |
| Augmentation | Training currently uses no augmentation. The focused transpose test verifies that axis-swapping symmetries exchange both colors, leaving side-relative value unchanged. |
| Dataset loader and loss | Loads the scalar unchanged and applies MSE to the bounded side-to-move prediction. |
| Checkpoint inference | The model and neural adapter return the encoded position's side-to-move value. |
| MCTS | A child estimate is negated when the child's side to move is the root player's opponent. |

Deterministic tests cover a decisive artifact whose targets alternate +1/-1,
the axis-swapping symmetry, dataset loading and zero-predictor MSE, checkpoint
inference, and MCTS child-to-root conversion. No encoding change was needed.

## Target balance and calibration

The full training distribution is 73,632 losses, 122,997 draws, and 75,261
wins from the side-to-move perspective; validation is 8,293 / 13,648 / 8,468.
The +1 and -1 classes stay closely balanced in every ply bucket. Draws become
dominant late because drawn games reach the board limit while decisive games
leave the population. This is expected survival conditioning, not a sign bug.

The CUDA checkpoint selected by validation value loss has:

| Split | MSE | MAE | Mean target | Mean prediction |
| --- | ---: | ---: | ---: | ---: |
| Train | 0.4292 | 0.5273 | 0.0060 | 0.0047 |
| Validation | 0.4610 | 0.5481 | 0.0058 | 0.0030 |

Validation MSE falls steadily by phase: 0.683 for plies 0-15, 0.560 for
16-31, 0.438 for 32-47, 0.287 for 48-63, 0.121 for 64-79, and 0.032 for
80-95. Early outcomes remain noisy under shallow self-play; later positions
are substantially more learnable. The train/validation MSE gap is 0.0318.
Full calibration bins and phase distributions are preserved in
[`value-diagnostics.json`](../experiments/issue-89/value-diagnostics.json).
The diagnostics command rejects a checkpoint whose recorded dataset digest
does not match the supplied manifest, preventing stale small-data results from
being attributed to this experiment.

MSE remains the appropriate baseline: it directly estimates expected outcome,
the decisive signs are balanced, and diagnostics do not support class
reweighting or a robust-loss response. The actionable training issue was that
`best.pt` previously minimized combined policy-plus-value loss. The trainer now
supports `--selection-metric value`; this run selected epoch 2 at validation
value MSE 0.4610 rather than allowing later policy improvements to mask value
overfitting. Training used CUDA on an NVIDIA GeForce RTX 4060, seed 890100,
batch size 128, 20 epochs, AdamW, learning rate 0.001, and weight decay 0.0001.
The checkpoint SHA-256 is
`4b722180316bdf3eb0ff709bda9491fcb72c3a1cb0096cb49be794ddd9dafd07`.

## Matched search evaluation

Every learned mode and non-neural MCTS used 20 simulations and rollout limit
4. Each matchup used 20 paired games with roles swapped under the same seed
schedule (seed 890200).

| Guidance | Random W-D-L | Heuristic W-D-L | Non-neural MCTS W-D-L |
| --- | ---: | ---: | ---: |
| Policy + value | 16-4-0 | 0-0-20 | 11-4-5 |
| Policy only | 15-4-1 | 0-0-20 | 12-4-4 |
| Value only | 16-4-0 | 0-0-20 | 12-4-4 |

Against non-neural MCTS, the prior 100-game checkpoint scored 1-5-14 with
policy+value and 0-2-18 with value only. The larger-data result therefore
shows that useful value guidance was recovered. It does not establish broad
strength: all learned modes still lost every game to heuristic search, and the
20-game Wilson intervals remain wide. The complete game schedule and confidence
intervals are in [`strength.json`](../experiments/issue-89/strength.json).

## Reproduction

From a checkout containing the Issue 88 dataset:

```bash
PYTHONHASHSEED=0 twixt-ai-train \
  --dataset experiments/issue-88/5k/dataset \
  --output-dir experiments/issue-89/training \
  --epochs 20 --batch-size 128 --learning-rate 0.001 \
  --weight-decay 0.0001 --seed 890100 --device cuda \
  --selection-metric value --channels 8 --residual-blocks 1 --value-hidden 16

PYTHONHASHSEED=0 twixt-ai-value-diagnostics \
  --dataset experiments/issue-88/5k/dataset \
  --checkpoint experiments/issue-89/training/best.pt \
  --output experiments/issue-89/value-diagnostics.json --device cuda

PYTHONHASHSEED=0 twixt-ai-mini-strength-evaluation \
  --checkpoint experiments/issue-89/training/best.pt \
  --output experiments/issue-89/strength.json \
  --games-per-matchup 20 --seed 890200 --simulations 20 \
  --rollout-limit 4 --device cuda
```

The fixed seed controls data order and match schedules. CUDA convolution
kernels may still introduce small run-to-run floating-point differences, so
the recorded checkpoint and SHA-256 identify the exact evaluated model.
