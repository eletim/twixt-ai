# Issue 118 Mini strength scaling

The immutable starting baseline is
`experiments/issue-57/baseline/best.pt`, SHA-256
`ce20f05c8a3d687fce4860595d40d710177f5901298c0bea95afb488cadcc3c8`.
Its fixed 20-simulation results are recorded in
`experiments/issue-118/baseline-strength.json`; later experiments do not
replace that baseline.

## Teacher diagnosis

The Issue 88 dataset contained 302,299 positions from 5,000 games, but its
four-simulation teacher gave 98.7% of positions exactly three supported policy
moves. Its mean maximum target probability was 0.503. The 5,000-game retraining
repaired value guidance, but every learned mode still lost 0-20 to the fixed
depth-1, 10,000-node heuristic search.

Budget and root-coverage screens used the CUDA-trained Issue 89 checkpoint
`4b722180316bdf3eb0ff709bda9491fcb72c3a1cb0096cb49be794ddd9dafd07`.
At the default widening constant, policy+value search only occasionally scored
against the heuristic at 64-256 simulations; policy-only search scored 0-16.
At 64 simulations, exploration 0.7, and widening constant 3.0, policy+value
search scored 9-11 in 20 paired games. Constant 12.0 scored 19-1, but was
rejected for generation because its policy targets were nearly uniform: mean
support 56.3 and mean maximum probability 0.050. Constant 3.0 retained mean
support 23.2 and mean maximum probability 0.399 in the screen.

The complete schedules, seeds, role swaps, and negative results are retained
under `experiments/issue-118/diagnostics`.

## Generation 1

Generation 1 used 1,000 games and the selected 64-simulation teacher with
exploration 0.7, progressive widening 3.0/0.5, and rollout limit 4. Eight CPU
game threads shared one CUDA model through batches of eight with a 2 ms maximum
wait. The run used CPython 3.10.12, PyTorch 2.8.0+cu128, CUDA 12.8, and an
NVIDIA GeForce RTX 4060.

Self-play completed all 1,000 games in 561.835 seconds (6,407.6 games/hour),
producing 32,388 positions. The dataset manifest SHA-256 is
`dad89888e94ea78ebe97022e6dc7e390dc79f921b0b5df434b489d6ce0be87c1`.
Targets have mean support 23.32, mean maximum probability 0.382, and mean
entropy 2.296 nats. Derived JSONL shards over GitHub's per-file size limit are
not checked in; the complete raw game artifacts, manifest, split seed, and
dataset configuration are retained so those shards can be rebuilt exactly.

Training warm-started the Issue 89 checkpoint and ran 20 AdamW epochs on CUDA
with batch size 128, learning rate 0.001, weight decay 0.0001, and seed 1185011.
Combined validation loss selected epoch 1. The later negative result is
retained: validation total/value loss worsened from 4.286/0.508 at epoch 1 to
4.394/0.619 at epoch 20, while validation policy loss moved only from 3.778 to
3.775. The selected checkpoint SHA-256 is
`b79f3a0b6745518d815b5873c65c2ded4102915b3dc50b01cb8ab1e6e8365b38`.

The predeclared promotion rule required candidate wins divided by all 40 paired
games to reach 55%, with draws in the denominator. The candidate scored 29-7-4
(72.5%) against the Issue 89 champion and was promoted. In a separate paired
comparison it scored 32-2-6 against the immutable Issue 57 baseline.

Under the fixed standard strength protocol, policy+value scored 18-0-2 against
Random, 14-6-0 against matched non-neural MCTS, and 1-19-0 against heuristic
search. Policy-only scored 19-0-1, 15-3-2, and 1-19-0 respectively. This is a
material improvement over the old champion and matched MCTS, but it does not
solve the heuristic gap.

## Reproduction

Run generation 1 from a source checkout with CUDA:

```bash
PYTHONHASHSEED=0 PYTHONPATH=src python3 -m twixt_ai.training.generations_cli \
  --initial-champion experiments/issue-89/training/best.pt \
  --output-dir experiments/issue-118/generation-1 \
  --generations 1 --games-per-generation 1000 --dataset-window 1 \
  --selfplay-simulations 64 --selfplay-exploration 0.7 \
  --selfplay-progressive-widening-constant 3.0 \
  --selfplay-progressive-widening-exponent 0.5 \
  --evaluation-games 40 --evaluation-simulations 20 --rollout-limit 4 \
  --workers 8 --inference-batch-size 8 \
  --inference-max-wait-seconds 0.002 \
  --epochs 20 --batch-size 128 --learning-rate 0.001 \
  --weight-decay 0.0001 --selection-metric total \
  --validation-fraction 0.1 --shard-size 10000 \
  --promotion-win-rate 0.55 --seed 1185000 --device cuda
```

CUDA kernels can produce small floating-point differences across runs. The
recorded checkpoint hash identifies the exact model used for all reported
head-to-head results.
