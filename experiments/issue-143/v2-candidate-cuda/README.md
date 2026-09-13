# Architecture-v2 CUDA candidate

This is the first trained candidate in the fresh architecture-v2 Mini lineage.
It warm-started the immutable architecture-v2 bootstrap checkpoint and trained
for 20 epochs on the architecture-v2 self-play dataset. No architecture-v1
checkpoint or data contributed weights or examples.

Training used AdamW, batch size 128, learning rate `0.001`, weight decay
`0.0001`, seed `143200`, and validation value loss for best-checkpoint
selection. The model retains the 10x10, encoding-v1, 8-channel, one-residual-
block architecture with widened 256-unit policy and value heads.

[`metrics.jsonl`](metrics.jsonl) records these four loss curves independently
at every epoch; [`summary.json`](summary.json) retains the same complete
history with the resolved configuration and device details:

- `train_policy_loss`
- `train_value_loss`
- `validation_policy_loss`
- `validation_value_loss`

| Point | Train policy | Train value | Validation policy | Validation value |
| --- | ---: | ---: | ---: | ---: |
| Epoch 1 | 4.3491025639 | 0.5756183843 | 4.3984234902 | 0.9392352255 |
| Best value, epoch 3 | 4.1628890780 | 0.1805610712 | 4.3549107338 | 0.8038753293 |
| Epoch 20 | 3.8562824794 | 0.0503989339 | 4.4024432390 | 1.4525887448 |

The best trained checkpoint is `best.pt`, SHA-256
`5a4af18afee078a87d30e72d894f16617e2efde71375214abe4e68a518365461`.
The final-epoch resumable checkpoint is `latest.pt`, SHA-256
`c8517ade6990532c671c6de4c42c8af81aec2ced22e72195630c52f174d5dbf9`.
The input dataset manifest SHA-256 is
`e79de1637a0ec0f852c150345ee0ea80b1fe8b15d9474b92c6f07aca29a402e4`.

CUDA ran on an NVIDIA GeForce RTX 4060 with driver `550.163.01`, PyTorch
`2.8.0+cu128`, and CUDA runtime `12.8`. The complete command took 5.46 seconds
of wall-clock time. Timed optimization work took 1.4009307760 seconds at
62401.37 examples/second and peaked at 33,388,032 CUDA allocation bytes.
[`run.json`](run.json) records these measurements, input identities, output
hashes, and the names of the four separate loss series.

Reproduce the run from the repository root with an available CUDA device and
an absent output directory:

```bash
PYTHONHASHSEED=0 PYTHONPATH="$PWD/src" \
  python3 -m twixt_ai.training.train_cli \
  --dataset experiments/issue-143/v2-selfplay-dataset/dataset \
  --output-dir experiments/issue-143/v2-candidate-cuda \
  --epochs 20 --batch-size 128 --learning-rate 0.001 \
  --weight-decay 0.0001 --seed 143200 --device cuda \
  --selection-metric value --channels 8 --residual-blocks 1 \
  --value-hidden 256 \
  --initial-checkpoint \
    experiments/issue-143/architecture-v2-bootstrap/champion.pt
```
