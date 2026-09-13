# Architecture-v2 value-head quality

This audit measures the first trained architecture-v2 candidate's value head
on fixed self-play probe positions, independently of the loss history written
by the trainer. The complete machine-readable output is retained in
[`value-head.json`](value-head.json), including the negative result and every
train and validation bucket.

## Method and artifact verification

The audit reuses the Issue 129 value-head diagnostic methodology without
changing the checkpoint or dataset. It runs fresh checkpoint inference over
every example, verifies the dataset manifest, both shard hashes, and all 4,702
example counts, and reports outcome calibration, MSE, and MAE overall and by
fixed 16-ply phase, normalized search-policy entropy, signed prediction, and
absolute prediction confidence.

- Candidate: `../v2-candidate-cuda/best.pt`, SHA-256
  `5a4af18afee078a87d30e72d894f16617e2efde71375214abe4e68a518365461`.
- Probe dataset: `../v2-selfplay-dataset/dataset`, generated from 100 fixed
  self-play games with seed `143100` and split seed
  `issue-143-v2-143100`.
- Dataset manifest SHA-256:
  `e79de1637a0ec0f852c150345ee0ea80b1fe8b15d9474b92c6f07aca29a402e4`.
- Probe positions: 4,371 train and 331 held-out validation positions. The
  checkpoint records the same manifest digest, and the diagnostic reports
  `dataset_sha256_matches: true`.

The validation probe is fixed by the committed dataset and was not selected
or modified in response to these results. CPU inference was used to make the
measurement portable; it evaluates the same immutable checkpoint selected at
training epoch 3.

## Results

| Split | Positions | MSE | MAE | Mean target | Mean prediction | Calibration error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 4,371 | 0.1655 | 0.2569 | +0.0075 | +0.0263 | +0.0187 |
| Validation | 331 | 0.8039 | 0.7125 | +0.0091 | +0.0130 | +0.0039 |

The independently recomputed validation MSE agrees with the checkpoint's
recorded selection value to floating-point precision, but the stratified
results show quality that the single training metric does not. Validation MSE
is 0.9221 at plies 0-15, 0.8241 at plies 16-31, 0.4784 at plies 32-47, and
0.0307 for the four positions at plies 48-63. The validation-minus-training
MSE gap is 0.6383.

| Absolute prediction | Positions | MSE | MAE | Calibration error |
| --- | ---: | ---: | ---: | ---: |
| 0.0-0.2 | 56 | 0.9784 | 0.9821 | +0.0994 |
| 0.2-0.4 | 52 | 0.8958 | 0.8995 | +0.0027 |
| 0.4-0.6 | 55 | 0.9537 | 0.8549 | -0.0327 |
| 0.6-0.8 | 86 | 0.5086 | 0.5113 | -0.0051 |
| 0.8-1.0 | 82 | 0.8355 | 0.5254 | -0.0266 |

## Negative findings and limits

The value head does not generalize reliably on this probe. Its small aggregate
calibration error comes from cancellation rather than uniformly calibrated
predictions: the `[-1.0, -0.8)` signed bin has calibration error -0.4146, and
the `[0.8, 1.0]` bin has error +0.2924. Across those two most-confident signed
bins, 19 of 82 predictions have the wrong outcome sign. Error is also not
monotonic in absolute confidence: the highest-confidence bucket has worse MSE
than the 0.6-0.8 bucket.

The training and validation distributions differ materially. Train includes
1,430 draw targets, while the validation probe contains 164 losses, no draws,
and 167 wins. The validation set represents only nine held-out source games;
the four-position late-phase bucket and the low- and medium-entropy buckets
are too small for broad conclusions. These limits make the measurement
inconclusive about value quality on draws and rare search-entropy regimes, but
they do not weaken the observed generalization and conditional-calibration
failures on the retained decisive-position probe.

## Reproduction

From the repository root:

```bash
PYTHONHASHSEED=0 PYTHONPATH="$PWD/src" python3 -m \
  twixt_ai.training.value_diagnostics_cli \
  --dataset experiments/issue-143/v2-selfplay-dataset/dataset \
  --checkpoint experiments/issue-143/v2-candidate-cuda/best.pt \
  --output experiments/issue-143/v2-value-quality/value-head.json \
  --ply-bucket-size 16 --calibration-bins 10 --difficulty-bins 3 \
  --confidence-bins 5 --batch-size 512 --device cpu
```
