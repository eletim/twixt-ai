# Generation-2 value-head diagnostics

This audit applies the extended `twixt-ai-value-diagnostics` tool to the
immutable generation-2 champion and the largest available self-play dataset
matched by the checkpoint's recorded dataset digest. The complete machine
output, including every train and validation bucket, is retained in
[`generation-2-value-head.json`](generation-2-value-head.json).

## Artifact verification

- Dataset: `experiments/issue-118/generation-2/generation-0001/dataset`, 1,000
  source games, 23,843 training positions, and 3,626 validation positions.
- Manifest SHA-256:
  `9b6fca3d3c08f08e5eccccfba67ca0e09c809dcaeaf9ed9ac73024f4b8a72fad`.
- Champion: `experiments/issue-118/generation-2/generation-0001/candidate/best.pt`.
- Checkpoint SHA-256:
  `742229c59caf251a07c7ecac6dc77ff75cbf09643a22cca16b92fe083df5a5ec`.
- The checkpoint metadata records the same manifest digest. The diagnostic run
  recomputed the manifest hash, verified all six shard hashes and all 27,469
  example counts, and reports `dataset_sha256_matches: true`.

Neither source artifact was modified. In particular, this is not the larger
Issue 88 dataset: that dataset does not match the generation-2 checkpoint and
the tool rejects a checkpoint carrying a different dataset digest.

## Definitions

Game phase uses fixed 16-ply buckets. Position difficulty is a reproducible
proxy derived from the training target: the entropy of the sparse MCTS visit
distribution divided by the log of its support. Zero means concentrated search
visits and one means uniform visits. It measures teacher-search ambiguity, not
intrinsic game complexity. Value-head confidence is the absolute bounded value
prediction, grouped into five equal-width buckets. Every bucket retains target
counts and fractions, mean target and prediction, signed calibration error
(`mean_prediction - mean_target`), MSE, and MAE.

## Results and negative findings

Overall validation MSE is 0.4239 and MAE is 0.4786, compared with training MSE
0.3356 and MAE 0.3998. The validation-minus-training MSE gap is 0.0883. The
overall validation calibration error is only +0.0045, but that average masks
substantial conditional errors.

| Validation phase | Examples | Draw share | MSE | MAE | Calibration error |
| --- | ---: | ---: | ---: | ---: | ---: |
| plies 0–15 | 1,967 | 5.7% | 0.4976 | 0.5187 | -0.0018 |
| plies 16–31 | 1,040 | 10.8% | 0.3445 | 0.4111 | +0.0282 |
| plies 32–47 | 247 | 45.3% | 0.4340 | 0.5356 | +0.0944 |
| plies 48–63 | 142 | 78.9% | 0.3026 | 0.4517 | -0.0414 |
| plies 64–79 | 128 | 87.5% | 0.2307 | 0.4108 | -0.0948 |
| plies 80–95 | 102 | 100.0% | 0.2014 | 0.3775 | -0.1451 |

Early positions remain the largest source of total error. Late validation
positions are not cleanly solved either: every target at plies 80–95 is a draw,
yet the mean prediction is -0.1451. By contrast, the corresponding training
bucket has MSE 0.0282 and calibration error +0.0384. This phase-specific
generalization failure is hidden by the globally small calibration error.

| Normalized search entropy | Examples | Draw share | MSE | MAE | Calibration error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0–1/3 | 9 | 100.0% | 0.1335 | 0.2669 | -0.1260 |
| 1/3–2/3 | 877 | 11.2% | 0.5144 | 0.5482 | -0.0587 |
| 2/3–1 | 2,740 | 20.3% | 0.3959 | 0.4570 | +0.0251 |

The difficulty proxy is not monotonic: medium-entropy positions have worse
error than high-entropy positions. The low-entropy sample contains only nine
all-draw positions, so it cannot support a broad claim. Search entropy is also
confounded with phase and target balance and should not be treated as a direct
measure of value difficulty.

| Absolute value prediction | Examples | Draw share | MSE | MAE | Calibration error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.0–0.2 | 329 | 41.0% | 0.5867 | 0.6231 | +0.0957 |
| 0.2–0.4 | 385 | 35.6% | 0.6316 | 0.6997 | +0.0557 |
| 0.4–0.6 | 937 | 22.5% | 0.5871 | 0.6650 | +0.0091 |
| 0.6–0.8 | 833 | 14.2% | 0.3790 | 0.4662 | -0.0072 |
| 0.8–1.0 | 1,142 | 5.3% | 0.2060 | 0.2185 | -0.0344 |

The middle confidence region is the weakest: predictions with magnitude
0.2–0.4 reach 0.6316 MSE, and the entire 0.0–0.6 region remains near 0.59 MSE
or worse. Even the highest-confidence bucket is imperfect at 0.2060 MSE.
Signed calibration bins in the full output further show a +0.1591 error for
predictions from +0.2 to +0.4. These findings do not justify treating the
champion's value output as uniformly reliable merely because its global mean
is calibrated.

The target signs themselves remain fairly balanced: training has 10,354
losses, 2,466 draws, and 11,023 wins, while validation has 1,435 losses, 662
draws, and 1,529 wins. However, validation has a larger draw share (18.3%
versus 10.3%), and draw prevalence changes sharply with phase and confidence.
Future value-learning comparisons therefore need to retain stratified metrics
rather than relying on one aggregate loss.

## Reproduction

The recorded run used CUDA on an NVIDIA GeForce RTX 4060 with PyTorch 2.8.0,
batch size 512, ten signed calibration bins, three difficulty buckets, and five
confidence buckets:

```bash
PYTHONPATH="$PWD/src" PYTHONHASHSEED=0 python3 -m \
  twixt_ai.training.value_diagnostics_cli \
  --dataset experiments/issue-118/generation-2/generation-0001/dataset \
  --checkpoint \
    experiments/issue-118/generation-2/generation-0001/candidate/best.pt \
  --output experiments/issue-125/diagnostics/generation-2-value-head.json \
  --ply-bucket-size 16 --calibration-bins 10 --difficulty-bins 3 \
  --confidence-bins 5 --batch-size 512 --device cuda
```
