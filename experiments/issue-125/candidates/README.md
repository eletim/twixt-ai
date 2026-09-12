# Value-guided candidates

Both candidates use the generation-2-matched dataset at
`experiments/issue-118/generation-2/generation-0001/dataset`, manifest SHA-256
`9b6fca3d3c08f08e5eccccfba67ca0e09c809dcaeaf9ed9ac73024f4b8a72fad`.
Each was warm-started only from the immutable generation-2 champion at
`experiments/issue-118/generation-2/generation-0001/candidate/best.pt`, SHA-256
`742229c59caf251a07c7ecac6dc77ff75cbf09643a22cca16b92fe083df5a5ec`.
The source checkpoint was never used as an output path.

Both runs used AdamW, batch size 128, weight decay 0.0001, no scheduler, value
validation loss for checkpoint selection, the unchanged 8-channel/1-block/16-
unit value-head Mini architecture, and CUDA on an NVIDIA GeForce RTX 4060 with
PyTorch 2.8.0+cu128 and CUDA runtime 12.8. Each `summary.json` contains the
complete training and model configuration, dataset digest, device metadata,
performance measurements, and every epoch; each `metrics.jsonl` retains the
same per-epoch metrics independently.

| Candidate | Seed | Schedule | Best validation value loss | Best checkpoint SHA-256 | Latest checkpoint SHA-256 | Decision |
| --- | ---: | --- | ---: | --- | --- | --- |
| [`value-selected-lr-1e-4`](value-selected-lr-1e-4/summary.json) | 125100 | 20 epochs, LR 0.0001 | 0.4239549535 (epoch 2) | `7a1830a71be9edce4a488bf3047151e8303aa980cec8ca2803198cd71cd6c733` | `7b806ac01512feb3f37bfe2d267078623e125e7d07211ae2c8eb7ee7c3a843fb` | inconclusive/rejected |
| [`value-selected-lr-3e-4`](value-selected-lr-3e-4/summary.json) | 125200 | 12 epochs, LR 0.0003 | 0.4257806858 (epoch 1) | `3e6c1dbba9761faa3c14b46b60b1a6037d758b3789431bda023d8eae327166a2` | `3ae7219dcfa3e31496aa0622d2bbf20bb4e08b920f4de1128e7f435b1eb9bacd` | negative/rejected |

The first run was effectively tied with the champion's 0.423941 validation
value loss and became worse after epoch 2, so the second schedule was tried.
The second was worse from epoch 1 and continued to regress. This supports the
earlier diagnosis that checkpoint selection and a gentler schedule alone do
not repair the value bottleneck on the same targets; changing target quality,
sampling, or value-head capacity is a higher-leverage next step.

The commands differed only in output, epochs, learning rate, and seed:

```bash
PYTHONHASHSEED=0 PYTHONPATH=src python3 -m twixt_ai.training.train_cli \
  --dataset experiments/issue-118/generation-2/generation-0001/dataset \
  --output-dir experiments/issue-125/candidates/value-selected-lr-1e-4 \
  --epochs 20 --batch-size 128 --learning-rate 0.0001 \
  --weight-decay 0.0001 --seed 125100 --device cuda \
  --selection-metric value --channels 8 --residual-blocks 1 \
  --value-hidden 16 \
  --initial-checkpoint \
    experiments/issue-118/generation-2/generation-0001/candidate/best.pt

PYTHONHASHSEED=0 PYTHONPATH=src python3 -m twixt_ai.training.train_cli \
  --dataset experiments/issue-118/generation-2/generation-0001/dataset \
  --output-dir experiments/issue-125/candidates/value-selected-lr-3e-4 \
  --epochs 12 --batch-size 128 --learning-rate 0.0003 \
  --weight-decay 0.0001 --seed 125200 --device cuda \
  --selection-metric value --channels 8 --residual-blocks 1 \
  --value-hidden 16 \
  --initial-checkpoint \
    experiments/issue-118/generation-2/generation-0001/candidate/best.pt
```
