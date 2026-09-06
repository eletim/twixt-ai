# Mini encoding cost comparison

The `twixt-ai-encoding-comparison` benchmark compares the version 1 22-plane
encoding with the side-to-move normalized version 2 10-plane encoding. Both
paths read the same versioned 10×10 training dataset and use identical trunk,
policy-head, value-head, batch, optimizer, seed, and thread settings. Only the
input plane count and encoding version differ.

Run the checked-in workload with:

```bash
PYTHONHASHSEED=0 twixt-ai-encoding-comparison \
  --dataset experiments/issue-56/baseline/dataset \
  --device cpu --device cuda \
  --output benchmarks/mini-encoding-comparison.json
```

Repeat `--device` to measure more than one device, for example `--device cpu
--device cuda`. With no device flags, the command measures CPU and also CUDA
when CUDA is available. CUDA timings are synchronized, and the report records
peak allocated device memory. The CPU report records exact input tensor and
model storage; it does not claim to measure process RSS or allocator overhead.

The versioned JSON artifact keeps encoding latency separate from single-position
and batched model-forward latency. It also reports matched training-step
throughput, encoded tensor bytes, parameter and buffer bytes, parameter counts,
the dataset manifest digest, model/workload configuration, Git revision, and
Python, PyTorch, CPU, and accelerator details. Positive throughput-change
percentages favor the 10-plane path; positive reduction percentages mean the
10-plane path used less time or storage.

These measurements quantify computational cost only. They do not support a
playing-strength claim; strength requires a controlled paired evaluation.

## Issue 74 measurement

The checked-in [benchmark artifact](../benchmarks/mini-encoding-comparison.json)
encoded all 1,961 training positions per repeat from the Issue 56 Mini baseline.
Forward and training steps reused the first 32 examples under both encodings,
giving a fixed matched batch. The host was an AMD Ryzen 9 7900X with an NVIDIA
RTX 4060 and PyTorch 2.8.0; PyTorch used one host thread. Selected results are:

| Metric | 22-plane v1 | 10-plane v2 |
| --- | ---: | ---: |
| Encoding latency (µs/position) | 86.4 | 69.0 |
| Encoded bytes/position | 8,800 | 4,000 |
| Model parameters | 24,547 | 23,683 |
| CPU single forward (positions/s) | 7,900 | 8,998 |
| CPU batch-32 forward (positions/s) | 51,633 | 65,130 |
| CPU training (positions/s) | 12,956 | 14,370 |
| CUDA single forward (positions/s) | 4,708 | 4,808 |
| CUDA batch-32 forward (positions/s) | 152,866 | 150,793 |
| CUDA training (positions/s) | 29,570 | 29,303 |

On this run, compact encoding reduced tensor storage by 54.5%, encoding latency
by 20.0%, and model parameters by 3.5%. CPU forward and training throughput
improved, while CUDA forward and training were effectively unchanged (within
2.2%). Timing results are samples of this machine and should
be rerun when hardware, PyTorch, or workload configuration changes.
