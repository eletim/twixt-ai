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
when CUDA is available. Only CPU and available CUDA devices are accepted;
unsupported or duplicate device aliases are rejected. CUDA timings are
synchronized, and the report records peak allocated device memory. The CPU
report records exact input tensor and model storage; it does not claim to
measure process RSS or allocator overhead.

The versioned JSON artifact keeps encoding latency separate from single-position
and batched model-forward latency. Each workload uses seven paired samples by
default. The member measured first alternates between v1 and v2 on each sample,
and every timed model sample starts from a fresh, deterministically initialized
model. Reports include every raw sample, the median, and the minimum, maximum,
and median absolute deviation for wall time, latency, and throughput.

The artifact also reports matched training-step throughput, encoded tensor
bytes, parameter and buffer bytes, parameter counts, the dataset manifest
digest, model/workload configuration, Git revision, and Python, PyTorch, CPU,
and accelerator details. Positive throughput-change percentages favor the
10-plane path; positive reduction percentages mean the 10-plane path used less
time or storage. Timing percentage comparisons are calculated within each
paired sample and report the median, range, and median absolute deviation.

These measurements quantify computational cost only. They do not support a
playing-strength claim; strength requires a controlled paired evaluation.

## Issue 74 measurement

The checked-in [benchmark artifact](../benchmarks/mini-encoding-comparison.json)
encoded all 1,961 training positions per repeat from the Issue 56 Mini baseline.
Forward and training steps reused the first 32 examples under both encodings,
giving a fixed matched batch. The host was an AMD Ryzen 9 7900X with an NVIDIA
RTX 4060 and PyTorch 2.8.0; PyTorch used one host thread. Selected results are:

Values below are medians with the seven-sample minimum–maximum range in
parentheses.

| Metric | 22-plane v1 | 10-plane v2 |
| --- | ---: | ---: |
| Encoding latency (µs/position) | 86.4 (86.1–86.9) | 75.9 (75.6–76.7) |
| Encoded bytes/position | 8,800 | 4,000 |
| Model parameters | 24,547 | 23,683 |
| CPU single forward (positions/s) | 8,482 (7,959–8,555) | 8,983 (8,527–9,165) |
| CPU batch-32 forward (positions/s) | 52,371 (51,263–53,185) | 65,826 (62,223–66,456) |
| CPU training (positions/s) | 12,982 (12,671–13,104) | 14,573 (14,154–14,596) |
| CUDA single forward (positions/s) | 4,846 (4,657–4,890) | 4,939 (4,853–5,036) |
| CUDA batch-32 forward (positions/s) | 153,353 (150,530–155,973) | 151,816 (150,118–153,666) |
| CUDA training (positions/s) | 30,940 (25,872–31,305) | 30,984 (28,151–31,596) |

Paired-sample medians show that compact encoding reduced tensor storage by
54.5%, encoding latency by 12.1% (0.3 percentage-point MAD), and model
parameters by 3.5%. CPU throughput increased by 5.8% for single forwards, 23.8%
for batch-32 forwards, and 11.6% for training; every paired CPU sample was
positive. CUDA changes were not material: +1.9% for single forwards, −1.0% for
batch-32 forwards, and +1.4% for training. CUDA training was the noisiest result
and its paired range crossed zero (−1.5% to +19.8%, 2.0 percentage-point MAD).
Timing results are samples of this machine and should be rerun when hardware,
PyTorch, or workload configuration changes.
