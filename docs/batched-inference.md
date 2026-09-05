# Batched policy/value inference

`NeuralInferenceBatcher` preserves the synchronous policy/value callback used
by `MCTSAgent` while collecting calls from concurrent search threads into one
PyTorch forward pass. It flushes when its configured batch size is reached or
when the maximum wait expires. Calling `flush()` forces queued work through;
using `batch_size=1` keeps a serialized synchronous path for tests and
debugging while retaining safe shutdown behavior for concurrent callers.

Concurrent games must use `BatchConfig(worker_mode="thread")` so their agent
factories can reference one in-process batcher. Process mode remains the
default for NN-free self-play and intentionally does not share Python objects.
A typical integration is:

```python
from functools import partial

from twixt_ai.search import MCTSAgent
from twixt_ai.search.neural import NeuralInferenceBatcher, NeuralPolicyValue
from twixt_ai.selfplay import BatchConfig, run_batch

with NeuralInferenceBatcher(NeuralPolicyValue(model), batch_size=16) as inference:
    factory = partial(MCTSAgent, simulations=400, policy_value=inference)
    run_batch(
        factory,
        factory,
        config=BatchConfig(games=32, workers=16, worker_mode="thread"),
        output_dir="selfplay",
    )
```

The batch size, wait threshold, seeded model initialization, device, and actual
batch counts are recorded by the reproducible benchmark:

```bash
PYTHONHASHSEED=0 twixt-ai-inference-benchmark \
  --output benchmarks/mini-inference-performance.json
```

The checked-in Issue 54 measurement used the available RTX 4060. It reports
positions per second for synchronous and batched paths, process CPU utilization,
an `nvidia-smi` GPU-utilization sample, peak CUDA allocation, and environment
details. The GPU sample is an instantaneous post-workload reading rather than a
time-weighted hardware trace; the throughput timings synchronize CUDA before
and after each workload.
