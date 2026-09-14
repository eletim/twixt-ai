# Architecture-v2 Mini champion bootstrap

`champion.pt` is the fresh starting champion for the architecture-v2 Mini
lineage. It was initialized on CPU with seed `143000`; it was not warm-started
from, converted from, or written over any architecture-v1 checkpoint. The
Issue 57 through Issue 128 lineage remains historical-only.

The checkpoint uses the 10x10, encoding-v1 (22-plane) Mini preset: an 8-channel
trunk with one residual block, a `800 -> 256 -> 256 -> 100` policy head, and a
`800 -> 256 -> 256 -> 1 -> tanh` value head. `manifest.json` records the full
model configuration, architecture name/version, PyTorch version, input/trunk/head
shapes, every state-dict tensor shape, and the checkpoint byte count and SHA-256.

Reproduce it with PyTorch 2.8.0+cu128 and the source at this revision, targeting
an absent or empty directory:

```bash
PYTHONPATH="$PWD/src" python3 -m twixt_ai.training.bootstrap_cli \
  --output-dir /tmp/architecture-v2-bootstrap \
  --seed 143000
sha256sum /tmp/architecture-v2-bootstrap/champion.pt
```

The expected SHA-256 is
`fd5a9ee7dcd6cd12af8429ab0370b858c5271f2a71afd20956a36ea21db3bb87`.
