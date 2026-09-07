# Training dataset format

`twixt-ai-dataset` converts version 1 headless match artifacts, including the
artifacts emitted by `twixt-ai-selfplay`, without importing the web backend or
UI. A build writes `manifest.json` plus deterministic JSON Lines shards under
`train/` and `validation/`.

## Dataset manifest, version 1

The manifest uses `format: "twixt-ai-training-dataset"` and `version: 1`. It
records the common source-game board dimensions in `board`; mixed-size sources
are rejected. It
records the shard size, validation fraction, split seed, caller metadata,
source-game and example counts, and each shard's relative path, row count, and
SHA-256 digest. Shard digests allow consumers to detect incomplete or modified
datasets.

Games, rather than individual positions, are assigned to a split by hashing
the match's canonical content digest together with `split_seed`. This makes the
split independent of input order and prevents positions from one game leaking
across train and validation sets. Examples and games are ordered by their
content-derived identifiers before sharding.

## Training example, version 1

Every JSONL row uses `format: "twixt-ai-training-example"` and `version: 1` and
contains:

- `id`: the source game SHA-256 plus the zero-based ply;
- `position`: the canonical versioned position before the action;
- `action`: the played coordinate;
- `outcome`: `1`, `0`, or `-1` from the position's side-to-move perspective;
- `source`: source game ID, ply, full match configuration, and the decision's
  seed and agent metadata;
- `policy`, when MCTS `root_moves` metadata has positive visit counts: a sparse
  coordinate/probability array normalized from those counts.

Unknown match versions, invalid match configurations or decision seeds, invalid
replay histories, misaligned decisions, and malformed search statistics are
rejected instead of being silently converted. Seeded matches must contain the
exact per-ply seed sequence derived from the match seed; unseeded matches must
contain only null decision seeds. An MCTS policy target requires a positive
integer `simulations` value equal to the sum of all root visit counts.

Build a dataset from a self-play run with:

```bash
twixt-ai-dataset --input selfplay-run --output-dir dataset \
  --shard-size 10000 --validation-fraction 0.1 --split-seed experiment-1
```

Python callers can use `twixt_ai.training.build_dataset` and attach
JSON-compatible version or experiment information through
`DatasetConfig(metadata=...)`.
