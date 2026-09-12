# Issue 128 Mini strength-scaling contract

The machine-readable source of truth is
[`experiments/issue-128/scaling-contract.json`](../experiments/issue-128/scaling-contract.json).
The primary curve freezes the current generation-3 champion at SHA-256
`aee1036dbda115eeec0e245909d30e1f8330454a82099e852e1b6a9c26c0dab9`
as the teacher, training initialization, and fixed promotion opponent for every
stage. A candidate never becomes the teacher for a later data-volume stage.

## Fixed experiment

Every primary stage uses the 10x10 Mini rules, encoding version 1 with 22
side-to-move channels, and the version-1 Mini residual policy/value network
(8 trunk channels, one residual block, and 16 value-hidden units). CUDA
self-play uses one shared model, eight game threads, batches of eight, and a
2 ms queue wait. Both colors use policy+value MCTS with 64 simulations,
exploration 0.7, progressive widening 3.0/0.5, and rollout limit 4. Moves use
the maximum root visit count with the engine's stable tie breaks; no sampling
temperature or root noise is applied.

Each stage builds only its own dataset with a game-level 90/10 split and
5,000-example shards. Training independently warm-starts the frozen champion
and runs 20 AdamW epochs with batch size 128, learning rate 0.001, weight decay
0.0001, no scheduler, and value-loss checkpoint selection. Thus game count and
the declared fresh stage seed are the only differences on the primary curve.

The historical Issue 125 generation-3 data are explicitly excluded as the 1k
control: those games were taught by the generation-2 checkpoint
`742229c59caf251a07c7ecac6dc77ff75cbf09643a22cca16b92fe083df5a5ec`.
Issue 128 must first generate a fresh 1,000-game dataset taught by the frozen
generation-3 champion. The 1k and 5k stages are required; 10k and 25k are
conditional, and 50k is a stretch stage.

## Evaluation and stop decisions

Every candidate receives 40 games as 20 identical-seed role-swapped pairs
against the starting champion, the previous retained stage candidate when one
exists, matched non-neural MCTS, and the unchanged depth-1, 10,000-node
heuristic search. Learned and non-neural MCTS use 20 simulations and rollout
limit 4. All stage/opponent comparisons reuse evaluation seed 1289000.
The generation command must pass `--evaluation-seed 1289000`; this same seed
drives the executable candidate-versus-starting-champion promotion gate, rather
than the generation pipeline's default derivation from the stage root seed.

A candidate is promoted only with at least 22 wins in 40 games against the
starting champion; draws remain in the denominator. After the mandatory 1k
and 5k points, a later stage is justified only when the latest candidate also
wins at least 22 of 40 against the previous retained stage candidate. Failure
of that latter gate declares saturation and is retained as negative evidence;
loss or calibration improvements cannot override playing strength. The 50k
stage additionally requires the 25k gain gate, at most 12 projected self-play
hours, and at most 25 GiB projected retained external storage.

## Evidence and retention

Generation reports now include the hashes, wall times, throughput, target
distributions, and storage totals needed by the scaling table. The inspection
command renders those fields:

```bash
PYTHONHASHSEED=0 PYTHONPATH=src python3 -m twixt_ai.training.generations_cli \
  --initial-champion experiments/issue-125/generation-3/generation-0001/candidate/best.pt \
  --output-dir /path/to/issue-128/matched-1k \
  --artifact-uri s3://bucket/issue-128/matched-1k \
  --generations 1 --games-per-generation 1000 --dataset-window 1 \
  --selfplay-simulations 64 --selfplay-exploration 0.7 \
  --selfplay-progressive-widening-constant 3.0 \
  --selfplay-progressive-widening-exponent 0.5 \
  --evaluation-games 40 --evaluation-simulations 20 --rollout-limit 4 \
  --workers 8 --inference-batch-size 8 \
  --inference-max-wait-seconds 0.002 \
  --epochs 20 --batch-size 128 --learning-rate 0.001 \
  --weight-decay 0.0001 --selection-metric value \
  --validation-fraction 0.1 --shard-size 5000 \
  --promotion-win-rate 0.55 --seed 1281000 \
  --evaluation-seed 1289000 --device cuda

twixt-ai-mini-report /path/to/issue-128/matched-1k \
  --output /path/to/issue-128/matched-1k-report.md
```

This is the complete matched-1k invocation; it does not rely on workflow
defaults. For a later stage, change only `--games-per-generation`, `--seed`,
`--output-dir`, and `--artifact-uri` to that stage's values in the contract.
Every stage remains a separate `--generations 1 --dataset-window 1` run from
the same initial checkpoint, so no candidate can become another stage's
teacher or enter another stage's dataset.

Commit the contract, aggregate reports, configs, manifests and shard hashes,
training summaries/metrics, each attempted stage's best checkpoint, and all
evaluation results—including rejection and saturation evidence. Raw games,
derived JSONL shards, and recovery-only checkpoints belong in durable external
artifact storage rather than Git at these scales. `--artifact-uri` records the
durable base URI in a versioned retention manifest. That manifest inventories
each retained object's relative path, SHA-256, and bytes, plus categorized and
total file/byte counts. The inspection report renders both these objects and
every dataset-shard and fixed-opponent evaluation identity. Training marks only
`inventory_complete`; supplying the URI does not claim or verify a transfer.
After upload, a separate storage verifier must attest the same external URI and
inventory SHA-256 with its identity and verification time. Inspection derives
pruning readiness only when the inventory digest and that attestation both
validate. Validation requires all four non-empty categories, safe unique
relative object paths, lowercase SHA-256 values, non-negative byte sizes, and
exact category/global file and byte rollups. Without a structurally complete
inventory and independent attestation, local pruning is forbidden.

Any run that changes the teacher, board/rules, encoding, architecture, search,
training, or evaluation semantics is a separately labelled diagnostic and
must not appear on the primary strength-vs-data curve.
