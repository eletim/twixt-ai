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
twixt-ai-mini-report /path/to/stage-run --output /path/to/stage-report.md
```

Commit the contract, aggregate reports, configs, manifests and shard hashes,
training summaries/metrics, each attempted stage's best checkpoint, and all
evaluation results—including rejection and saturation evidence. Raw games,
derived JSONL shards, and recovery-only checkpoints belong in durable external
artifact storage rather than Git at these scales. Before local pruning, record
the external URI, hashes, file counts, and bytes so the retained evidence can
be verified and the datasets can be recovered.

Any run that changes the teacher, board/rules, encoding, architecture, search,
training, or evaluation semantics is a separately labelled diagnostic and
must not appear on the primary strength-vs-data curve.
