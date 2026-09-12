# Issue 128 matched 1k stage

This directory records the fresh 1,000-game reference stage from the fixed
Issue 128 scaling contract. Self-play and candidate training both started from
the immutable generation-3 checkpoint
`aee1036dbda115eeec0e245909d30e1f8330454a82099e852e1b6a9c26c0dab9`.
The resulting candidate is retained as this stage's result only: it does not
replace that checkpoint as the teacher or training initialization for any
later scaling stage.

The CUDA run completed 1,000 games in 441.574 seconds (8,152.7 games/hour)
and produced 25,535 examples. The dataset manifest SHA-256 is
`9d79b587e041f044084d8f14933b98baca31b9fbc0c799b358322e6160d5d6f7`.
Mean policy support was 23.55, mean maximum target probability was 0.385, and
mean target entropy was 2.280 nats. Value targets contained 11,172 losses,
2,853 draws, and 11,510 wins from the recorded position's perspective.

Training warm-started the immutable teacher and ran all 20 AdamW epochs on
CUDA. Value-loss selection chose epoch 9 at validation value loss 0.299410;
the selected checkpoint SHA-256 is
`5320f9ca4c146b1c925054ac4bd1e5e605d5f228c2dfb1d8c2a8a490d0981af2`.
Under the fixed seed-1289000 paired gate, the candidate scored 30-8-2 against
the starting champion and cleared the predeclared 22-win threshold. This gate
result does not change the fixed-teacher rule above.

[`report.md`](report.md) renders the complete configuration, hashes, lineage,
timings, losses, evaluation games, and retention inventory. The raw games,
derived JSONL shards, and recovery-only `latest.pt` remain outside Git. Their
544,379,916-byte inventory is recorded against
`s3://twixt-ai/issue-128/matched-1k` with inventory SHA-256
`51b183947f4995b068b40dc7a8de51f7c384d21001e3727806cc01394ebbaff8`.
No storage attestation is recorded, so the report correctly marks the
inventory as not pruning-ready.

Reproduce the stage from the repository root with the complete command frozen
in the [Issue 128 scaling contract](../../../docs/issue-128-strength-scaling-contract.md),
using `experiments/issue-128/matched-1k` as the output directory and
`s3://twixt-ai/issue-128/matched-1k` as the artifact URI.
