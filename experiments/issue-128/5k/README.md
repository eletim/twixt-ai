# Issue 128 5k stage

This directory records the required 5,000-game stage from the fixed Issue 128
scaling contract. Self-play and candidate training both started from the
immutable generation-3 checkpoint
`aee1036dbda115eeec0e245909d30e1f8330454a82099e852e1b6a9c26c0dab9`;
the matched-1k candidate was used only as an evaluation opponent.

The CUDA run completed 5,000 games in 2,304.013 seconds (7,812.5 games/hour)
and produced 130,825 examples. The dataset manifest SHA-256 is
`d0ab6252d29945b1e25735615460cc42f9528191ce46e5e4e2c25f29d2360af8`.
Mean policy support was 23.46, mean maximum target probability was 0.384, and
mean target entropy was 2.276 nats. Value targets contained 55,857 losses,
17,408 draws, and 57,560 wins from the recorded position's perspective.

Training warm-started the immutable teacher and ran all 20 AdamW epochs on
CUDA. Value-loss selection chose epoch 12 at validation value loss 0.321488;
the selected checkpoint SHA-256 is
`60087bdb8c11fdd04c665775b6dc0e5595b206a8cac26859f4330ad8bc14db19`.
Under the fixed seed-1289000 paired gate, it scored 29-10-1 against the
starting champion and passed the 22-win promotion threshold.

The same 40-game paired schedule produced 19-21-0 against the retained
matched-1k candidate, 34-3-3 against matched non-neural MCTS, and 14-26-0
against the unchanged heuristic search. The 19 wins against matched-1k miss
the predeclared 22-win meaningful-scaling-gain gate, so the result is retained
as negative saturation evidence and the optional 10k stage is not run.

[`report.md`](report.md) renders the complete configuration, hashes, lineage,
timings, losses, evaluation games, and retention inventory. Raw games, derived
JSONL shards, and the recovery-only `latest.pt` remain outside Git at
`file:///home/eletim/twixt-ai-artifacts/issue-128/5k/generation-0001`. An independent
byte-for-byte audit verified all 5,037 objects and 2,769,910,005 bytes against
inventory SHA-256
`5447d23e68d0df76348c4077d502a8e5fd227f55f236349544e9d1abdbbc03e1`.

The stage used the complete command frozen in the
[Issue 128 scaling contract](../../../docs/issue-128-strength-scaling-contract.md),
changing only game volume, stage seed, output directory, and artifact URI from
the matched-1k invocation.
