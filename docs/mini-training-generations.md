# Iterative Mini training generations

The Issue 59 workflow turns the one-shot Mini experiments into a small,
inspectable AlphaZero-style loop. Each generation performs these stages in
order:

1. Generate self-play games using the current champion checkpoint.
2. Build a training dataset from the newest configured window of generations.
3. Warm-start and train a candidate from the current champion.
4. Run a fixed, paired candidate-versus-champion evaluation with roles swapped.
5. Promote the candidate only when its wins divided by all evaluation games is
   at least the configured threshold. Draws stay in the denominator.

The default schedule runs two generations, retains up to five generations of
self-play, and requires a 55% candidate win rate. Seeds for every stage are
derived from the recorded root seed. Set `PYTHONHASHSEED=0` so data ordering and
the complete schedule are reproducible.

Use `--evaluation-seed` when a protocol requires the same promotion schedule
across otherwise independent runs. When omitted, the original per-generation
derivation from `--seed` remains in effect. The resolved seed is recorded in
both the generation's `seeds` and `resolved_config` objects.

Self-play search strength can be changed independently of the fixed promotion
evaluation with `--selfplay-exploration` and the two
`--selfplay-progressive-widening-*` options. The generation report records the
resolved search settings, self-play summary and dataset manifest SHA-256,
policy-target support, entropy and mean maximum probability, and value-target
counts/fractions. It also records per-stage wall time, self-play and training
throughput, evaluation artifact identity, and retained file/byte totals. These
diagnostics make it possible to reject a search configuration that plays
strongly but emits nearly uniform or badly imbalanced training targets.
`--selection-metric` controls whether candidate checkpoint selection uses
combined policy/value loss or value loss.

## Run

The examples below record how the architecture-v1 lineage was run. The current
architecture-v2 loader intentionally rejects the referenced Issue 57 checkpoint
and all later champions through Issue 128 because the widened heads have
different weights. To continue the workflow with current code, first bootstrap
a fresh architecture-v2 champion and supply that checkpoint instead; do not use
a pre-v2 checkpoint as `--initial-champion`.

The historical lineage started from the measured Issue 57 Mini checkpoint:

```bash
PYTHONHASHSEED=0 twixt-ai-mini-generations \
  --initial-champion experiments/issue-57/baseline/best.pt \
  --output-dir mini-generations
```

For a quick two-generation smoke run, reduce the work while keeping the same
stage boundaries:

```bash
PYTHONHASHSEED=0 twixt-ai-mini-generations \
  --initial-champion experiments/issue-57/baseline/best.pt \
  --output-dir mini-generations-smoke \
  --generations 2 --games-per-generation 2 \
  --selfplay-simulations 1 --evaluation-games 2 \
  --evaluation-simulations 1 --epochs 1 --workers 1
```

Output directories must be empty or absent. The workflow never overwrites or
renames a champion checkpoint. A promoted candidate becomes the input path for
the next generation; a rejected candidate remains under its generation
directory.

For large runs, pass `--artifact-uri` with the durable base location. The
generation report then includes a retention manifest with categorized
file/byte totals and a relative path, SHA-256, and byte size for every retained
object. Generation records `inventory_complete` and a canonical inventory
SHA-256, but never claims that an external transfer succeeded. After transfer,
a separate storage verifier must add an attestation matching the external URI
and inventory digest; inspection reports pruning readiness only when both the
inventory and that attestation validate.

## Artifacts and recovery

`config.json` records the immutable schedule, while the root `report.json`
records overall status, runtime, checkpoint identities, and lineage. Each
`generation-NNNN` directory contains self-play games, its windowed dataset, all
candidate training checkpoints and metrics, `evaluation.json`, and a generation
`report.json` with stage runtimes and the promotion decision.

The root and generation reports are atomically refreshed at stage boundaries.
If a stage fails, the relevant report identifies the failed stage and exception
while all completed games, datasets, checkpoints, and rejected candidates stay
available for inspection. Recovery uses a new empty output directory; an
existing run is never silently modified.

## Inspect a run

Generate a Markdown summary from the stored artifacts after a completed or
partially completed run:

```bash
twixt-ai-mini-report mini-generations --output mini-generations/report.md
```

Omit `--output` to print the report. The command identifies the exact source
report, complete configuration, checkpoint and artifact hashes and lineage;
summarizes stage timing, self-play/training throughput, dataset and retained
storage sizes, policy/value target distributions, loss curves, search budgets,
candidate-vs-parent promotion evaluations, and generation-over-generation
champion changes; and evaluates
every available checkpoint on the versioned `mini-fixed-positions-v1` probe
set. Checkpoints whose recorded hash does not match are flagged and never
evaluated. The command only reads source artifacts (apart from the explicitly
requested Markdown output), so those artifacts remain the source of truth.
