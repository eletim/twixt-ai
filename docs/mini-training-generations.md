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

## Run

Start from the measured Issue 57 Mini checkpoint:

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
report, complete configuration, checkpoint hashes and lineage; summarizes
self-play throughput, dataset sizes, loss curves, search budgets, promotion
evaluations, and generation-over-generation win-rate changes; and evaluates
every available checkpoint on the versioned `mini-fixed-positions-v1` probe
set. Checkpoints whose recorded hash does not match are flagged and never
evaluated. The command only reads source artifacts (apart from the explicitly
requested Markdown output), so those artifacts remain the source of truth.
