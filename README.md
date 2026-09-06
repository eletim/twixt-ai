# twixt-ai

An AI-first implementation of [Twixt](https://en.wikipedia.org/wiki/TwixT).
The canonical game and AI stack lives in Python and runs without a browser; the
web application is only a play and inspection client.

The project is at the architecture-skeleton stage. See
[docs/architecture.md](docs/architecture.md) for responsibility boundaries,
dependency rules, the planned AI progression, and the initial directory layout.
The persisted position and replay schemas are documented in
[docs/game-record-format.md](docs/game-record-format.md).
The checkpoint-stable CNN feature planes and augmentation transforms are defined
in [docs/neural-input-encoding.md](docs/neural-input-encoding.md).
The reproducible 10-plane versus 22-plane Mini cost benchmark is documented in
[docs/encoding-comparison.md](docs/encoding-comparison.md).
The versioned, reproducible training shard schema is defined in
[docs/training-data-format.md](docs/training-data-format.md).
The model training, metrics, and resume workflow is documented in
[docs/model-training.md](docs/model-training.md).

## Experiment presets

All browser and headless workflows share named board presets. `standard` is
the compatible 24×24 default and `mini` is the 10×10 Mini Twixt experiment;
both use the canonical v0.0.1 automatic-link rules with no pie/swap rule.
Select Mini Twixt in the browser's Board control, or use `--preset mini` with
the match, self-play, agent benchmark, and engine benchmark commands. Explicit
`--width` and `--height` values override individual preset dimensions.

For example, a complete reproducible Mini pipeline starts with:

```bash
twixt-ai-selfplay --preset mini --games 100 --workers 4 --seed 1234 \
  --red mcts --black mcts --output-dir mini-selfplay
twixt-ai-dataset --input mini-selfplay --output-dir mini-dataset
twixt-ai-train --dataset mini-dataset --output-dir mini-training --seed 1234
```

The first measured 100-game Mini MCTS dataset and its exact reproduction command
are documented in
[`docs/mini-dataset-experiment.md`](docs/mini-dataset-experiment.md).
The first learned Mini model, optimization sanity checks, and measured training
run are documented in
[`docs/mini-training-experiment.md`](docs/mini-training-experiment.md).
Its paired strength evaluation against non-neural baselines, including
policy-only and value-only ablations, is documented in
[`docs/mini-strength-evaluation.md`](docs/mini-strength-evaluation.md).
The repeatable self-play, windowed training, paired evaluation, and explicit
checkpoint-promotion loop is documented in
[`docs/mini-training-generations.md`](docs/mini-training-generations.md).
Summarize one of those runs, including checkpoint lineage, promotion-driven
champion changes, and fixed-position policy/value probes, with
`twixt-ai-mini-report RUN`.

Match, self-play, and benchmark artifacts record their board dimensions. The
dataset manifest, training summary, and policy/value checkpoints also carry the
dimensions, so 10×10 and 24×24 artifacts cannot be silently mixed.

## Development

Python 3.10 or newer is required. From a checkout:

```bash
python -m pip install -e '.[dev]'
pytest
```

Engine-only installations have no PyTorch dependency. Install `.[models]` when
using the neural encoding or learned-model package without development tools.

Importing `twixt_ai.game`, `twixt_ai.agents`, or `twixt_ai.evaluation` never
requires the backend or UI.

## Headless matches

Run a reproducible random-agent match and emit a machine-readable JSON artifact:

```bash
twixt-ai-match --red random --black random --seed 1234 --output match.json
```

Use `--preset mini` for 10×10, `--width` and `--height` for custom boards, or omit `--output` to
write the artifact to standard output. Python callers can use
`twixt_ai.evaluation.run_match` with any agents that implement the common
agent protocol.

Generate a reproducible batch of games in parallel with one match artifact per
game and an aggregate `summary.json` manifest:

```bash
twixt-ai-selfplay --games 100 --workers 4 --seed 1234 \
  --red random --black random --output-dir selfplay-run
```

The `random`, bounded heuristic `search`, and Monte Carlo `mcts` agents are
available from the CLI.
Failures are isolated to their game and recorded both beside successful game
artifacts and in the summary. Python callers can supply any agent factories to
`twixt_ai.selfplay.run_batch`.

Convert a completed run into deterministic train/validation JSONL shards:

```bash
twixt-ai-dataset --input selfplay-run --output-dir dataset \
  --shard-size 10000 --validation-fraction 0.1 --split-seed experiment-1
```

Splits are assigned at game granularity, and examples retain match
configuration, decision seeds, and agent metadata. MCTS visit counts are
normalized into policy targets when present.

Train the policy/value network with reproducible shuffling, recorded optimizer
and scheduler settings, and resumable best/latest checkpoints:

```bash
twixt-ai-train --dataset dataset --output-dir training-run \
  --epochs 20 --batch-size 64 --learning-rate 0.001 --seed 1234
```

Compare two agents head-to-head, or repeat `--agent` three or more times for a
round robin:

```bash
twixt-ai-benchmark --agent baseline=random --agent candidate=search \
  --games-per-pair 20 --seed 1234 --elo --output benchmark.json
```

Games are paired under the same seed with red/black roles swapped. The JSON
artifact records each entrant's package version and construction settings,
every game seed, W/L/D and color splits, Wilson confidence intervals, explicit
first-player results, and (when requested) Elo-style ratings. Search entrants
use `--search-depth` and `--node-budget`; these settings apply to every search
entrant in a CLI run. Python callers can supply distinct factories and recorded
configurations for each entrant.

Profile the canonical engine with a deterministic mid-game workload:

```bash
PYTHONHASHSEED=0 twixt-ai-engine-benchmark --output engine-benchmark.json
```

The artifact reports legal-move, link-update, win-check, and full-transition
rates, including positions per second. See
[`docs/engine-performance.md`](docs/engine-performance.md) for methodology,
baseline results, and identified hotspots.

Reproduce the separate NN-free 10×10 Mini Twixt self-play baseline with one
command:

```bash
PYTHONHASHSEED=0 twixt-ai-selfplay-benchmark \
  --output mini-selfplay-performance.json
```

The versioned JSON report measures legal moves and move application, MCTS move
latency plus simulations/nodes per second at 100/400/1600 simulations, complete
game latency and games/hour, CPU utilization, and fixed-workload scaling from
one to two workers. It also projects the measured wall-clock cost of 1,000 and
10,000 games and identifies the slowest measured paths. See
[`docs/mini-performance.md`](docs/mini-performance.md) for methodology and the
checked-in baseline. The existing `twixt-ai-engine-benchmark` command remains
the independent standard 24×24 engine workload.

MCTS is the primary non-neural search baseline. It uses a reproducible
simulation-count budget, seeded random rollouts with a finite default horizon,
heuristic evaluation at non-terminal cutoffs, and progressive widening so PUCT
revisits actions on the wide standard board. It reports root visit/value
statistics suitable for evaluation and later training targets:

```python
from twixt_ai.search import MCTSAgent

agent = MCTSAgent(simulations=400, rollout_limit=4)
```

Use `--agent candidate=mcts --simulations 400 --rollout-limit 4` in the
benchmark CLI; self-play accepts the same budget flags. A `policy_value`
callback can supply move priors and leaf values without replacing the MCTS
orchestration layer. The initial learned path uses a small residual PyTorch
network and an inference adapter:

```python
from twixt_ai.models import load_policy_value_checkpoint
from twixt_ai.search import MCTSAgent
from twixt_ai.search.neural import NeuralPolicyValue

checkpoint = load_policy_value_checkpoint("model.pt")
agent = MCTSAgent(
    simulations=400,
    policy_value=NeuralPolicyValue(checkpoint.model),
)
```

The network emits raw logits for training; the adapter encodes canonical
positions, masks illegal moves, normalizes legal priors, and supplies a value
in `[-1, 1]` from the side-to-move perspective. Checkpoints record encoding,
architecture, and configuration versions and reject incompatible loads. See
[`docs/policy-value-network.md`](docs/policy-value-network.md) for the action
mapping and checkpoint contract. Python callers may explicitly set
`rollout_limit=None` for terminal playouts.

`twixt_ai.search.HeuristicSearchAgent` (also available as `SearchAgent`) uses
the shared position heuristic with alpha-beta minimax. Its search limits and
move ordering are configurable:

```python
from twixt_ai.search import HeuristicSearchAgent

agent = HeuristicSearchAgent(depth=2, node_budget=20_000)
```

## Browser play

Start the minimal local human-vs-agent UI after installing the project:

```bash
twixt-ai-web
```

Open <http://127.0.0.1:8000>, choose a side and an available agent, then start a
new game. The browser only renders state and submits human/agent turn requests;
the canonical Python engine owns validation, links, turns, results, and agent
selection. Session revisions prevent delayed browser clicks from changing a
newer position. Enable **AI inspection** to overlay candidate scores or
probabilities and the selected move, plus the agent-provided value estimate and
search statistics. The overlay is off by default and does not evaluate moves in
the browser.
