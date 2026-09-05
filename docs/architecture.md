# Architecture

## Principles

`twixt-ai` is an AI project first and a web application second. Python is the
source of truth for board state, legal moves, link placement, turn order,
terminal-state detection, and scoring. The engine and every AI workflow must
run in a headless process. A browser may display or request actions, but it must
not decide whether an action is valid or reproduce game semantics.

PyTorch is the default machine-learning framework. The intended progression is:

1. random agent;
2. heuristic and conventional search agents;
3. Monte Carlo tree search (MCTS);
4. MCTS guided by a small CNN/ResNet-style policy/value network.

A Transformer is not part of the initial learned baseline. This sequence keeps
each stronger agent measurable against a simpler, deterministic baseline.

## Modules and responsibilities

| Path | Owns | May depend on |
| --- | --- | --- |
| `twixt_ai.game` | Immutable state, moves, rules, transitions, terminal results, and canonical serialization | Python standard library only |
| `twixt_ai.agents` | The agent protocol plus random and heuristic agents | `game` |
| `twixt_ai.search` | Search trees, policies, MCTS, and reusable search utilities | `game`, `agents` protocols |
| `twixt_ai.models` | PyTorch policy/value networks, tensor encoding, and checkpoints | `game`; PyTorch |
| `twixt_ai.selfplay` | Headless game generation and trajectory records | `game`, `agents`, `search`, `models` |
| `twixt_ai.training` | Dataset preparation, optimization, checkpoint production, and training commands | `game`, `models`, `selfplay` |
| `twixt_ai.evaluation` | Headless matches, tournaments, metrics, and promotion decisions | `game`, `agents`, `search`, `models` |
| `twixt_ai.backend` | HTTP boundary, request validation, session orchestration, and conversion to/from canonical Python values | all Python modules as needed |
| `ui/` | Board rendering, controls, and inspection of API responses | backend HTTP API only |

The table is a dependency direction, not merely a directory list. In
particular, `game` never imports an agent, framework, backend, or UI module;
AI/evaluation modules never import browser code; and no Python module imports
from `ui/`.

The `agents` package owns the narrow agent interface (choose a move from a
state). Search implementations can satisfy that interface without placing
search concerns in the engine. Neural-network code stays in `models`, keeping
PyTorch optional for engine-only consumers.

## State and execution boundaries

The canonical state is a Python value produced by `twixt_ai.game`. All state
changes pass through an engine transition function, including moves submitted
by a human through the UI. Agents receive state snapshots and return proposed
moves; the engine validates and applies those moves. Self-play records canonical
states, actions, and outcomes rather than DOM or presentation data.

Training consumes persisted self-play trajectories and produces versioned model
checkpoints. Evaluation loads explicit agent/checkpoint configurations and emits
machine-readable results. Both are command-line/headless workflows and do not
call the backend.

## Backend and web UI

The first transport is a small JSON-over-HTTP API. Request/response endpoints
for creating a game, reading its state, submitting a move, and requesting an AI
move are sufficient for turn-based play; polling or a later event stream can be
added only if needed. Wire objects carry coordinates, player identifiers, links,
and result data, while the backend converts them to canonical Python types.

The server rejects illegal moves using `twixt_ai.game` and returns the updated
canonical state. Frontend checks may improve interaction feedback, but they are
never authoritative. Training, self-play, and evaluation APIs are deliberately
excluded from the initial browser boundary.

## Initial directory layout

```text
.
├── docs/
│   └── architecture.md
├── src/twixt_ai/
│   ├── game/
│   ├── agents/
│   ├── search/
│   ├── models/
│   ├── selfplay/
│   ├── training/
│   ├── evaluation/
│   └── backend/
├── tests/
└── ui/
```

Code shared by two domains should live in the lowest appropriate Python module,
not in a generic utilities package. Executable entry points should be thin:
they parse configuration, call package code, and serialize results. Tests mirror
the package layout; cross-module tests additionally enforce headless imports and
end-to-end game/agent behavior.
