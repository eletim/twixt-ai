# twixt-ai

An AI-first implementation of [Twixt](https://en.wikipedia.org/wiki/TwixT).
The canonical game and AI stack lives in Python and runs without a browser; the
web application is only a play and inspection client.

The project is at the architecture-skeleton stage. See
[docs/architecture.md](docs/architecture.md) for responsibility boundaries,
dependency rules, the planned AI progression, and the initial directory layout.
The persisted position and replay schemas are documented in
[docs/game-record-format.md](docs/game-record-format.md).

## Development

Python 3.10 or newer is required. From a checkout:

```bash
python -m pip install -e '.[dev]'
pytest
```

Importing `twixt_ai.game`, `twixt_ai.agents`, or `twixt_ai.evaluation` never
requires the backend or UI.

## Headless matches

Run a reproducible random-agent match and emit a machine-readable JSON artifact:

```bash
twixt-ai-match --red random --black random --seed 1234 --output match.json
```

Use `--width` and `--height` for nonstandard boards, or omit `--output` to
write the artifact to standard output. Python callers can use
`twixt_ai.evaluation.run_match` with any agents that implement the common
agent protocol.

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
newer position.
