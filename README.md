# twixt-ai

An AI-first implementation of [Twixt](https://en.wikipedia.org/wiki/TwixT).
The canonical game and AI stack lives in Python and runs without a browser; the
web application is only a play and inspection client.

The project is at the architecture-skeleton stage. See
[docs/architecture.md](docs/architecture.md) for responsibility boundaries,
dependency rules, the planned AI progression, and the initial directory layout.

## Development

Python 3.10 or newer is required. From a checkout:

```bash
python -m pip install -e '.[dev]'
pytest
```

Importing `twixt_ai.game`, `twixt_ai.agents`, or `twixt_ai.evaluation` never
requires the backend or UI.
