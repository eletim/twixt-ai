# Web UI

This is a thin, dependency-free play and inspection client. It renders the
canonical state returned by `twixt_ai.backend`; every attempted move is sent to
the Python engine for validation and transition.

From an installed development checkout, run:

```bash
twixt-ai-web
```

Then open <http://127.0.0.1:8000>. Pass `--host` or `--port` to change the bind
address. The server keeps one human-vs-human game in memory and reset preserves
its board dimensions.
