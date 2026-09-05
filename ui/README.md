# Web UI

The thin, dependency-free play and inspection client is packaged from
`src/twixt_ai/ui` so an installed wheel contains everything the web executable
serves. It renders canonical state returned by `twixt_ai.backend`; every
attempted move is sent to the Python engine for validation and transition.

After installing the project, run:

```bash
twixt-ai-web
```

Then open <http://127.0.0.1:8000>. Pass `--host` or `--port` to change the bind
address. The server keeps one human-vs-human game in memory and reset preserves
its board dimensions.
