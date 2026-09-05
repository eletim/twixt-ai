# Game state and replay format

The engine persists positions and complete games as compact, deterministic
JSON. Writers sort object keys, use no insignificant whitespace, and emit
canonical peg/link ordering. Readers validate every field and reject unknown
format versions rather than guessing how to interpret them.

## Position format, version 1

A position uses `format: "twixt-ai-state"` and `version: 1`, followed by the
board dimensions, canonical peg and link arrays, side to move, and result:

```json
{"board":{"height":6,"width":6},"format":"twixt-ai-state","links":[],"pegs":[],"result":"in_progress","side_to_move":"red","version":1}
```

Use `GameState.to_json()` to save a position and `GameState.from_json()` to
load it.

## Game record format, version 1

A game record uses `format: "twixt-ai-game-record"` and `version: 1`. It stores
the canonical starting position, the ordered peg-placement history, and the
canonical final position. Including both positions permits a record to start
from an arbitrary saved position and makes its replay result independently
checkable.

```json
{
  "format": "twixt-ai-game-record",
  "version": 1,
  "initial_state": {"format": "twixt-ai-state", "version": 1, "board": {"height": 4, "width": 4}, "pegs": [], "links": [], "side_to_move": "red", "result": "in_progress"},
  "moves": [{"player": "red", "coordinate": {"x": 1, "y": 1}}],
  "final_state": {"format": "twixt-ai-state", "version": 1, "board": {"height": 4, "width": 4}, "pegs": [{"owner": "red", "coordinate": {"x": 1, "y": 1}}], "links": [], "side_to_move": "black", "result": "in_progress"}
}
```

Create a verified record with `GameRecord.from_moves(initial_state, moves)`.
`GameRecord.from_json()` validates the schema, replays every move through the
authoritative transition function, and rejects illegal histories or a stored
final position that differs from the replay result. `GameRecord.replay()`
returns the reproduced final state.

Future incompatible schemas must use a new integer `version`. Version 1
readers intentionally reject those records until explicit migration support is
implemented.
