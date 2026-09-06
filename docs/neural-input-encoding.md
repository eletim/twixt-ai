# Neural input encodings

Encoding versions are checkpoint compatibility data. The original 22-plane
format remains version 1 and the compact side-to-move normalized format is
version 2. A checkpoint must record the version it was trained with; callers
must never select an encoder from tensor dimensions alone.

## Version 1: 22 planes

Learned agents consume a channel-first PyTorch `float32` tensor with shape
`[22, height, width]` (or `[22, 24, 24]` for the standard preset). Rows are
board `y` coordinates and columns are board `x`
coordinates. Values are binary. The encoding depends only on the canonical
Python `GameState`, never backend or browser values.

The version is exposed as `twixt_ai.models.ENCODING_VERSION`. Channel order and
meaning are checkpoint compatibility data and must not change within version 1.

| Channels | Meaning |
| --- | --- |
| 0-1 | Red pegs, Black pegs |
| 2-9 | Red links in the direction order listed below |
| 10-17 | Black links in the same direction order |
| 18-19 | Red-to-move, Black-to-move (one constant one-hot plane) |
| 20-21 | Red north/south goal borders, Black west/east goal borders |

Each undirected link appears twice: at each endpoint in the plane directed
toward the other endpoint. This explicitly preserves connectivity while making
geometric augmentation a channel permutation plus a spatial transform. Goal
planes exclude the four corners because neither player may place there.

The link direction order is `(-2,-1)`, `(-2,+1)`, `(-1,-2)`, `(-1,+2)`,
`(+1,-2)`, `(+1,+2)`, `(+2,-1)`, `(+2,+1)`.

`BoardSymmetry` defines all eight symmetries of the square. Reflections across
x or y and a half-turn retain player identities. Quarter-turns and diagonal
reflections exchange Red and Black, including the side to move, terminal
winner, link ownership, and goal planes. `transform_state` maps canonical game
values, while `transform_encoding` performs the equivalent operation directly
on encoded tensors and supports leading batch dimensions.

Standard 24-by-24 and Mini 10-by-10 positions use identical feature semantics.
The policy/value model configuration and checkpoints record the spatial
dimensions and reject positions from a different board size.
No auxiliary heuristic features are included in v1: occupancy, connectivity,
turn, and immutable goal geometry are sufficient primitive inputs, and derived
features would increase compatibility surface without adding information.

## Version 2: 10 normalized planes

The compact Mini encoder is exposed as `encode_mini_position`, with version
`MINI_ENCODING_VERSION`. On the validated 10-by-10 preset its tensor shape is
`[10, 10, 10]`. It has no history, last-move, turn, or goal-border planes.

| Channels | Meaning |
| --- | --- |
| 0-1 | Side-to-move (`self`) pegs, opponent pegs |
| 2-5 | Self links in the four canonical orientations |
| 6-9 | Opponent links in the four canonical orientations |

Red's north/south goal orientation is canonical. A Red-to-move position is
unchanged; a Black-to-move position is transposed (`normalized x = game y`,
`normalized y = game x`). Player colors are then expressed semantically as
`self` and `opponent`. Thus the current player always connects the normalized
north and south edges. `game_to_normalized_coordinate` and
`normalized_to_game_coordinate` apply this exact transform.

Each undirected link is stored exactly once. After normalization, endpoints
are ordered left-to-right and a bit is placed at the left endpoint in one of
the `(dx,dy)` planes `(+1,-2)`, `(+1,+2)`, `(+2,-1)`, or `(+2,+1)`.
`decode_mini_position` reconstructs all pegs and links when given the original
side to move, which is the perspective metadata inherently required by a
normalized representation.

Policy logits use row-major coordinates in the normalized frame.
`game_coordinate_to_normalized_action_index` maps game actions into that frame
and `normalized_action_index_to_game_coordinate` reverses the mapping. Board
dimensions are configurable; for rectangular Black positions the normalized
width and height are exchanged.
