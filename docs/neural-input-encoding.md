# Neural input encoding v1

Learned agents consume a channel-first PyTorch `float32` tensor with shape
`[22, 24, 24]`. Rows are board `y` coordinates and columns are board `x`
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

Only standard 24-by-24 positions can be encoded. This intentional validation
prevents a checkpoint from silently receiving incompatible spatial dimensions.
No auxiliary heuristic features are included in v1: occupancy, connectivity,
turn, and immutable goal geometry are sufficient primitive inputs, and derived
features would increase compatibility surface without adding information.
