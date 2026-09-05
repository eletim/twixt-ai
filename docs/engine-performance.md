# Engine performance

The canonical engine has a versioned, deterministic microbenchmark workload.
It constructs a standard 24×24 position at ply 160 using seed `2401`, then
measures legal move generation, automatic link updates, win checks, and full
move application independently. Run it with:

```bash
PYTHONHASHSEED=0 twixt-ai-engine-benchmark --output engine-benchmark.json
```

The JSON artifact records the complete workload, fixture size, Python and
platform information, per-operation latency and throughput, and full move
application as `positions_per_second`. Timings naturally vary across machines;
the seed, ply, board, warmups, repeats, and iterations make the work itself
repeatable. Compare results on the same otherwise-idle machine and Python build.

## Issue 24 baseline

The following before/after measurements were collected on CPython 3.10.12,
Linux x86-64, using one warmup and the median of five 200-iteration repeats.
They are a project baseline, not a cross-machine performance threshold.

| Operation | `dev/v0.0.1` | optimized | change |
| --- | ---: | ---: | ---: |
| Legal move generation | 2,035 calls/s | 2,773 calls/s | 1.36× |
| Automatic link updates | 10,485 calls/s | 10,641 calls/s | 1.01× |
| Win check | 21,270 calls/s | 24,079 calls/s | 1.13× |
| Move application | 1,286 positions/s | 3,466 positions/s | 2.70× |

Profiling the baseline over mixed mid-game positions attributed about 70% of
full transition time to legal move generation invoked solely for draw
detection. That path allocated every legal `Coordinate` and `PegPlacement`
after every move even though the transition only needed to know whether one
empty point existed. Canonical `GameState` sorting and validation accounted for
about 14%, automatic link generation and crossing checks about 9%, and win
detection about 4% in that workload.

The optimized transition counts occupied playable points for draw detection,
without materializing replies. Full legal generation now iterates only the
current player's playable rows or columns, preserving row-major ordering, and
the boolean win predicate avoids the predecessor map and neighbor sorting used
to return a deterministic path witness. State validation and link-crossing
scans remain intentional costs and are visible as candidates for future work;
changing either requires care because they enforce canonical public semantics.
