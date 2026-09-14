# Architecture-v2 data scaling

Issue 151 tests whether the fixed widened-head Mini model from Issue 143 needs
more self-play data. This directory retains completed stages without changing
the architecture, teacher/search semantics, optimizer schedule, or paired
evaluation contract.

- [`1k/`](1k/) is the completed 1,000-game stage. It improved held-out value
  MSE and the value-guided playing ablations, but did not catch the fixed
  non-neural baselines or the architecture-v1 matched-1k reference.

Later scale stages are intentionally outside this mini task.
