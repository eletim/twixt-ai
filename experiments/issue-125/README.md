# Issue 125 starting baseline

Issue 125 starts from the immutable Issue 118 generation-2 champion at
[`../issue-118/generation-2/generation-0001/candidate/best.pt`](../issue-118/generation-2/generation-0001/candidate/best.pt),
SHA-256 `742229c59caf251a07c7ecac6dc77ff75cbf09643a22cca16b92fe083df5a5ec`.
The checkpoint stays at its original path and must remain read-only.

On `dev/v0.0.8` commit `cd7f2714011231cc6a38fe99110e34b0c54bc627`, its hash
and deterministic paired evaluations were re-verified without training or modifying
either checkpoint. The results reproduced as 28-8-4 versus generation 1
([promotion report](../issue-118/generation-2/generation-0001/report.json)), 39-0-1
versus the [Issue 57 baseline](../issue-118/generation-2/baseline-comparison.json),
and 6-14-0 policy+value, 5-15-0 policy-only, and 0-19-1 value-only versus
heuristic search ([strength report](../issue-118/generation-2/strength.json)).

The immutable Issue 57 reference remains
[`../issue-57/baseline/best.pt`](../issue-57/baseline/best.pt), SHA-256
`ce20f05c8a3d687fce4860595d40d710177f5901298c0bea95afb488cadcc3c8`.
Both checkpoints are read-only reference points for every later Issue 125 comparison.

The deeper value-head audit is retained in
[`diagnostics/`](diagnostics/README.md). It uses the largest self-play dataset
whose manifest hash matches the generation-2 champion, and records phase,
search-ambiguity, and value-confidence breakdowns without changing either
artifact.
