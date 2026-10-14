"""Known-variance rules of the Biometrika version, copied unchanged (package `uai` -> `latentcov.kv`).

extremal    closed forms over the extremal family and the one-sided constants
certify     branch-and-bound upper bounds for R_{p,q} (double precision, margin 1e-9)
procedures  Fay-Herriot, shape-free Markov rule, HetLDC, simple shrink and helpers

These are used by the comparison experiments only. `procedures.CERTIFIED_SLACK_LEVEL` and
`certified_rank` are superseded by `latentcov.levels` and `latentcov.rank`.
"""
