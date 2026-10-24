# Reference oracles

`cfr_joint.py` is a verbatim copy (2026-10-24) of
`OneDrive - aus.edu/Scalability_Final_Rev/Polynomials/studies/cfr_interband/cfr_joint.py`.
`tests/test_joint_compression.py` imports it by path (`CFR_INTERBAND_DIR` overrides the
directory) and checks `adrvtrx.metrics.joint_compression` against `composite_papr` /
`pe_papr`. It is a frozen oracle, excluded from ruff and black: do not edit it.
