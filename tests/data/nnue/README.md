# NNUE Test Asset

`volkrix-halfkp128x2-test.volknnue` is the retained Phase 12 synthetic integration net.

It is:

- clean-room and Volkrix-owned
- deterministic
- sized for the retained `HalfKP 128x2` topology
- intended only for parser, accumulator, and inference validation
- not a production playing net

The weights are synthetic and primarily encode simple material-count signals so the Phase 12 tests can validate:

- loader and metadata checks
- accumulator full-build and incremental-update correctness
- score orientation
- runtime `EvalFile` activation
