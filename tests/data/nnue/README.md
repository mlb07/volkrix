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

`phase13-fixture.fens` is the retained Phase 13 tiny deterministic fixture corpus.

It is:

- a small in-repo FEN-lines corpus
- intended for export/training/packing smoke validation only
- not intended to be a strength corpus
- kept separate from any real locally trained candidate-net artifacts
