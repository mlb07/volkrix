# Bench Scaffolding

The built-in benchmark uses the same evaluator and search-service path as UCI.
Its first output line records the thread count and evaluator provenance:

```bash
cargo run --locked --release -- bench \
  --threads 1 --hash-mb 64 --evalfile classical
```

Use an absolute network path instead of `classical` to benchmark NNUE. Run the
same explicit evaluator at every thread count being compared.

## Strength Testing

Search benchmarks are regression and performance checks; changes in nodes, depth,
or wall time are not evidence of playing strength. Candidate promotion requires a
paired engine match in which every opening is played twice with colors reversed.

The built-in comparison command preserves input order and performs that color
reversal deterministically:

```bash
cargo run --locked --release -p volkrix-nnue -- compare-engines \
  --openings /absolute/path/to/openings.fens \
  --baseline /absolute/path/to/baseline \
  --candidate /absolute/path/to/candidate \
  --baseline-evalfile /absolute/path/to/network.nnue \
  --candidate-evalfile /absolute/path/to/network.nnue \
  --artifacts /absolute/path/to/new-run-directory \
  --movetime-ms 100 \
  --max-plies 240
```

Its report includes:

- the five pentanomial pair buckets, from two candidate losses through two wins
- candidate score and a paired Wilson-style 95% confidence interval
- an Elo estimate and interval when the score bounds are finite
- natural termination counts for checkmate and every board-rule draw
- separate counts for draws imposed by `--max-plies`

Fixed-depth matches are the deterministic debugging mode. Movetime matches model
real play but necessarily include operating-system scheduling noise. In either
mode, fewer than 100 opening pairs is an exploratory smoke test, not a promotion
result. Serious search changes should use a large, varied, license-compatible
opening suite and a sequential probability ratio test through Fastchess or
OpenBench. Archive the exact binaries, commit IDs, compiler version, CPU, command,
opening-suite checksum, and complete result log with every experiment.

`--max-plies` is an adjudication boundary. Games that reach it are reported as
max-ply-adjudicated draws rather than silently combined with repetitions,
stalemates, fifty-move draws, or insufficient-material draws. A high max-ply draw
rate means the test configuration is not producing enough information to support
an Elo claim.

The complete release smoke, immutable FastChess SPRT, PGO, and OpenBench workflows
are documented in [`docs/strength-testing.md`](../docs/strength-testing.md).
