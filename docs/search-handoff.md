# Search Change Acceptance

This document is the handoff protocol for search and evaluation experiments. It
replaces historical lists of tiny local matches: those results are useful lab
notes, but they are not durable evidence that the current tree is stronger.

## Non-negotiable correctness gate

Before measuring strength, a candidate must pass:

```bash
cargo fmt --all -- --check
cargo test --workspace --all-features --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --release --features internal-testing --locked
cargo test --release --test perft -- --ignored
```

Use focused tests during iteration, but run the complete gate before promotion.
Replay printed PVs as legal move sequences, exercise stop/deadline behavior, and
test `Threads=1` plus representative multithread counts after touching shared
state.

## Performance gate

Build the baseline and candidate with the same compiler, target features, release
profile, and environment. Record:

- both commit IDs and exact binaries
- `rustc -Vv`, CPU model, OS, and power mode
- command line, thread/hash/evaluator settings, and network checksum
- several warm runs of `volkrix bench`
- NPS and completed depth at fixed wall times on representative positions

Node count can rise because a search became stronger or fall because a pruning
became unsound. Wall time can move independently of both. A benchmark is a
regression detector and profiler, never an Elo substitute.

## Strength gate

Every opening must be played twice with colors reversed. Hold `Hash`, `Threads`,
`Move Overhead`, `SyzygyPath`, `EvalFile`, adjudication, and time control constant
unless that option is the experiment itself.

The built-in paired harness is suitable for smoke tests:

```bash
cargo run --locked --release -p volkrix-nnue -- compare-engines \
  --openings /absolute/path/to/openings.fens \
  --baseline /absolute/path/to/baseline \
  --candidate /absolute/path/to/candidate \
  --baseline-evalfile /absolute/path/to/network.nnue \
  --candidate-evalfile /absolute/path/to/network.nnue \
  --artifacts /absolute/path/to/new-run-directory \
  --movetime-ms 100 \
  --hash-mb 64 \
  --max-plies 240
```

It reports pentanomial pair buckets, score, a paired confidence interval, an Elo
interval where finite, termination reasons, and max-ply adjudications. Treat a
small run as plumbing validation only. For promotion, use a large varied and
license-compatible opening suite with Fastchess or OpenBench and a predeclared
SPRT or confidence threshold.

Do not tune on one opening set and quote the same set as confirmation. Keep a
held-out suite, test at more than one time scale, and check that gains survive
both colors and common hardware targets.

## Experiment isolation

Change one coherent mechanism at a time. Search, evaluation, NNUE architecture,
SMP, and time-management changes should normally be separate candidates. When a
bundle is unavoidable, keep intermediate binaries so regressions can be bisected.

For classical tuning, emit a candidate weights JSON first and compare it without
rewriting engine defaults:

```bash
cargo run --release -p volkrix-nnue -- compare-classical-weights \
  --openings <openings.fens> \
  --candidate-weights <weights.json> \
  --depth 4
```

For `VOLKNNUE`, compare a packed candidate directly with the classical fallback:

```bash
cargo run --release -p volkrix-nnue -- compare-fallback \
  --openings <openings.fens> \
  --candidate <net.volknnue> \
  --movetime-ms 100
```

## Promotion record

A retained result should include the raw game log and:

- hypothesis and exact code/config difference
- test protocol chosen before the run
- opening-suite source, license, and checksum
- W/D/L and pentanomial counts
- score/Elo interval and stopping rule
- crashes, time losses, illegal moves, and adjudication counts
- performance delta and correctness commands

If a candidate is neutral, uncertain, or hardware-specific, record that plainly.
There is no valid “it cannot get better” endpoint for a chess engine; the honest
release standard is that each retained change has survived the strongest
available falsification attempt.

See [`strength-testing.md`](strength-testing.md) for the fail-closed UCI release
smoke, FastChess SPRT wrapper, PGO production build, and OpenBench template.
