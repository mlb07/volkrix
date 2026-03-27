# Volkrix

Volkrix is a clean-room Rust UCI chess engine built for long-term strength, reproducibility, and public development.

## Status

The repository currently includes:

- Phase 0 repo scaffolding, CI, docs, licensing, and developer scripts
- Phase 1 core chess types, position model, FEN parsing/serialization, legal move application for standard chess, and a minimal UCI shell
- Phase 2 attack generation, staged legal move generation, canonical perft support, and divide regression artifacts
- Phase 3 exact make/unmake restoration, Zobrist, repetition, rules/status helpers, and SEE
- Phase 4 deterministic single-thread search with iterative deepening, alpha-beta, quiescence, PV tracking, and a simple classical evaluation
- Phase 5 single-thread transposition table integration, deterministic TT-on/TT-off search paths, and a reproducible bench path tied to the real search core
- Phase 6 search-strength layering with stronger move ordering, aspiration windows, deterministic internal heuristic toggles, and documented Phase 5 baseline comparisons
- Phase 7 practical UCI usability with timed search, real `stop`, `Hash` / `Clear Hash`, and persistent TT reuse across UCI searches

What is intentionally not here yet:

- SMP or other shared search infrastructure
- advanced pruning and reduction heuristics
- tablebases or NNUE

## Clean-Room Provenance

Volkrix is an independent clean-room implementation released under dual MIT/Apache-2.0 licensing. GPL engine code, translated code, or closely paraphrased code must not be copied into this repository unless the project owner explicitly authorizes a licensing change in writing.

Contributors must preserve that boundary. Public concepts, papers, high-level algorithms, and non-code descriptions may inform design, but implementation must be written from scratch.

## Development

Requirements:

- stable Rust
- no nightly features
- no external chess crates in the engine core

Common commands:

```bash
python3 scripts/dev.py fmt
python3 scripts/dev.py clippy
python3 scripts/dev.py test
python3 scripts/dev.py bench
python3 scripts/dev.py release --target aarch64-apple-darwin
```

Direct Cargo commands:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo build
cargo test
cargo run --release -- bench
```

## Current UCI Surface

The engine currently supports:

- `uci`
- `isready`
- `ucinewgame`
- `position`
- `setoption name Hash value <mb>`
- `setoption name Clear Hash`
- `go depth`
- `go movetime <ms>`
- `go wtime <ms> btime <ms> [winc <ms>] [binc <ms>] [movestogo <n>]`
- `go infinite`
- `stop`
- `quit`

Malformed FEN strings, invalid moves, and malformed UCI commands are handled without panicking or corrupting engine state. The UCI runtime keeps the search core single-threaded, reuses TT state across UCI searches, and still preserves the documented deterministic fixed-depth bench and regression path outside the UCI shell.

## Documentation

- `docs/architecture.md`
- `docs/roadmap.md`
- `docs/search.md`
