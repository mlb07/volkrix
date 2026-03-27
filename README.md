# Volkrix

Volkrix is a clean-room Rust UCI chess engine built for long-term strength, reproducibility, and public development.

## Status

The repository currently includes:

- Phase 0 repo scaffolding, CI, docs, licensing, and developer scripts
- Phase 1 core chess types, position model, FEN parsing/serialization, legal move application for standard chess, and a minimal UCI shell
- Phase 2 attack generation, staged legal move generation, canonical perft support, and divide regression artifacts

What is intentionally not here yet:

- optimized attack tables or staged move ordering
- search heuristics beyond placeholder legal move selection
- transposition tables, SMP, tablebases, or NNUE

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
```

## Current UCI Surface

The engine currently supports:

- `uci`
- `isready`
- `ucinewgame`
- `position`
- `go depth`
- `stop`
- `quit`

Malformed FEN strings, invalid moves, and malformed UCI commands are handled without panicking or corrupting engine state.

## Documentation

- `docs/architecture.md`
- `docs/roadmap.md`
