# Volkrix

Volkrix is a UCI chess engine written in Rust. It is a clean-room implementation focused on long-term strength, reproducibility, and open development.

## Features

- Standard chess move generation with perft-verified legality
- Make/unmake with Zobrist hashing, repetition detection, and SEE
- Iterative deepening alpha-beta search with quiescence, PV tracking, aspiration windows, and quiet-move LMR
- Tapered classical evaluation: material, mobility, king safety, pawn structure, passed pawns, bishop pair, rook files, and lightweight threats
- Persistent transposition table with `Hash` / `Clear Hash` controls
- Lazy SMP via the `Threads` option
- Optional Syzygy tablebase probing (`SyzygyPath`)
- Optional NNUE evaluation (`EvalFile`) using the in-tree `VOLKNNUE` format
- Reproducible `bench` command tied to the real search

## Building

Volkrix builds on stable Rust with no external chess crates in the engine core.

```bash
cargo build --release
```

The release binary is written to `target/release/volkrix`.

A small helper script wraps the common workflows:

```bash
python3 scripts/dev.py fmt
python3 scripts/dev.py clippy
python3 scripts/dev.py test
python3 scripts/dev.py bench
python3 scripts/dev.py release --target aarch64-apple-darwin
```

## Usage

Volkrix speaks the [Universal Chess Interface](https://www.shredderchess.com/chess-features/uci-universal-chess-interface.html) and works with any UCI-compatible GUI (Cute Chess, Arena, Banksia, etc.).

From a terminal:

```
$ volkrix
uci
position startpos moves e2e4 e7e5
go movetime 1000
```

Run a deterministic search benchmark:

```bash
volkrix bench
```

### Supported UCI commands

`uci`, `isready`, `ucinewgame`, `position`, `go`, `stop`, `quit`.

`go` accepts `depth`, `movetime`, `infinite`, and the standard time-control fields (`wtime`, `btime`, `winc`, `binc`, `movestogo`).

### Options

| Option        | Default | Description                                       |
| ------------- | ------- | ------------------------------------------------- |
| `Hash`        | 16      | Transposition table size in MB                    |
| `Clear Hash`  | —       | Button: clear the transposition table             |
| `Threads`     | 1       | Number of search threads (Lazy SMP)               |
| `SyzygyPath`  | ""      | Directory containing Syzygy tablebase files       |
| `EvalFile`    | ""      | Path to a `VOLKNNUE` network; empty uses classical eval |

Malformed FEN strings, illegal moves, and malformed UCI input are rejected without panicking or corrupting engine state.

## Releases

Publishing a GitHub Release triggers `.github/workflows/release.yml`, which builds `volkrix` and uploads per-platform archives to the release.

## Project layout

```
src/        engine core (board, movegen, search, eval, UCI)
tools/      offline tooling, including the NNUE packer
trainer/    NNUE training pipeline
benches/    Criterion benches
tests/      integration and perft tests
docs/       design notes
```

Further reading lives in `docs/`:

- [`architecture.md`](docs/architecture.md) — module layout and data flow
- [`search.md`](docs/search.md) — search algorithm details
- [`perft.md`](docs/perft.md) — move generation verification
- [`nnue-training.md`](docs/nnue-training.md) — NNUE training and packing workflow
- [`roadmap.md`](docs/roadmap.md) — planned work

## Clean-room provenance

Volkrix is an independent clean-room implementation. GPL-licensed engine code, machine translations, or close paraphrases of GPL sources must not be copied into this repository. Contributors may study public papers, algorithms, and high-level descriptions, but implementation must be written from scratch.

## License

Dual-licensed under either of:

- MIT license ([LICENSE-MIT](LICENSE-MIT))
- Apache License 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.
