# Volkrix

Volkrix is a clean-room UCI chess engine written in Rust. The project combines a
verified chess core with a selective alpha-beta search, parallel search, Syzygy
probing, and optional neural evaluation without requiring a Volkrix-specific
neural network to be trained first.

## Highlights

- Bitboard/mailbox position representation with incremental Zobrist hashing,
  repetition tracking, make/unmake, null moves, static exchange evaluation, and
  perft-tested legal move generation
- Iterative deepening PVS with aspiration windows, quiescence search, mate-distance
  pruning, staged move ordering, history feedback, LMR, null-move pruning,
  ProbCut, and conservative futility/SEE/history pruning
- Lock-free, four-entry clustered transposition table shared by all search threads
- Persistent worker pool with deterministic root-move sharding for helper searches
- Tapered classical fallback evaluation and optional incremental NNUE evaluation
- Root DTZ and in-search WDL probing for castling-free Syzygy positions with up to
  seven pieces
- Defensive UCI parsing, cooperative cancellation, clock management, configurable
  move overhead, and live depth/seldepth/PV/search statistics
- Reproducible search bench plus corpus, tuning, network-packing, and paired-match
  tools

These features describe the implementation, not an Elo claim. Search changes are
promoted through paired games and confidence-aware match reports; see
[`benches/README.md`](benches/README.md) and
[`docs/strength-testing.md`](docs/strength-testing.md).

## Build

Volkrix builds on stable Rust:

```bash
cargo build --locked --release
```

The optimized binary is `target/release/volkrix`. The release profile enables
full LTO and a single codegen unit, so release builds take longer than development
builds.

Common checks are available through `scripts/dev.py`:

```bash
python3 scripts/dev.py fmt
python3 scripts/dev.py clippy
python3 scripts/dev.py test
python3 scripts/dev.py bench
```

## Run

Volkrix speaks the [Universal Chess Interface](https://www.shredderchess.com/chess-features/uci-universal-chess-interface.html)
and can be launched directly by a UCI-compatible GUI or from a terminal:

```text
$ target/release/volkrix
uci
position startpos moves e2e4 e7e5
go movetime 1000
```

Run the built-in deterministic search workload with:

```bash
target/release/volkrix bench
```

The default benchmark explicitly uses the classical evaluator. Use
`--evalfile /absolute/path/to/network.nnue` to benchmark NNUE, and use
`--threads`, `--hash-mb`, `--depth`, or `--no-tt` to select the exact
configuration. The result prints those evaluator and thread choices.

### UCI surface

Supported commands are `uci`, `isready`, `ucinewgame`, `position`, `setoption`,
`go`, `stop`, `ponderhit`, `quit`, and `debug on|off`.

`go` supports:

- `depth <plies>`
- `nodes <count>`, alone or as an additional search cap
- `movetime <milliseconds>`
- `wtime`, `btime`, `winc`, `binc`, and `movestogo`
- `infinite`, terminated by `stop`
- `ponder` with clock controls, released by `ponderhit` or terminated by `stop`
- `searchmoves <move...>` as a root filter

`go mate` is not currently supported and is rejected explicitly instead of being
silently ignored.

| Option | Default | Range / behavior |
| --- | ---: | --- |
| `Hash` | 16 | 1–512 MiB; resizing replaces the current TT |
| `Threads` | 1 | 1–64 search threads |
| `Move Overhead` | 10 | 0–5000 ms reserved from time controls |
| `Ponder` | false | Advertises GUI-controlled pondering; `go ponder` uses clock controls |
| `SyzygyPath` | empty | Tablebase directory; empty disables probing |
| `SyzygyProbeLimit` | 7 | 0–7 pieces; `0` disables probing without unloading files |
| `Syzygy50MoveRule` | true | Honor the rule-50 clock and cursed-win/blessed-loss outcomes |
| `EvalFile` | discovered sibling or empty | `VOLKNNUE` or supported Stockfish `.nnue`; empty uses classical evaluation |
| `SmallEvalFile` | empty | Optional secondary evaluator; loading it alone does not change search |
| `DualEvalPolicy` | `off` | `off` preserves big-only search; `small-fallback` enables the experimental dual policy |
| `DualEvalThreshold` | 200 | 0–2000 cp ambiguity band for big-network fallback |
| `Clear Hash` | — | Clears all TT entries |

Malformed FENs, illegal moves, invalid options, and unsupported search arguments
produce `info string error` responses without changing the current position.

## Use a pretrained NNUE network

`EvalFile` auto-detects Volkrix's `VOLKNNUE` header. Other files are parsed by the
pinned MIT-licensed `nnue-rs` 0.4.0 backend, which supports SFNNv10,
HalfKAv2_hm, HalfKAv2, and HalfKP networks.

Official release archives include the large Stockfish 18 network beside the
Volkrix executable. That layout is discovered automatically at startup, while an
explicit `EvalFile` still takes precedence.

For a local source build, install the same network beside the optimized binary.
The helper downloads from the Stockfish testing service, verifies the complete
SHA-256 digest, and refuses to install any mismatch:

```bash
scripts/fetch-stockfish18-net.sh target/release/nn-c288c895ea92.nnue
```

The expected large-network SHA-256 is
`c288c895ea924429ea9092e3f36b2b3c1f00f2a3a4c759ff7e57e79e3b43e4a7`.

For maximum measured playing strength, use the large network above. On AArch64,
the official 3.4 MiB small network is a faster, lower-memory alternative and can
be fetched with:

```bash
scripts/fetch-stockfish18-net.sh --small ./nn-37f18f62d772.nnue
```

The small network is opt-in and never replaces the bundled large network. Select
either file explicitly when you do not want the automatically discovered one:

```text
setoption name EvalFile value /absolute/path/nn-c288c895ea92.nnue
isready
```

Dual-network evaluation is an experimental, default-off A/B seam. It keeps both
lazy accumulator stacks synchronized, evaluates the small net first, and falls
back to the large net when the small score is within the configured ambiguity
band:

```text
setoption name EvalFile value /absolute/path/nn-c288c895ea92.nnue
setoption name SmallEvalFile value /absolute/path/nn-37f18f62d772.nnue
setoption name DualEvalThreshold value 200
setoption name DualEvalPolicy value small-fallback
isready
```

This policy is available for paired testing; it is not promoted as stronger than
the large network without a passing match gate. The threshold-200 candidate was
explicitly rejected after a direct sanity match: 2 wins, 5 draws, and 41 losses
in 48 completed games (9.4%). The advertised threshold is an experiment
parameter, not a strength recommendation; big-only remains the default.

External network files are separate artifacts and retain their own licenses. The
networks fetched by the helper and shipped in official release archives are
distributed under CC0. `nnue-rs` has AVX2 acceleration on x86-64 and stable
NEON/DotProd acceleration with scalar tails on AArch64. The SIMD kernels are
checked bit-for-bit against scalar reference implementations.

Volkrix's own `VOLKNNUE` runtime supports HalfKP 128x2 and 256x2 networks and uses
preallocated incremental accumulators. Its training/export workflow is documented
in [`docs/nnue-training.md`](docs/nnue-training.md).

## Syzygy

Point `SyzygyPath` at a directory containing Syzygy files:

```text
setoption name SyzygyPath value /absolute/path/to/syzygy
setoption name SyzygyProbeLimit value 7
setoption name Syzygy50MoveRule value true
```

The vendored MIT-licensed Fathom backend probes root DTZ and non-root WDL results.
Volkrix restricts probes to castling-free positions with at most seven pieces and
to the cardinality actually loaded by Fathom. A successful path change reports
the detected maximum cardinality. Search completion reports probe attempts, hits,
misses, and errors before `bestmove`; a probe failure falls back to normal search
and is surfaced through `info string` instead of silently changing the result.

With `Syzygy50MoveRule=true`, root DTZ receives the current halfmove clock and
non-root WDL probing is restricted to positions whose clock is zero, as required
by Fathom's WDL interface. Setting it to `false` deliberately ignores the rule-50
clock and treats cursed wins/losses as unconditional wins/losses. Leave it enabled
for ordinary play.

Tablebase files are not bundled. For a minimal integration test, download the
checksum-pinned KQvK WDL and DTZ pair into a temporary directory and run the
ignored real-backend tests:

```bash
scripts/fetch-syzygy-smoke.sh /tmp/volkrix-syzygy-smoke
VOLKRIX_SYZYGY_PATH=/tmp/volkrix-syzygy-smoke \
  cargo test --locked --lib real_tablebase_root_resolution -- --ignored
VOLKRIX_SYZYGY_PATH=/tmp/volkrix-syzygy-smoke \
  cargo test --locked --lib real_fathom_wdl_probes_survive_concurrent_reconfiguration \
  -- --ignored --test-threads=1
```

## Project layout

```text
src/core/              board state, rules, move generation, SEE, perft
src/search/            search, evaluation, TT, SMP, NNUE, tablebases
src/uci/               protocol parser and asynchronous stdio runtime
tools/volkrix-nnue/    corpus, training, packing, tuning, and match tools
trainer/               pinned Bullet training support
tests/                 integration, protocol, perft, and regression tests
benches/               performance and strength-testing guidance
docs/                  architecture and development documentation
vendor/fathom/         pinned Fathom source and upstream license
```

Start with [`docs/architecture.md`](docs/architecture.md),
[`docs/search.md`](docs/search.md), and [`docs/perft.md`](docs/perft.md).

## Releases

Publishing a GitHub Release triggers builds for Linux x86-64, macOS AArch64, and
Windows x86-64. Each archive contains the executable, licenses and notices, the
fetch helper, and the checksum-verified large Stockfish 18 network beside the
executable. The workflow also publishes a SHA-256 sidecar for every archive.

The network remains a separate CC0 data artifact: it is neither compiled nor
linked into Volkrix. Syzygy files are not included; production deployments must
provision them separately and set `SyzygyPath`.

## Provenance and licensing

Volkrix is an independent clean-room implementation. Do not copy GPL-licensed
engine source, machine translations of it, or close paraphrases into this
repository. Public papers, algorithms, specifications, test positions, and
high-level descriptions may be studied; implementation contributed to Volkrix
must be written independently and must respect the license of every dependency
and data artifact.

Volkrix is available under either the [MIT license](LICENSE-MIT) or the
[Apache License 2.0](LICENSE-APACHE), at your option. Bundled components and
optional external artifacts are listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
