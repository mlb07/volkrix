# Strength Testing and Production Builds

Volkrix uses three different gates because correctness, speed, and playing
strength are different properties. A candidate must pass all applicable gates;
a faster benchmark or a positive short match is not by itself evidence of Elo.

## Release and UCI smoke gate

Run a real process-level protocol check against the evaluator that will ship:

```bash
python3 scripts/uci_smoke.py \
  --engine /absolute/path/to/volkrix \
  --evalfile /absolute/path/to/nn-c288c895ea92.nnue \
  --transcript /absolute/path/to/uci-smoke.log
```

Use `--evalfile classical` to test the fallback deliberately. The probe verifies
the required UCI options, applies every production setting, waits for `readyok`,
checks a finite search for a legal move and `info` output, verifies `stop` on an
infinite search, and requires a prompt exit after `quit`. It has hard deadlines
and treats `info string error`, malformed output, crashes, and hangs as failures.

CI runs the same probe with the checksum-verified small Stockfish 18 network.
Release jobs run it against the packaged binary and packaged large network before
creating the archive. This catches packaging, sibling-file, parser, and protocol
failures that unit tests alone cannot.

### Real Syzygy backend smoke

Syzygy data is intentionally not part of release archives or the repository. A
small checksum-pinned KQvK pair is sufficient to exercise Fathom initialization,
root DTZ decoding, and both one-thread and SMP service paths:

```bash
scripts/fetch-syzygy-smoke.sh /tmp/volkrix-syzygy-smoke
VOLKRIX_SYZYGY_PATH=/tmp/volkrix-syzygy-smoke \
  cargo test --locked --lib real_tablebase_root_resolution -- --ignored --nocapture
VOLKRIX_SYZYGY_PATH=/tmp/volkrix-syzygy-smoke \
  cargo test --locked --lib real_fathom_wdl_probes_survive_concurrent_reconfiguration \
  -- --ignored --nocapture --test-threads=1
```

The helper downloads from the Sesse tablebase mirror over HTTP, then rejects the
files unless their complete SHA-256 digests match the pinned KQvK fixtures. These
files are for an opt-in smoke test only and are never copied into a build or
release artifact. Production testing should point `SyzygyPath` at the deployment's
full licensed tablebase set and confirm that UCI reports the expected loaded
cardinality and nonzero hit counts.

## Reproducible built-in match

The built-in runner is intended for local plumbing checks and compact experiments:

```bash
cargo run --locked --release -p volkrix-nnue -- compare-engines \
  --openings /absolute/path/to/openings.fens \
  --baseline /absolute/path/to/baseline \
  --candidate /absolute/path/to/candidate \
  --baseline-evalfile /absolute/path/to/network.nnue \
  --candidate-evalfile /absolute/path/to/network.nnue \
  --candidate-small-evalfile /absolute/path/to/small-network.nnue \
  --candidate-dual-policy small-fallback \
  --candidate-dual-threshold 200 \
  --artifacts /absolute/path/to/new-run-directory \
  --clock-ms 1000 --increment-ms 10 \
  --threads 1 --hash-mb 64 --move-overhead-ms 10 \
  --syzygy-path none --max-plies 240
```

The artifact directory contains the exact manifest and checksums, JSONL game
records, PGN, full protocol traffic, and an atomic opening-pair checkpoint.
`--resume on` resumes only when those immutable inputs match. The checkpoint is
the commit marker: recovery discards an orphan or partial game beyond its pair
boundary and rebuilds PGN from the committed JSONL records before continuing.
Each opening is played twice with colors reversed. Engine crashes, hangs, time
forfeits, protocol errors, and illegal moves are losses rather than disappearing
from the sample.

## Promotion SPRT with FastChess

Use a current [FastChess](https://github.com/Disservin/fastchess) build and a
license-compatible EPD or PGN suite. Predeclare bounds and do not inspect results
to change the stopping rule:

```bash
scripts/run-strength-sprt.sh \
  --fastchess /absolute/path/to/fastchess \
  --baseline /absolute/path/to/volkrix-base \
  --candidate /absolute/path/to/volkrix-dev \
  --evalfile /absolute/path/to/nn-c288c895ea92.nnue \
  --book /absolute/path/to/UHO_Lichess_4852_v1.epd \
  --output-dir /absolute/path/to/results/change-name-stc \
  --tc 10+0.1 --rounds 100000 --concurrency 4 \
  --threads 1 --hash-mb 64 \
  --elo0 0 --elo1 3 --alpha 0.05 --beta 0.05
```

The output directory must be new. Before play, the wrapper records SHA-256 values
for FastChess, both binaries, the book, and both networks; records host, options,
SPRT bounds, and the exact shell-escaped command; and runs FastChess's UCI
compliance checker. It uses `-repeat -games 2`, sequential openings, and the
normalized pentanomial SPRT model. Pass `--dry-run` to inspect the frozen command
without starting games. FastChess checkpoints every completed pair, and the
final `summary.json` classifies abnormal PGN termination tags instead of allowing
crashes, stalls, time forfeits, or illegal moves to disappear into the score.

The wrapper explicitly freezes and preflights `Threads`, `Hash`, `Move Overhead`,
`SyzygyPath`, `SyzygyProbeLimit`, and `Syzygy50MoveRule` on both engines. Use
`--syzygy-probe-limit` and `--syzygy-50-move-rule` when deviating from the
production defaults; the exact values are preserved in both the command and run
manifest. FastChess cannot encode an empty `option.Name=` engine token, so an
explicitly empty value such as the default `SyzygyPath` is retained in the
manifest and UCI preflight but omitted from the FastChess command. The preflight
therefore verifies the intended default-empty state without producing an invalid
FastChess argument.

Use separate `--baseline-evalfile` and `--candidate-evalfile` only when the
network itself is the declared experiment. Otherwise both sides must use the same
network. Keep the complete directory, including PGN and engine-level log.

## PGO production build

Install Rust's matching LLVM profiling tools:

```bash
rustup component add llvm-tools-preview
```

Then build with a new work directory and a new output path:

```bash
python3 scripts/build_pgo.py \
  --evalfile /absolute/path/to/nn-c288c895ea92.nnue \
  --threads 4 \
  --work-dir /absolute/path/to/new-pgo-work \
  --output /absolute/path/to/volkrix-pgo
```

The script uses isolated Cargo target trees. It builds with
`-Cprofile-generate`, runs the deterministic benchmark at one thread and the
requested SMP width plus the UCI smoke workload, merges every raw profile with
the `llvm-profdata` from the active Rust toolchain, then rebuilds with
`-Cprofile-use` and missing-function warnings enabled. The final binary is
smoke-tested again. `manifest.json` records the toolchain, evaluator checksum,
profile checksum, build flags, workload, and output checksum.
It also records the HEAD commit, Cargo lockfile checksum, binary Git-diff
checksum, untracked path/content checksum, and a composite source-tree identity.
Dirty builds compile `HEAD-dirty-<digest>` into the engine's source-provenance
constant instead of recording the clean commit as the complete identity. The
script verifies that this source identity is unchanged after both PGO stages
and refuses provenance if another process modified the tree during the build.

Do not distribute a `target-cpu=native` build to different hardware. If
`--base-rustflags` contains CPU features, build and test a separate artifact for
each compatible target. PGO is a performance candidate and still needs an A/B
SPRT: it is not automatically stronger under every time control.

On the Apple M4 validation host, the final native PGO artifact improved the
production SFNNv10 depth-7 median from 451 ms to 419 ms at one thread and from
about 506.6k to 542.1k NPS at two threads. A 100-game PGO-versus-generic
match scored 26 wins, 50 draws, and 24 losses; treat that game result as
inconclusive and the deterministic throughput improvement as hardware-specific.

## OpenBench

[`openbench/Makefile`](../openbench/Makefile) satisfies the public-engine contract
for `make EXE=Engine-ABCDEFGH`. When the worker supplies `EVALFILE`, the exact
network is embedded into the single retained executable, with its SHA-256 and
size exposed in the default UCI `EvalFile` label. Ordinary Cargo builds remain
external-network builds. Copy [`openbench/Volkrix.json.example`](../openbench/Volkrix.json.example)
into an OpenBench server's `Engines/` directory, then:

1. replace `nps` with the value calibrated on that server's reference worker;
2. upload and select the checksum-verified production network;
3. upload the named, licensed opening book;
4. provision stable Rust and Cargo on every eligible worker;
5. verify the `main` base branch and workload sizes;
6. run several identical benches before accepting the configuration.

The full deployment, isolated-binary smoke procedure, STC/LTC/regression policy,
and resumable external calibration lab are documented in
[`docs/openbench.md`](openbench.md). CI tests the embedded build contract on
Linux, Windows, and macOS with a real Stockfish-format network.

The OpenBench requirements and current engine-config schema are documented in
the [official OpenBench repository](https://github.com/AndyGrant/OpenBench).

## Promotion record

Retain the hypothesis, commit IDs, build flags, compiler, CPU/power state, exact
binaries and networks, book source/license/checksum, time controls, concurrency,
W/D/L and pentanomial counts, confidence or SPRT decision, crash/time/illegal
counts, PGN, and full logs. Run a held-out opening suite and at least one longer
time control before making a broad strength claim.
