# OpenBench and the Volkrix Strength Lab

Volkrix supports two complementary testing environments:

- OpenBench distributes commit-versus-commit SPRTs and SPSA workloads across
  trusted workers.
- `scripts/strength_lab.py` runs fixed-size external-engine calibration
  gauntlets locally or on one controlled host.

Neither tool downloads third-party engines or opening books. The operator is
responsible for acquiring them under compatible licenses and preserving their
source and license records.

## Production NNUE OpenBench builds

Current OpenBench workers download the configured network and invoke the engine
Makefile with `EVALFILE=/absolute/path/to/network`. They retain only the output
executable. `openbench/Makefile` therefore builds with the opt-in
`embedded-nnue` feature whenever `EVALFILE` is present. The build script hashes
the bytes and records the SHA-256 and size in the binary's default `EvalFile`
label. If OpenBench's network path is named by its 64-character digest, the
build independently rejects a content mismatch.

Normal Cargo and release builds do not embed a network. Their sibling-file and
explicit `setoption name EvalFile` behavior is unchanged. In an embedded build,
the compiled evaluator is authoritative at startup: `VOLKRIX_EVAL_FILE` and a
sibling `nn-c288c895ea92.nnue` are deliberately ignored, so ambient worker state
cannot silently change a match. An explicit UCI `setoption name EvalFile` can
still replace it after startup.

Before deployment, validate the exact contract with a small network:

```bash
scripts/fetch-stockfish18-net.sh --small target/release/nn-37f18f62d772.nnue
mkdir -p target/openbench-smoke
make -C openbench \
  EXE="$PWD/target/openbench-smoke/volkrix" \
  EVALFILE="$PWD/target/release/nn-37f18f62d772.nnue" TUNE=0
python3 scripts/uci_smoke.py \
  --engine target/openbench-smoke/volkrix \
  --evalfile embedded \
  --transcript target/openbench-smoke/uci.log
```

The smoke must succeed without a network beside the output binary. Production
workers should use the verified large Stockfish 18 SFNNv10 network.

## Server deployment checklist

1. Deploy a pinned OpenBench server revision and record it.
2. Copy `openbench/Volkrix.json.example` to `Engines/Volkrix.json`.
3. Upload `nn-c288c895ea92.nnue` through the network administration page and
   mark it as Volkrix's default. Confirm SHA-256
   `c288c895ea924429ea9092e3f36b2b3c1f00f2a3a4c759ff7e57e79e3b43e4a7`.
4. Upload the named opening books, retaining their source, license, and digest.
5. Install stable Rust/Cargo on every worker. The example accepts Linux,
   Windows, and Darwin because the build contract is tested on all three.
6. Run at least five reference benches on the designated reference worker and
   replace the example `nps` value with the stable median.
7. Run a no-change STC test. It should be statistically neutral and must show
   zero crashes, stalls, time forfeits, and illegal moves.
8. Use STC SPRT for iteration, LTC SPRT plus a held-out book before promotion,
   and regression presets for release candidates. Test SMP changes separately.

Budget worker memory from measurements of the exact deployed binary and
network. On the Apple M4 validation host, the large-network embedded binary used
about 263 MB resident memory at depth 1, versus about 154 MB for the
external-file binary. Treat roughly 263 MB as the embedded process baseline and
add the configured hash plus match-runner overhead. These
figures are platform-dependent; calibrate concurrency conservatively on every
worker class and leave headroom for simultaneous engines and the operating
system.

OpenBench builds default to `TUNE=1`, which adds the `spsa-tuning` feature;
`TUNE=0` creates a smaller production-only option surface. The 26 bounded
`Tune*` options cover aspiration, LMR, null move, reverse futility, futility,
late-move, SEE and history pruning, ProbCut, and time management. Their defaults
are the production defaults, so merely advertising them does not alter play.
`setoption name TuneManifest` prints the stable schema, values, and checksum.
Do not start SPSA unless that manifest matches the intended baseline and a
no-change match is neutral.

Actual server creation, account provisioning, opening-book upload, and worker
connection require infrastructure credentials and machines outside this
repository. They cannot be completed by a source checkout.

## External calibration gauntlet

Copy `openbench/strength-lab.example.json` outside the repository, replace every
path and digest, and list each legally obtained opponent. List every evaluator
network and other file-valued runtime input in `assets`; Volkrix `EvalFile`
options are rejected unless their file is frozen there. Keep engine-specific
options on that opponent; profile options are applied first and engine options
override them.

```bash
python3 scripts/strength_lab.py prepare \
  --config /absolute/path/to/gauntlet.json \
  --output /absolute/path/to/new-volkrix-gauntlet
python3 scripts/strength_lab.py run \
  --lab /absolute/path/to/new-volkrix-gauntlet --dry-run
python3 scripts/strength_lab.py run \
  --lab /absolute/path/to/new-volkrix-gauntlet
python3 scripts/strength_lab.py verify \
  --lab /absolute/path/to/new-volkrix-gauntlet
```

Preparation resolves all paths, verifies optional expected SHA-256 values, and
freezes a matrix of profile/opponent commands. Relative `EvalFile` values become
verified absolute paths. Before accepting the lab, it launches every distinct
engine/effective-option vector through `uci`, `setoption`, `isready`, and a
depth-one search; missing protocol acknowledgements, option errors, timeouts,
and nonzero exits fail preparation. The exact preflight input, transcript, and
digest are preserved. Every opening is played as a color-reversed pair with
sequential ordering. FastChess saves recovery state every pair. If interrupted,
rerun the same `run` command: a pending job resumes from its frozen recovery
file, while completed jobs are skipped only after their marker and artifact
hashes verify.

Each job retains the full PGN, engine protocol log, console output, FastChess
recovery state, exact command, summary, and artifact hashes. PGN `Termination`
tags are classified as crash/stall (`abandoned` in current FastChess), time
forfeit, illegal move, interrupted, or unknown abnormal termination. A job cannot be marked complete
unless it contains exactly two finished games per requested `Round` tag, with
the candidate on opposite colors. Pairing never depends on PGN completion order.
Summaries include W/D/L, score, pentanomial counts, logistic Elo difference, and
a pair-aware approximate 95% interval. If an opponent has an optional `rating`,
the report also gives the corresponding estimated candidate rating; record the
rating list, pool, hardware, and date because that calibration is not universal.

External gauntlets calibrate Volkrix's approximate competitive level; they are
not promotion tests because opponents, hardware efficiency, and evaluation
networks differ. Promote an internal change only with paired same-engine SPRTs
where the tested change is the sole controlled variable.

## Initial external calibration (2026-08-08)

The Apple M4 validation host ran the hardened PGO control with the large SF18
network, one thread, 64 MiB hash, and paired `8moves_v3.pgn` openings. The direct
Berserk 14 match used `1+0.01`, no adjudication, and 100 games: Volkrix scored
2 wins, 6 draws, and 92 losses (5.0%, logistic difference about -511.5 Elo).
Every PGN termination was normal. This is the most useful external result in
the initial set and establishes a large remaining gap to that elite engine.

For a coarse ladder only, Stockfish 18 was run with `UCI_LimitStrength=true`:

| Stockfish setting | TC | Games | Volkrix W-D-L | Score | Logistic difference |
| --- | --- | ---: | ---: | ---: | ---: |
| `UCI_Elo=2400` | `1+0.01` | 40 | 36-0-4 | 90.0% | +381.7 |
| `UCI_Elo=2800` | `1+0.01` | 40 | 19-9-12 | 58.75% | +61.4 |
| `UCI_Elo=3000` | `1+0.01` | 40 | 8-9-23 | 31.25% | -137.0 |
| `UCI_Elo=2900` | `10+0.1` | 20 | 12-3-5 | 67.5% | +127.0 |

These nominal `UCI_Elo` values are difficulty controls, not ratings on a shared
universal scale. The samples are small, the LTC row has only ten opening pairs,
and FastChess reported repeated "bestmove does not match beginning of last PV"
warnings from limited-strength Stockfish. All listed games still completed
normally, but those warnings and the deliberately weakened move selection make
the ladder unsuitable for claiming a Volkrix rating. Its purpose is regression
orientation and selecting future opponents near a useful score band.

The local, gitignored evidence is under `target/external-ladder/`: complete PGNs,
engine protocol logs, and FastChess recovery configurations. Future published
calibrations should be generated through `strength_lab.py` so binaries, books,
networks, commands, and final artifacts receive immutable checksums as well.
