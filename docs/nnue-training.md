# NNUE Runtime and Training

Volkrix has two NNUE paths with different purposes:

- The Stockfish-format path loads an existing `.nnue` through pinned `nnue-rs`
  0.4.0. It is the practical route to a strong pretrained evaluator without
  training a Volkrix model.
- The `VOLKNNUE` path is Volkrix's clean-room HalfKP format and offline training
  pipeline. It exists for controlled experimentation and reproducible ownership
  of feature, export, packing, and loader semantics.

Official release bundles enable the large Stockfish-format network through
same-directory discovery. A source build does the same when that file is installed
beside the executable. `EvalFile` selects another network at runtime, and an empty
value restores the classical evaluator. Tablebase and board-rule results remain
authoritative over both evaluators.

## Pretrained Stockfish-format network

The repository includes a fetcher for the Stockfish 18 network published under
CC0:

```bash
scripts/fetch-stockfish18-net.sh target/release/nn-c288c895ea92.nnue
```

Official release archives already contain that large network beside the engine,
which lets Volkrix discover it automatically. Source builds use the command above
to create the same layout. An explicit `EvalFile` overrides sibling discovery;
an explicit empty value selects classical evaluation for that session.

Fetch the official smaller HalfKAv2_hm evaluator, which is the lower-latency,
lower-memory AArch64 alternative with the current scalar backend:

```bash
scripts/fetch-stockfish18-net.sh --small ./nn-37f18f62d772.nnue
```

The small network is opt-in and is not substituted into official bundles.

The script downloads from the Stockfish testing service and requires this exact
SHA-256 before installing the file:

```text
c288c895ea924429ea9092e3f36b2b3c1f00f2a3a4c759ff7e57e79e3b43e4a7
```

Configure the verified file through UCI:

```text
setoption name EvalFile value /absolute/path/nn-c288c895ea92.nnue
isready
```

Format detection reads the file prefix once. Non-`VOLKNNUE` files are delegated
to Volkrix's pinned MIT fork of `nnue-rs` 0.4.0, which validates and parses
SFNNv10 threat nets, HalfKAv2_hm, HalfKAv2, and HalfKP. Each search thread
pre-creates accumulator frames for the full search horizon. A move push records
only a compact `Move` + `UndoState` delta. The accumulator is materialized on
first evaluation, avoiding both eager inference work and the former two-board,
64-square diff. An unevaluated chain refreshes only the requested leaf.

Inference uses AVX2 on supported x86-64 systems. AArch64 uses bit-exact NEON for
activation and accumulator kernels and the ARM dot-product instruction when the
CPU reports `FEAT_DotProd`; all paths retain a scalar fallback and parity tests.
The evaluator also preserves PSQT and positional components for explicit score
scaling. `SmallEvalFile` can keep synchronized big/small lazy states under the
default-off `DualEvalPolicy` A/B seam. The `small-fallback` policy uses the large
network when the small score lies inside `DualEvalThreshold`; bench counters
record both selections. No dual policy is promoted without paired match evidence.
The first threshold-200 candidate was rejected at 2W/5D/41L in 48 completed
games, so the large network remains authoritative by default.

The post-Stockfish-18 development network format beginning with version
`0x6a448afa` is intentionally rejected. It includes additional pair features and
a different downstream architecture, not merely a renamed SFNNv10 header.
Volkrix will enable it only after a permissively licensed format specification or
independent implementation plus differential oracle coverage establishes exact
behavior; the project does not copy GPL engine implementation code or guess file
layout constants.

## `VOLKNNUE` runtime

The version-1 format begins with the eight-byte `VOLKNNUE` magic and records its
topology and dimensions. Supported topologies are:

- HalfKP 128x2, retained for deterministic integration fixtures
- HalfKP 256x2, the production topology used by the offline packer

The feature space contains 40,960 sparse inputs: perspective king square × ten
own/enemy non-king piece buckets × piece square. The runtime shares immutable
weights and gives each search thread cache-line-aligned, topology-sized
accumulator slabs reserved for all plies. Ordinary push/evaluate/pop performs no
heap allocation and no per-edge `Vec` clone. King moves rebuild the affected
perspective as required.

The checked-in `tests/data/nnue/volkrix-halfkp128x2-test.volknnue` is synthetic.
It validates parsing, topology checks, orientation, and incremental updates; it is
not a playing-strength network.

## Tooling boundary

Offline commands live in the separate `tools/volkrix-nnue` workspace member.
Rust is authoritative for:

- FEN normalization and corpus filtering
- HalfKP feature indices and score orientation
- versioned example and checkpoint metadata
- `VOLKNNUE` quantization, packing, and validation
- paired engine-match reporting

Training uses the CPU backend from Bullet pinned to git revision
`feab6443fc523c9d349427bca2d5bb3c04369420`. Bullet is not linked into the engine
binary.

## End-to-end `VOLKNNUE` workflow

### 1. Prepare positions

Inputs are normalized FEN lines, one position per line. Expand curated seeds:

```bash
cargo run --release -p volkrix-nnue -- expand-fens \
  --input <seed.fens> \
  --output <expanded.fens> \
  --max-plies 4 \
  --branching 3 \
  --max-positions 100000
```

Or generate positions through Volkrix self-play:

```bash
cargo run --release -p volkrix-nnue -- selfplay-fens \
  --input <seed.fens> \
  --output <selfplay.fens> \
  --depth 4 \
  --hash-mb 64 \
  --max-plies 160 \
  --max-positions 1000000
```

`selfplay-fens` also accepts `--movetime-ms` instead of `--depth`. Seed sources,
licenses, engine commit, and generation settings should be archived with every
corpus.

### 2. Export supervised examples

```bash
cargo run --release -p volkrix-nnue -- export-examples \
  --input <positions.fens> \
  --output <examples.txt> \
  --label-mode search \
  --label-depth 4 \
  --tt off \
  --workers 8 \
  --position-filter quiet \
  --label-timeout-ms 30000
```

Search and static-eval label modes are supported. The manifest records the source
commit, topology, label environment, search depth, TT/hash settings, filter, and
timeout. Fresh search services isolate labels from previous TT, network, or
tablebase state. Eligible tablebase-scope positions are excluded.

Targets use the engine's side-to-move score orientation and are clipped to
`[-2000, 2000]` centipawns. Mate scores therefore saturate at the clip boundary.

The versioned text format is:

1. `VOLKRIX_EXAMPLES<TAB>1`
2. a JSON manifest prefixed by `# `
3. a fixed tab-separated column header
4. rows containing FEN, normalized FEN, side to move, raw/target score, and sparse
   active/passive feature lists

Normalized FEN is produced by parsing and reserializing all six fields. The split
is deterministic and independent of input order:

```text
fnv1a64(normalized_fen_utf8) % 10 == 0  -> validation
otherwise                               -> training
```

### 3. Train

```bash
cargo run --release -p volkrix-nnue -- train-bullet \
  --examples <examples.txt> \
  --checkpoint-dir <checkpoint-dir> \
  --superbatches 64 \
  --batch-size 512 \
  --initial-lr 0.001 \
  --final-lr 0.0001
```

Optional flags also control save rate, trainer/loader thread counts, queue size,
and evaluation scale. Continue from a prior compatible checkpoint with:

```bash
cargo run --release -p volkrix-nnue -- train-bullet \
  --examples <stage2.examples> \
  --checkpoint-dir <stage2-checkpoint> \
  --init-from-checkpoint-dir <stage1-checkpoint> \
  --superbatches 64 \
  --initial-lr 0.0002 \
  --final-lr 0.00002
```

The checkpoint directory contains the manifest, little-endian float tensors,
Bullet datasets and metadata, raw Bullet checkpoints, and training logs. Parent
checkpoint provenance is carried into a continued run.

The Bullet graph uses dual-perspective sparse HalfKP inputs, clipped ReLU hidden
activations, AdamW, and a sigmoid score target. Packing quantizes the trained
parameters into the integer layout expected by the runtime.

### 4. Pack and validate

```bash
cargo run --release -p volkrix-nnue -- pack-volknnue \
  --checkpoint-dir <checkpoint-dir> \
  --output <candidate.volknnue>

cargo run --release -p volkrix-nnue -- validate-volknnue \
  --evalfile <candidate.volknnue>
```

Packing writes a `<candidate.volknnue>.manifest.json` sidecar. Traceability is kept
outside the locked version-1 payload so runtime compatibility is not changed by
new training metadata.

## Classical Texel tuning

The same exported examples can tune the parameterized classical evaluator:

```bash
cargo run --release -p volkrix-nnue -- texel-tune \
  --examples <examples.txt> \
  --output <weights.json> \
  --iterations 12 \
  --step 8 \
  --regularization 0.000001
```

Treat emitted weights as a candidate artifact. Validate with
`compare-classical-weights` before changing defaults.

## Strength validation

A lower training or validation loss does not establish engine strength. Compare a
packed candidate against the classical fallback with paired color-reversed games:

```bash
cargo run --release -p volkrix-nnue -- compare-fallback \
  --openings <held-out.fens> \
  --candidate <candidate.volknnue> \
  --movetime-ms 100 \
  --hash-mb 64 \
  --max-plies 240
```

The report includes pentanomial counts, confidence intervals, Elo bounds when
finite, and termination/adjudication counts. Small runs are smoke tests only.
Promote a network only after a large held-out Fastchess/OpenBench experiment with
a predeclared stopping rule.

The largest previously documented local `VOLKNNUE` candidates lost all 16 games
in each of two tiny comparisons against the classical fallback. That result proves
the pipeline exercised real networks; it does not validate those networks for
play. Keep them external, and do not describe the current `VOLKNNUE` trainer output
as stronger until new match evidence says so.

## Reproducibility checklist

For every retained network, archive:

- corpus sources, licenses, normalized corpus checksum, and generation command
- exporter and engine commits plus complete manifest
- split rule and train/validation counts
- Bullet revision, configuration, logs, and parent checkpoint chain
- packed network and sidecar checksums
- compiler/CPU, match binaries, openings checksum, command, and raw game log
- W/D/L, pentanomial counts, confidence interval, time losses, crashes, and
  adjudications

Production `.nnue`, `.volknnue`, Bullet datasets, and checkpoints are intentionally
ignored by git unless a separately reviewed distribution decision is made.
