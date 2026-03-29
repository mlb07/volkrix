# NNUE Training

Phase 13 adds Volkrix's first retained offline NNUE training and net-iteration path. It does not add new runtime or UCI controls.

## Retained Runtime Baseline

Authoritative retained baseline:

- commit `594cb6bcc6b29dd2d13647b1162b882a66cde264`
- `EvalFile=""`, `SyzygyPath=""`, `Threads=1` remains the exact fixed-depth deterministic baseline
- tablebases still outrank NNUE on retained eligible positions
- TT remains the only shared mutable search structure

## Offline Tooling Boundary

Phase 13 uses a locked hybrid split:

- Rust remains authoritative for:
  - HalfKP feature semantics
  - example-format semantics
  - VOLKNNUE packing
  - final load validation
- Python is allowed only for offline optimization/training
- no Python dependency is introduced into the engine runtime path

Retained Rust offline tooling lives in the isolated workspace member `tools/volkrix-nnue`.

Retained Python training code lives under `tools/nnue/`.

## Retained Corpus Input

The retained larger-corpus input format is FEN lines:

- one position per line
- no retained PGN ingestion
- no retained UCI move-sequence ingestion

Retained tiny in-repo fixture corpus:

- `tests/data/nnue/phase13-fixture.fens`

## Retained Label Environment

Phase 13 labels are generated only in this explicit environment:

- `Threads=1`
- `SyzygyPath=""`
- `EvalFile=""`
- `SearchLimits::new(5).without_tt()`

No retained label may depend on hidden runtime state.

The exporter creates a fresh isolated search service for each label so prior TT contents, network state, tablebase state, and deferred UCI/runtime configuration cannot leak into labeling.

## Retained Target Semantics

The retained scalar target scheme is:

- deterministic fixed-depth 5 Volkrix search
- score orientation matches the engine's existing side-to-move convention
- direct rules remain authoritative
- retained Phase 11 tablebase-scope positions are excluded from the corpus

Target conversion:

- take the side-to-move search output
- convert it directly into the retained scalar target orientation
- clip to `[-2000, 2000]` centipawns

Phase 13 Layer I does not preserve a separate mate-distance target scheme. Mate scores saturate into the retained clip band.

## Normalized FEN And Split Rule

For the retained split rule, "normalized FEN" means:

- parse the source line with `Position::from_fen`
- reserialize it with `Position::to_fen`
- use that canonical six-field FEN string exactly

The retained train/validation split is:

- `fnv1a64(normalized_fen_utf8) % 10 == 0` => validation
- all other positions => training

Record-order splitting is not retained.

## Retained Example Format

The retained example file is a versioned text file:

1. line 1: `VOLKRIX_EXAMPLES<TAB>1`
2. line 2: JSON manifest prefixed with `# `
3. line 3: fixed tab-separated column header
4. remaining lines: tab-separated examples

Retained manifest fields include:

- exporter version
- source engine commit
- topology id
- feature count
- hidden size
- output input count
- label depth
- target clip
- explicit label environment
- normalized-FEN rule

Retained row columns are:

- `fen`
- `normalized_fen`
- `side_to_move`
- `raw_score_cp`
- `target_cp`
- `active_features`
- `passive_features`

Feature columns are comma-separated sparse HalfKP feature indices already derived from the authoritative Rust feature encoder and already oriented for active/passive side-to-move use.

## Checkpoint And Pack Artifacts

Retained trainer checkpoint directory contents:

- `manifest.json`
- `input_weights.f32le`
- `hidden_biases.f32le`
- `output_weights.f32le`
- `output_bias.f32le`

Retained checkpoint manifest fields include:

- trainer version
- seed
- optimizer
- loss
- split rule
- example counts
- retained topology metadata
- source export manifest

Retained packed net traceability lives alongside the packed VOLKNNUE file only:

- packed net: `<name>.volknnue`
- pack manifest sidecar: `<name>.volknnue.manifest.json`

It is not embedded inside the packed VOLKNNUE payload because the Phase 12 VOLKNNUE file format remains locked.

## Commands

Export retained examples:

```bash
cargo run -p volkrix-nnue -- export-examples \
  --input tests/data/nnue/phase13-fixture.fens \
  --output /tmp/volkrix-phase13-fixture.examples
```

Train retained checkpoint:

```bash
python3 tools/nnue/train_halfkp128x2.py \
  --examples /tmp/volkrix-phase13-fixture.examples \
  --checkpoint-dir /tmp/volkrix-phase13-checkpoint
```

Pack retained VOLKNNUE:

```bash
cargo run -p volkrix-nnue -- pack-volknnue \
  --checkpoint-dir /tmp/volkrix-phase13-checkpoint \
  --output /tmp/volkrix-phase13.volknnue
```

Validate retained VOLKNNUE:

```bash
cargo run -p volkrix-nnue -- validate-volknnue \
  --evalfile /tmp/volkrix-phase13.volknnue
```

## Fixture Smoke Path

Retained fixture smoke flow:

1. export from `tests/data/nnue/phase13-fixture.fens`
2. train the retained Python checkpoint
3. pack to VOLKNNUE
4. validate load through the retained Phase 12 loader

This is a smoke path only. It is not a strength corpus.

## Real Candidate-Net Validation

Phase 13 also requires one real locally trained candidate net outside the repo:

- loaded through `EvalFile`
- validated at `Threads=1`
- optionally smoke-checked at `Threads=2`

Required candidate-vs-fallback sanity comparison:

- candidate net through `EvalFile`
- fallback with `EvalFile=""`
- `Threads=1`
- `Hash=64`
- `SyzygyPath=""`

This is practical sanity evidence, not an Elo claim.

Manual validation hooks:

- `cargo test real_net_smoke_threads_one -- --ignored --nocapture`
- `cargo test real_net_smoke_threads_two -- --ignored --nocapture`
- `cargo test phase_thirteen_candidate_vs_fallback_sanity_report -- --ignored --nocapture`

Set `VOLKRIX_EVALFILE` to the local trained candidate-net path before running those commands.

## Synthetic Test Net Versus Real Trained Nets

Phase 12 synthetic integration asset:

- `tests/data/nnue/volkrix-halfkp128x2-test.volknnue`

That file remains:

- deterministic
- synthetic
- intended only for parser/integration validation

Phase 13 real trained candidate nets are separate local artifacts produced by the retained export/train/pack workflow and are not checked into the repo.
