# NNUE Training

Phase 13 adds Volkrix's first retained offline NNUE training and net-iteration path. It does not add new runtime or UCI controls.

## Retained Runtime Baseline

Authoritative retained baseline:

- commit `594cb6bcc6b29dd2d13647b1162b882a66cde264`
- `EvalFile=""`, `SyzygyPath=""`, `Threads=1` remains the exact fixed-depth deterministic baseline
- tablebases still outrank NNUE on retained eligible positions
- TT remains the only shared mutable search structure

## Offline Tooling Boundary

Phase 13 uses a retained Rust-first split:

- Rust remains authoritative for:
  - HalfKP feature semantics
  - example-format semantics
  - VOLKNNUE packing
  - final load validation
- the isolated offline trainer uses Bullet's CPU backend through the `volkrix-nnue` tool crate
- no Bullet dependency is introduced into the engine runtime path

Retained Rust offline tooling lives in the isolated workspace member `tools/volkrix-nnue`.

## Retained Corpus Input

The retained larger-corpus input format is FEN lines:

- one position per line
- no retained PGN ingestion
- no retained UCI move-sequence ingestion

Retained tiny in-repo fixture corpus:

- `tests/data/nnue/phase13-fixture.fens`

## Retained Label Environment

Every exported label is generated only in an explicit isolated environment:

- `Threads=1`
- `SyzygyPath=""`
- `EvalFile=""`
- an explicit search/static label mode recorded in the export manifest
- explicit search depth, TT setting, hash size, worker count, position filter, and optional timeout recorded in the export manifest

No retained label may depend on hidden runtime state.

The exporter creates fresh isolated search services for labeling so prior TT contents, network state, tablebase state, and deferred UCI/runtime configuration cannot leak into the corpus.

## Retained Target Semantics

The retained scalar target scheme is:

- explicit Volkrix-generated scalar supervision recorded by the export manifest
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
- `train.bulletdata`
- `validation.bulletdata`
- `bullet-training.json`
- `bullet-checkpoints/`

Retained checkpoint manifest fields include:

- trainer version
- seed
- optimizer
- loss
- split rule
- example counts
- retained topology metadata
- source export manifest
- optional parent-checkpoint traceability for two-stage training:
  - `init_from_mode`
  - `init_from_checkpoint_dir`
  - `init_from_checkpoint_trainer_version`
  - `init_from_checkpoint_epochs`
  - `init_from_checkpoint_examples_path`

Retained Bullet sidecar metadata records:

- Bullet git revision
- backend (`Bullet CPU`)
- activation (`crelu` in the training graph, matching runtime clipped ReLU after `QA=255` scaling)
- eval scale
- batch size
- batches per superbatch
- superbatch range
- learning-rate schedule endpoints
- data file paths
- optional parent-checkpoint traceability:
  - `init_from_mode`
  - `init_from_checkpoint_dir`
  - `init_from_bullet_checkpoint`
  - `init_from_checkpoint_trainer_version`
  - `init_from_checkpoint_epochs`
  - `init_from_checkpoint_examples_path`

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

Train retained checkpoint with Bullet:

```bash
cargo run -p volkrix-nnue -- train-bullet \
  --examples /tmp/volkrix-phase13-fixture.examples \
  --checkpoint-dir /tmp/volkrix-phase13-checkpoint
```

Initialize a new Bullet run from a prior Volkrix checkpoint:

```bash
cargo run -p volkrix-nnue -- train-bullet \
  --examples /tmp/volkrix-phase13-fixture.examples \
  --checkpoint-dir /tmp/volkrix-phase13-stage2-checkpoint \
  --init-from-checkpoint-dir /tmp/volkrix-phase13-checkpoint \
  --superbatches 64 \
  --initial-lr 0.0002 \
  --final-lr 0.00002
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
2. train the retained Bullet checkpoint
3. pack to VOLKNNUE
4. validate load through the retained Phase 12 loader

This is a smoke path only. It is not a strength corpus.

## Bullet Training Notes

The retained Bullet trainer is pinned to:

- Bullet git revision `feab6443fc523c9d349427bca2d5bb3c04369420`
- CPU backend only
- retained HalfKP `40960 -> 256x2 -> 1` production graph
- deterministic normalized-FEN split
- `AdamW`
- sigmoid-squared-error value loss

The training graph uses normalized hidden-layer parameters with `QA=255` and exports them back into Volkrix's runtime integer layout after training:

- `l0` weights and biases are multiplied by `255`
- `l1` weights are multiplied by `400 * 64 / 255`
- `l1` bias is multiplied by `400 * 64`

That scaling is what keeps the retained VOLKNNUE pack path compatible with the runtime loader without changing the engine.

Two-stage training uses checkpoint initialization, not raw Bullet schedule continuation:

- `--init-from-checkpoint-dir` resolves the prior Volkrix checkpoint directory
- the tool loads the prior Bullet weights from that checkpoint
- stage 2 then trains over the full new corpus from batch 0 with its own local superbatch schedule
- parent-checkpoint traceability is recorded in both `manifest.json` and `bullet-training.json`

This keeps the fine-tune corpus fully visible to stage 2 instead of inheriting Bullet's dataset-offset behavior from schedule resume.

## Canonical Two-Stage Workflow

Canonical current experiment:

1. Stage 1 pretrain on a large self-play search-labeled corpus with cheap depth-1 labels
2. Stage 2 fine-tune from that checkpoint on a smaller higher-quality depth-2 quiet self-play search corpus
3. pack and validate the final VOLKNNUE
4. compare it against fallback at `Threads=1`, `Hash=64`, `SyzygyPath=""`

Example commands:

```bash
target/release/volkrix-nnue export-examples \
  --input /tmp/volkrix-bullet-selfplay-40000-d1.fens \
  --output /tmp/volkrix-bullet-selfplay-40000-d1-search1-w8.examples \
  --label-mode search \
  --label-depth 1 \
  --tt on \
  --hash-mb 64 \
  --workers 8

target/release/volkrix-nnue train-bullet \
  --examples /tmp/volkrix-bullet-selfplay-40000-d1-search1-w8.examples \
  --checkpoint-dir /tmp/volkrix-bullet-two-stage-stage1-checkpoint \
  --batch-size 512 \
  --superbatches 128

target/release/volkrix-nnue train-bullet \
  --examples /tmp/volkrix-bullet-selfplay-40000-d2w8-quiet-timeout100.examples \
  --checkpoint-dir /tmp/volkrix-bullet-two-stage-stage2-checkpoint \
  --init-from-checkpoint-dir /tmp/volkrix-bullet-two-stage-stage1-checkpoint \
  --batch-size 512 \
  --superbatches 64 \
  --initial-lr 0.0002 \
  --final-lr 0.00002

target/release/volkrix-nnue pack-volknnue \
  --checkpoint-dir /tmp/volkrix-bullet-two-stage-stage2-checkpoint \
  --output /tmp/volkrix-bullet-two-stage-stage2.volknnue

target/release/volkrix-nnue validate-volknnue \
  --evalfile /tmp/volkrix-bullet-two-stage-stage2.volknnue

target/release/volkrix-nnue compare-fallback \
  --openings /tmp/volkrix-bullet-eval-openings.fens \
  --candidate /tmp/volkrix-bullet-two-stage-stage2.volknnue \
  --movetime-ms 50 \
  --hash-mb 64 \
  --max-plies 120
```

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
- a runtime compatibility asset for the older `HalfKP 128x2` shape
- intended only for parser/integration validation

Phase 13+ real trained candidate nets are separate local artifacts produced by the retained export/train/pack workflow. New retained production checkpoints and packed nets target `HalfKP 256x2` and are not checked into the repo.
