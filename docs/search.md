# Search

Phases 4 through 13 establish Volkrix's deterministic single-thread baseline, its TT-backed search layers, practical UCI runtime behavior, the first conservative SMP layer, the first optional tablebase / probe integration, the first optional NNUE evaluator path, and now the first offline NNUE training / packing layer on top of that retained engine.

Phase 13 does not widen the engine runtime surface. The retained runtime shape documented below remains authoritative. The offline export / training / packing workflow is documented separately in `docs/nnue-training.md`.

## Current Shape

- iterative deepening at the root
- alpha-beta negamax core
- quiescence search for tactical stabilization
- principal variation tracking through the existing root/PV bookkeeping
- tapered classical evaluation as the retained fallback evaluator
- an optional NNUE evaluator boundary controlled by `EvalFile`
- transposition table integration with deterministic TT-on and TT-off paths at `Threads=1`
- stronger move ordering through root PV hints, SEE-informed capture buckets, killer moves, and quiet history
- aspiration windows around iterative deepening
- basic quiet-only late move reductions at eligible later quiet moves only
- a conservative Lazy SMP Layer I when `Threads > 1`
- an optional tablebase boundary controlled by `SyzygyPath`
- cooperative stop, movetime, clocked search, and infinite-search control in the UCI runtime
- terminal handling for checkmate, stalemate, repetition, fifty-move draw, and insufficient-material draw

## Current Classical Eval Status

The retained fallback evaluator is still the strongest practical Volkrix evaluator today when `EvalFile=""`.

Recent classical-eval additions on top of the existing tapered material / piece-square / mobility / king-safety / pawn-structure base include:

- pawn-island penalties
- pawn-phalanx bonuses
- protected passed-pawn bonuses
- rook-on-seventh bonuses
- supported knight-outpost bonuses

What is currently proved:

- the targeted eval test suite covers these terms directly in `tests/eval.rs`
- the eval path remains deterministic and does not mutate position state in the covered tests

What is not yet proved:

- these terms do not yet have match evidence proving Elo gain
- they should be treated as plausible classical improvements until proxy bench and match testing support them

Future-agent guidance:

- record any new classical-eval terms and their evidence here or in a more specific eval handoff note
- do not claim Elo gain from classical-eval edits unless match evidence supports it
- if search work is in flight elsewhere, keep eval changes isolated from search-logic edits

## Current NNUE Runtime Model

The retained NNUE runtime design is still deliberately narrow:

- `EvalFile` is the only new public control surface
- NNUE is optional and disabled by default
- the runtime owns an optional internal NNUE service behind `search::nnue`
- the retained network format is a clean-room Volkrix-owned `VOLKNNUE` binary format only
- the runtime supports only retained clean-room HalfKP topologies
- retained production topology is `HalfKP 256x2`
- retained compatibility support includes the synthetic in-repo `HalfKP 128x2` test net
- one active network only
- one retained feature scheme only
- one retained accumulator/update architecture only
- no external `.nnue` compatibility in this phase
- helpers remain silent and non-authoritative for user-visible publication
- TT remains the only shared mutable search structure
- network weights are shared read-only across workers

When `EvalFile` is empty:

- `Threads=1` preserves the authoritative retained Phase 11 fixed-depth deterministic baseline exactly
- `Threads>1` preserves the retained Phase 11 SMP behavior
- the classical evaluator remains the active path
- runtime/deferred-command semantics remain unchanged

## Retained HalfKP-Like Feature Scheme

The retained feature space is:

- `64` normalized king squares
- `10` non-king piece buckets
- `64` normalized piece squares
- total feature count: `64 * 10 * 64 = 40,960`

The `10` retained non-king buckets are explicitly:

1. own pawn
2. own knight
3. own bishop
4. own rook
5. own queen
6. enemy pawn
7. enemy knight
8. enemy bishop
9. enemy rook
10. enemy queen

Squares are normalized per perspective so each accumulator sees its own side from the same board orientation. Kings are not encoded as input pieces; instead, the king square selects the active `HalfKP` slice for that perspective.

## Retained Topology and Score Orientation

The retained production topology is:

- one shared input-to-hidden matrix: `40960 x 256`
- one hidden bias vector: `256`
- one output head over concatenated perspective activations: `512 -> 1`

The runtime remains compatibility-capable for the synthetic in-repo `HalfKP 128x2` test asset, but new retained production checkpoints and packed nets target `HalfKP 256x2`.

The retained numeric path is:

- input weights: `i16`
- hidden biases: `i16`
- accumulator lanes: `i32`
- output weights: `i16`
- output bias: `i32`
- activation: clipped ReLU to `[0, 255]`
- final score: output sum divided by the stored output scale

Final NNUE score orientation in engine terms:

- positive scores favor the side to move
- negative scores favor the opponent

That matches Volkrix's retained classical static-eval convention, so search integration does not need a separate score-orientation bridge.

## Evaluator Boundary and Authority Rules

Phase 12 introduces a clean evaluator boundary:

- retained classical evaluation when `EvalFile` is empty
- NNUE evaluation when a network is loaded successfully

The retained authority rules remain unchanged:

- direct mate/stalemate/repetition/fifty-move/insufficient-material handling stays authoritative before evaluator choice matters
- when `SyzygyPath` is enabled and a position is tablebase-resolved within the retained Phase 11 scope, tablebase handling remains authoritative and NNUE must not override that result
- helpers do not emit user-visible info lines
- helpers do not own or publish final `bestmove` or user-visible PV state

## Thread-Local Accumulator / Update Architecture

Accumulator state is deliberately kept out of `Position`.

The retained model is:

- search-local accumulator state stored in `SearchContext`
- one accumulator for White-king perspective
- one accumulator for Black-king perspective
- exact root build from the current position at search start
- stack-based restoration on unmake

Retained incremental update rules:

- ordinary non-king moves patch both perspectives incrementally by removing the old feature and adding the new feature
- captures, promotions, and en passant apply exact piece add/remove deltas
- if a king moves, that side's perspective accumulator is rebuilt from the child position instead of patching king-indexed features incrementally
- castling uses the king-move rebuild for the moving side's perspective and a simple rook delta for the opposite perspective
- unmake restoration uses accumulator stack pop, not reverse-delta reconstruction

## Tiny Test Net and Real-Net Policy

Phase 12 includes one deterministic in-repo integration net:

- `tests/data/nnue/volkrix-halfkp128x2-test.volknnue`

This file is:

- clean-room and Volkrix-owned
- minimal and synthetic
- a compatibility asset, not the retained production topology target
- intended for parser, accumulator, and inference validation only
- explicitly not treated as a production playing net

Optional ignored real-net smoke tests may be run with `VOLKRIX_EVALFILE`, but that is validation convenience only and is not required for Phase 12 completion.

## Determinism and Validation Rules

- `EvalFile` empty / `Threads=1` fixed-depth benchmark and profile paths remain the authoritative reproducible baseline
- NNUE-enabled runs are correctness, integration, and benefit checks, not checksum-equality requirements
- `Threads>1` NNUE-enabled runs are not required to preserve deterministic move order or checksum
- `Threads>1` NNUE-enabled runs must still remain correct

## Phase 12 Evidence

No-network fixed-depth baseline preservation:

| Profile | Nodes | Checksum |
| --- | ---: | --- |
| Retained Phase 11 baseline / `EvalFile` empty / `SyzygyPath` empty / `Threads=1` | 505147 | `244a71a65613ec7f` |
| Phase 12 default / `EvalFile` empty / `SyzygyPath` empty / `Threads=1` | 505147 | `244a71a65613ec7f` |

Targeted tiny-test-net validation:

| Scenario | Purpose |
| --- | --- |
| `tests/eval.rs::tiny_nnue_eval_returns_finite_scores_on_curated_positions` | finite-score inference sanity |
| `src/search/nnue.rs` incremental-update tests | full rebuild vs incremental exactness |
| `tests/uci.rs::nnue_enabled_go_depth_returns_a_legal_move` | public `EvalFile` activation and legal move production |
| `src/search/service.rs` NNUE threaded tests | `Threads=1` and `Threads=2` correctness with shared read-only weights |

Manual benchmark/report hooks are available through:

- `cargo test --test tt phase_twelve_nnue_profile_report -- --ignored --nocapture`

That report prints:

- retained Phase 11 baseline / `EvalFile` empty / `SyzygyPath` empty / `Threads=1`
- Phase 12 default / `EvalFile` empty / `SyzygyPath` empty / `Threads=1`
- targeted tiny-test-net checks at `Threads=1`
- targeted tiny-test-net checks at `Threads=2`

## Deferred Beyond Phase 12

Still deferred beyond this first NNUE engine-integration layer:

- external `.nnue` compatibility
- training pipeline work
- self-play data generation
- tuner infrastructure
- network architecture search
- broader feature-family experimentation
- extra public NNUE knobs
- broad classical-eval deletion
- broader eval/search co-design work

The Phase 12 goal is a still-trusted retained Phase 11 engine when `EvalFile` is empty, plus a clean, optional, testable NNUE inference path that can later support training-pipeline and tuning work without destabilizing the current search/runtime substrate.
