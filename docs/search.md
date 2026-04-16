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
- conservative non-PV selective pruning through null move, a tightened shallow reverse futility guard, shallow futility, and shallow late-move pruning
- a conservative Lazy SMP Layer I when `Threads > 1`
- an optional tablebase boundary controlled by `SyzygyPath`
- cooperative stop, movetime, clocked search, and infinite-search control in the UCI runtime
- terminal handling for checkmate, stalemate, repetition, fifty-move draw, and insufficient-material draw

## Current Search Candidate Status

The current committed search bundle reintroduced a staged move picker and selective-search work on top of the older stronger classical baseline.

Current tuning rule:

- for new search work, compare candidate builds directly against current `HEAD`
- older baseline comparisons below are historical context, not promotion targets

Current search-specific changes in the committed tree include:

- staged move picking in `src/search/movepicker.rs`
- principal variation search with scout-window re-search at root and in the main alpha-beta path
- previous-iteration PV reuse below the root when the current search prefix still matches
- continuation-history scoring for quiet replies
- null-move pruning backed by reversible null moves in `Position`
- a slightly more aggressive null-move reduction step from depth `6` upward
- reverse futility pruning for shallow non-PV nodes
- shallow futility pruning for quiet non-check moves
- shallow late-move pruning for very late quiet non-check moves
- qsearch and root search refactors to consume the staged picker
- an external-engine comparison tool in `tools/volkrix-nnue` for same-machine A/B match testing

What is currently proved against the immediate pre-search-change baseline `e9a5a06`:

- proxy bench improved from `623756` nodes / checksum `5873b4276c1d4c51` to `655322` nodes / checksum `5873b44162c8b651`
- wall-clock bench time on the same machine dropped from `803 ms` to `482 ms`
- `cargo test --quiet --lib search::root` passed
- `cargo test --quiet --test search` passed
- `cargo test --quiet --test uci` passed

What is not yet proved:

- no Elo gain is established yet
- quick same-machine matches versus `e9a5a06` were fully neutral in the first samples
- a later local PVS/PV-reuse refresh versus the immediate pre-change binary was small and mixed:
- `24` games over `12` expanded openings at `--movetime-ms 10 --max-plies 60`: `1W 23D 0L`, score `52.1%`, approximate Elo `+14.5`
- `48` games over `24` expanded openings at `--movetime-ms 10 --max-plies 60`: `0W 48D 0L`, score `50.0%`
- `16` games over `8` expanded openings at `--movetime-ms 50 --max-plies 60`: `1W 15D 0L`, score `53.1%`, approximate Elo `+21.7`
- a later local selective-pruning refresh versus the immediate pre-pruning binary was also small and mixed:
- `24` games over `12` expanded openings at `--movetime-ms 10 --max-plies 60`: `1W 23D 0L`, score `52.1%`, approximate Elo `+14.5`
- `16` games over `8` expanded openings at `--movetime-ms 50 --max-plies 60`: `1W 14D 1L`, score `50.0%`
- a later tuned selective-pruning refresh with a more conservative reverse-futility guard was positive in the next local samples:
- `96` games over `48` expanded openings at `--movetime-ms 10 --max-plies 60`: `3W 93D 0L`, score `51.6%`, approximate Elo `+10.9`
- `48` games over `24` expanded openings at `--movetime-ms 50 --max-plies 60`: `1W 47D 0L`, score `51.0%`, approximate Elo `+7.2`
- `4` games over `2` expanded openings at `--movetime-ms 10 --max-plies 40`: `0W 4D 0L`
- `16` games over `8` expanded openings at `--movetime-ms 10 --max-plies 40`: `0W 16D 0L`
- `8` games over `4` expanded openings at `--movetime-ms 50 --max-plies 60`: `0W 8D 0L`

Current interpretation:

- the search bundle clearly changes node counts, checksums, and wall time
- the search bundle is not currently justified as an Elo improvement
- the newer PVS/PV-reuse refresh looks promising in very small samples but remains far too small to treat as a proved strength gain
- the newer selective-pruning refresh looked mixed at first, but the tightened reverse-futility variant was positive in both follow-up samples and is the first `#2` form worth keeping for broader validation
- until a longer match produces a real score edge, treat this as a performance-positive but strength-unproven candidate

Future-agent guidance:

- do not claim Elo gain from this search bundle based on bench movement alone
- if search tuning resumes, compare directly against `e9a5a06` or another explicitly recorded strong baseline binary
- prefer the external-engine match workflow over checksum-only reasoning
- `cargo run -p volkrix-nnue -- expand-fens --input tests/data/nnue/phase13-fixture.fens --output /tmp/volkrix-openings.fens --max-plies 4 --branching 3 --max-positions 96`
- `cargo run -p volkrix-nnue -- compare-engines --openings /tmp/volkrix-openings.fens --baseline <baseline-bin> --candidate <candidate-bin> --movetime-ms <n> --max-plies <n> --max-openings <n>`
- when the handoff notes and the code disagree, trust the code and re-document from the tree instead of inheriting the old note

## Recent Rejected Search Experiments

These local search experiments were tested directly against the committed `ef06683` baseline and were not kept:

- conservative check extension:
- `24` games over `12` openings at `--movetime-ms 10 --max-plies 60`: `0W 24D 0L`
- `16` games over `8` openings at `--movetime-ms 50 --max-plies 80`: `0W 15D 1L`, approximate Elo `-21.7`
- quiet-history maluses for failed quiets:
- `24` games over `12` openings at `--movetime-ms 10 --max-plies 60`: `0W 24D 0L`
- `16` games over `8` openings at `--movetime-ms 50 --max-plies 80`: `0W 16D 0L`
- root fail-high beta-cut inside root aspiration windows:
- `24` games over `12` openings at `--movetime-ms 10 --max-plies 60`: `0W 24D 0L`
- `16` games over `8` openings at `--movetime-ms 50 --max-plies 80`: `1W 15D 0L`, approximate Elo `+21.7`
- larger follow-up `96` games over `48` openings at `--movetime-ms 10 --max-plies 60`: `2W 89D 5L`, approximate Elo `-10.9`
- more aggressive depth-scaled late-move reduction:
- `24` games over `12` openings at `--movetime-ms 10 --max-plies 60`: `0W 24D 0L`
- `16` games over `8` openings at `--movetime-ms 50 --max-plies 80`: `1W 14D 1L`, score `50.0%`

Current interpretation for these rejected passes:

- none of the four changes above showed a durable strength gain against the immediate committed baseline
- node savings alone were not a sufficient promotion signal
- the root fail-high experiment looked promising in a tiny slower bucket and then failed the larger fast follow-up
- treat all four as explored-and-rejected unless a materially different variant is tested

## Current Local Search Keep Candidate

The current in-tree search candidate is a root aspiration-window widening change:

- `search_root_with_aspiration_core` now preserves the non-failing side of the window and widens only the side that failed instead of rebuilding a symmetric window around the original guess on every retry
- intent: reduce avoidable root re-search churn after one-sided aspiration misses without changing the underlying root search or selective-pruning guards
- targeted validation stayed clean:
- `cargo test --quiet --lib search::root`
- `cargo test --quiet --test search`
- `cargo test --quiet --test uci`
- same-machine engine evidence against the latest pre-change `HEAD` snapshot is positive:
- `96` games over `48` openings at `--movetime-ms 10 --max-plies 60`: `4W 91D 1L`, score `51.6%`, approximate Elo `+10.9`
- `192` games over `96` openings at `--movetime-ms 10 --max-plies 60`: `10W 181D 1L`, score `52.3%`, approximate Elo `+16.3`
- `96` games over `48` openings at `--movetime-ms 50 --max-plies 80`: `9W 82D 5L`, score `52.1%`, approximate Elo `+14.5`

Current interpretation for this keep candidate:

- this is still local same-machine evidence, not a large statistically hardened Elo claim
- unlike the recent rejected `HEAD`-only experiments, it stayed positive in the larger fast follow-up and also positive in the slower confirmation bucket
- if search work pauses here, this is the current search-side change worth keeping from this round

Previous retained change from the same `HEAD`-only round:

- `null_move_reduction(depth)` changed from `if depth >= 7 { 3 } else { 2 }` to `if depth >= 6 { 3 } else { 2 }`
- evidence at promotion time:
- `96` games over `48` openings at `--movetime-ms 10 --max-plies 60`: `4W 91D 1L`, score `51.6%`, approximate Elo `+10.9`
- `192` games over `96` openings at `--movetime-ms 10 --max-plies 60`: `7W 184D 1L`, score `51.6%`, approximate Elo `+10.9`
- `96` games over `48` openings at `--movetime-ms 50 --max-plies 80`: `6W 86D 4L`, score `51.0%`, approximate Elo `+7.2`

Additional rejected search experiments from the same round:

- exact countermove ordering for quiet replies:
- `24` games over `12` openings at `--movetime-ms 10 --max-plies 60`: `0W 24D 0L`
- `16` games over `8` openings at `--movetime-ms 50 --max-plies 80`: `1W 14D 1L`, score `50.0%`
- countermove ordering plus qsearch TT exact reuse / depth-0 stores:
- `24` games over `12` openings at `--movetime-ms 10 --max-plies 60`: `0W 24D 0L`
- `16` games over `8` openings at `--movetime-ms 50 --max-plies 80`: `0W 16D 0L`
- root TT stores without changing the root beta-cut behavior:
- `24` games over `12` openings at `--movetime-ms 10 --max-plies 60`: `1W 23D 0L`, score `52.1%`
- `96` games over `48` openings at `--movetime-ms 10 --max-plies 60`: `6W 84D 6L`, score `50.0%`
- `16` games over `8` openings at `--movetime-ms 50 --max-plies 80`: `1W 14D 1L`, score `50.0%`
- root TT stores plus root fail-high beta-cuts:
- `24` games over `12` openings at `--movetime-ms 10 --max-plies 60`: `0W 24D 0L`
- `16` games over `8` openings at `--movetime-ms 50 --max-plies 80`: `2W 14D 0L`, score `56.2%`
- `48` games over `24` openings at `--movetime-ms 50 --max-plies 80`: `3W 41D 4L`, score `49.0%`
- PV-only internal iterative deepening on missing move hints:
- `24` games over `12` openings at `--movetime-ms 10 --max-plies 60`: `0W 23D 1L`, score `47.9%`, approximate Elo `-14.5`

## Current Classical Eval Status

The retained fallback evaluator is still the strongest practical Volkrix evaluator today when `EvalFile=""`.

Recent classical-eval additions on top of the existing tapered material / piece-square / mobility / king-safety / pawn-structure base include:

- pawn-island penalties
- pawn-phalanx bonuses
- backward-pawn penalties
- protected passed-pawn bonuses
- rook-on-seventh bonuses
- supported knight-outpost bonuses

What is currently proved:

- the targeted eval test suite covers these terms directly in `tests/eval.rs`
- the eval path remains deterministic and does not mutate position state in the covered tests
- the classical evaluator now exposes a parameterized `ClassicalEvalWeights` surface while preserving default behavior, so offline tuning can vary weights without changing engine code shape
- the offline toolchain now includes a first-pass `texel-tune` command that reads retained examples files and fits classical weights offline
- a first tuned candidate on `96` depth-2 search-labeled expanded positions improved offline loss:
- train log-loss `0.694556 -> 0.693712`
- validation log-loss `0.691864 -> 0.690567`
- a stronger follow-up corpus on `256` expanded positions with quiet-filtered depth-2 search labels produced `84` examples, but the tuned candidate was not better:
- train log-loss `0.692744 -> 0.692738`
- validation log-loss `0.692891 -> 0.692899`
- a depth-3 pass on a smaller curated expanded corpus improved offline loss but the direct tuned weights were too aggressive to promote unchanged
- a broader depth-2 pass on a `64`-position expanded curated corpus improved offline loss:
- `any` corpus: train log-loss `0.694906 -> 0.693533`, validation log-loss `0.690616 -> 0.688720`
- `quiet` corpus: train log-loss `0.692755 -> 0.692683`
- the low-risk `quiet`-subset candidate was promoted into the current default classical weights with these effective deltas from the old defaults:
- knight mobility `4 -> 7`
- bishop mobility `5 -> 8`
- queen mobility `1 -> 3`
- phalanx pawn bonus `8 -> 12`
- pawn-threat-vs-minor `12 -> 13`
- that candidate produced the first positive same-machine match score against the frozen pre-tuning baseline binary:
- `64` games over `32` expanded curated openings at `--movetime-ms 10 --max-plies 60`: `4W 58D 2L`, score `51.6%`, approximate Elo `+10.9`
- a later conservative backward-pawn penalty was added to the default evaluator with targeted eval coverage:
- `cargo test --quiet --test eval`
- `cargo test --quiet --test search`
- `cargo test --quiet --test uci`
- `cargo test --quiet -p volkrix-nnue parameter_specs_are_unique`
- same-machine engine evidence against the frozen pre-backward-pawn binary is mildly positive but still small:
- `24` games over `12` self-play openings at `--movetime-ms 10 --max-plies 60`: `1W 23D 0L`, score `52.1%`, approximate Elo `+14.5`
- `16` games over `8` self-play openings at `--movetime-ms 50 --max-plies 80`: `0W 16D 0L`, score `50.0%`

What is not yet proved:

- the current positive match result is still a small same-machine sample, not a statistically solid Elo claim
- the backward-pawn term also remains strength-unproven beyond small local samples
- the current tuner is a score-target logistic tuning pass over retained examples data, not yet a large-scale game-result Texel workflow with proven Elo gains
- the first tuned candidate was neutral in a same-machine engine match and was not kept as the default evaluator:
- `16` games over `8` openings at `--movetime-ms 50 --max-plies 60`: `0W 16D 0L`
- several more aggressive depth-3-derived candidates stayed neutral or regressed on broader follow-up matches, so the current default keeps only the low-risk quiet-subset deltas above

Future-agent guidance:

- record any new classical-eval terms and their evidence here or in a more specific eval handoff note
- do not claim Elo gain from classical-eval edits unless match evidence supports it
- if search work is in flight elsewhere, keep eval changes isolated from search-logic edits
- for Texel tuning, start from the new parameterized classical weights in `src/search/eval.rs` instead of editing constants in-place
- current first-pass workflow:
- `cargo run -p volkrix-nnue -- export-examples --input <fens.txt> --output /tmp/volkrix.examples [--label-mode search|static]`
- `cargo run -p volkrix-nnue -- texel-tune --examples /tmp/volkrix.examples --output /tmp/volkrix-weights.json [--iterations N] [--step N] [--sigmoid-scale F] [--regularization F] [--max-examples N]`
- after tuning, treat the emitted weights JSON as a candidate artifact first; do not replace default eval weights without a match result
- the most useful workflow so far was:
- `cargo run -p volkrix-nnue -- expand-fens --input <seed-fens> --output /tmp/volkrix-expanded.fens --max-plies 4 --branching 3 --max-positions 64`
- `cargo run -p volkrix-nnue -- export-examples --input /tmp/volkrix-expanded.fens --output /tmp/volkrix-expanded-d2-quiet.examples --label-depth 2 --label-mode search --workers 4 --tt off --position-filter quiet`
- `cargo run -p volkrix-nnue -- texel-tune --examples /tmp/volkrix-expanded-d2-quiet.examples --output /tmp/volkrix-expanded-d2-quiet.weights.json --iterations 12 --step 8 --regularization 0.000001`
- promote only the low-risk overlapping deltas that survive eval tests and then validate with `compare-engines`
- for baseline-vs-candidate classical testing, prefer `cargo run -p volkrix-nnue -- compare-engines --openings <fens.txt> --baseline <baseline-bin> --candidate <candidate-bin> [--movetime-ms N | --depth N]`

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
