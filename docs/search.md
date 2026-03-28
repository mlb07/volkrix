# Search

Phases 4 through 8 establish Volkrix's deterministic single-thread search baseline, its first transposition-table layer, the first correctness-first strength pass on top of that baseline, a practical time-controlled UCI runtime around the same single-thread core, and a disciplined classical-eval bridge on top of that search/runtime foundation.

## Current Shape

- iterative deepening at the root
- alpha-beta negamax core
- quiescence search for tactical stabilization
- principal variation tracking
- tapered classical evaluation with middlegame/endgame blending
- transposition table integration with TT-on and TT-off deterministic paths
- stronger move ordering through root PV hints, SEE-informed capture buckets, killer moves, and quiet history
- aspiration windows around iterative deepening
- debug-only internal profile hooks for exact Phase 5 fallback and Phase 6 on/off regression work
- reproducible bench path tied to the real search core
- UCI-only persistent TT reuse with `Hash` / `Clear Hash`
- cooperative stop, movetime, clocked search, and infinite-search control in the UCI runtime
- terminal handling for checkmate, stalemate, repetition, fifty-move draw, and insufficient-material draw
- a compact Phase 8 classical eval term set: mobility, king safety, pawn structure, passed pawns, bishop pair, rook file terms, and small static threats

## Phase 8 Eval Bridge

The retained Phase 8 eval terms are:

- tapered material and piece-square scoring
- pseudo-legal mobility for knights, bishops, rooks, and queens using existing attack/query helpers only
- king safety through pawn-shield and open-file exposure terms around the king
- pawn-structure penalties for isolated and doubled pawns
- rank-scaled passed-pawn bonuses with stronger endgame weighting
- bishop-pair bonus
- rook bonuses on open and semi-open files
- basic static threats from pawn attacks on enemy minor/rook/queen targets and minor-piece attacks on enemy rook/queen targets

Intentionally omitted optional subterms in Phase 8:

- legal-move or make/unmake-dependent mobility
- king attack-pressure counting beyond the simple static shelter/file terms
- backward pawns
- pawn islands
- passed-pawn support/blockage adjustments
- rook on the seventh rank
- eval caches or pawn hash tables

These were omitted because they were either noisier, more expensive, harder to test cleanly, or pushed Phase 8 beyond the intended bridge scope.

## Phase 7 Runtime Notes

- the search core remains single-threaded
- the stdio runtime uses one helper thread only to observe `stop` and `quit` while search is active
- the main thread remains the sole owner of TT resize/clear operations and all position mutation
- `Hash`, `Clear Hash`, `position`, and `ucinewgame` never race with live TT probe/store activity because they are applied only after the active search fully unwinds
- fixed-depth bench and regression helpers still run through the deterministic fresh-TT path

## Phase 7 vs Phase 8 Bench Note

Observed `cargo run -- bench` comparison at depth 5 on the built-in four-position suite:

| Version | Nodes | Checksum | Time (ms) | NPS |
| --- | ---: | --- | ---: | ---: |
| Phase 7 baseline | 339774 | `b1ac360363bad479` | 6587 | 51582 |
| Phase 8 current | 541650 | `244a715de801bc83` | 13621 | 39765 |

The richer Phase 8 eval intentionally changes search trajectories, so the historical Phase 5 through Phase 7 node counts and checksums are no longer expected to stay fixed. What remains required is deterministic reproducibility within the current Phase 8 code, plus a documented comparison against the Phase 7 baseline.

## Phase 6 Retained Heuristics

Current deterministic depth-5 profile comparison under the Phase 8 eval:

| Profile | Nodes | Checksum |
| --- | ---: | --- |
| Phase 5 baseline | 450433 | `244a7eff15a69f5c` |
| PV move ordering | 497836 | `244a7eef3b1587b0` |
| SEE capture buckets | 450630 | `244a7eff108a9f5c` |
| Killer moves | 411418 | `244a780e123c6557` |
| Quiet history | 403866 | `244a78f494312dc9` |
| Aspiration windows | 564407 | `244a778a9615c8a8` |
| Full Phase 6 retained set | 541650 | `244a715de801bc83` |

The retained Phase 6 set is:

- root PV move ordering
- SEE-informed capture bucket ordering
- killer moves for quiet beta cutoffs
- quiet-history ordering
- aspiration windows

The debug-only `phase5_baseline` profile still preserves the old Phase 5 search behavior, including Phase 5 quiescence move ordering. Under the richer Phase 8 eval it is used as a deterministic fallback/profile path, not as a promise that pre-Phase-8 bench signatures remain unchanged.

## Phase 6 Deferred Or Rejected

Not retained in Phase 6:

- countermove ordering
- null-move pruning
- late move reductions

Still deferred beyond Phase 6:

- move-count pruning
- futility pruning
- razoring
- internal iterative reductions
- singular extensions
- SMP
- NNUE and tablebases

## Intentional Limits

- no SMP
- no null move or LMR in the kept Phase 6 set
- no public UCI-surface controls for internal heuristics
- no SMP or shared-search work in the Phase 7 runtime model
- no attempt to turn Phase 8 into a full kitchen-sink eval rewrite

The goal of these phases is correctness-first search infrastructure, measured and debuggable search-strength growth, a practical but still tightly constrained UCI shell, and a disciplined classical-eval bridge before any SMP, NNUE, tablebase, or tuning work.
