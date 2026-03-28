# Search

Phases 4 through 9 establish Volkrix's deterministic single-thread search baseline, its first transposition-table layer, the first correctness-first strength pass on top of that baseline, a practical time-controlled UCI runtime around the same single-thread core, a disciplined classical-eval bridge, and the next narrow depth/selectivity layer on top of that Phase 8 baseline.

## Current Shape

- iterative deepening at the root
- alpha-beta negamax core
- quiescence search for tactical stabilization
- principal variation tracking through the existing root/PV bookkeeping
- tapered classical evaluation with middlegame/endgame blending
- transposition table integration with deterministic TT-on and TT-off paths
- stronger move ordering through root PV hints, SEE-informed capture buckets, killer moves, and quiet history
- aspiration windows around iterative deepening
- basic quiet-only late move reductions at eligible later quiet moves only
- debug-only internal profile hooks for exact Phase 8 baseline, LMR-only, and full retained Phase 9 comparisons
- documented null-move evaluation evidence, but no retained null-move pruning in the current tree
- reproducible bench path tied to the real search core
- UCI-only persistent TT reuse with `Hash` / `Clear Hash`
- cooperative stop, movetime, clocked search, and infinite-search control in the UCI runtime
- terminal handling for checkmate, stalemate, repetition, fifty-move draw, and insufficient-material draw

## Phase 9 Retained Heuristics

The retained Phase 9 set is:

- basic quiet-only LMR

### Basic quiet-only LMR

LMR is allowed only when all of the following are true:

- node is not root
- node is not on the preserved PV path
- side to move is not in check
- remaining depth is at least 4 plies
- move is quiet
- move is not a promotion
- move does not give check
- move is not the hash move
- move is not among the first 3 ordered quiet moves searched at the node

Phase 9 keeps LMR conservative:

- only a 1-ply reduction
- reduced search first
- mandatory full-depth re-search if the reduced result improves alpha or fails high
- no multi-level reductions
- no reduction tables
- no reductions on captures, promotions, checking moves, PV moves, or root moves

## Phase 9 Null-Move Evaluation

Guarded null-move pruning was evaluated during Phase 9 and rejected from the retained engine.

The evaluated null-move variant used the intended conservative guardrails:

- node was not root
- node was not on the preserved PV path
- side to move was not in check
- remaining depth was at least 3 plies
- direct draw/mate/stalemate handling ran before null-move eligibility
- no repeated consecutive null moves
- no null move when the side to move had no non-pawn material
- null-window search only
- reduction `R = 2` at depths 3 through 5
- reduction `R = 3` at depths 6 and above
- no verification search
- no adaptive null-move formulas

It was not retained because the evidence did not clear a stricter future/Elo-oriented bar:

- as a standalone profile it reduced nodes only modestly versus the exact Phase 8 baseline but ran slower
- when added on top of LMR, the combined path still ran slower than LMR-only on the same deterministic bench
- that was not strong enough to justify keeping the extra complexity in the retained single-thread search

## Phase 9 Evidence Bench

Observed deterministic depth-5 profile comparison on the built-in four-position suite:

| Profile | Nodes | Checksum | Time (ms) |
| --- | ---: | --- | ---: |
| Phase 8 baseline | 541650 | `244a715de801bc83` | 6934 |
| Null move only, evaluated and rejected | 536751 | `244a716f0e89aa83` | 7405 |
| LMR only | 505147 | `244a71a65613ec7f` | 6111 |
| LMR plus null move, evaluated but not retained | 496914 | `244a71b8d70abc7f` | 6509 |

The current retained `phase9_default` path matches the LMR-only profile at `505147` nodes and checksum `244a71a65613ec7f`.

Retention reasoning:

- LMR was retained because the conservative quiet-only form stayed correctness-safe, remained locally explainable, and improved node and time behavior clearly versus the exact Phase 8 baseline
- guarded null move was rejected because it did not demonstrate a clear enough benefit beyond LMR-only to justify its added complexity in this engine
- the current retained `phase9_default` path therefore matches the LMR-only profile

## Phase 7 Runtime Notes

- the search core remains single-threaded
- the stdio runtime uses one helper thread only to observe `stop` and `quit` while search is active
- the main thread remains the sole owner of TT resize/clear operations and all position mutation
- `Hash`, `Clear Hash`, `position`, and `ucinewgame` never race with live TT probe/store activity because they are applied only after the active search fully unwinds
- fixed-depth bench and regression helpers still run through deterministic internal profile paths

## Deferred Or Rejected

Still deferred beyond Phase 9:

- move-count pruning
- futility pruning
- razoring
- probcut / multi-cut
- singular extensions
- countermove history
- internal iterative deepening as a new feature
- SMP
- NNUE and tablebases

Rejected or not added in Phase 9:

- guarded null-move pruning
- null-move verification search
- adaptive null-move formulas
- multi-level or table-driven LMR
- public UCI-surface controls for null move or LMR

## Intentional Limits

- no SMP or shared-search work in the retained Phase 9 set
- no public UCI-surface heuristic controls
- no protocol expansion beyond the Phase 7 UCI surface
- no classical eval expansion in Phase 9
- no attempt to turn Phase 9 into a kitchen-sink heuristic dump

The goal of Phase 9 is a stronger but still disciplined single-thread search: a narrow depth/selectivity layer with exact Phase 8 baseline preservation, explicit debug isolation, and evidence-backed retention of only the heuristics that clearly earn their keep.
