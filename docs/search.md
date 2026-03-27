# Search

Phases 4 through 6 establish Volkrix's deterministic single-thread search baseline, its first transposition-table layer, and the first correctness-first strength pass on top of that baseline.

## Current Shape

- iterative deepening at the root
- alpha-beta negamax core
- quiescence search for tactical stabilization
- principal variation tracking
- simple classical evaluation based on material and square bonuses
- transposition table integration with TT-on and TT-off deterministic paths
- stronger move ordering through root PV hints, SEE-informed capture buckets, killer moves, and quiet history
- aspiration windows around iterative deepening
- debug-only internal profile hooks for exact Phase 5 fallback and Phase 6 on/off regression work
- reproducible bench path tied to the real search core
- terminal handling for checkmate, stalemate, repetition, fifty-move draw, and insufficient-material draw

## Phase 6 Retained Heuristics

Deterministic bench comparison against the Phase 5 baseline at depth 5:

| Profile | Nodes | Checksum |
| --- | ---: | --- |
| Phase 5 baseline | 443712 | `b1ac3c8fc22df05f` |
| PV move ordering | 419646 | `b1ac31913a29371c` |
| SEE capture buckets | 443627 | `b1ac3c8fc579f05f` |
| Killer moves | 395956 | `b1ac308520268256` |
| Quiet history | 382507 | `9bd9952d43de544e` |
| Aspiration windows | 403124 | `b1ac3c9e6a216548` |
| Full Phase 6 retained set | 339774 | `b1ac360363bad479` |

The retained Phase 6 set is:

- root PV move ordering
- SEE-informed capture bucket ordering
- killer moves for quiet beta cutoffs
- quiet-history ordering
- aspiration windows

The debug-only `phase5_baseline` profile preserves the old Phase 5 behavior, including Phase 5 quiescence move ordering.

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
- time-management expansion beyond `go depth`
- evaluation overhaul
- SMP
- NNUE and tablebases

## Intentional Limits

- no SMP
- no null move or LMR in the kept Phase 6 set
- no public UCI-surface controls for internal heuristics
- no time management expansion beyond `go depth`

The goal of these phases is correctness-first search infrastructure and measured, debuggable search-strength growth, not SMP or tuning sprawl.
