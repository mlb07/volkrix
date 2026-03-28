# Search

Phases 4 through 10 establish Volkrix's deterministic single-thread search baseline, its first transposition-table layer, the first correctness-first strength passes on top of that baseline, a practical time-controlled UCI runtime, a disciplined classical-eval bridge, and the first conservative SMP / Lazy SMP layer on top of the retained Phase 9 engine.

## Current Shape

- iterative deepening at the root
- alpha-beta negamax core
- quiescence search for tactical stabilization
- principal variation tracking through the existing root/PV bookkeeping
- tapered classical evaluation with middlegame/endgame blending
- transposition table integration with deterministic TT-on and TT-off paths at `Threads=1`
- stronger move ordering through root PV hints, SEE-informed capture buckets, killer moves, and quiet history
- aspiration windows around iterative deepening
- basic quiet-only late move reductions at eligible later quiet moves only
- a conservative Lazy SMP Layer I when `Threads > 1`
- debug-only internal profile hooks for exact Phase 8 baseline, retained Phase 9 default, and Phase 10 thread-count comparisons
- cooperative stop, movetime, clocked search, and infinite-search control in the UCI runtime
- terminal handling for checkmate, stalemate, repetition, fifty-move draw, and insufficient-material draw

## Phase 10 Retained SMP Model

The retained Phase 10 design is deliberately narrow:

- `Threads` is the only new public control surface
- `Threads=1` remains the authoritative retained Phase 9 debug baseline
- one main search thread remains authoritative for orchestration, final `bestmove`, and user-visible PV/info output
- helper workers search cloned root position/state plus thread-local search state
- helper workers must not share mutable `Position` or state-history objects across workers
- helper workers must not emit user-visible info lines
- helper workers must not own or publish final `bestmove` or user-visible PV state
- TT is the only shared mutable search structure
- the worker-pool model is persistent and runtime-owned rather than recreated for every search

Helper diversification remains conservative:

- helpers use the same retained Phase 9 search code and heuristics
- helpers rotate the non-hinted portion of the root move order to avoid exact duplication
- no split points
- no YBWC
- no work stealing
- no MultiPV or separate user-visible analyses

## Shared State and Determinism

Phase 10 keeps shared mutable state intentionally small:

- TT is the only intended shared mutable search structure
- TT synchronization is localized to TT internals with per-cluster locking
- all other search state remains thread-local in workers

Determinism rules:

- `Threads=1` benchmark/profile paths remain the authoritative reproducible baseline
- `Threads>1` results are not required to preserve deterministic move order or checksum
- `Threads>1` results must still remain correct and measurably beneficial

## Phase 10 Benchmark Evidence

Observed fixed-depth depth-5 comparison on the built-in four-position suite:

| Profile | Threads | Nodes | Checksum | Time (ms) | NPS |
| --- | ---: | ---: | --- | ---: | ---: |
| Retained Phase 9 baseline | 1 | 505147 | `244a71a65613ec7f` | 16336 | 30922 |
| Phase 10 default | 1 | 505147 | `244a71a65613ec7f` | 11903 | 42438 |
| Phase 10 default | 2 | 442898 | `244a723163d23d4b` | 12934 | 34242 |
| Phase 10 default | 4 | 370831 | `244a735ef6371ec5` | 14317 | 25901 |

Observed fixed-time comparison at 50 ms per position:

| Profile | Threads | Depth Sum | Nodes | Checksum | Time (ms) |
| --- | ---: | ---: | ---: | --- | ---: |
| Retained Phase 9 baseline | 1 | 10 | 7781 | `a78c4124670c793e` | 205 |
| Phase 10 default | 1 | 10 | 8604 | `a78c413e1f0c793e` | 216 |
| Phase 10 default | 2 | 11 | 11646 | `a78c41611d1f8b3e` | 335 |
| Phase 10 default | 4 | 12 | 10617 | `a78c41411d1e713e` | 358 |

Interpretation:

- `Threads=1` preserves the retained Phase 9 default bench signature exactly
- on this validation machine, the retained SMP Layer I design does not improve fixed-depth elapsed time over the retained `Threads=1` default row
- `Threads=2` and `Threads=4` do improve fixed-time completed depth, reaching depth sums `11` and `12` versus `10` at `Threads=1`
- the retained Layer I SMP design is accepted because `Threads>1` stays correct and shows practical time-to-depth benefit without weakening the authoritative `Threads=1` baseline

## Runtime Notes

- the stdio runtime still uses one input helper thread only to observe `stop` and `quit`
- the main thread remains the sole owner of command application boundaries
- `Hash`, `Clear Hash`, `Threads`, `position`, and `ucinewgame` still apply only after the active search fully unwinds
- worker threads are owned by the persistent search service and shut down explicitly with the runtime

## Deferred Beyond Phase 10

Still deferred beyond this Layer I SMP pass:

- split-point search
- YBWC-style parallel search
- work stealing
- distributed search
- pondering
- MultiPV expansion
- tablebases
- NNUE
- further eval expansion
- further selectivity expansion unrelated to SMP

The Phase 10 goal is a stronger but still disciplined engine: a trusted `Threads=1` baseline plus a conservative helper-worker SMP mode that remains easy to reason about, easy to disable, and safe under the existing runtime/control model.
