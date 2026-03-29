# Search

Phases 4 through 11 establish Volkrix's deterministic single-thread baseline, its TT-backed search layers, practical UCI runtime behavior, the first conservative SMP layer, and the first optional tablebase / probe integration on top of that retained engine.

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
- an optional tablebase boundary controlled by `SyzygyPath`
- cooperative stop, movetime, clocked search, and infinite-search control in the UCI runtime
- terminal handling for checkmate, stalemate, repetition, fifty-move draw, and insufficient-material draw

## Phase 11 Retained Tablebase Model

The retained Phase 11 design is deliberately narrow:

- `SyzygyPath` is the only new public control surface
- tablebase probing is optional and disabled by default
- the runtime owns an optional internal tablebase service behind a backend-agnostic boundary
- the approved retained backend is a vendored copy of `jdart1/Fathom`
- tablebase files are user-provided runtime assets, never repo-managed artifacts
- helpers must not emit user-visible info lines
- helpers must not own or publish final `bestmove` or user-visible PV state
- TT remains the only shared mutable search structure
- tablebase access is shared read-only state only

The retained supported probe scope is:

- no castling rights
- at most 6 pieces total
- backend-specific exact eligibility checks

Current backend-specific exact eligibility checks for the approved Fathom Layer I integration are:

- root resolution requires loaded Fathom tables for the current material cardinality
- non-root outcome substitution requires loaded Fathom tables for the current material cardinality
- non-root outcome substitution is limited to positions with `halfmove_clock == 0`, because the retained threaded non-root probe path uses Fathom's thread-safe WDL probing API

## Retained Probe Semantics

- direct mate/stalemate/repetition/fifty-move/insufficient-material handling stays authoritative and ahead of any probe attempt
- root positions may resolve directly through Fathom root probing on the main thread only
- helper workers do not perform root probe resolution and do not publish final root state
- eligible non-root positions may substitute an exact tablebase outcome score instead of normal expansion
- the non-root substitution path uses backend-provided WDL semantics, not naive handcrafted endgame rules

The retained tablebase score mapping is:

- `Win` maps into a positive tablebase score band with ply-aware shaping
- `Loss` maps into the corresponding negative tablebase score band
- `Draw`, `CursedWin`, and `BlessedLoss` map to `0` so fifty-move-aware draw semantics stay correct in search
- the current tablebase score band is `20000`, explicitly kept below mate-score thresholds so tablebase wins/losses do not collide semantically with mate handling

## Backend Boundary and Approval

Phase 11 keeps the backend boundary explicit:

- search/runtime/UCI code talks only to the internal `search::tablebase` service boundary
- the approved retained backend is vendored exactly from `jdart1/Fathom` revision `c9c6fef0dddc05d2e242c183acf5833149ab676d`
- no probing code was transliterated, re-ported, or cherry-picked from Stockfish, Cfish, Pyrrhic, `pyrrhic-rs`, `shakmaty-syzygy`, `python-chess`, or any other unapproved source
- the vendored backend remains localized behind the internal boundary so later backend changes, if ever approved, do not require protocol or search-architecture rewrites

## Disabled-Path Preservation

When `SyzygyPath` is empty:

- `Threads=1` preserves the authoritative retained Phase 10 fixed-depth deterministic baseline exactly
- `Threads>1` preserves the retained Phase 10 SMP behavior
- root-state preservation remains unchanged
- existing runtime/deferred-command semantics remain unchanged

## Determinism and Validation Rules

- no-tablebase `Threads=1` fixed-depth benchmark/profile paths remain the authoritative reproducible baseline
- tablebase-enabled runs are correctness and benefit checks, not checksum-equality requirements
- `Threads>1` tablebase-enabled runs are not required to preserve deterministic move order or checksum
- `Threads>1` tablebase-enabled runs must still remain correct

## Phase 11 Evidence

No-tablebase fixed-depth baseline preservation:

| Profile | Nodes | Checksum |
| --- | ---: | --- |
| Retained Phase 10 baseline / `SyzygyPath` empty / `Threads=1` | 505147 | `244a71a65613ec7f` |
| Phase 11 default / `SyzygyPath` empty / `Threads=1` | 505147 | `244a71a65613ec7f` |

Mock-backed targeted root-resolution validation on a legal 3-piece KQK position:

| Scenario | Best Move | Nodes |
| --- | --- | ---: |
| No tablebase / `Threads=1` | `d3a6` | 38100 |
| Mock tablebase enabled / `Threads=1` | `d3d7` | 0 |
| Mock tablebase enabled / `Threads=2` | `d3d7` | 0 |

Real asset-backed Fathom validation completed on this machine with:

- `/tmp/volkrix-syzygy-min/KQvK.rtbw`
- `/tmp/volkrix-syzygy-min/KQvK.rtbz`
- env-gated ignored Fathom tests passing for both `Threads=1` and `Threads=2`

Direct UCI sanity on the same legal 3-piece KQK position:

| Scenario | Best Move | Nodes | Info |
| --- | --- | ---: | --- |
| Real Fathom enabled / `Threads=1` | `d3a3` | 0 | `info depth 0 score cp 20000 ... pv d3a3` |
| Real Fathom enabled / `Threads=2` | `d3a3` | 0 | `info depth 0 score cp 20000 ... pv d3a3` |

Interpretation:

- the authoritative no-tablebase baseline remains unchanged
- the retained root-resolution semantics are validated under both `Threads=1` and `Threads=2`
- the mock-backed validation rows remain useful correctness-focused checks, not reproducibility requirements
- the real asset-backed validation rows confirm that the approved Fathom backend resolves eligible root positions immediately under both `Threads=1` and `Threads=2`

## Deferred Beyond Phase 11

Still deferred beyond this first tablebase / probe layer:

- repo-managed tablebase assets
- tablebase download/install tooling
- extra public tablebase knobs
- broader endgame tooling
- NNUE
- further eval expansion
- search/selectivity expansion unrelated to this probe layer

The Phase 11 goal is a still-trusted no-tablebase engine by default, plus a clean, testable, license-safe tablebase integration that uses the approved Fathom backend without destabilizing the retained Phase 10 search/runtime substrate.
