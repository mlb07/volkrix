# Search Handoff

This note tracks the accepted non-NNUE, non-eval search work that landed after the original Phase 9 baseline.

## Scope

- search flow, ordering, pruning, and qsearch only
- no classical-eval edits
- no NNUE runtime edits
- no UCI-surface edits beyond what existing search paths already consume

## Accepted Search Changes

- root fail-high handling now beta-cuts correctly
- principal variation search is active at root and in the main alpha-beta path
- move ordering uses a staged move picker instead of repeated full-list rescoring
- previous-iteration PV reuse now works below the root when the current prefix still matches
- countermove ordering is active for quiet replies
- quiet alpha-improving best moves feed back into quiet-history ordering
- capture ordering uses SEE buckets plus a light victim/aggressor tie-break
- late move reductions scale with move lateness instead of always reducing by one ply
- qsearch prunes clearly losing captures with SEE
- qsearch probes/stores depth-0 TT entries and honors TT move hints
- qsearch uses delta pruning for low-gain capture and promotion candidates
- null-move pruning is enabled with a real reversible null move in `Position`
- reverse futility pruning is enabled for shallow non-PV nodes
- shallow futility pruning is enabled for quiet non-check moves
- shallow late-move pruning is enabled for very late quiet non-check moves

## Supporting Engine Changes

- `Position` now supports `make_null_move` / `unmake_null_move`
- `Position::has_non_pawn_material` exists to guard selective pruning in low-material cases
- `SearchHeuristics` now exposes explicit toggles for the accepted selective-search features

## Validation Pattern

Accepted search changes were kept only when they passed:

- `cargo test --quiet --lib search::root`
- `cargo test --quiet --test search`
- `cargo test --quiet --test uci`
- `cargo run --quiet --release -- bench`

## Current Caveat

Exact retained benchmark signatures in `tests/tt.rs` are sensitive to concurrent eval work. If classical-eval edits are in flight, treat immediate same-tree before/after comparisons as authoritative for search experiments, then rebaseline `tests/tt.rs` once eval and search are settled together.
