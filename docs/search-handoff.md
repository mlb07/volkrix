# Search Handoff

This note tracks the accepted non-NNUE, non-eval search work that landed after the original Phase 9 baseline.

## Scope

- search flow, ordering, pruning, and qsearch only
- no classical-eval edits
- no NNUE runtime edits
- no UCI-surface edits beyond what existing search paths already consume
- for new tuning work, promote changes only if they beat current `HEAD`
- if an initial result is close, require a larger follow-up bucket before promotion

## Accepted Search Changes

- root fail-high handling now beta-cuts correctly
- principal variation search is active at root and in the main alpha-beta path
- move ordering uses a staged move picker instead of repeated full-list rescoring
- previous-iteration PV reuse now works below the root when the current prefix still matches
- previous-iteration PV hints below the root are now only reused on PV nodes
- countermove ordering is active for quiet replies
- quiet alpha-improving best moves feed back into quiet-history ordering
- capture ordering uses SEE buckets plus a light victim/aggressor tie-break
- late move reductions scale with move lateness instead of always reducing by one ply
- null-move pruning is enabled with a real reversible null move in `Position`
- reverse futility pruning is enabled for shallow non-PV nodes
- shallow futility pruning is no longer enabled in the current phase-9 default
- shallow late-move pruning is enabled for very late quiet non-check moves
- shallow late-move pruning now stops at depth `2` instead of depth `3`
- late move reductions now start at depth `5` instead of depth `4`
- null-move pruning now uses the deeper `R=3` reduction from depth `6` upward instead of depth `7`
- null-move pruning now requires `static_eval >= beta + 32` before it is eligible
- null-move pruning now skips depth `3` nodes and only starts at depth `4`
- reverse futility pruning now uses a more conservative `140 * depth` margin instead of `120 * depth`
- root aspiration re-search now widens only the side that failed instead of rebuilding a symmetric window around the original guess
- qsearch now skips non-promotion captures with `SEE <= 0` when not in check

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

## Current Evidence

- current `HEAD` keep candidate: qsearch now skips non-promotion captures only when `SEE < 0`
- targeted validation stayed clean:
- `cargo test --quiet --lib search::root`
- `cargo test --quiet --test search`
- `cargo test --quiet --test uci`
- `cargo run --quiet --release -- bench`
- direct same-machine engine evidence versus the latest pre-change `HEAD` snapshot is positive:
- `96` games over `48` openings at `--movetime-ms 10 --max-plies 60`: `3W 92D 1L`, score `51.0%`, approximate Elo `+7.2`
- `192` games over `96` openings at `--movetime-ms 10 --max-plies 60`: `9W 178D 5L`, score `51.0%`, approximate Elo `+7.2`
- `96` games over `48` openings at `--movetime-ms 50 --max-plies 80`: `5W 87D 4L`, score `50.5%`, approximate Elo `+3.6`

Previous retained search evidence from the same round:

- `alpha_beta_core` now only applies `previous_pv_move(ply)` when `node_state.is_pv`
- `96` games over `48` openings at `--movetime-ms 10 --max-plies 60`: `2W 94D 0L`, score `51.0%`, approximate Elo `+7.2`
- `192` games over `96` openings at `--movetime-ms 10 --max-plies 60`: `7W 181D 4L`, score `50.8%`, approximate Elo `+5.4`
- `96` games over `48` openings at `--movetime-ms 50 --max-plies 80`: `10W 82D 4L`, score `53.1%`, approximate Elo `+21.7`

- `SearchHeuristics::phase9_default()` now sets `futility_pruning: false`
- `96` games over `48` openings at `--movetime-ms 10 --max-plies 60`: `2W 93D 1L`, score `50.5%`, approximate Elo `+3.6`
- `192` games over `96` openings at `--movetime-ms 10 --max-plies 60`: `6W 182D 4L`, score `50.5%`, approximate Elo `+3.6`
- `96` games over `48` openings at `--movetime-ms 50 --max-plies 80`: `8W 83D 5L`, score `51.6%`, approximate Elo `+10.9`

- late-move pruning now only fires through depth `2` instead of depth `3`
- `96` games over `48` openings at `--movetime-ms 10 --max-plies 60`: `2W 94D 0L`, score `51.0%`, approximate Elo `+7.2`
- `192` games over `96` openings at `--movetime-ms 10 --max-plies 60`: `10W 174D 8L`, score `50.5%`, approximate Elo `+3.6`
- `96` games over `48` openings at `--movetime-ms 50 --max-plies 80`: `9W 85D 2L`, score `53.6%`, approximate Elo `+25.4`
- late move reductions now start at depth `5` instead of depth `4`
- `96` games over `48` openings at `--movetime-ms 10 --max-plies 60`: `4W 91D 1L`, score `51.6%`, approximate Elo `+10.9`
- `192` games over `96` openings at `--movetime-ms 10 --max-plies 60`: `9W 179D 4L`, score `51.3%`, approximate Elo `+9.0`
- `96` games over `48` openings at `--movetime-ms 50 --max-plies 80`: `5W 87D 4L`, score `50.5%`, approximate Elo `+3.6`
- null-move pruning now requires `static_eval >= beta + 32` before it is eligible
- `96` games over `48` openings at `--movetime-ms 10 --max-plies 60`: `4W 89D 3L`, score `50.5%`, approximate Elo `+3.6`
- `192` games over `96` openings at `--movetime-ms 10 --max-plies 60`: `9W 175D 8L`, score `50.3%`, approximate Elo `+1.8`
- `96` games over `48` openings at `--movetime-ms 50 --max-plies 80`: `10W 82D 4L`, score `53.1%`, approximate Elo `+21.7`
- qsearch now skips non-promotion captures with `SEE <= 0` when not in check
- `96` games over `48` openings at `--movetime-ms 10 --max-plies 60`: `2W 93D 1L`, score `50.5%`, approximate Elo `+3.6`
- `192` games over `96` openings at `--movetime-ms 10 --max-plies 60`: `10W 174D 8L`, score `50.5%`, approximate Elo `+3.6`
- `96` games over `48` openings at `--movetime-ms 50 --max-plies 80`: `6W 86D 4L`, score `51.0%`, approximate Elo `+7.2`
- qsearch now skips clearly losing non-promotion captures by SEE when not in check
- `96` games over `48` openings at `--movetime-ms 10 --max-plies 60`: `2W 94D 0L`, score `51.0%`, approximate Elo `+7.2`
- `192` games over `96` openings at `--movetime-ms 10 --max-plies 60`: `5W 185D 2L`, score `50.8%`, approximate Elo `+5.4`
- `96` games over `48` openings at `--movetime-ms 50 --max-plies 80`: `8W 87D 1L`, score `53.6%`, approximate Elo `+25.4`
- `search_root_with_aspiration_core` now keeps the non-failing side of the window fixed and widens only the side that missed
- `96` games over `48` openings at `--movetime-ms 10 --max-plies 60`: `4W 91D 1L`, score `51.6%`, approximate Elo `+10.9`
- `192` games over `96` openings at `--movetime-ms 10 --max-plies 60`: `10W 181D 1L`, score `52.3%`, approximate Elo `+16.3`
- `96` games over `48` openings at `--movetime-ms 50 --max-plies 80`: `9W 82D 5L`, score `52.1%`, approximate Elo `+14.5`
- `null_move_reduction(depth)` changed from `if depth >= 7 { 3 } else { 2 }` to `if depth >= 6 { 3 } else { 2 }`
- `96` games over `48` openings at `--movetime-ms 10 --max-plies 60`: `4W 91D 1L`, score `51.6%`, approximate Elo `+10.9`
- `192` games over `96` openings at `--movetime-ms 10 --max-plies 60`: `7W 184D 1L`, score `51.6%`, approximate Elo `+10.9`
- `96` games over `48` openings at `--movetime-ms 50 --max-plies 80`: `6W 86D 4L`, score `51.0%`, approximate Elo `+7.2`

## Current Caveat

Exact retained benchmark signatures in `tests/tt.rs` are sensitive to concurrent eval work. If classical-eval edits are in flight, treat immediate same-tree before/after comparisons as authoritative for search experiments, then rebaseline `tests/tt.rs` once eval and search are settled together.
