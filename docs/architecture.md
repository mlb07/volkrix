# Volkrix Architecture

Volkrix is organized around one authoritative mutable `Position`, reversible
state transitions, and a search service that owns long-lived shared resources.
The Rust library crate exists for the binary, tests, and offline tools; it is not
yet a stable public API.

## Runtime data flow

```text
UCI stdin
   |
   v
UciEngine ---- position/options ----> UciSearchService
                                         |
                 +-----------------------+-----------------------+
                 |                       |                       |
                 v                       v                       v
          main search thread      persistent helpers      optional services
                 |                (root-sharded SMP)       NNUE / Fathom
                 +-----------------------+-----------------------+
                                         |
                                         v
                               shared lock-free TT
```

The stdio runtime uses a small input thread so `stop` and `quit` can interrupt a
search. The main runtime remains the owner of the UCI state and applies position
or option changes only after the current search unwinds.

## Chess core

`Position` maintains redundant representations deliberately:

- a 64-square mailbox
- piece bitboards and color/all occupancies
- fixed-capacity piece lists and cached king squares
- side to move, castling rights, en-passant square, and move clocks
- an incrementally maintained Zobrist key
- bounded repetition history

Moves, captures, promotions, en passant, and castling all use the same
make/unmake path used by search. `UndoState` contains exactly the information
needed to restore the previous state. Null moves have a separate reversible
state used only by selective search.

Legal move generation is staged for all moves, captures, quiets, or evasions.
Pinned pieces and check masks are computed directly; en-passant candidates use a
temporary make/unmake validation because removing both pawns can reveal a distant
line attack on the king.

The position's normal Zobrist key describes board state. Its TT search key adds
the rule-50 clock in constant time. Repetition history is intentionally not part
of that key: path-dependent draws are checked before a TT probe and are not
stored, allowing genuine transpositions reached through different histories to
share entries.

## Search resources

`UciSearchService` owns:

- the configured hash size and shared transposition table
- a persistent helper-worker pool sized on demand
- an optional read-only NNUE network service
- an optional Syzygy/Fathom service
- optional classical weights used by offline tooling

Each search thread owns its position copy, search stack, move-ordering histories,
PV storage, and NNUE accumulator state. The TT is the shared hot-path structure.
It is a cache-line-aligned, four-way clustered table with atomic 16-byte logical
entries, checksum-style key verification, depth/age replacement, and no mutex on
probe or store.

For `Threads > 1`, the main thread remains authoritative for the published move
and PV. Helpers receive deterministic root-move shards, search silently, and
contribute TT knowledge. Node, TT-hit, and seldepth statistics include completed
helper work. Scheduling and shared-TT timing still mean multithreaded searches
should not be treated as bit-for-bit deterministic.

## Evaluation order

Search resolves values in this order:

1. board-rule terminal states and path-dependent draws
2. eligible Syzygy results
3. configured NNUE evaluation
4. tapered classical evaluation

The default classical evaluator includes tapered material and piece-square terms,
mobility, king safety, pawn structure, passed pawns, rook activity, outposts, and
static threats. `EvalFile` replaces its static score but does not bypass chess
rules or a tablebase result.

The in-tree `VOLKNNUE` backend owns the file format and HalfKP feature semantics.
The Stockfish-format backend is an adapter over pinned `nnue-rs`. Both share
weights across workers and allocate per-thread accumulator frames up front.

## Tablebases

The vendored Fathom C source is built by `build.rs` and hidden behind
`TablebaseService`. Fathom has process-global state, so configuration changes are
exclusive against every probe; concurrent WDL calls retain Fathom's documented
thread safety through a shared read guard. A service identity prevents an old
`Arc` from probing after reconfiguration. Root DTZ calls have their additional
non-thread-safe guard.
Search uses root DTZ probing where available and non-root WDL probing. Volkrix
admits only castling-free positions within `SyzygyProbeLimit` and the cardinality
captured when Fathom loaded the files. Atomics account for root/WDL attempts,
hits, misses, and errors without adding a hot-path mutex. Fathom failures fall
back to search and remain visible through per-search UCI diagnostics.

## UCI and failure boundaries

The parser validates commands before mutating engine state. Unsupported `go`
arguments are errors. A failed `SyzygyPath` or `EvalFile` load leaves the prior
working service in place. During active stdio search, state-changing commands are
deferred until the search has stopped.

The engine reports iterative `info` lines containing depth, seldepth, score,
nodes, NPS, elapsed time, TT hits, and a legal PV. A search that attempted
tablebase probes also reports the probe delta and last failure, if any, followed
by one `bestmove`.

## Build and dependency boundaries

- stable Rust, edition 2024
- no external chess crate in board, rules, move generation, or search logic
- `nnue-rs` is used only for external NNUE parsing and inference
- vendored Fathom is used only for Syzygy probing
- Bullet and corpus tooling stay in the separate `volkrix-nnue` workspace member
- production networks and tablebases are external artifacts, not repository data

See [`search.md`](search.md) for the search design and
[`nnue-training.md`](nnue-training.md) for the offline network workflow.
