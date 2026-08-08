# Search

Volkrix uses iterative-deepening principal-variation search with a classical or
NNUE static evaluator. The search is designed around three constraints: chess
rules are resolved before cache reuse, all speculative state changes are exactly
reversible, and strength changes require game evidence rather than benchmark
movement alone.

## Iterative deepening and root search

Depths are searched in order so each completed iteration supplies the next one
with a score, root ordering, and PV. From depth two onward the root begins with an
aspiration window around the previous score and expands only the bound that
failed. The root and internal nodes use PVS: the first candidate receives a full
window; later candidates receive a scout window and are re-searched when they can
raise alpha.

The main thread publishes only fully completed iterative results. Cooperative
soft deadlines stop after a completed iteration; hard deadlines and `stop`
interrupt node expansion. Mate scores are ply-normalized when stored in the TT and
mate-distance bounds tighten the alpha-beta window as the search descends.

## Move ordering

The staged picker prioritizes:

1. a legal TT/PV move
2. tactical moves ordered by SEE-informed capture buckets
3. killer moves
4. quiets ordered by history and continuation history

Successful cutoffs reward the responsible move. Failed quiet candidates receive
history maluses, and capture-history updates are implemented for controlled
tuning even though capture-history scoring is not enabled by default. Previous
iteration PV hints are reused only while the current PV prefix still matches and
only at PV nodes.

## Selectivity

The default search combines:

- precomputed logarithmic late-move reductions, with a full-depth re-search after
  a reduced move improves alpha; contextual adjustments are implemented for
  controlled tuning but disabled by default after profiling regressed
- guarded null-move pruning in positions with non-pawn material
- reverse futility, shallow futility, and late-move pruning
- SEE and quiet-history pruning for sufficiently late moves
- ProbCut on promising tactical candidates
- internal iterative reduction when a deeper node has no TT move

Two additional mechanisms remain experimental and default-off:

- correction history reconstructs pawn and per-color non-pawn structure keys
  only when explicitly enabled. It never changes the raw static evaluation stored
  in the TT. A 300-game paired test scored 70 wins, 137 draws, and 93 losses
  (46.17%, about -26.7 Elo), so the default path allocates no correction tables
  and pays no incremental position-key maintenance.
- singular extension performs a reduced null-window search at a non-root node
  while excluding one legal TT move. It requires depth 8, an exact/lower TT
  result within three plies of the requested depth, and an ordinary non-mate,
  non-tablebase score. Exclusion nodes cannot reuse or publish a TT result for
  the incomplete move set, probe tablebases, recursively exclude another move,
  or apply forward pruning. Only a verified fail-low extends the TT move by one
  ply. Multi-cut behavior is deliberately not implemented.

The initial four-position depth-9 release profile for singular extension reduced
nodes from 3,447,998 to 1,194,220 and runtime from 4.58 s to 1.94 s, with 99
verification searches and 10 extensions. Two evaluations changed while all four
best moves remained stable. Its subsequent 200-game paired match scored 60 wins,
76 draws, and 64 losses (49.0%, about -7 Elo). The result did not justify
promotion, so singular extension remains disabled.

Every pruning family has explicit depth, PV, check, mate-score, and move-type
guards. The rules are covered by focused tests and can be toggled internally for
A/B profiling. They are heuristics, not proofs; the validation process in
[`search-handoff.md`](search-handoff.md) is the authority for retaining a tuning
change.

## Quiescence

At the nominal depth limit, quiescence continues tactical play to reduce horizon
effects. A non-check node starts from stand pat and searches legal tactical moves;
an in-check node searches all evasions and cannot stand pat. Losing captures can
be rejected by SEE. Quiescence probes and stores depth-zero TT entries, reuses a
cached static evaluation where safe, honors mate-distance bounds, and checks the
same rule/tablebase terminal conditions as the main search.

## Draws, tablebases, and TT order

The order of operations is deliberate:

1. stop/deadline checks
2. repetition, fifty-move, insufficient-material, mate, and stalemate handling
3. eligible tablebase result
4. TT probe
5. search or static evaluation

Repetition is path-dependent, so a repetition result is never stored. The TT key
contains the board Zobrist key and the rule-50 clock but not the repetition path.
This preserves correct rule-50 values while allowing transpositions reached by
different move orders to share search work.

Syzygy probing uses root DTZ and internal WDL results for castling-free positions
with no more than seven pieces, subject to `SyzygyProbeLimit` and the cardinality
actually loaded by Fathom. `SyzygyProbeLimit=0` disables probes without unloading
the configured files. Exact board rules and tablebase values outrank NNUE or
classical evaluation.

With `Syzygy50MoveRule=true`, root DTZ receives the position's halfmove clock.
Because Fathom's non-root WDL entry point has no rule-50-clock argument, those
probes are conservatively limited to a zero halfmove clock. Disabling the option
allows WDL probes at any clock and promotes cursed/blessed outcomes to
unconditional wins/losses. Probe errors never become tablebase scores: search
continues normally and UCI diagnostics report the failure.

## Transposition table

The TT is shared by all threads and contains cache-line-aligned clusters of four
entries. A logical entry occupies two `AtomicU64` words:

- a compact payload with move, score, static eval, depth, bound, and generation
- a verification word formed from the position key and payload

A reader accepts an entry only when the two words reconstruct its requested key.
This checksum-style protocol makes a concurrently mixed read behave like an
ordinary 64-bit collision instead of accepting a torn payload. Probe, store, and
clear use atomics rather than a cluster mutex. Replacement favors the matching
key, then an empty slot, then the lowest depth adjusted for generation age.

The TT is disposable acceleration. Correctness must not depend on an entry being
present, surviving replacement, or arriving in a particular order.

## Parallel search

`Threads=1` runs the main search only. With more threads, a persistent pool starts
helper searches over deterministic round-robin root-move shards. All workers have
private positions, heuristics, PV buffers, and accumulator stacks; only the TT and
immutable evaluator/tablebase services are shared.

The main thread remains authoritative for `bestmove` and user-visible PV output.
Helpers warm the TT and their completed nodes, TT hits, and seldepth are folded
into the final statistics. A helper panic is contained and accounted for so the
pool can recover instead of leaving an active-worker count stuck.

Root sharding prevents every helper from beginning with the same root list, but
this remains shared-TT Lazy SMP rather than split-point work stealing. OS
scheduling can change which TT entries win races, so multithreaded runs are not a
fixed-node reproducibility mode.

## Evaluation

When `EvalFile` is empty, search uses the tapered classical evaluator. When a
network is loaded, each thread owns a reusable incremental state:

- `VOLKNNUE`: topology-sized, cache-line-aligned accumulator slabs reserved for
  the full search horizon; ordinary push/pop performs no allocation or `Vec` clone
- Stockfish format: pre-created `nnue-rs` accumulator frames and a zero-copy view
  of the parent board reconstructed from the child position, move, and undo state

Weights are immutable and shared through `Arc`. King moves trigger the refreshes
required by the selected feature architecture; ordinary moves use incremental
updates. Both paths are tested against fresh evaluation across normal captures,
en passant, castling, promotions, and long make/unmake sequences.

An optional production dual state can synchronize `EvalFile` and
`SmallEvalFile`. `DualEvalPolicy=off` is the default and selects the original
big-only path without constructing the dual wrapper. The experimental
`small-fallback` policy evaluates the small network first and uses the big
network inside `DualEvalThreshold` centipawns. Bench output reports
`small_selected` and `big_fallbacks` so thresholds can be frozen before paired
strength testing:

```bash
target/release/volkrix bench --depth 6 --threads 1 \
  --evalfile /tmp/nn-c288c895ea92.nnue \
  --small-evalfile /tmp/nn-37f18f62d772.nnue \
  --dual-policy small-fallback --dual-threshold 200
```

Throughput does not imply strength here. A direct 48-game sanity match rejected
threshold 200 with 2 wins, 5 draws, and 41 losses (9.4%) against big-only. The
policy therefore remains default-off. Higher fallback thresholds may be tested
as separate candidates, but none is promoted without new paired match evidence.

The Stockfish adapter accepts the architectures supported by the pinned
`nnue-rs` fork: SFNNv10 threat networks, HalfKAv2_hm, HalfKAv2, and HalfKP. It
uses AVX2 on x86-64 and stable NEON/DotProd kernels with scalar tails on AArch64.
All optimized kernels have scalar parity tests.

Representative Apple M4 release measurements (one thread, August 2026):

| Artifact | Four-position depth-7 nodes | Median time | NPS |
| --- | ---: | ---: | ---: |
| Generic release, SFNNv10 big | 187,557 | 451 ms | 415.9k |
| M4-native PGO, SFNNv10 big | 187,557 | 419 ms | 447.6k |

The structural lazy/direct-delta and SIMD work reduced an isolated 10,000-cycle
quiet push/evaluate/pop median from 50.125 ms to 16.503 ms. The PGO artifact was
about 7.1% faster at one thread and 7.0% faster at two threads in the production
bench; its 100-game generic-build A/B was directionally positive but inconclusive
(26 wins, 50 draws, 24 losses).

These performance observations are not Elo measurements. A separate final
paired, color-reversed Fastchess gate on the same Apple M4 used 100 games from
the official `8moves_v3.pgn` suite at `0.1+0.01`, `Threads=1`, `Hash=64`, and the
default 10 ms move overhead. The big SFNNv10 network scored 88 wins, 8 losses,
and 4 draws (90.0%) against the small network, with no time forfeits. This short
local match is not a universal rating, but it decisively makes the big network
the maximum-strength recommendation; the small network remains the useful
throughput/memory alternative.

Against the frozen pre-roadmap Volkrix binary in a separate 100-game paired,
color-reversed match at `1+0.01`, `Threads=1`, and `Hash=64`, the final M4 PGO
artifact scored 45 wins, 34 draws, and 21 losses (62.0%, about +85.0 Elo,
99.96% likelihood of superiority), with no crashes, hangs, illegal moves, or
time forfeits.

## Time management and reporting

`go movetime` uses one hard deadline after subtracting `Move Overhead`. Clock mode
reserves the configured overhead plus a small fraction of remaining time, derives
a soft budget from moves-to-go and increment, and caps a larger hard budget by
the available clock. Completed-iteration best-move and score stability contract
the soft target for stable positions or extend unstable searches toward—but
never beyond—the hard deadline. The previous iteration cost is used to avoid
starting work unlikely to finish in the remaining adaptive budget. Fixed
`movetime` searches are deliberately exempt from this contraction. Integer
arithmetic is saturating and deadlines are checked for overflow.

`go nodes N` uses one exact atomic budget shared by the main thread and every
helper. It can be combined with a depth or time limit; the first exhausted limit
stops the search.

Each completed iteration may publish:

```text
info depth D seldepth S score cp|mate V nodes N nps R time MS tthits H pv ...
```

With SMP, the last line is rewritten to include completed helper statistics.
Only the main thread emits protocol output.

## Current limitations

- No MultiPV or `go mate`
- No split-point/work-stealing SMP
- No bundled tablebase files
- No statistically hardened public Elo rating is asserted by this repository

These are release facts, not hidden roadmap phases. Priorities are kept in
[`roadmap.md`](roadmap.md).
