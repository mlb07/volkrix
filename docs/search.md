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

Nine additional mechanisms remain experimental and default-off. Four new
candidate families are deliberately compiled only in tests, debug builds, or an
`internal-testing` build, so a production release has neither their UCI surface
nor their hot-path branches:

- `ExperimentalOrderedProbCut` replaces raw legal-order probing with a bounded
  tactical prefix ordered by legal TT/PV priority, capture history, SEE bucket,
  and victim/attacker value. Promotions and SEE-negative captures are filtered
  before the eight-candidate limit is applied. The ordinary full-search picker
  remains unchanged.
- `ExperimentalCaptureHistory` gives the existing capture table independently
  tunable success and failure scales, bounded decay interval and retention,
  periodic saturation control, and guarded quiescence cutoff training. It
  remains a Stockfish-18-NNUE candidate rather than inheriting the earlier
  classical fixed-depth rejection.
- `ExperimentalMultiPlyContinuationHistory` adds independent two- and four-ply
  continuation tables to the proven one-ply table. Lookup weights are 1, 1/2,
  and 1/4; training weights are 1, 3/4, and 1/2. Tables are allocated only when
  the candidate is enabled.
- `ExperimentalContextualLmr` retains the same full-depth re-search safety rule
  while exposing bounded cut-node, PV-node, improving, and positive/negative
  history adjustments to SPSA. The logarithmic base reduction and every search
  guard remain intact.

Each option is default-false and clears the TT when changed. Fixed-depth profiles
are screening evidence only; candidates may be promoted one at a time only by
held-out color-reversed game tests.

- capture LMR extends Volkrix's proven quiet-move reduction/re-search protocol
  to late captures. Current Stockfish applies its LMR stage to every move after
  the first and then adjusts the reduction using move-specific evidence
  ([source](https://github.com/official-stockfish/Stockfish/blob/5062aee519a1ba262d472d8ab139851ced56573e/src/search.cpp#L1324-L1396)).
  Volkrix starts more conservatively: only non-PV captures at depth 5 or higher,
  after two moves have already been searched, are eligible. Checks, promotions,
  hash moves, and nodes in check retain full depth. Non-losing captures lose at
  most one ply; losing captures lose at most two. Any reduced result that raises
  alpha is re-searched at full depth. The reduction uses the same active
  `lmr_divisor_pct` as quiet LMR, including in SPSA builds. The heuristic field,
  counters, candidate classification, reduction branch, and UCI option are
  compiled only for tests, debug builds, or `internal-testing`. The default-false
  `ExperimentalCaptureLmr` option clears the TT whenever it changes.

  Frozen SF18-big-NNUE profiling was deterministic but deliberately treated as
  screening rather than strength evidence. At depth 8 the candidate expanded
  520,496 nodes versus 478,607 (+8.75%), with all four curated best moves and
  scores unchanged. At depth 9 it expanded 4,661,095 versus 4,801,401 (-2.92%);
  three curated positions retained their move and score, while one high-advantage
  line changed from `c5d6`/+1408 to `c5e7`/+1470 and collapsed from 9,509,712 to
  1,680,274 nodes. Because the evidence was mixed, the predeclared decision test
  used 500 color-reversed pairs at `1+0.01`, openings 34001 through 34500, the
  same frozen native-M4 binary and big network on both sides, and only the
  experimental option differing. The authoritative 1,000-game run scored 366
  wins, 272 draws, and 362 losses (50.20%, +1.39 Elo, pair-aware 95% CI
  [-14.192, +16.977], pentanomial `[45, 78, 251, 80, 46]`) with zero crashes,
  stalls, illegal moves, time forfeits, or other abnormal terminations. This is
  neutral evidence, not a promotion signal, so capture LMR remains default-off
  and does not advance to LTC or PGO testing.

- razoring performs a guarded quiescence verification when a shallow non-PV
  node's static evaluation is far below alpha. The experiment uses Stockfish
  18's conservative quadratic margin (`485 + 281 * depth^2`) at depth four or
  less, but returns early only when quiescence confirms a fail-low. Checks,
  exclusion searches, PV nodes, and mate-score windows bypass it. Stockfish 18
  [uses the same fail-low idea and margin](https://github.com/official-stockfish/Stockfish/blob/cb3d4ee9b47d0c5aae855b12379378ea1439675c/src/search.cpp#L870-L874),
  while Berserk 14 independently
  [uses quiescence-verified razoring](https://github.com/jhonnold/berserk/blob/8ae895a6151695be4a50d4fb65b0c131659c513a/src/search.c#L524-L529).
  Volkrix's field, counters, branch, constants, and helpers are compiled only
  for tests, debug builds, or `internal-testing`, so the production default has
  no hot-path branch. On the frozen SF18 big network, the four-position depth-8
  profile fell from 478,607 to 436,626 nodes (-8.77%); all four best moves and
  scores were unchanged. The resulting held-out, color-reversed 1,000-game STC
  match nevertheless scored 377 wins, 186 draws, and 437 losses (47.00%,
  -20.87 Elo, pair-aware 95% CI [-37.50, -4.33], pentanomial
  `[65, 85, 242, 61, 47]`) with zero abnormal terminations. The predeclared
  promotion rule therefore rejected razoring, demonstrating why a fixed-depth
  node reduction is not sufficient strength evidence. An
  `internal-testing` build advertises the default-false UCI option
  `ExperimentalRazoring`; changing it clears the TT so both sides of a paired
  test can safely use the exact same executable.

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
  ply.
- Multi-Cut probes at most six ordered candidates three plies shallower at
  guarded non-PV cut nodes. It requires three reduced fail-highs, non-pawn
  material, an ordinary score window, a plausible static evaluation, and depth
  7. Probe children suppress recursive Multi-Cut and null-move pruning. A cutoff
  publishes only the reduced-depth bound. This remains a match-test seam because
  Multi-Cut is probabilistic forward pruning; the original algorithm and its
  tradeoffs are described in
  [Björnsson and Marsland](https://staff.ru.is/yngvi/pdf/BjornssonM01a.pdf).
  The entire seam, including its heuristic field, counters, branch, and probe
  code, is compiled only for tests, debug builds, or `internal-testing`; normal
  release and OpenBench binaries pay no per-node cost for the rejected idea.

The isolated four-position depth-8 Multi-Cut profile was deterministic but did
not provide a promotion signal: it expanded 564,278 nodes versus 553,304 for the
default (+1.98%). Runtime was noisy and slightly lower in the median local run,
which is insufficient evidence for a selective-search change. It therefore
remains off pending a paired SPRT rather than contaminating the validated
default.

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

With exactly two threads, the adaptive default uses a young-brothers-wait root
split: the main thread searches the eldest move and releases disjoint sibling
work only after establishing the iteration bound. At three or more threads,
Adaptive assigns helpers deterministic overlapping circular root windows while
the main thread remains authoritative. A held-out 1,000-game SF18 STC match at
four threads scored 446 wins, 174 draws, and 380 losses (53.30%, +22.964 Elo,
pair-aware 95% CI [+7.670, +38.348], pentanomial `[35, 57, 275, 73, 60]`) with
zero abnormal terminations or time forfeits. Independent LTC then used 2,000
held-out color-reversed pairs at `1+0.01`. Across both frozen stages the new
policy scored 1,811 wins, 660 draws, and 1,529 losses (53.525%, +24.535 Elo,
pair-aware 95% CI [+16.004, +33.095], pentanomial
`[195, 230, 983, 282, 310]`) with zero crashes, stalls, illegal moves, or time
forfeits. It therefore cleared the fixed-candidate promotion policy and is the
production wider-thread default.

The internal-only `ExperimentalSmpDiversification` seam remains semantically
explicit for identical-binary regression matches: its advertised default
`false` selects legacy Lazy SMP, while `true` selects Diversified. Production
builds expose no experimental option and use Adaptive directly. OS scheduling
can change which TT entries win races, so multithreaded runs are not a fixed-node
reproducibility mode.

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

An internal-only `ExperimentalTimeManagement` candidate keeps this production
policy unchanged by default while testing a richer deterministic policy. Its
clock allocator accounts for increment, explicit moves-to-go, the opponent's
clock, and a separate emergency reserve; every derived budget is capped by the
usable clock even for overflowing UCI inputs. The companion search-instability
model tracks completed-iteration best-move and PV churn, rolling score
volatility, second-best margin, aspiration re-search cost, and observed
iteration growth. Its aspiration windows widen fail-low more aggressively than
fail-high. These behavior changes require paired clock-match evidence at the
intended controls before any production default can change.

A four-position depth-8 classical screening run preserved every final score and
three of four best moves; the equal-scored start position selected a different
move. Aggregate nodes fell from 721,194 to 709,296 (-1.65%). This verifies that
the candidate is connected, deterministic, and modest in search-tree impact;
it is not evidence that the clock policy gains Elo.

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
