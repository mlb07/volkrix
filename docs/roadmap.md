# Volkrix Roadmap

The foundational engine is implemented: legal chess, selective PVS, a lock-free
TT, adaptive two-thread root splitting and wider Lazy SMP, time management,
classical and neural evaluation,
seven-piece Syzygy probing, offline training/tuning tools, and paired match
reporting. Future work is prioritized by measured strength and reliability rather
than numbered phases.

## Release-critical

- Run the complete correctness, clippy, release, perft, UCI-compliance, and
  sanitizer-equivalent test matrix on every supported release target.
- Establish a statistically meaningful Fastchess/OpenBench baseline for
  `Threads=1` and representative SMP configurations.
- Validate bundled Stockfish 18 inference end to end on every release target and
  continue publishing separate classical/NNUE performance numbers.
- Automate dependency license auditing; third-party notices and the verified CC0
  network are already included in release archives.
- Treat crashes, illegal moves, hangs, time losses, and corrupted PV output as
  release blockers regardless of match score.

## Highest-value strength work

- Tune modern search heuristics with SPRT: LMR, ProbCut, futility margins, history
  aging, aspiration behavior, and capture-history ordering.
- Keep correction history available only as an experimental toggle: its initial
  300-game match scored 46.17% (about -26.7 Elo), so it is not part of the
  default profile. Singular extension is also experimental/default-off after its
  first 200-game match scored 49.0% (about -7 Elo).
- Improve helper diversification and evaluate whether more sophisticated root or
  split-point parallelism outperforms the current shared-TT design.
- Build a larger, diverse, license-clean supervised corpus and train a competitive
  `VOLKNNUE` candidate; the existing small local candidates are not strength
  evidence.
- Continue profiling the AArch64 NEON/DotProd backend and tune only changes that
  preserve bit-exact scalar parity.
- Tune time allocation and stability heuristics through game outcomes and time-loss
  rates rather than fixed formulas alone.

## Protocol and usability

- `go mate`
- MultiPV and optional WDL reporting
- Chess960 castling and the corresponding UCI option
- richer Syzygy controls and diagnostics
- reproducible Syzygy setup in release documentation

## Engineering follow-through

- Expand the deterministic FEN/UCI and make/unmake stress corpus whenever a new
  parser or move-state regression is found; the dependency-free long profile now
  runs weekly and on demand.
- Add long-running TT/SMP stress jobs and fault-injection tests for helper failure.
- Add profile-guided optimization experiments and hardware-specific benchmarks.
- Preserve exact provenance for vendored code, model files, corpora, and opening
  books.
- Keep the public Rust API explicitly unstable until its ownership and compatibility
  contract are designed.

Completed mechanisms are described in [`architecture.md`](architecture.md) and
[`search.md`](search.md). The evidence required to keep new work is defined in
[`search-handoff.md`](search-handoff.md).
