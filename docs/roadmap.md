# Volkrix Roadmap

## Completed

- Phase 0 repo scaffolding, CI, licensing, and clean-room provenance language
- Phase 1 core types, position model, FEN parsing, move parsing/application, and minimal UCI shell
- Phase 2 attacks, staged move generation, legality/perft hardening, and divide artifact support
- Phase 3 make/unmake state expansion, Zobrist, repetition, rules, and SEE
- Phase 4 deterministic single-thread search baseline with iterative deepening, alpha-beta, quiescence, PV tracking, and simple classical evaluation
- Phase 5 transposition table and search infrastructure on top of the single-thread baseline
- Phase 6 search-strength layering with stronger move ordering, aspiration windows, deterministic heuristic toggles, and documented Phase 5 baseline comparisons
- Phase 7 time management and practical UCI usability on top of the correct TT-backed single-thread baseline
- Phase 8 classical eval bridge with tapered evaluation and a disciplined first expansion of the static eval terms
- Phase 9 search depth and selectivity layer II with exact Phase 8 baseline preservation, conservative quiet-only LMR, and evidence-driven rejection of guarded null-move pruning
- Phase 10 SMP / Lazy SMP Layer I with minimal `Threads` control, persistent helper-worker pool, shared-TT-only SMP, and authoritative `Threads=1` baseline preservation
- Phase 11 Tablebases / Probe Layer I with optional `SyzygyPath`, approved vendored `jdart1/Fathom` probing behind the internal tablebase boundary, and exact disabled-path baseline preservation
- Phase 12 NNUE Engine Integration Layer I with minimal `EvalFile` control, clean evaluator boundary, retained clean-room Volkrix NNUE format, thread-local accumulator/update architecture, and exact disabled-path preservation when no network is configured

## Next

- Phase 13 Training Pipeline and Net Iteration Layer I pending signoff: isolated offline export/pack/validate tooling, deterministic fixed-depth 5 label generation, hybrid Rust/Python training flow, and first real candidate-net validation through `EvalFile`
- Later phases, not yet locked in order: broader net iteration / tuning / release polish

## Intentional Boundaries

The current implementation still stops before:

- split-point or work-stealing SMP
- broader tablebase features beyond the approved probe-only Fathom Layer I design
- broader NNUE work beyond the retained single-network integration and first offline training layer
- advanced pruning and reduction heuristics
