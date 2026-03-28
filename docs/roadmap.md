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

## Next

- Phase 9 search depth and selectivity layer II pending signoff: exact Phase 8 baseline preservation, basic quiet-only LMR as the retained path, and documented guarded null-move evaluation evidence
- Later phases, not yet locked in order: deeper classical eval work, SMP, tablebases, NNUE, training, and tuning

## Intentional Boundaries

The current implementation still stops before:

- SMP and shared-search infrastructure
- advanced pruning and reduction heuristics
