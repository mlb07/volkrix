# Volkrix Roadmap

## Completed

- Phase 0 repo scaffolding, CI, licensing, and clean-room provenance language
- Phase 1 core types, position model, FEN parsing, move parsing/application, and minimal UCI shell
- Phase 2 attacks, staged move generation, legality/perft hardening, and divide artifact support
- Phase 3 make/unmake state expansion, Zobrist, repetition, rules, and SEE
- Phase 4 deterministic single-thread search baseline with iterative deepening, alpha-beta, quiescence, PV tracking, and simple classical evaluation
- Phase 5 transposition table and search infrastructure on top of the single-thread baseline

## Next

- Phase 6 search-strength layering and selectivity on top of the correct TT-backed single-thread baseline
- Later phases: time management, classical eval growth, SMP, tablebases, NNUE, training, and tuning

## Intentional Boundaries

The current implementation still stops before:

- SMP and shared-search infrastructure
- advanced pruning and reduction heuristics
