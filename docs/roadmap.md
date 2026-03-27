# Volkrix Roadmap

## Completed

- Phase 0 repo scaffolding, CI, licensing, and clean-room provenance language
- Phase 1 core types, position model, FEN parsing, move parsing/application, and minimal UCI shell
- Phase 2 attacks, staged move generation, legality/perft hardening, and divide artifact support
- Phase 3 make/unmake state expansion, Zobrist, repetition, rules, and SEE
- Phase 4 deterministic single-thread search baseline with iterative deepening, alpha-beta, quiescence, PV tracking, and simple classical evaluation

## Next

- Phase 5 transposition table integration and search-strength layering on top of the correct single-thread baseline
- Later phases: selectivity, time management, classical eval growth, SMP, tablebases, NNUE, training, and tuning

## Intentional Boundaries

The current implementation still stops before:

- search heuristics beyond placeholder legal move selection
- incremental Zobrist/material/phase state
