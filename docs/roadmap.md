# Volkrix Roadmap

## Completed

- Phase 0 repo scaffolding, CI, licensing, and clean-room provenance language
- Phase 1 core types, position model, FEN parsing, move parsing/application, and minimal UCI shell
- Phase 2 attacks, staged move generation, legality/perft hardening, and divide artifact support

## Next

- Phase 3 make/unmake state expansion, Zobrist, repetition, rules, and SEE
- Later phases: TT, search, selectivity, time management, classical eval, SMP, tablebases, NNUE, training, and tuning

## Intentional Boundaries

The current implementation still stops before:

- search heuristics beyond placeholder legal move selection
- incremental Zobrist/material/phase state
