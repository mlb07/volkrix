# Volkrix Roadmap

## Completed

- Phase 0 repo scaffolding, CI, licensing, and clean-room provenance language
- Phase 1 core types, position model, FEN parsing, move parsing/application, and minimal UCI shell

## Next

- Phase 2 attacks, full move generation validation, legality/perft hardening, and canonical divide output
- Phase 3 make/unmake state expansion, Zobrist, repetition, rules, and SEE
- Later phases: TT, search, selectivity, time management, classical eval, SMP, tablebases, NNUE, training, and tuning

## Intentional Boundaries

Phase 1 stops before:

- optimized sliding attack infrastructure
- canonical perft/debug command surface
- search heuristics beyond placeholder legal move selection
- incremental Zobrist/material/phase state

