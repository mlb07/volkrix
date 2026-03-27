# Search

Phases 4 and 5 establish Volkrix's first real deterministic single-thread search baseline and its first transposition-table layer.

## Current Shape

- iterative deepening at the root
- alpha-beta negamax core
- quiescence search for tactical stabilization
- principal variation tracking
- simple classical evaluation based on material and square bonuses
- transposition table integration with TT-on and TT-off deterministic paths
- reproducible bench path tied to the real search core
- terminal handling for checkmate, stalemate, repetition, fifty-move draw, and insufficient-material draw

## Intentional Limits

- no SMP
- no null move, LMR, or aspiration windows
- no time management expansion beyond `go depth`

The goal of these phases is correctness-first search infrastructure, not SMP or strength tuning.
