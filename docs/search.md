# Search

Phase 4 adds Volkrix's first real deterministic single-thread search baseline.

## Current Shape

- iterative deepening at the root
- alpha-beta negamax core
- quiescence search for tactical stabilization
- principal variation tracking
- simple classical evaluation based on material and square bonuses
- terminal handling for checkmate, stalemate, repetition, fifty-move draw, and insufficient-material draw

## Intentional Limits

- no transposition table
- no SMP
- no null move, LMR, or aspiration windows
- no time management expansion beyond `go depth`

The goal of this phase is correctness and a playable engine baseline, not strength tuning.
