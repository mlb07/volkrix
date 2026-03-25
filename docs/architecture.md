# Volkrix Architecture

## Phase 0 and Phase 1

Volkrix currently exposes a single public surface: a command-line UCI engine binary.

The Rust library crate exists to share code between the binary and tests. It is internal-only at this stage and is not a promised public Rust API.

## Constraints

- stable Rust only
- no nightly features
- no external chess crates for engine core logic
- minimal dependencies, with dev-only additions only when justified
- clean-room implementation under dual MIT/Apache-2.0

## Engine Core Shape

The current position model uses the long-term hybrid representation target:

- bitboards by color and piece type
- a mailbox square-to-piece array
- fixed-capacity piece lists
- cached king squares
- occupancy bitboards

Phase 1 intentionally omits partial incremental subsystems such as Zobrist, material, and phase until they are needed and can be implemented coherently.

## Move Handling

The move path is intentionally not a throwaway scaffold:

- UCI move parsing supports all standard legal move kinds
- move application supports quiets, captures, promotions, double pawn pushes, en passant, and castling
- legality is checked through the same make/unmake path used by move generation

Phase 2 will extend this path with stronger move generation infrastructure, attack tables, and perft coverage rather than replacing it.

## UCI

Supported commands in Phase 1:

- `uci`
- `isready`
- `ucinewgame`
- `position`
- `go depth`
- `stop`
- `quit`

Malformed input must never panic or corrupt engine state.

