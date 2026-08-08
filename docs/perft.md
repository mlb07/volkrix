# Perft

Perft is Volkrix's primary legal move-generation oracle. It recursively counts
legal leaf positions without evaluation, pruning, TT reuse, or tablebases.

## Canonical totals

The integration suite checks these reference positions:

| Position | d1 | d2 | d3 | d4 | d5 | d6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Start position | 20 | 400 | 8,902 | 197,281 | 4,865,609 | 119,060,324 |
| Kiwipete | 48 | 2,039 | 97,862 | 4,085,603 | 193,690,690 | — |
| Position 3 | 14 | 191 | 2,812 | 43,238 | 674,624 | 11,030,083 |
| Position 4 | 6 | 264 | 9,467 | 422,333 | 15,833,292 | — |
| Position 5 | 44 | 1,486 | 62,379 | 2,103,487 | — | — |
| Position 6 | 46 | 2,079 | 89,890 | 3,894,594 | — | — |

Run the default suite with:

```bash
cargo test --locked --test perft
```

The deepest totals are marked ignored to keep ordinary CI latency reasonable.
Run them explicitly before a move-generation release:

```bash
cargo test --locked --release --test perft -- --ignored
```

## Divide artifacts

`divide()` returns one node count per legal root move and sorts the result by UCI
move text. Checked-in outputs under `tests/fixtures/divide/` localize a mismatch to
a specific root branch and are generated or externally verified artifacts, not
hand-maintained totals.

## Why generation remains mutable

`generate_legal_moves(&mut self, ...)` is mutable because en-passant legality uses
the authoritative temporary make/unmake path. Removing the capturing pawn and the
captured pawn simultaneously can expose a rook, bishop, or queen attack that is
not visible from either pawn's original occupancy alone.

## Deterministic state and parser stress

The feature-gated `volkrix-stress` binary complements fixed perft totals with
reproducible randomized state-machine walks. It compares the optimized legal
generator with an independent pseudo-legal plus checked-make oracle, verifies
that generation is non-mutating, checks every selected move through both direct
and UCI application, validates incremental Zobrist and repetition state after
each transition, round-trips FEN, and exactly unwinds every walk.

The mandatory root corpus includes both castling directions, legal and pinned
en passant, quiet and capture promotions for both colors, an actual threefold
repetition path, Kiwipete, and an endgame perft position. The same run sends
bounded deterministic garbage and malformed command families through the FEN and
UCI parsers and requires errors to leave the current position untouched.

Run the quick property profile through the normal test suite, or invoke a larger
reproducible job directly:

```bash
cargo test --locked --all-features deterministic_stress_quick_profile
cargo run --locked --release --features internal-testing \
  --bin volkrix-stress -- \
  --seed 0x6a09e667f3bcc909 --walks 256 --plies 512 --parser-cases 50000
```

Failures print the seed, corpus name, walk, ply, and move context needed to replay
the same path. Successful runs print a deterministic trace digest. GitHub's
`Deterministic Stress` workflow runs the long profile against four fixed seeds
weekly and can also be started manually.
