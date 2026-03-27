# Perft

## Canonical Totals

Phase 2 validates the following reference totals:

- Startpos: d1 `20`, d2 `400`, d3 `8902`, d4 `197281`, d5 `4865609`, d6 `119060324`
- Kiwipete: d1 `48`, d2 `2039`, d3 `97862`, d4 `4085603`, d5 `193690690`
- Position 3: d1 `14`, d2 `191`, d3 `2812`, d4 `43238`, d5 `674624`, d6 `11030083`
- Position 4: d1 `6`, d2 `264`, d3 `9467`, d4 `422333`, d5 `15833292`
- Position 5: d1 `44`, d2 `1486`, d3 `62379`, d4 `2103487`
- Position 6: d1 `46`, d2 `2079`, d3 `89890`, d4 `3894594`

Position 5 and Position 6 intentionally stop at depth 4 in Phase 2. That is an explicit runtime boundary for the current regression suite, not an omitted fixture.

## Divide Artifacts

`divide()` is implemented for debugging and regression support.

Checked-in divide outputs are treated as generated or directly verified regression artifacts under `tests/fixtures/divide/`, not hand-maintained prose. Root-move ordering is deterministic because `divide()` sorts output by UCI move text before returning.

Phase 2 requires canonical perft totals. Divide artifacts are an additional regression layer for selected positions where the artifact was generated or verified cleanly.

## Mutable Move Generation

`generate_legal_moves(&mut self, ...)` remains mutable in Phase 2 because en passant legality still uses temporary `make_move` / `unmake_move` validation. That edge case can expose a rook, bishop, or queen attack on the king only after both pawns are removed from their original line, and the generator intentionally reuses the authoritative state-transition path for that check.

