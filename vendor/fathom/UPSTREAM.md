Vendored upstream: `jdart1/Fathom`

- Repository: <https://github.com/jdart1/Fathom>
- Approved upstream revision: `c9c6fef0dddc05d2e242c183acf5833149ab676d`
- License basis: upstream MIT license

Phase 11 guardrails for this vendored copy:

- Keep this backend behind Volkrix's internal `search::tablebase` boundary.
- Do not transliterate, re-port, or cherry-pick probing code from other backends.
- Do not widen the public surface beyond `SyzygyPath`.
