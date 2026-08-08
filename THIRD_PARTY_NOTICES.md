# Third-Party Notices

Volkrix itself is licensed under `MIT OR Apache-2.0`. The components and optional
artifacts below retain their own licenses. This file is informational and does not
replace their license texts or change their terms.

## Fathom

Volkrix vendors Fathom revision
`c9c6fef0dddc05d2e242c183acf5833149ab676d` for Syzygy probing.

- Source: <https://github.com/jdart1/Fathom>
- License: MIT
- Copyright (c) 2013-2018 Ronald de Man
- Copyright (c) 2015 basil00
- Copyright (c) 2016-2025 Jon Dart

The canonical copy shipped in the source tree is `vendor/fathom/LICENSE`.

## nnue-rs

Volkrix vendors and pins `nnue-rs` 0.4.0 for parsing and evaluating external
Stockfish-format NNUE networks. The local fork starts from upstream commit
`64ba58a93224e18ae48c19a6c58f34026f237730` and adds compact move deltas,
component-preserving output, lazy integration support, and bit-exact AArch64
NEON/dot-product kernels. Those Volkrix changes remain available under the
upstream MIT terms.

- Source: <https://github.com/hedgeg0d/nnue-rs>
- License: MIT
- Copyright (c) 2026 hedgeg0d
- Canonical vendored license: `vendor/nnue-rs/LICENSE`
- Fork provenance: `vendor/nnue-rs/VOLKRIX_FORK.md`

## Bullet

The separate `volkrix-nnue` offline workspace tool uses Bullet revision
`feab6443fc523c9d349427bca2d5bb3c04369420`. Bullet is not linked into the Volkrix
engine executable.

- Source: <https://github.com/jw1912/bullet>
- License: MIT
- Copyright (c) 2023 Jamie Whiting

## MIT license text for the components above

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Stockfish 18 neural network

`scripts/fetch-stockfish18-net.sh` optionally downloads
`nn-c288c895ea92.nnue` or the smaller `nn-37f18f62d772.nnue` from the Stockfish
testing service. Official Volkrix release archives place the large network beside
the executable for automatic discovery. It remains a separate data artifact and
is not compiled or linked into the Volkrix binary. The small network is available
only as an opt-in download.

- Source: <https://tests.stockfishchess.org/nns>
- License: [Creative Commons CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)
- Large-network SHA-256: `c288c895ea924429ea9092e3f36b2b3c1f00f2a3a4c759ff7e57e79e3b43e4a7`
- Small-network SHA-256: `37f18f62d772f3107e1d6aaca3898c130c3c86f2ab63e6555fbbca20635a899d`

CC0 dedicates the work to the public domain to the extent legally possible and
provides the work without warranties. Refer to the linked legal code for the
complete terms.

## Other Rust dependencies

The separate `volkrix-nnue` offline tool also resolves `montyformat 0.9.2`
(`AGPL-3.0`) and `sfbinpack 0.6.2` (`GPL-3.0`). They are not dependencies of or
linked into the Volkrix engine binary. Distributing the offline tool or a derived
binary requires complying with their respective copyleft terms.

- `montyformat`: <https://crates.io/crates/montyformat/0.9.2>
- `sfbinpack`: <https://github.com/Disservin/binpack-rust>

The authoritative dependency graph is `Cargo.lock`. `scripts/audit_third_party.py`
checks every locked package's declared SPDX expression, confines the reviewed
copyleft exceptions to the offline-tool graph, pins Git sources, and verifies the
complete vendored Fathom/nnue-rs trees and documented network provenance.
