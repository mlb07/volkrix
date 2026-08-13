# Volkrix nnue-rs fork provenance

This directory is a source-vendored fork of `nnue-rs` 0.4.0 from upstream
commit `64ba58a93224e18ae48c19a6c58f34026f237730`:

<https://github.com/hedgeg0d/nnue-rs/tree/64ba58a93224e18ae48c19a6c58f34026f237730>

The upstream and modified sources are MIT licensed; the canonical license is
[`LICENSE`](LICENSE). Volkrix vendors the dependency so its NNUE parser and hot
path remain reproducible and so architecture-specific optimizations can be
tested in the same repository.

Volkrix-specific changes are deliberately limited to independently implemented
integration and execution features:

- compact caller-supplied board deltas;
- lazy accumulator materialization support;
- separate PSQT and positional evaluation components;
- AArch64 NEON and ARM dot-product kernels, including a DotProd remainder
  path for the 16--32-element later network layers, with scalar parity tests
  and an ignored Apple-Silicon microprofile.

No Stockfish GPL source code is included in or copied into this fork. External
network data retains its own license and checksum requirements.

## SFNNv16 / PP_3Wide clean-room status

Investigation date: 2026-08-08.

The current official big network `nn-ab28990d4ea3.nnue` has SHA-256
`ab28990d4ea3d5c97f7d3918bc5dd5061609330369fe00c2d93a34d4777b5552`,
format version `0x6a448afa`, architecture hash `0xa85b2205`, and feature-
transformer hash `0xcb685313`. Stockfish's published NNUE documentation says
that SFNNv16 added `PP_3Wide`, but it does not publish the exact PP_3Wide
feature-index mapping, incremental-update semantics, or serialized block order:

<https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html#sfnnv16-architecture>

The authoritative engine and trainer implementations are GPL-3.0:

- <https://github.com/official-stockfish/Stockfish>
- <https://github.com/official-stockfish/nnue-pytorch>

A GitHub code search for `PP_3Wide` on the investigation date found no
permissively licensed implementation or complete independent specification.
The additional implementations checked (`ppigazzini/zfish`,
`Hengyu/Swiftfish`, and `Tors3/Triumviratus`) are also GPL-3.0. Their source was
not used. Repository license metadata was checked before implementation files.

Consequently this fork does not implement SFNNv16 by inference or by copying
GPL code. Version `0x6a448afa` is rejected as
`Error::UnsupportedPp3Wide`, with a diagnostic that names SFNNv16/PP_3Wide and
confirms that stable SFNNv10 remains supported. A clean implementation can be
reconsidered when a sufficiently complete permissive specification or
implementation becomes available; it must then pass exact Stockfish-oracle
tests on curated and random legal positions plus scalar/SIMD parity before the
support claim is changed.
