// Compile the pinned nnue-rs fork as private in-tree modules. Cargo rewrites
// publishable path dependencies to their registry versions, which would lose
// Volkrix's audited fork extensions in a source package. This keeps checkout
// and `.crate` builds on the same vendored source files. The fork targets Rust
// 2021, so its established lint policy also contains Rust 2024 compatibility
// warnings at this module boundary.
#[path = "../vendor/nnue-rs/src/error.rs"]
#[rustfmt::skip]
#[allow(dead_code)]
mod error;
#[path = "../vendor/nnue-rs/src/feature.rs"]
#[rustfmt::skip]
#[allow(dead_code)]
mod feature;
#[path = "../vendor/nnue-rs/src/fen.rs"]
#[rustfmt::skip]
#[allow(dead_code)]
mod fen;
#[path = "../vendor/nnue-rs/src/leb128.rs"]
#[rustfmt::skip]
#[allow(dead_code, clippy::op_ref)]
mod leb128;
#[path = "../vendor/nnue-rs/src/network.rs"]
#[rustfmt::skip]
#[allow(
    dead_code,
    clippy::if_same_then_else,
    clippy::manual_is_multiple_of,
    clippy::manual_range_contains,
    clippy::needless_range_loop,
    clippy::precedence,
    clippy::too_many_arguments
)]
mod network;
#[path = "../vendor/nnue-rs/src/simd.rs"]
#[rustfmt::skip]
#[allow(
    dead_code,
    unsafe_op_in_unsafe_fn,
    clippy::manual_is_multiple_of,
    clippy::op_ref,
    clippy::precedence
)]
mod simd;
#[path = "../vendor/nnue-rs/src/threats.rs"]
#[rustfmt::skip]
#[allow(
    dead_code,
    clippy::if_same_then_else,
    clippy::manual_range_contains
)]
mod threats;
#[path = "../vendor/nnue-rs/src/types.rs"]
#[rustfmt::skip]
#[allow(dead_code)]
mod types;

#[allow(unused_imports)]
mod nnue_rs {
    pub(crate) use crate::feature::Arch;
    pub(crate) use crate::network::{Accumulator, BoardDelta, Evaluation, Network};
    pub(crate) use crate::types::{Board, Color, Piece, PieceKind};
}

pub mod core;
#[cfg(feature = "offline-tools")]
#[doc(hidden)]
pub mod nnue_training;
pub mod search;
#[cfg(any(test, feature = "internal-testing"))]
#[doc(hidden)]
pub mod stress;
pub mod uci;
pub mod util;

pub const ENGINE_AUTHOR: &str = "Monty Bognar";
pub const ENGINE_NAME: &str = "Volkrix";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const SOURCE_COMMIT: &str = env!("VOLKRIX_SOURCE_COMMIT");
