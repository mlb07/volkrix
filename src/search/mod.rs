mod bench;
pub mod eval;
#[cfg(any(debug_assertions, feature = "internal-testing"))]
#[doc(hidden)]
pub mod internal;
pub mod limits;
mod movepicker;
pub(crate) mod nnue;
mod qsearch;
mod root;
#[cfg(feature = "offline-tools")]
#[doc(hidden)]
pub mod service;
#[cfg(not(feature = "offline-tools"))]
pub(crate) mod service;
mod stockfish_nnue;
pub(crate) mod tablebase;
mod tt;

pub use bench::{BenchConfig, BenchResult, DualBenchStats, run_bench};
pub use eval::evaluate;
pub use limits::SearchLimits;
pub(crate) use root::PonderState;
pub use root::{SearchResult, SearchStats, search};
pub use service::{DEFAULT_DUAL_EVAL_THRESHOLD, MAX_DUAL_EVAL_THRESHOLD};
