mod bench;
pub mod eval;
#[cfg(any(debug_assertions, feature = "internal-testing"))]
#[doc(hidden)]
pub mod internal;
pub mod limits;
mod qsearch;
mod root;
mod tt;

pub use bench::{BenchConfig, BenchResult, run_bench};
pub use eval::evaluate;
pub use limits::SearchLimits;
pub use root::{SearchResult, SearchStats, search};
