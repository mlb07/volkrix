pub mod core;
#[cfg(feature = "offline-tools")]
#[doc(hidden)]
pub mod nnue_training;
pub mod search;
pub mod uci;
pub mod util;

pub const ENGINE_AUTHOR: &str = "Volkrix contributors";
pub const ENGINE_NAME: &str = "Volkrix";
pub const SOURCE_COMMIT: &str = env!("VOLKRIX_SOURCE_COMMIT");
