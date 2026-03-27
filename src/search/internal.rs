use super::{BenchConfig, BenchResult, SearchLimits, limits::SearchHeuristics, run_bench};

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeuristicProfile {
    Phase5Baseline,
    PvMoveOrdering,
    CaptureBuckets,
    KillerMoves,
    QuietHistory,
    AspirationWindows,
    Phase6Default,
}

impl HeuristicProfile {
    const fn heuristics(self) -> SearchHeuristics {
        let baseline = SearchHeuristics::phase5_baseline();
        match self {
            Self::Phase5Baseline => baseline,
            Self::PvMoveOrdering => baseline.with_pv_move_ordering(true),
            Self::CaptureBuckets => baseline.with_capture_buckets(true),
            Self::KillerMoves => baseline.with_killer_moves(true),
            Self::QuietHistory => baseline.with_quiet_history(true),
            Self::AspirationWindows => baseline.with_aspiration_windows(true),
            Self::Phase6Default => SearchHeuristics::phase6_default(),
        }
    }
}

#[doc(hidden)]
pub fn phase5_baseline_limits(depth: u8) -> SearchLimits {
    SearchLimits::new(depth).with_phase5_baseline()
}

#[doc(hidden)]
pub fn no_aspiration_limits(depth: u8) -> SearchLimits {
    SearchLimits::new(depth)
        .with_heuristics(SearchHeuristics::phase6_default().with_aspiration_windows(false))
}

#[doc(hidden)]
pub fn run_profile_bench(depth: u8, profile: HeuristicProfile) -> BenchResult {
    run_bench(BenchConfig::new(depth).with_heuristics(profile.heuristics()))
}
