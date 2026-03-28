use super::{BenchConfig, BenchResult, SearchLimits, limits::SearchHeuristics, run_bench};

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeuristicProfile {
    Phase8Baseline,
    LmrOnly,
    Phase9Default,
}

impl HeuristicProfile {
    const fn heuristics(self) -> SearchHeuristics {
        let baseline = SearchHeuristics::phase8_baseline();
        match self {
            Self::Phase8Baseline => baseline,
            Self::LmrOnly => baseline.with_late_move_reductions(true),
            Self::Phase9Default => SearchHeuristics::phase9_default(),
        }
    }
}

#[doc(hidden)]
pub fn phase8_baseline_limits(depth: u8) -> SearchLimits {
    SearchLimits::new(depth).with_phase8_baseline()
}

#[doc(hidden)]
pub fn lmr_only_limits(depth: u8) -> SearchLimits {
    SearchLimits::new(depth)
        .with_heuristics(SearchHeuristics::phase8_baseline().with_late_move_reductions(true))
}

#[doc(hidden)]
pub fn phase9_default_limits(depth: u8) -> SearchLimits {
    SearchLimits::new(depth).with_heuristics(SearchHeuristics::phase9_default())
}

#[doc(hidden)]
pub fn no_aspiration_limits(depth: u8) -> SearchLimits {
    SearchLimits::new(depth)
        .with_heuristics(SearchHeuristics::phase9_default().with_aspiration_windows(false))
}

#[doc(hidden)]
pub fn run_profile_bench(depth: u8, profile: HeuristicProfile) -> BenchResult {
    run_bench(BenchConfig::new(depth).with_heuristics(profile.heuristics()))
}
