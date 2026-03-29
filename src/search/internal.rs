use crate::core::{Position, Score};

use super::{
    BenchConfig, BenchResult, SearchLimits,
    bench::TimedBenchResult,
    limits::SearchHeuristics,
    nnue::{NnueService, tiny_test_evalfile_path},
    run_bench,
    service::{SearchRequest, UciSearchService},
};

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

#[doc(hidden)]
pub fn run_threaded_profile_bench(
    depth: u8,
    profile: HeuristicProfile,
    threads: usize,
) -> BenchResult {
    run_bench(
        BenchConfig::new(depth)
            .with_heuristics(profile.heuristics())
            .with_threads(threads),
    )
}

#[doc(hidden)]
pub fn run_threaded_timed_profile_bench(
    movetime_ms: u64,
    profile: HeuristicProfile,
    threads: usize,
) -> TimedBenchResult {
    super::bench::run_timed_bench(
        BenchConfig::new(127)
            .with_heuristics(profile.heuristics())
            .with_threads(threads),
        movetime_ms,
    )
}

#[doc(hidden)]
pub fn phase12_test_evalfile_path() -> String {
    tiny_test_evalfile_path()
        .to_str()
        .expect("tiny test eval file path must be UTF-8")
        .to_owned()
}

#[doc(hidden)]
pub fn debug_evaluate_with_tiny_nnue(position: &Position) -> Score {
    let service = NnueService::open_eval_file(&phase12_test_evalfile_path())
        .expect("tiny deterministic NNUE test net must load");
    let accumulators = service.build_accumulator(position);
    service.evaluate(position, &accumulators)
}

#[doc(hidden)]
pub fn run_threaded_tiny_nnue_bench(depth: u8, threads: usize) -> BenchResult {
    let mut service = UciSearchService::new();
    service.set_threads(threads);
    service
        .set_eval_file(&phase12_test_evalfile_path())
        .expect("tiny deterministic NNUE test net must load");

    let started = std::time::Instant::now();
    let mut total_nodes = 0u64;
    let mut checksum = 0u64;
    for fen in [
        crate::core::STARTPOS_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 2 3",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    ] {
        let mut position = Position::from_fen(fen).expect("bench FEN must parse");
        service.clear_hash();
        let result = service.search(
            &mut position,
            SearchRequest {
                limits: SearchLimits::new(depth)
                    .with_hash_mb(super::tt::DEFAULT_HASH_MB)
                    .with_tt(true)
                    .with_heuristics(SearchHeuristics::phase9_default()),
                soft_deadline: None,
                hard_deadline: None,
                stop_flag: None,
            },
        );
        total_nodes += result.nodes;
        let best_move_hash = result
            .best_move
            .map(|mv| {
                let mut hash = 0xcbf2_9ce4_8422_2325u64;
                for byte in mv.to_string().bytes() {
                    hash ^= byte as u64;
                    hash = hash.wrapping_mul(0x1000_0000_01b3);
                }
                hash
            })
            .unwrap_or(0);
        checksum = checksum.rotate_left(9)
            ^ (result.score.0 as i64 as u64)
            ^ best_move_hash
            ^ result.nodes;
    }

    BenchResult {
        depth,
        positions: 4,
        total_nodes,
        checksum,
        tt_enabled: true,
        hash_mb: super::tt::DEFAULT_HASH_MB,
        elapsed_ms: started.elapsed().as_millis(),
    }
}
