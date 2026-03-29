use std::time::Instant;

use crate::core::Position;

use super::{
    SearchLimits,
    limits::SearchHeuristics,
    root::{SearchControl, search, search_with_control},
    service::{SearchRequest, UciSearchService},
};

const BENCH_FENS: [&str; 4] = [
    crate::core::STARTPOS_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 2 3",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BenchConfig {
    pub depth: u8,
    pub tt_enabled: bool,
    pub hash_mb: usize,
    pub(crate) heuristics: SearchHeuristics,
    pub(crate) threads: usize,
}

impl BenchConfig {
    pub const fn new(depth: u8) -> Self {
        Self {
            depth,
            tt_enabled: true,
            hash_mb: super::tt::DEFAULT_HASH_MB,
            heuristics: SearchHeuristics::phase9_default(),
            threads: 1,
        }
    }

    pub const fn without_tt(mut self) -> Self {
        self.tt_enabled = false;
        self
    }

    pub const fn with_hash_mb(mut self, hash_mb: usize) -> Self {
        self.hash_mb = hash_mb;
        self
    }

    pub(crate) const fn with_heuristics(mut self, heuristics: SearchHeuristics) -> Self {
        self.heuristics = heuristics;
        self
    }

    pub(crate) const fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self::new(5)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BenchResult {
    pub depth: u8,
    pub positions: usize,
    pub total_nodes: u64,
    pub checksum: u64,
    pub tt_enabled: bool,
    pub hash_mb: usize,
    pub elapsed_ms: u128,
}

impl BenchResult {
    pub fn nps(&self) -> u64 {
        if self.elapsed_ms == 0 {
            self.total_nodes
        } else {
            (self.total_nodes as u128 * 1000 / self.elapsed_ms) as u64
        }
    }

    pub fn render_lines(&self) -> Vec<String> {
        vec![
            format!(
                "bench depth {} positions {} tt {} hash {}",
                self.depth,
                self.positions,
                if self.tt_enabled { "on" } else { "off" },
                self.hash_mb
            ),
            format!("bench nodes {}", self.total_nodes),
            format!("bench checksum {:016x}", self.checksum),
            format!("bench time_ms {}", self.elapsed_ms),
            format!("bench nps {}", self.nps()),
        ]
    }
}

pub fn run_bench(config: BenchConfig) -> BenchResult {
    if config.threads <= 1 || !config.tt_enabled {
        return run_single_thread_bench(config);
    }

    run_threaded_bench(config)
}

fn run_single_thread_bench(config: BenchConfig) -> BenchResult {
    let started = Instant::now();
    let mut total_nodes = 0u64;
    let mut checksum = 0u64;

    for fen in BENCH_FENS {
        let mut position = Position::from_fen(fen).expect("bench FEN must parse");
        let limits = SearchLimits::new(config.depth)
            .with_hash_mb(config.hash_mb)
            .with_tt(config.tt_enabled)
            .with_heuristics(config.heuristics);
        let result = search(&mut position, limits);
        total_nodes += result.nodes;

        let best_move_hash = result
            .best_move
            .map(|mv| hash_text(&mv.to_string()))
            .unwrap_or(0);
        checksum = checksum.rotate_left(9)
            ^ (result.score.0 as i64 as u64)
            ^ best_move_hash
            ^ result.nodes;
    }

    BenchResult {
        depth: config.depth,
        positions: BENCH_FENS.len(),
        total_nodes,
        checksum,
        tt_enabled: config.tt_enabled,
        hash_mb: config.hash_mb,
        elapsed_ms: started.elapsed().as_millis(),
    }
}

fn run_threaded_bench(config: BenchConfig) -> BenchResult {
    let started = Instant::now();
    let mut total_nodes = 0u64;
    let mut checksum = 0u64;
    let mut service = UciSearchService::new();
    service.resize_hash(config.hash_mb);
    service.set_threads(config.threads);

    for fen in BENCH_FENS {
        let mut position = Position::from_fen(fen).expect("bench FEN must parse");
        service.clear_hash();
        let result = service.search(
            &mut position,
            SearchRequest {
                limits: SearchLimits::new(config.depth)
                    .with_hash_mb(config.hash_mb)
                    .with_tt(true)
                    .with_heuristics(config.heuristics),
                soft_deadline: None,
                hard_deadline: None,
                stop_flag: None,
            },
        );
        total_nodes += result.nodes;

        let best_move_hash = result
            .best_move
            .map(|mv| hash_text(&mv.to_string()))
            .unwrap_or(0);
        checksum = checksum.rotate_left(9)
            ^ (result.score.0 as i64 as u64)
            ^ best_move_hash
            ^ result.nodes;
    }

    BenchResult {
        depth: config.depth,
        positions: BENCH_FENS.len(),
        total_nodes,
        checksum,
        tt_enabled: config.tt_enabled,
        hash_mb: config.hash_mb,
        elapsed_ms: started.elapsed().as_millis(),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TimedBenchResult {
    pub movetime_ms: u64,
    pub positions: usize,
    pub total_nodes: u64,
    pub checksum: u64,
    pub total_completed_depth: u64,
    pub elapsed_ms: u128,
}

pub(crate) fn run_timed_bench(config: BenchConfig, movetime_ms: u64) -> TimedBenchResult {
    if config.threads <= 1 || !config.tt_enabled {
        return run_single_thread_timed_bench(config, movetime_ms);
    }

    run_threaded_timed_bench(config, movetime_ms)
}

fn run_single_thread_timed_bench(config: BenchConfig, movetime_ms: u64) -> TimedBenchResult {
    let started = Instant::now();
    let mut total_nodes = 0u64;
    let mut checksum = 0u64;
    let mut total_completed_depth = 0u64;

    for fen in BENCH_FENS {
        let mut position = Position::from_fen(fen).expect("bench FEN must parse");
        let result = search_with_control(
            &mut position,
            SearchLimits::new(127)
                .with_hash_mb(config.hash_mb)
                .with_tt(config.tt_enabled)
                .with_heuristics(config.heuristics),
            None,
            None,
            None,
            SearchControl {
                stop_flag: None,
                helper_stop_flag: None,
                soft_deadline: Some(Instant::now() + std::time::Duration::from_millis(movetime_ms)),
                hard_deadline: Some(Instant::now() + std::time::Duration::from_millis(movetime_ms)),
                role: super::root::SearchThreadRole::Main,
            },
        );
        total_nodes += result.nodes;
        total_completed_depth += result.depth as u64;
        checksum = checksum.rotate_left(9)
            ^ (result.score.0 as i64 as u64)
            ^ result.nodes
            ^ hash_text(
                &result
                    .best_move
                    .map(|mv| mv.to_string())
                    .unwrap_or_else(|| "0000".to_owned()),
            );
    }

    TimedBenchResult {
        movetime_ms,
        positions: BENCH_FENS.len(),
        total_nodes,
        checksum,
        total_completed_depth,
        elapsed_ms: started.elapsed().as_millis(),
    }
}

fn run_threaded_timed_bench(config: BenchConfig, movetime_ms: u64) -> TimedBenchResult {
    let started = Instant::now();
    let mut total_nodes = 0u64;
    let mut checksum = 0u64;
    let mut total_completed_depth = 0u64;
    let mut service = UciSearchService::new();
    service.resize_hash(config.hash_mb);
    service.set_threads(config.threads);

    for fen in BENCH_FENS {
        let mut position = Position::from_fen(fen).expect("bench FEN must parse");
        service.clear_hash();
        let deadline = Instant::now() + std::time::Duration::from_millis(movetime_ms);
        let result = service.search(
            &mut position,
            SearchRequest {
                limits: SearchLimits::new(127)
                    .with_hash_mb(config.hash_mb)
                    .with_tt(true)
                    .with_heuristics(config.heuristics),
                soft_deadline: Some(deadline),
                hard_deadline: Some(deadline),
                stop_flag: None,
            },
        );
        total_nodes += result.nodes;
        total_completed_depth += result.depth as u64;
        checksum = checksum.rotate_left(9)
            ^ (result.score.0 as i64 as u64)
            ^ result.nodes
            ^ hash_text(
                &result
                    .best_move
                    .map(|mv| mv.to_string())
                    .unwrap_or_else(|| "0000".to_owned()),
            );
    }

    TimedBenchResult {
        movetime_ms,
        positions: BENCH_FENS.len(),
        total_nodes,
        checksum,
        total_completed_depth,
        elapsed_ms: started.elapsed().as_millis(),
    }
}

fn hash_text(text: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in text.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}
