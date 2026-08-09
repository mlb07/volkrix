use std::time::Instant;

use crate::core::Position;

use super::{
    SearchLimits,
    limits::SearchHeuristics,
    service::{
        DEFAULT_DUAL_EVAL_THRESHOLD, DualEvalPolicy, SearchRequest, SmpStrategy, UciSearchService,
    },
};

const BENCH_FENS: [&str; 4] = [
    crate::core::STARTPOS_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 2 3",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
];

#[derive(Clone, Debug, Eq, PartialEq)]
enum BenchEvaluator {
    Discovered,
    Classical,
    EvalFile(String),
}

impl BenchEvaluator {
    fn label(&self) -> String {
        match self {
            Self::Discovered => "default".to_owned(),
            Self::Classical => "classical".to_owned(),
            Self::EvalFile(path) => path.clone(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BenchConfig {
    pub depth: u8,
    pub tt_enabled: bool,
    pub hash_mb: usize,
    pub(crate) heuristics: SearchHeuristics,
    pub(crate) threads: usize,
    pub(crate) smp_strategy: SmpStrategy,
    evaluator: BenchEvaluator,
    small_evaluator: Option<String>,
    dual_eval_policy: DualEvalPolicy,
    dual_eval_threshold: i32,
}

impl BenchConfig {
    pub fn new(depth: u8) -> Self {
        Self {
            depth,
            tt_enabled: true,
            hash_mb: super::tt::DEFAULT_HASH_MB,
            heuristics: SearchHeuristics::phase9_default(),
            threads: 1,
            smp_strategy: SmpStrategy::default(),
            evaluator: BenchEvaluator::Classical,
            small_evaluator: None,
            dual_eval_policy: DualEvalPolicy::Off,
            dual_eval_threshold: DEFAULT_DUAL_EVAL_THRESHOLD,
        }
    }

    pub fn without_tt(mut self) -> Self {
        self.tt_enabled = false;
        self
    }

    pub fn with_hash_mb(mut self, hash_mb: usize) -> Self {
        self.hash_mb = hash_mb;
        self
    }

    #[cfg_attr(
        not(any(test, debug_assertions, feature = "internal-testing")),
        allow(dead_code)
    )]
    pub(crate) fn with_heuristics(mut self, heuristics: SearchHeuristics) -> Self {
        self.heuristics = heuristics;
        self
    }

    #[cfg_attr(
        not(any(test, debug_assertions, feature = "internal-testing")),
        allow(dead_code)
    )]
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads.max(1);
        self
    }

    #[cfg_attr(
        not(any(test, debug_assertions, feature = "internal-testing")),
        allow(dead_code)
    )]
    pub(crate) fn with_smp_strategy(mut self, strategy: SmpStrategy) -> Self {
        self.smp_strategy = strategy;
        self
    }

    pub fn with_eval_file(mut self, path: impl Into<String>) -> Self {
        self.evaluator = BenchEvaluator::EvalFile(path.into());
        self
    }

    #[doc(hidden)]
    pub fn with_discovered_eval(mut self) -> Self {
        self.evaluator = BenchEvaluator::Discovered;
        self
    }

    pub fn with_classical_eval(mut self) -> Self {
        self.evaluator = BenchEvaluator::Classical;
        self
    }

    pub fn with_small_eval_file(mut self, path: impl Into<String>) -> Self {
        self.small_evaluator = Some(path.into());
        self
    }

    pub fn with_dual_small_fallback(mut self, ambiguity_threshold: i32) -> Self {
        self.dual_eval_policy = DualEvalPolicy::SmallFallback;
        self.dual_eval_threshold = ambiguity_threshold;
        self
    }

    pub fn enable_dual_small_fallback(mut self) -> Self {
        self.dual_eval_policy = DualEvalPolicy::SmallFallback;
        self
    }

    pub fn with_dual_eval_threshold(mut self, ambiguity_threshold: i32) -> Self {
        self.dual_eval_threshold = ambiguity_threshold;
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
    pub threads: usize,
    pub evaluator: String,
    pub dual_eval: Option<DualBenchStats>,
    pub elapsed_ms: u128,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DualBenchStats {
    pub small_evaluator: String,
    pub ambiguity_threshold: i32,
    pub small_selected: u64,
    pub big_fallbacks: u64,
}

impl BenchResult {
    pub fn nps(&self) -> u64 {
        (self.total_nodes as u128 * 1000)
            .checked_div(self.elapsed_ms)
            .unwrap_or(self.total_nodes as u128) as u64
    }

    pub fn render_lines(&self) -> Vec<String> {
        let mut lines = vec![
            format!(
                "bench depth {} positions {} tt {} hash {} threads {} evaluator {}",
                self.depth,
                self.positions,
                if self.tt_enabled { "on" } else { "off" },
                self.hash_mb,
                self.threads,
                self.evaluator
            ),
            format!("bench nodes {}", self.total_nodes),
            format!("bench checksum {:016x}", self.checksum),
            format!("bench time_ms {}", self.elapsed_ms),
            format!("bench nps {}", self.nps()),
        ];
        if let Some(dual) = &self.dual_eval {
            lines.push(format!(
                "bench dual_policy small-fallback small_evaluator {} threshold {} small_selected {} big_fallbacks {}",
                dual.small_evaluator,
                dual.ambiguity_threshold,
                dual.small_selected,
                dual.big_fallbacks,
            ));
        }
        lines
    }
}

pub fn run_bench(config: BenchConfig) -> BenchResult {
    run_threaded_bench(config)
}

fn run_threaded_bench(config: BenchConfig) -> BenchResult {
    let started = Instant::now();
    let mut total_nodes = 0u64;
    let mut checksum = 0u64;
    let mut service = UciSearchService::new();
    service.resize_hash(config.hash_mb);
    service.set_threads(config.threads);
    service.set_smp_strategy(config.smp_strategy);
    configure_evaluator(&mut service, &config.evaluator);
    configure_dual_evaluator(&mut service, &config);

    for fen in BENCH_FENS {
        let mut position = Position::from_fen(fen).expect("bench FEN must parse");
        service.clear_hash();
        let result = service.search(
            &mut position,
            SearchRequest {
                limits: SearchLimits::new(config.depth)
                    .with_hash_mb(config.hash_mb)
                    .with_tt(config.tt_enabled)
                    .with_heuristics(config.heuristics),
                soft_deadline: None,
                hard_deadline: None,
                stop_flag: None,
                root_moves: None,
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

    let dual_eval = dual_bench_stats(&service, &config);
    let evaluator = if config.evaluator == BenchEvaluator::Discovered {
        if service.eval_file().is_empty() {
            "classical".to_owned()
        } else {
            service.eval_file().to_owned()
        }
    } else {
        config.evaluator.label()
    };
    BenchResult {
        depth: config.depth,
        positions: BENCH_FENS.len(),
        total_nodes,
        checksum,
        tt_enabled: config.tt_enabled,
        hash_mb: config.hash_mb,
        threads: config.threads,
        evaluator,
        dual_eval,
        elapsed_ms: started.elapsed().as_millis(),
    }
}

#[cfg_attr(
    not(any(test, debug_assertions, feature = "internal-testing")),
    allow(dead_code)
)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TimedBenchResult {
    pub movetime_ms: u64,
    pub positions: usize,
    pub total_nodes: u64,
    pub checksum: u64,
    pub total_completed_depth: u64,
    pub elapsed_ms: u128,
}

#[cfg_attr(
    not(any(test, debug_assertions, feature = "internal-testing")),
    allow(dead_code)
)]
pub(crate) fn run_timed_bench(config: BenchConfig, movetime_ms: u64) -> TimedBenchResult {
    run_threaded_timed_bench(config, movetime_ms)
}

#[cfg_attr(
    not(any(test, debug_assertions, feature = "internal-testing")),
    allow(dead_code)
)]
fn run_threaded_timed_bench(config: BenchConfig, movetime_ms: u64) -> TimedBenchResult {
    let started = Instant::now();
    let mut total_nodes = 0u64;
    let mut checksum = 0u64;
    let mut total_completed_depth = 0u64;
    let mut service = UciSearchService::new();
    service.resize_hash(config.hash_mb);
    service.set_threads(config.threads);
    service.set_smp_strategy(config.smp_strategy);
    configure_evaluator(&mut service, &config.evaluator);
    configure_dual_evaluator(&mut service, &config);

    for fen in BENCH_FENS {
        let mut position = Position::from_fen(fen).expect("bench FEN must parse");
        service.clear_hash();
        let deadline = Instant::now() + std::time::Duration::from_millis(movetime_ms);
        let result = service.search(
            &mut position,
            SearchRequest {
                limits: SearchLimits::new(127)
                    .with_hash_mb(config.hash_mb)
                    .with_tt(config.tt_enabled)
                    .with_heuristics(config.heuristics),
                soft_deadline: Some(deadline),
                hard_deadline: Some(deadline),
                stop_flag: None,
                root_moves: None,
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

fn configure_evaluator(service: &mut UciSearchService, evaluator: &BenchEvaluator) {
    let path = match evaluator {
        BenchEvaluator::Discovered => return,
        BenchEvaluator::Classical => "",
        BenchEvaluator::EvalFile(path) => path,
    };
    service
        .set_eval_file(path)
        .unwrap_or_else(|error| panic!("bench evaluator '{path}' failed to load: {error}"));
}

fn configure_dual_evaluator(service: &mut UciSearchService, config: &BenchConfig) {
    if let Some(path) = config.small_evaluator.as_deref() {
        service
            .set_small_eval_file(path)
            .unwrap_or_else(|error| panic!("bench small evaluator '{path}' failed: {error}"));
    }
    service
        .set_dual_eval_threshold(config.dual_eval_threshold)
        .unwrap_or_else(|error| panic!("bench dual evaluator threshold failed: {error}"));
    service
        .set_dual_eval_policy(config.dual_eval_policy)
        .unwrap_or_else(|error| panic!("bench dual evaluator policy failed: {error}"));
}

fn dual_bench_stats(service: &UciSearchService, config: &BenchConfig) -> Option<DualBenchStats> {
    (config.dual_eval_policy == DualEvalPolicy::SmallFallback).then(|| {
        let counters = service.dual_eval_counters();
        DualBenchStats {
            small_evaluator: config
                .small_evaluator
                .clone()
                .expect("enabled dual bench must configure SmallEvalFile"),
            ambiguity_threshold: config.dual_eval_threshold,
            small_selected: counters.small_selected,
            big_fallbacks: counters.big_fallbacks,
        }
    })
}

fn hash_text(text: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in text.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}
