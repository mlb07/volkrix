use std::{
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::{self, Receiver, Sender},
    },
    thread::{self, JoinHandle},
    time::Instant,
};

#[cfg(not(volkrix_embedded_nnue))]
use std::path::PathBuf;

use crate::core::{Move, MoveList, Position};

use super::{
    SearchLimits, SearchResult, eval,
    nnue::{DualEvalCounterSnapshot, DualEvalCounters, NnueService},
    root::{
        self, NodeBudget, PonderState, RootSplitCoordinator, SearchControl, SearchResources,
        SearchThreadRole,
    },
    tablebase::{MAX_SYZYGY_PIECES, TablebaseProbeStats, TablebaseService},
    tt::{DEFAULT_HASH_MB, TranspositionTable},
};

pub(crate) const DEFAULT_THREADS: usize = 1;
pub(crate) const MAX_THREADS: usize = 64;
#[cfg(not(volkrix_embedded_nnue))]
pub(crate) const STOCKFISH_18_NETWORK_FILE: &str = "nn-c288c895ea92.nnue";
pub const DEFAULT_DUAL_EVAL_THRESHOLD: i32 = 200;
pub const MAX_DUAL_EVAL_THRESHOLD: i32 = 2_000;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum DualEvalPolicy {
    #[default]
    Off,
    SmallFallback,
}

impl DualEvalPolicy {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::SmallFallback => "small-fallback",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum SmpStrategy {
    /// Previous root-sharded Lazy SMP behavior, retained as an A/B baseline.
    Lazy,
    /// Young-brothers-wait root splitting with disjoint helper sibling shards.
    RootSplit,
    /// Use root splitting where it measured stronger (two total threads) and
    /// retain high-throughput Lazy SMP on wider machines.
    #[default]
    Adaptive,
}

impl SmpStrategy {
    const fn resolve(self, threads: usize) -> Self {
        match self {
            Self::Adaptive if threads == 2 => Self::RootSplit,
            Self::Adaptive => Self::Lazy,
            strategy => strategy,
        }
    }
}

#[derive(Clone)]
#[doc(hidden)]
pub struct SearchRequest {
    pub limits: SearchLimits,
    pub soft_deadline: Option<Instant>,
    pub hard_deadline: Option<Instant>,
    pub stop_flag: Option<Arc<AtomicBool>>,
    pub root_moves: Option<Vec<Move>>,
}

struct WorkerJob {
    position: Position,
    limits: SearchLimits,
    tt: Arc<TranspositionTable>,
    nnue: Option<Arc<NnueService>>,
    tablebases: Option<Arc<TablebaseService>>,
    classical_weights: Option<eval::ClassicalEvalWeights>,
    control: SearchControl,
    done_sender: Sender<WorkerCompletion>,
    active_helpers: Arc<AtomicUsize>,
}

struct HelperSearchSpec<'a> {
    helper_count: usize,
    position: &'a Position,
    limits: SearchLimits,
    tt: Arc<TranspositionTable>,
    nnue: Option<Arc<NnueService>>,
    tablebases: Option<Arc<TablebaseService>>,
    classical_weights: Option<eval::ClassicalEvalWeights>,
    stop_flag: Option<Arc<AtomicBool>>,
    helper_stop_flag: Arc<AtomicBool>,
    soft_deadline: Option<Instant>,
    hard_deadline: Option<Instant>,
    ponder_state: Option<Arc<PonderState>>,
    node_budget: Option<Arc<NodeBudget>>,
    root_moves: Option<Vec<Move>>,
    strategy: SmpStrategy,
    root_split: Option<Arc<RootSplitCoordinator>>,
}

enum WorkerCommand {
    Search(Box<WorkerJob>),
    Shutdown,
}

enum WorkerCompletion {
    Complete(SearchResult),
    Panicked,
}

#[derive(Default)]
struct HelperSearchSummary {
    nodes: u64,
    tt_hits: u64,
    seldepth: u8,
    panics: usize,
}

struct WorkerHandle {
    sender: Sender<WorkerCommand>,
    join_handle: Option<JoinHandle<()>>,
}

struct WorkerPool {
    workers: Vec<WorkerHandle>,
    active_helpers: Arc<AtomicUsize>,
    helper_panics: usize,
    last_helper_nodes: u64,
}

impl WorkerPool {
    fn new() -> Self {
        Self {
            workers: Vec::new(),
            active_helpers: Arc::new(AtomicUsize::new(0)),
            helper_panics: 0,
            last_helper_nodes: 0,
        }
    }

    fn ensure_capacity(&mut self, helper_count: usize) {
        while self.workers.len() < helper_count {
            let worker_index = self.workers.len() + 1;
            self.workers.push(spawn_worker(worker_index));
        }
    }

    fn start_helpers(&mut self, spec: HelperSearchSpec<'_>) -> (usize, Receiver<WorkerCompletion>) {
        self.ensure_capacity(spec.helper_count);
        let (done_sender, done_receiver) = mpsc::channel();
        let helper_root_moves = helper_root_move_sets(
            spec.position,
            spec.root_moves.as_deref(),
            spec.helper_count,
            spec.strategy,
        );
        let mut dispatched = 0usize;

        for (worker_index, root_moves) in helper_root_moves.into_iter().enumerate() {
            let command = WorkerCommand::Search(Box::new(WorkerJob {
                position: spec.position.clone(),
                limits: spec.limits,
                tt: Arc::clone(&spec.tt),
                nnue: spec.nnue.clone(),
                tablebases: spec.tablebases.clone(),
                classical_weights: spec.classical_weights,
                control: SearchControl {
                    stop_flag: spec.stop_flag.clone(),
                    helper_stop_flag: Some(Arc::clone(&spec.helper_stop_flag)),
                    soft_deadline: spec.soft_deadline,
                    hard_deadline: spec.hard_deadline,
                    ponder_state: spec.ponder_state.clone(),
                    node_budget: spec.node_budget.clone(),
                    role: SearchThreadRole::Helper(worker_index + 1),
                    root_moves,
                    root_split: spec.root_split.clone(),
                },
                done_sender: done_sender.clone(),
                active_helpers: Arc::clone(&self.active_helpers),
            }));

            self.active_helpers.fetch_add(1, Ordering::Relaxed);
            match self.workers[worker_index].sender.send(command) {
                Ok(()) => dispatched += 1,
                Err(error) => {
                    self.active_helpers.fetch_sub(1, Ordering::Relaxed);
                    self.replace_worker(worker_index);
                    self.active_helpers.fetch_add(1, Ordering::Relaxed);
                    match self.workers[worker_index].sender.send(error.0) {
                        Ok(()) => dispatched += 1,
                        Err(_) => {
                            self.active_helpers.fetch_sub(1, Ordering::Relaxed);
                            self.helper_panics += 1;
                        }
                    }
                }
            }
        }

        drop(done_sender);
        (dispatched, done_receiver)
    }

    fn finish_helpers(
        &mut self,
        helper_count: usize,
        done_receiver: Receiver<WorkerCompletion>,
    ) -> HelperSearchSummary {
        let mut summary = HelperSearchSummary::default();
        for _ in 0..helper_count {
            match done_receiver.recv() {
                Ok(WorkerCompletion::Complete(result)) => {
                    summary.nodes = summary.nodes.saturating_add(result.nodes);
                    summary.tt_hits = summary.tt_hits.saturating_add(result.tt_hits);
                    summary.seldepth = summary.seldepth.max(result.seldepth);
                }
                Ok(WorkerCompletion::Panicked) => summary.panics += 1,
                Err(_) => {
                    summary.panics += 1;
                    break;
                }
            }
        }
        self.helper_panics = self.helper_panics.saturating_add(summary.panics);
        self.last_helper_nodes = summary.nodes;
        self.active_helpers.store(0, Ordering::Relaxed);
        summary
    }

    fn replace_worker(&mut self, worker_index: usize) {
        if let Some(join_handle) = self.workers[worker_index].join_handle.take() {
            let _ = join_handle.join();
        }
        self.workers[worker_index] = spawn_worker(worker_index + 1);
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    fn worker_count(&self) -> usize {
        self.workers.len()
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    fn active_helper_count(&self) -> usize {
        self.active_helpers.load(Ordering::Relaxed)
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    #[cfg_attr(not(test), allow(dead_code))]
    fn helper_panic_count(&self) -> usize {
        self.helper_panics
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    #[cfg_attr(not(test), allow(dead_code))]
    fn last_helper_nodes(&self) -> u64 {
        self.last_helper_nodes
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        for worker in &self.workers {
            let _ = worker.sender.send(WorkerCommand::Shutdown);
        }
        for worker in &mut self.workers {
            if let Some(join_handle) = worker.join_handle.take() {
                let _ = join_handle.join();
            }
        }
    }
}

#[doc(hidden)]
pub struct UciSearchService {
    hash_mb: usize,
    threads: usize,
    syzygy_path: String,
    syzygy_probe_limit: u8,
    syzygy_50_move_rule: bool,
    eval_file: String,
    eval_discovery_diagnostic: Option<String>,
    tt: Arc<TranspositionTable>,
    nnue: Option<Arc<NnueService>>,
    small_nnue: Option<Arc<NnueService>>,
    dual_nnue: Option<Arc<NnueService>>,
    small_eval_file: String,
    dual_eval_policy: DualEvalPolicy,
    dual_eval_threshold: i32,
    dual_eval_counters: Arc<DualEvalCounters>,
    tablebases: Option<Arc<TablebaseService>>,
    classical_weights: Option<eval::ClassicalEvalWeights>,
    smp_strategy: SmpStrategy,
    workers: WorkerPool,
}

impl UciSearchService {
    pub fn new() -> Self {
        let environment_path = std::env::var("VOLKRIX_EVAL_FILE").ok();
        let executable_path = std::env::current_exe().ok();
        Self::new_with_eval_discovery(environment_path.as_deref(), executable_path.as_deref())
    }

    pub(crate) fn new_with_eval_discovery(
        environment_path: Option<&str>,
        executable_path: Option<&Path>,
    ) -> Self {
        let mut service = Self {
            hash_mb: DEFAULT_HASH_MB,
            threads: DEFAULT_THREADS,
            syzygy_path: String::new(),
            syzygy_probe_limit: MAX_SYZYGY_PIECES,
            syzygy_50_move_rule: true,
            eval_file: String::new(),
            eval_discovery_diagnostic: None,
            tt: Arc::new(TranspositionTable::new_mb(DEFAULT_HASH_MB)),
            nnue: None,
            small_nnue: None,
            dual_nnue: None,
            small_eval_file: String::new(),
            dual_eval_policy: DualEvalPolicy::Off,
            dual_eval_threshold: DEFAULT_DUAL_EVAL_THRESHOLD,
            dual_eval_counters: Arc::new(DualEvalCounters::default()),
            tablebases: None,
            classical_weights: None,
            smp_strategy: SmpStrategy::default(),
            workers: WorkerPool::new(),
        };

        #[cfg(volkrix_embedded_nnue)]
        {
            // OpenBench retains only the built executable. Its compiled network
            // is therefore authoritative at startup: ambient host state must
            // never replace it or turn a bad external path into classical play.
            let _ = (environment_path, executable_path);
            let label = format!(
                "<embedded:{}:{}>",
                env!("VOLKRIX_EMBEDDED_NNUE_SHA256"),
                env!("VOLKRIX_EMBEDDED_NNUE_SIZE")
            );
            let nnue = NnueService::open_embedded_eval()
                .expect("build-validated embedded EvalFile must load at startup");
            service.eval_file = label;
            service.nnue = Some(nnue);
            service
        }

        #[cfg(not(volkrix_embedded_nnue))]
        {
            if let Some(candidate) = default_eval_candidate(environment_path, executable_path) {
                let display_path = candidate.path.to_string_lossy().into_owned();
                match NnueService::open_eval_file(&display_path) {
                    Ok(nnue) => {
                        service.eval_file = display_path;
                        service.nnue = Some(nnue);
                    }
                    Err(error) => {
                        service.eval_discovery_diagnostic = Some(format!(
                            "automatic EvalFile '{}' was ignored; using classical evaluation: {error}",
                            candidate.path.display()
                        ));
                    }
                }
            }
            service
        }
    }

    pub(crate) fn hash_mb(&self) -> usize {
        self.hash_mb
    }

    #[cfg_attr(
        not(any(test, debug_assertions, feature = "internal-testing")),
        allow(dead_code)
    )]
    pub(crate) fn threads(&self) -> usize {
        self.threads
    }

    #[cfg_attr(
        not(any(test, debug_assertions, feature = "internal-testing")),
        allow(dead_code)
    )]
    pub(crate) fn syzygy_path(&self) -> &str {
        &self.syzygy_path
    }

    pub(crate) const fn syzygy_probe_limit(&self) -> u8 {
        self.syzygy_probe_limit
    }

    pub(crate) const fn syzygy_50_move_rule(&self) -> bool {
        self.syzygy_50_move_rule
    }

    pub(crate) fn syzygy_loaded_cardinality(&self) -> Option<u8> {
        self.tablebases
            .as_ref()
            .and_then(|tablebases| tablebases.loaded_cardinality())
    }

    pub(crate) fn syzygy_probe_stats(&self) -> TablebaseProbeStats {
        self.tablebases
            .as_ref()
            .map_or_else(TablebaseProbeStats::default, |tablebases| {
                tablebases.probe_stats()
            })
    }

    pub(crate) fn syzygy_last_probe_error(&self) -> Option<String> {
        self.tablebases
            .as_ref()
            .and_then(|tablebases| tablebases.last_probe_error())
    }

    #[cfg_attr(
        not(any(test, debug_assertions, feature = "internal-testing")),
        allow(dead_code)
    )]
    pub(crate) fn eval_file(&self) -> &str {
        &self.eval_file
    }

    pub(crate) fn small_eval_file(&self) -> &str {
        &self.small_eval_file
    }

    pub(crate) const fn dual_eval_policy(&self) -> DualEvalPolicy {
        self.dual_eval_policy
    }

    pub(crate) const fn dual_eval_threshold(&self) -> i32 {
        self.dual_eval_threshold
    }

    pub(crate) fn dual_eval_counters(&self) -> DualEvalCounterSnapshot {
        self.dual_eval_counters.snapshot()
    }

    pub(crate) fn eval_discovery_diagnostic(&self) -> Option<&str> {
        self.eval_discovery_diagnostic.as_deref()
    }

    pub fn set_threads(&mut self, threads: usize) {
        self.threads = threads.clamp(1, MAX_THREADS);
    }

    pub(crate) fn set_smp_strategy(&mut self, strategy: SmpStrategy) {
        self.smp_strategy = strategy;
    }

    #[cfg(feature = "offline-tools")]
    #[doc(hidden)]
    pub fn set_classical_weights(&mut self, weights: Option<eval::ClassicalEvalWeights>) {
        self.classical_weights = weights;
    }

    pub(crate) fn set_syzygy_path(&mut self, path: &str) -> Result<(), String> {
        let path = path.trim();
        if path.is_empty() {
            self.syzygy_path.clear();
            self.tablebases = None;
            return Ok(());
        }

        if path == self.syzygy_path {
            return Ok(());
        }

        let tablebases = TablebaseService::open_syzygy_path(path, self.tablebases.as_ref())?;
        tablebases.set_probe_limit(self.syzygy_probe_limit);
        tablebases.set_rule50_enabled(self.syzygy_50_move_rule);
        self.syzygy_path = path.to_owned();
        self.tablebases = Some(tablebases);
        Ok(())
    }

    pub(crate) fn set_syzygy_probe_limit(&mut self, limit: u8) {
        self.syzygy_probe_limit = limit.min(MAX_SYZYGY_PIECES);
        if let Some(tablebases) = &self.tablebases {
            tablebases.set_probe_limit(self.syzygy_probe_limit);
        }
    }

    pub(crate) fn set_syzygy_50_move_rule(&mut self, enabled: bool) {
        self.syzygy_50_move_rule = enabled;
        if let Some(tablebases) = &self.tablebases {
            tablebases.set_rule50_enabled(enabled);
        }
    }

    pub(crate) fn set_eval_file(&mut self, path: &str) -> Result<(), String> {
        let path = path.trim();
        if path.is_empty() {
            self.eval_file.clear();
            self.nnue = None;
            self.dual_eval_policy = DualEvalPolicy::Off;
            self.rebuild_dual_nnue();
            self.eval_discovery_diagnostic = None;
            return Ok(());
        }

        if path == self.eval_file {
            return Ok(());
        }

        let nnue = NnueService::open_eval_file(path)?;
        self.eval_file = path.to_owned();
        self.nnue = Some(nnue);
        self.rebuild_dual_nnue();
        self.eval_discovery_diagnostic = None;
        Ok(())
    }

    pub(crate) fn set_small_eval_file(&mut self, path: &str) -> Result<(), String> {
        let path = path.trim();
        if path.is_empty() {
            self.small_eval_file.clear();
            self.small_nnue = None;
            self.dual_eval_policy = DualEvalPolicy::Off;
            self.rebuild_dual_nnue();
            return Ok(());
        }
        if path == self.small_eval_file {
            return Ok(());
        }
        let nnue = NnueService::open_eval_file(path)
            .map_err(|error| format!("failed to load SmallEvalFile '{path}': {error}"))?;
        self.small_eval_file = path.to_owned();
        self.small_nnue = Some(nnue);
        self.rebuild_dual_nnue();
        Ok(())
    }

    pub(crate) fn set_dual_eval_policy(&mut self, policy: DualEvalPolicy) -> Result<(), String> {
        if policy == DualEvalPolicy::SmallFallback
            && (self.nnue.is_none() || self.small_nnue.is_none())
        {
            return Err(
                "DualEvalPolicy small-fallback requires both EvalFile and SmallEvalFile".to_owned(),
            );
        }
        self.dual_eval_policy = policy;
        self.rebuild_dual_nnue();
        Ok(())
    }

    pub(crate) fn set_dual_eval_threshold(&mut self, threshold: i32) -> Result<(), String> {
        if !(0..=MAX_DUAL_EVAL_THRESHOLD).contains(&threshold) {
            return Err(format!(
                "DualEvalThreshold must be between 0 and {MAX_DUAL_EVAL_THRESHOLD}"
            ));
        }
        self.dual_eval_threshold = threshold;
        self.rebuild_dual_nnue();
        Ok(())
    }

    fn rebuild_dual_nnue(&mut self) {
        self.dual_nnue = match (
            self.dual_eval_policy,
            self.nnue.as_ref(),
            self.small_nnue.as_ref(),
        ) {
            (DualEvalPolicy::SmallFallback, Some(big), Some(small)) => {
                Some(NnueService::with_small_fallback(
                    Arc::clone(big),
                    Arc::clone(small),
                    self.dual_eval_threshold,
                    Arc::clone(&self.dual_eval_counters),
                ))
            }
            _ => None,
        };
    }

    fn active_nnue(&self) -> Option<Arc<NnueService>> {
        self.dual_nnue.clone().or_else(|| self.nnue.clone())
    }

    pub fn resize_hash(&mut self, hash_mb: usize) {
        let hash_mb = hash_mb.max(1);
        self.hash_mb = hash_mb;
        self.tt = Arc::new(TranspositionTable::new_mb(hash_mb));
    }

    pub fn clear_hash(&mut self) {
        self.tt.clear();
    }

    pub fn search(&mut self, position: &mut Position, request: SearchRequest) -> SearchResult {
        self.search_with_info(position, request, None)
    }

    pub fn search_with_info<'a>(
        &mut self,
        position: &mut Position,
        request: SearchRequest,
        info_reporter: root::InfoReporter<'a>,
    ) -> SearchResult {
        self.search_with_info_and_ponder(position, request, info_reporter, None)
    }

    pub(crate) fn search_with_info_and_ponder<'a>(
        &mut self,
        position: &mut Position,
        request: SearchRequest,
        mut info_reporter: root::InfoReporter<'a>,
        ponder_state: Option<Arc<PonderState>>,
    ) -> SearchResult {
        let search_started = Instant::now();
        let limits = request.limits.with_hash_mb(self.hash_mb);
        let node_budget = limits
            .node_limit
            .map(|limit| Arc::new(NodeBudget::new(limit)));
        let active_nnue = self.active_nnue();
        let effective_threads = self.effective_threads(limits.tt_enabled);
        if effective_threads <= 1 {
            return root::search_with_control(
                position,
                limits,
                SearchResources {
                    tt: limits.tt_enabled.then(|| Arc::clone(&self.tt)),
                    nnue: active_nnue,
                    tablebases: self.tablebases.clone(),
                    classical_weights: self.classical_weights,
                },
                SearchControl {
                    stop_flag: request.stop_flag,
                    helper_stop_flag: None,
                    soft_deadline: request.soft_deadline,
                    hard_deadline: request.hard_deadline,
                    ponder_state,
                    node_budget,
                    role: SearchThreadRole::Main,
                    root_moves: request.root_moves,
                    root_split: None,
                },
                info_reporter.take(),
            );
        }

        let helper_count = effective_threads - 1;
        let helper_stop_flag = Arc::new(AtomicBool::new(false));
        let active_strategy = self.smp_strategy.resolve(effective_threads);
        let root_split = (active_strategy == SmpStrategy::RootSplit)
            .then(|| Arc::new(RootSplitCoordinator::new()));
        let (started_helpers, done_receiver) = self.workers.start_helpers(HelperSearchSpec {
            helper_count,
            position,
            limits,
            tt: Arc::clone(&self.tt),
            nnue: active_nnue.clone(),
            tablebases: self.tablebases.clone(),
            classical_weights: self.classical_weights,
            stop_flag: request.stop_flag.clone(),
            helper_stop_flag: Arc::clone(&helper_stop_flag),
            soft_deadline: request.soft_deadline,
            hard_deadline: request.hard_deadline,
            ponder_state: ponder_state.clone(),
            node_budget: node_budget.clone(),
            root_moves: request.root_moves.clone(),
            strategy: active_strategy,
            root_split: root_split.clone(),
        });

        let main_search = catch_unwind(AssertUnwindSafe(|| {
            root::search_with_control(
                position,
                limits,
                SearchResources {
                    tt: Some(Arc::clone(&self.tt)),
                    nnue: active_nnue,
                    tablebases: self.tablebases.clone(),
                    classical_weights: self.classical_weights,
                },
                SearchControl {
                    stop_flag: request.stop_flag,
                    helper_stop_flag: Some(Arc::clone(&helper_stop_flag)),
                    soft_deadline: request.soft_deadline,
                    hard_deadline: request.hard_deadline,
                    ponder_state,
                    node_budget,
                    role: SearchThreadRole::Main,
                    root_moves: request.root_moves,
                    root_split: root_split.clone(),
                },
                info_reporter.take(),
            )
        }));

        helper_stop_flag.store(true, Ordering::Relaxed);
        if let Some(root_split) = root_split.as_ref() {
            root_split.cancel();
        }
        let helper_summary = self.workers.finish_helpers(started_helpers, done_receiver);

        match main_search {
            Ok(mut result) => {
                result.nodes = result.nodes.saturating_add(helper_summary.nodes);
                result.tt_hits = result.tt_hits.saturating_add(helper_summary.tt_hits);
                result.seldepth = result.seldepth.max(helper_summary.seldepth);
                let (nodes, tt_hits, seldepth) = (result.nodes, result.tt_hits, result.seldepth);
                if let Some(final_info) = result.info_lines.last_mut() {
                    rewrite_info_statistics(
                        final_info,
                        nodes,
                        tt_hits,
                        seldepth,
                        search_started.elapsed(),
                    );
                }
                result
            }
            Err(payload) => resume_unwind(payload),
        }
    }

    fn effective_threads(&self, tt_enabled: bool) -> usize {
        if !tt_enabled {
            return 1;
        }

        self.threads.clamp(1, MAX_THREADS)
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    pub(crate) fn debug_tt_entry_count(&self) -> usize {
        self.tt.debug_entry_count()
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    pub(crate) fn debug_worker_count(&self) -> usize {
        self.workers.worker_count()
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    pub(crate) fn debug_active_helper_count(&self) -> usize {
        self.workers.active_helper_count()
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn debug_helper_panic_count(&self) -> usize {
        self.workers.helper_panic_count()
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn debug_last_helper_nodes(&self) -> u64 {
        self.workers.last_helper_nodes()
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn debug_nnue_is_enabled(&self) -> bool {
        self.nnue.is_some()
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn debug_nnue_path(&self) -> &str {
        &self.eval_file
    }

    #[cfg(test)]
    pub(crate) fn debug_install_tablebases(
        &mut self,
        path: &str,
        tablebases: Arc<TablebaseService>,
    ) {
        tablebases.set_probe_limit(self.syzygy_probe_limit);
        tablebases.set_rule50_enabled(self.syzygy_50_move_rule);
        self.syzygy_path = path.to_owned();
        self.tablebases = Some(tablebases);
    }

    #[cfg(test)]
    pub(crate) fn debug_install_nnue(&mut self, path: &str, nnue: Arc<NnueService>) {
        self.eval_file = path.to_owned();
        self.nnue = Some(nnue);
        self.rebuild_dual_nnue();
    }
}

#[cfg(not(volkrix_embedded_nnue))]
struct DefaultEvalCandidate {
    path: PathBuf,
}

#[cfg(not(volkrix_embedded_nnue))]
fn default_eval_candidate(
    environment_path: Option<&str>,
    executable_path: Option<&Path>,
) -> Option<DefaultEvalCandidate> {
    if let Some(path) = environment_path
        .map(str::trim)
        .filter(|path| !path.is_empty())
    {
        return Some(DefaultEvalCandidate {
            path: PathBuf::from(path),
        });
    }

    let sibling = executable_path?.parent()?.join(STOCKFISH_18_NETWORK_FILE);
    sibling
        .is_file()
        .then_some(DefaultEvalCandidate { path: sibling })
}

impl Default for UciSearchService {
    fn default() -> Self {
        Self::new()
    }
}

fn spawn_worker(worker_index: usize) -> WorkerHandle {
    let (sender, receiver) = mpsc::channel();
    let join_handle = thread::Builder::new()
        .name(format!("volkrix-smp-{worker_index}"))
        .spawn(move || worker_loop(receiver))
        .expect("SMP worker thread must spawn");
    WorkerHandle {
        sender,
        join_handle: Some(join_handle),
    }
}

fn worker_loop(receiver: Receiver<WorkerCommand>) {
    while let Ok(command) = receiver.recv() {
        match command {
            WorkerCommand::Search(job) => {
                let WorkerJob {
                    mut position,
                    limits,
                    tt,
                    nnue,
                    tablebases,
                    classical_weights,
                    control,
                    done_sender,
                    active_helpers,
                } = *job;
                let search_result = catch_unwind(AssertUnwindSafe(|| {
                    root::search_with_control(
                        &mut position,
                        limits,
                        SearchResources {
                            tt: Some(tt),
                            nnue,
                            tablebases,
                            classical_weights,
                        },
                        control,
                        None,
                    )
                }));
                active_helpers.fetch_sub(1, Ordering::Relaxed);
                let completion = match search_result {
                    Ok(result) => WorkerCompletion::Complete(result),
                    Err(_) => WorkerCompletion::Panicked,
                };
                let _ = done_sender.send(completion);
            }
            WorkerCommand::Shutdown => break,
        }
    }
}

fn helper_root_move_sets(
    position: &Position,
    requested_root_moves: Option<&[Move]>,
    helper_count: usize,
    strategy: SmpStrategy,
) -> Vec<Option<Vec<Move>>> {
    if helper_count == 0 {
        return Vec::new();
    }
    let strategy = strategy.resolve(helper_count + 1);

    let root_moves = if let Some(requested) = requested_root_moves {
        requested.to_vec()
    } else {
        let mut scratch = position.clone();
        let mut legal_moves = MoveList::new();
        scratch.generate_legal_moves(&mut legal_moves);
        legal_moves.as_slice().to_vec()
    };

    if strategy == SmpStrategy::Lazy {
        if helper_count == 1 || root_moves.len() <= 1 {
            return (0..helper_count)
                .map(|_| requested_root_moves.map(|moves| moves.to_vec()))
                .collect();
        }
        return (0..helper_count)
            .map(|helper_index| {
                let mut shard = root_moves
                    .iter()
                    .copied()
                    .skip(helper_index)
                    .step_by(helper_count)
                    .collect::<Vec<_>>();
                if shard.is_empty() {
                    shard.push(root_moves[helper_index % root_moves.len()]);
                }
                Some(shard)
            })
            .collect();
    }

    // The main thread owns the eldest generated move. The release gate in
    // `root` ensures helpers do not start siblings before main establishes a
    // bound. Main remains authoritative and eventually verifies all moves, so
    // helper ordering cannot affect correctness.
    let root_moves = root_moves.into_iter().skip(1).collect::<Vec<_>>();

    if root_moves.is_empty() {
        return Vec::new();
    }

    let dispatched_helpers = helper_count.min(root_moves.len());

    (0..dispatched_helpers)
        .map(|helper_index| {
            let shard = root_moves
                .iter()
                .copied()
                .skip(helper_index)
                .step_by(dispatched_helpers)
                .collect::<Vec<_>>();
            debug_assert!(!shard.is_empty());
            Some(shard)
        })
        .collect()
}

fn rewrite_info_statistics(
    line: &mut String,
    nodes: u64,
    tt_hits: u64,
    seldepth: u8,
    elapsed: std::time::Duration,
) {
    let mut fields = line
        .split_whitespace()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let elapsed_ms = elapsed.as_millis();
    let nps = (nodes as u128)
        .saturating_mul(1_000)
        .checked_div(elapsed_ms)
        .unwrap_or_else(|| nodes.saturating_mul(1_000) as u128)
        .min(u64::MAX as u128) as u64;

    replace_value_after(&mut fields, "seldepth", seldepth.to_string());
    replace_value_after(&mut fields, "nodes", nodes.to_string());
    replace_value_after(&mut fields, "nps", nps.to_string());
    replace_value_after(&mut fields, "tthits", tt_hits.to_string());
    replace_value_after(&mut fields, "time", elapsed_ms.to_string());
    *line = fields.join(" ");
}

#[cfg(test)]
fn value_after<'a>(fields: &'a [String], name: &str) -> Option<&'a str> {
    let index = fields.iter().position(|field| field == name)?;
    fields.get(index + 1).map(String::as_str)
}

fn replace_value_after(fields: &mut [String], name: &str, value: String) {
    if let Some(index) = fields.iter().position(|field| field == name)
        && let Some(field) = fields.get_mut(index + 1)
    {
        *field = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::SearchLimits;
    use crate::search::nnue::{NnueService, tiny_test_evalfile_path};
    use crate::search::tablebase::{MockTablebaseBackend, TablebaseService, WdlOutcome};
    use std::{
        sync::Arc,
        time::{Duration, Instant},
    };

    fn mock_tablebases(fen: &str, best_move: &str) -> Arc<TablebaseService> {
        TablebaseService::from_backend_for_tests(
            "/mock/syzygy",
            Arc::new(MockTablebaseBackend::new().with_root_probe(
                fen,
                best_move,
                WdlOutcome::Win,
                Some(1),
            )),
        )
    }

    fn tiny_test_nnue() -> Arc<NnueService> {
        NnueService::open_eval_file(
            tiny_test_evalfile_path()
                .to_str()
                .expect("tiny test eval file path must be UTF-8"),
        )
        .expect("tiny deterministic NNUE test net must load")
    }

    #[cfg(volkrix_embedded_nnue)]
    #[test]
    fn embedded_startup_ignores_environment_evalfile_even_when_invalid() {
        let service = UciSearchService::new_with_eval_discovery(
            Some("/definitely/missing/ambient-network.nnue"),
            None,
        );
        assert!(service.debug_nnue_is_enabled());
        assert!(service.debug_nnue_path().starts_with("<embedded:"));
        assert!(service.eval_discovery_diagnostic().is_none());
    }

    #[cfg(volkrix_embedded_nnue)]
    #[test]
    fn embedded_startup_ignores_sibling_but_explicit_setoption_path_can_replace_it() {
        let directory = std::env::temp_dir().join(format!(
            "volkrix-embedded-precedence-{}",
            std::process::id()
        ));
        std::fs::create_dir(&directory).expect("test directory must create");
        let sibling = directory.join("nn-c288c895ea92.nnue");
        std::fs::write(&sibling, b"corrupt ambient sibling").expect("sibling fixture must write");

        let mut service =
            UciSearchService::new_with_eval_discovery(None, Some(&directory.join("volkrix")));
        assert!(service.debug_nnue_path().starts_with("<embedded:"));
        let replacement = tiny_test_evalfile_path();
        service
            .set_eval_file(replacement.to_str().expect("test path must be UTF-8"))
            .expect("explicit UCI evaluator replacement must remain supported");
        assert_eq!(service.debug_nnue_path(), replacement.to_string_lossy());

        std::fs::remove_file(sibling).expect("sibling fixture must remove");
        std::fs::remove_dir(directory).expect("test directory must remove");
    }

    #[test]
    fn worker_pool_scales_and_helpers_return_to_idle() {
        let mut service = UciSearchService::new();
        service.set_threads(4);
        let mut position = Position::startpos();

        let result = service.search(
            &mut position,
            SearchRequest {
                limits: SearchLimits::new(5),
                soft_deadline: None,
                hard_deadline: None,
                stop_flag: None,
                root_moves: None,
            },
        );

        assert!(result.best_move.is_some());
        assert_eq!(service.debug_active_helper_count(), 0);
        assert!(service.debug_worker_count() >= service.effective_threads(true) - 1);
        assert_eq!(service.debug_helper_panic_count(), 0);
        assert!(service.debug_last_helper_nodes() > 0);
        assert!(result.nodes >= service.debug_last_helper_nodes());
        let final_info = result
            .info_lines
            .last()
            .expect("completed search must report final info");
        let final_fields = final_info
            .split_whitespace()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let expected_nodes = result.nodes.to_string();
        assert_eq!(
            value_after(&final_fields, "nodes"),
            Some(expected_nodes.as_str())
        );
    }

    #[test]
    fn aggregate_info_uses_end_to_end_elapsed_time_for_time_and_nps() {
        let mut line =
            "info depth 8 seldepth 12 score cp 31 nodes 100 nps 1000 tthits 4 time 100 pv e2e4"
                .to_owned();

        rewrite_info_statistics(&mut line, 1_200, 55, 17, Duration::from_millis(800));

        let fields = line
            .split_whitespace()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        assert_eq!(value_after(&fields, "seldepth"), Some("17"));
        assert_eq!(value_after(&fields, "nodes"), Some("1200"));
        assert_eq!(value_after(&fields, "nps"), Some("1500"));
        assert_eq!(value_after(&fields, "tthits"), Some("55"));
        assert_eq!(value_after(&fields, "time"), Some("800"));
    }

    #[test]
    fn threaded_node_limit_is_an_exact_aggregate_budget() {
        let mut service = UciSearchService::new();
        service.set_threads(4);
        let mut position = Position::startpos();
        let before = position.to_fen();

        let result = service.search(
            &mut position,
            SearchRequest {
                limits: SearchLimits::new(127).with_node_limit(Some(4_001)),
                soft_deadline: None,
                hard_deadline: None,
                stop_flag: None,
                root_moves: None,
            },
        );

        assert_eq!(result.nodes, 4_001);
        assert!(result.best_move.is_some());
        assert_eq!(position.to_fen(), before);
        assert_eq!(service.debug_active_helper_count(), 0);
    }

    #[test]
    fn helper_root_shards_cover_each_legal_move_once() {
        let position = Position::startpos();
        let shards = helper_root_move_sets(&position, None, 3, SmpStrategy::Lazy);
        assert_eq!(shards.len(), 3);

        let mut all_moves = Vec::new();
        for shard in shards {
            let shard = shard.expect("multi-helper search must use explicit root shards");
            assert!(!shard.is_empty());
            for mv in shard {
                assert!(!all_moves.contains(&mv), "root shards must not overlap");
                all_moves.push(mv);
            }
        }

        let mut scratch = position.clone();
        let mut legal_moves = MoveList::new();
        scratch.generate_legal_moves(&mut legal_moves);
        assert_eq!(all_moves.len(), legal_moves.len());
        assert!(
            legal_moves
                .as_slice()
                .iter()
                .all(|mv| all_moves.contains(mv))
        );
    }

    #[test]
    fn helper_root_shards_preserve_searchmoves_filter() {
        let position = Position::startpos();
        let mut scratch = position.clone();
        let mut legal_moves = MoveList::new();
        scratch.generate_legal_moves(&mut legal_moves);
        let requested = &legal_moves.as_slice()[..2];

        let shards = helper_root_move_sets(&position, Some(requested), 4, SmpStrategy::Lazy);
        assert_eq!(shards.len(), 4);
        assert!(shards.iter().all(|shard| {
            shard
                .as_deref()
                .is_some_and(|moves| moves.len() == 1 && requested.contains(&moves[0]))
        }));
    }

    #[test]
    fn root_split_reserves_eldest_and_shards_siblings_once() {
        let position = Position::startpos();
        let mut scratch = position.clone();
        let mut legal_moves = MoveList::new();
        scratch.generate_legal_moves(&mut legal_moves);
        let eldest = legal_moves.as_slice()[0];

        let shards = helper_root_move_sets(&position, None, 64, SmpStrategy::RootSplit);
        assert_eq!(shards.len(), legal_moves.len() - 1);

        let mut siblings = Vec::new();
        for shard in shards {
            let shard = shard.expect("root split helpers must receive explicit sibling shards");
            assert_eq!(shard.len(), 1);
            assert_ne!(shard[0], eldest);
            assert!(!siblings.contains(&shard[0]));
            siblings.push(shard[0]);
        }
        assert_eq!(siblings.len(), legal_moves.len() - 1);
    }

    #[test]
    fn repeated_threaded_searches_reuse_existing_workers() {
        let mut service = UciSearchService::new();
        service.set_threads(3);
        let mut first = Position::startpos();
        let _ = service.search(
            &mut first,
            SearchRequest {
                limits: SearchLimits::new(2),
                soft_deadline: None,
                hard_deadline: None,
                stop_flag: None,
                root_moves: None,
            },
        );
        let worker_count = service.debug_worker_count();

        let mut second = Position::startpos();
        let _ = service.search(
            &mut second,
            SearchRequest {
                limits: SearchLimits::new(2),
                soft_deadline: None,
                hard_deadline: None,
                stop_flag: None,
                root_moves: None,
            },
        );

        assert_eq!(service.debug_worker_count(), worker_count);
        assert_eq!(service.debug_active_helper_count(), 0);
    }

    #[test]
    fn smp_strategy_seam_does_not_change_threads_one() {
        let fen = "k7/8/1QK5/8/8/8/8/8 w - - 0 1";
        let mut lazy = UciSearchService::new_with_eval_discovery(None, None);
        lazy.set_smp_strategy(SmpStrategy::Lazy);
        let mut root_split = UciSearchService::new_with_eval_discovery(None, None);
        root_split.set_smp_strategy(SmpStrategy::RootSplit);
        let request = SearchRequest {
            limits: SearchLimits::new(4),
            soft_deadline: None,
            hard_deadline: None,
            stop_flag: None,
            root_moves: None,
        };

        let lazy_result = lazy.search(
            &mut Position::from_fen(fen).expect("FEN must parse"),
            request.clone(),
        );
        let root_split_result = root_split.search(
            &mut Position::from_fen(fen).expect("FEN must parse"),
            request,
        );

        assert_eq!(root_split_result.best_move, lazy_result.best_move);
        assert_eq!(root_split_result.score, lazy_result.score);
        assert_eq!(root_split_result.depth, lazy_result.depth);
        assert_eq!(root_split_result.seldepth, lazy_result.seldepth);
        assert_eq!(root_split_result.nodes, lazy_result.nodes);
        assert_eq!(root_split_result.pv, lazy_result.pv);
        assert_eq!(root_split_result.tt_hits, lazy_result.tt_hits);
        assert_eq!(lazy.debug_worker_count(), 0);
        assert_eq!(root_split.debug_worker_count(), 0);
    }

    #[test]
    fn root_split_helpers_preserve_authoritative_unique_bestmove() {
        let fen = "k7/8/1QK5/8/8/8/8/8 w - - 0 1";
        let mut service = UciSearchService::new_with_eval_discovery(None, None);
        service.set_threads(4);
        service.set_smp_strategy(SmpStrategy::RootSplit);
        let mut position = Position::from_fen(fen).expect("FEN must parse");
        let before = position.to_fen();

        let result = service.search(
            &mut position,
            SearchRequest {
                limits: SearchLimits::new(4),
                soft_deadline: None,
                hard_deadline: None,
                stop_flag: None,
                root_moves: None,
            },
        );

        assert_eq!(
            result.best_move.map(|mv| mv.to_string()),
            Some("b6b7".to_owned())
        );
        assert!(service.debug_worker_count() > 0);
        assert_eq!(service.debug_active_helper_count(), 0);
        assert_eq!(position.to_fen(), before);
    }

    #[test]
    fn threaded_search_preserves_root_state_and_main_only_info_lines() {
        let mut service = UciSearchService::new();
        service.set_threads(2);
        let mut position =
            Position::from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 2 3")
                .expect("FEN parse must succeed");
        let before = position.to_fen();
        let before_key = position.zobrist_key();
        let before_search_key = position.debug_search_key();
        let before_history = position.debug_repetition_history_snapshot();

        let result = service.search(
            &mut position,
            SearchRequest {
                limits: SearchLimits::new(3),
                soft_deadline: None,
                hard_deadline: None,
                stop_flag: None,
                root_moves: None,
            },
        );

        assert!(result.best_move.is_some());
        assert_eq!(result.info_lines.len(), result.depth as usize);
        assert_eq!(position.to_fen(), before);
        assert_eq!(position.zobrist_key(), before_key);
        assert_eq!(position.debug_search_key(), before_search_key);
        assert_eq!(position.debug_repetition_history_snapshot(), before_history);
        assert_eq!(service.debug_active_helper_count(), 0);
        position.validate().expect("position must remain valid");
    }

    #[test]
    fn mock_tablebase_root_resolution_is_correct_in_threads_one() {
        let fen = "8/8/8/8/8/3Q4/2K5/k7 w - - 0 1";
        let mut service = UciSearchService::new();
        service.debug_install_tablebases("/mock/syzygy", mock_tablebases(fen, "d3d7"));
        let mut position = Position::from_fen(fen).expect("FEN parse must succeed");
        let before = position.to_fen();

        let result = service.search(
            &mut position,
            SearchRequest {
                limits: SearchLimits::new(5),
                soft_deadline: None,
                hard_deadline: None,
                stop_flag: None,
                root_moves: None,
            },
        );

        assert_eq!(
            result.best_move.map(|mv| mv.to_string()),
            Some("d3d7".to_owned())
        );
        assert_eq!(result.nodes, 0);
        assert_eq!(position.to_fen(), before);
        assert_eq!(service.debug_active_helper_count(), 0);
    }

    #[test]
    fn mock_tablebase_root_resolution_is_correct_in_threads_two() {
        let fen = "8/8/8/8/8/3Q4/2K5/k7 w - - 0 1";
        let mut service = UciSearchService::new();
        service.set_threads(2);
        service.debug_install_tablebases("/mock/syzygy", mock_tablebases(fen, "d3d7"));
        let mut position = Position::from_fen(fen).expect("FEN parse must succeed");
        let before = position.to_fen();

        let result = service.search(
            &mut position,
            SearchRequest {
                limits: SearchLimits::new(5),
                soft_deadline: None,
                hard_deadline: None,
                stop_flag: None,
                root_moves: None,
            },
        );

        assert_eq!(
            result.best_move.map(|mv| mv.to_string()),
            Some("d3d7".to_owned())
        );
        assert!(service.debug_worker_count() >= 1);
        assert_eq!(service.debug_active_helper_count(), 0);
        assert_eq!(position.to_fen(), before);
    }

    #[test]
    fn eval_file_reconfiguration_preserves_previous_service_on_failure() {
        let mut service = UciSearchService::new();
        let eval_file = tiny_test_evalfile_path();
        let eval_file = eval_file
            .to_str()
            .expect("tiny test eval file path must be UTF-8");

        service
            .set_eval_file(eval_file)
            .expect("tiny deterministic NNUE test net must load");
        assert!(service.debug_nnue_is_enabled());
        assert_eq!(service.debug_nnue_path(), eval_file);

        let error = service
            .set_eval_file("/tmp/missing-network.volknnue")
            .expect_err("missing network must be rejected");
        assert!(error.contains("failed to read EvalFile"));
        assert!(service.debug_nnue_is_enabled());
        assert_eq!(service.debug_nnue_path(), eval_file);
    }

    #[test]
    fn nnue_enabled_search_preserves_root_state_in_threads_one() {
        let mut service = UciSearchService::new();
        service.debug_install_nnue("/mock/nnue", tiny_test_nnue());
        let mut position =
            Position::from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 2 3")
                .expect("FEN parse must succeed");
        let before = position.to_fen();
        let before_key = position.zobrist_key();
        let before_search_key = position.debug_search_key();
        let before_history = position.debug_repetition_history_snapshot();

        let result = service.search(
            &mut position,
            SearchRequest {
                limits: SearchLimits::new(3),
                soft_deadline: None,
                hard_deadline: None,
                stop_flag: None,
                root_moves: None,
            },
        );

        assert!(result.best_move.is_some());
        assert_eq!(result.info_lines.len(), result.depth as usize);
        assert_eq!(position.to_fen(), before);
        assert_eq!(position.zobrist_key(), before_key);
        assert_eq!(position.debug_search_key(), before_search_key);
        assert_eq!(position.debug_repetition_history_snapshot(), before_history);
        assert_eq!(service.debug_active_helper_count(), 0);
    }

    #[test]
    fn configured_small_network_is_inert_while_dual_policy_is_off() {
        let eval_file = tiny_test_evalfile_path();
        let eval_file = eval_file
            .to_str()
            .expect("tiny test net path must be UTF-8");
        let mut big_only = UciSearchService::new_with_eval_discovery(None, None);
        big_only
            .set_eval_file(eval_file)
            .expect("big test net must load");
        let mut dual_off = UciSearchService::new_with_eval_discovery(None, None);
        dual_off
            .set_eval_file(eval_file)
            .expect("big test net must load");
        dual_off
            .set_small_eval_file(eval_file)
            .expect("small test net must load");
        assert_eq!(dual_off.dual_eval_policy(), DualEvalPolicy::Off);

        let request = SearchRequest {
            limits: SearchLimits::new(4),
            soft_deadline: None,
            hard_deadline: None,
            stop_flag: None,
            root_moves: None,
        };
        let big_result = big_only.search(&mut Position::startpos(), request.clone());
        let off_result = dual_off.search(&mut Position::startpos(), request);

        assert_eq!(off_result.best_move, big_result.best_move);
        assert_eq!(off_result.score, big_result.score);
        assert_eq!(off_result.depth, big_result.depth);
        assert_eq!(off_result.seldepth, big_result.seldepth);
        assert_eq!(off_result.nodes, big_result.nodes);
        assert_eq!(off_result.pv, big_result.pv);
        assert_eq!(off_result.tt_hits, big_result.tt_hits);
        assert_eq!(
            dual_off.dual_eval_counters(),
            DualEvalCounterSnapshot::default()
        );
    }

    #[test]
    fn nnue_enabled_search_preserves_root_state_in_threads_two() {
        let mut service = UciSearchService::new();
        service.set_threads(2);
        service.debug_install_nnue("/mock/nnue", tiny_test_nnue());
        let mut position =
            Position::from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 2 3")
                .expect("FEN parse must succeed");
        let before = position.to_fen();
        let before_key = position.zobrist_key();
        let before_search_key = position.debug_search_key();
        let before_history = position.debug_repetition_history_snapshot();

        let result = service.search(
            &mut position,
            SearchRequest {
                limits: SearchLimits::new(3),
                soft_deadline: None,
                hard_deadline: None,
                stop_flag: None,
                root_moves: None,
            },
        );

        assert!(result.best_move.is_some());
        assert_eq!(result.info_lines.len(), result.depth as usize);
        assert!(service.debug_worker_count() >= 1);
        assert_eq!(service.debug_active_helper_count(), 0);
        assert_eq!(position.to_fen(), before);
        assert_eq!(position.zobrist_key(), before_key);
        assert_eq!(position.debug_search_key(), before_search_key);
        assert_eq!(position.debug_repetition_history_snapshot(), before_history);
    }

    #[test]
    fn tablebase_root_resolution_remains_authoritative_when_nnue_is_enabled() {
        let fen = "8/8/8/8/8/3Q4/2K5/k7 w - - 0 1";
        let mut service = UciSearchService::new();
        service.debug_install_nnue("/mock/nnue", tiny_test_nnue());
        service.debug_install_tablebases("/mock/syzygy", mock_tablebases(fen, "d3d7"));
        let mut position = Position::from_fen(fen).expect("FEN parse must succeed");

        let result = service.search(
            &mut position,
            SearchRequest {
                limits: SearchLimits::new(5),
                soft_deadline: None,
                hard_deadline: None,
                stop_flag: None,
                root_moves: None,
            },
        );

        assert_eq!(
            result.best_move.map(|mv| mv.to_string()),
            Some("d3d7".to_owned())
        );
        assert_eq!(result.nodes, 0);
    }

    #[test]
    #[ignore = "manual real-net smoke for Phase 13 Threads=1"]
    fn real_net_smoke_threads_one() {
        let eval_file = std::env::var("VOLKRIX_EVALFILE")
            .expect("VOLKRIX_EVALFILE must point to a real NNUE file");
        let mut service = UciSearchService::new();
        service
            .set_eval_file(&eval_file)
            .expect("real NNUE file must load");
        let mut position = Position::startpos();
        let result = service.search(
            &mut position,
            SearchRequest {
                limits: SearchLimits::new(3),
                soft_deadline: None,
                hard_deadline: None,
                stop_flag: None,
                root_moves: None,
            },
        );
        assert!(result.best_move.is_some());
        assert!(result.score.0.abs() < super::root::INF);
    }

    #[test]
    #[ignore = "manual real-net smoke for Phase 13 Threads=2"]
    fn real_net_smoke_threads_two() {
        let eval_file = std::env::var("VOLKRIX_EVALFILE")
            .expect("VOLKRIX_EVALFILE must point to a real NNUE file");
        let mut service = UciSearchService::new();
        service.set_threads(2);
        service
            .set_eval_file(&eval_file)
            .expect("real NNUE file must load");
        let mut position = Position::startpos();
        let result = service.search(
            &mut position,
            SearchRequest {
                limits: SearchLimits::new(3),
                soft_deadline: None,
                hard_deadline: None,
                stop_flag: None,
                root_moves: None,
            },
        );
        assert!(result.best_move.is_some());
        assert!(result.score.0.abs() < super::root::INF);
        assert_eq!(service.debug_active_helper_count(), 0);
    }

    #[test]
    #[ignore = "manual Phase 13 candidate-vs-fallback sanity comparison"]
    fn phase_thirteen_candidate_vs_fallback_sanity_report() {
        let eval_file = std::env::var("VOLKRIX_EVALFILE")
            .expect("VOLKRIX_EVALFILE must point to a real NNUE file");
        let positions = [
            crate::core::STARTPOS_FEN,
            "r2q1rk1/ppp2ppp/2npbn2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8",
            "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 2 3",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        ];

        let mut fallback = UciSearchService::new();
        fallback.resize_hash(64);
        fallback.set_threads(1);

        let mut candidate = UciSearchService::new();
        candidate.resize_hash(64);
        candidate.set_threads(1);
        candidate
            .set_eval_file(&eval_file)
            .expect("real NNUE file must load");

        for fen in positions {
            let mut fallback_position = Position::from_fen(fen).expect("FEN parse must succeed");
            let mut candidate_position = Position::from_fen(fen).expect("FEN parse must succeed");

            let fallback_result = fallback.search(
                &mut fallback_position,
                SearchRequest {
                    limits: SearchLimits::new(5),
                    soft_deadline: None,
                    hard_deadline: None,
                    stop_flag: None,
                    root_moves: None,
                },
            );
            let candidate_result = candidate.search(
                &mut candidate_position,
                SearchRequest {
                    limits: SearchLimits::new(5),
                    soft_deadline: None,
                    hard_deadline: None,
                    stop_flag: None,
                    root_moves: None,
                },
            );

            println!(
                "candidate_vs_fallback fen \"{fen}\" fallback bestmove {} score {} nodes {} | candidate bestmove {} score {} nodes {}",
                fallback_result
                    .best_move
                    .map(|mv| mv.to_string())
                    .unwrap_or_else(|| "0000".to_owned()),
                fallback_result.score.0,
                fallback_result.nodes,
                candidate_result
                    .best_move
                    .map(|mv| mv.to_string())
                    .unwrap_or_else(|| "0000".to_owned()),
                candidate_result.score.0,
                candidate_result.nodes,
            );

            assert!(fallback_result.best_move.is_some());
            assert!(candidate_result.best_move.is_some());
            assert!(fallback_result.score.0.abs() < super::root::INF);
            assert!(candidate_result.score.0.abs() < super::root::INF);
        }
    }

    #[test]
    #[ignore = "manual mock-backed tablebase validation report for Phase 11"]
    fn phase_eleven_mock_tablebase_report() {
        let fen = "8/8/8/8/8/3Q4/2K5/k7 w - - 0 1";

        let mut baseline_service = UciSearchService::new();
        let mut baseline_position = Position::from_fen(fen).expect("FEN parse must succeed");
        let baseline_started = Instant::now();
        let baseline = baseline_service.search(
            &mut baseline_position,
            SearchRequest {
                limits: SearchLimits::new(5),
                soft_deadline: None,
                hard_deadline: None,
                stop_flag: None,
                root_moves: None,
            },
        );
        println!(
            "mock_tb baseline threads1: bestmove {} nodes {} time_ms {}",
            baseline
                .best_move
                .map(|mv| mv.to_string())
                .unwrap_or_else(|| "0000".to_owned()),
            baseline.nodes,
            baseline_started.elapsed().as_millis()
        );

        let mut tb_threads1 = UciSearchService::new();
        tb_threads1.debug_install_tablebases("/mock/syzygy", mock_tablebases(fen, "d3d7"));
        let mut tb_position_1 = Position::from_fen(fen).expect("FEN parse must succeed");
        let tb_started_1 = Instant::now();
        let result_1 = tb_threads1.search(
            &mut tb_position_1,
            SearchRequest {
                limits: SearchLimits::new(5),
                soft_deadline: None,
                hard_deadline: None,
                stop_flag: None,
                root_moves: None,
            },
        );
        println!(
            "mock_tb enabled threads1: bestmove {} nodes {} time_ms {}",
            result_1
                .best_move
                .map(|mv| mv.to_string())
                .unwrap_or_else(|| "0000".to_owned()),
            result_1.nodes,
            tb_started_1.elapsed().as_millis()
        );

        let mut tb_threads2 = UciSearchService::new();
        tb_threads2.set_threads(2);
        tb_threads2.debug_install_tablebases("/mock/syzygy", mock_tablebases(fen, "d3d7"));
        let mut tb_position_2 = Position::from_fen(fen).expect("FEN parse must succeed");
        let tb_started_2 = Instant::now();
        let result_2 = tb_threads2.search(
            &mut tb_position_2,
            SearchRequest {
                limits: SearchLimits::new(5),
                soft_deadline: None,
                hard_deadline: None,
                stop_flag: None,
                root_moves: None,
            },
        );
        println!(
            "mock_tb enabled threads2: bestmove {} nodes {} time_ms {}",
            result_2
                .best_move
                .map(|mv| mv.to_string())
                .unwrap_or_else(|| "0000".to_owned()),
            result_2.nodes,
            tb_started_2.elapsed().as_millis()
        );
    }

    #[test]
    #[ignore = "requires VOLKRIX_SYZYGY_PATH with real Syzygy files"]
    fn real_tablebase_root_resolution_is_correct_in_threads_one() {
        let path = std::env::var("VOLKRIX_SYZYGY_PATH")
            .expect("VOLKRIX_SYZYGY_PATH must be set for real tablebase tests");
        let fen = "8/8/8/8/8/3Q4/2K5/k7 w - - 0 1";
        let mut service = UciSearchService::new();
        service
            .set_syzygy_path(&path)
            .expect("approved Fathom backend must initialize");
        let mut position = Position::from_fen(fen).expect("FEN parse must succeed");
        let before = position.to_fen();

        let result = service.search(
            &mut position,
            SearchRequest {
                limits: SearchLimits::new(5),
                soft_deadline: None,
                hard_deadline: None,
                stop_flag: None,
                root_moves: None,
            },
        );

        assert!(result.best_move.is_some());
        assert_eq!(result.nodes, 0);
        assert_eq!(position.to_fen(), before);
        assert_eq!(service.debug_active_helper_count(), 0);
    }

    #[test]
    #[ignore = "requires VOLKRIX_SYZYGY_PATH with real Syzygy files"]
    fn real_tablebase_root_resolution_is_correct_in_threads_two() {
        let path = std::env::var("VOLKRIX_SYZYGY_PATH")
            .expect("VOLKRIX_SYZYGY_PATH must be set for real tablebase tests");
        let fen = "8/8/8/8/8/3Q4/2K5/k7 w - - 0 1";
        let mut service = UciSearchService::new();
        service.set_threads(2);
        service
            .set_syzygy_path(&path)
            .expect("approved Fathom backend must initialize");
        let mut position = Position::from_fen(fen).expect("FEN parse must succeed");
        let before = position.to_fen();

        let result = service.search(
            &mut position,
            SearchRequest {
                limits: SearchLimits::new(5),
                soft_deadline: None,
                hard_deadline: None,
                stop_flag: None,
                root_moves: None,
            },
        );

        assert!(result.best_move.is_some());
        assert_eq!(result.nodes, 0);
        assert_eq!(position.to_fen(), before);
        assert_eq!(service.debug_active_helper_count(), 0);
    }
}
