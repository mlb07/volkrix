use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::{self, Receiver, Sender},
    },
    thread::{self, JoinHandle},
    time::Instant,
};

use crate::core::Position;

use super::{
    SearchLimits, SearchResult,
    nnue::NnueService,
    root::{self, SearchControl, SearchThreadRole},
    tablebase::TablebaseService,
    tt::{DEFAULT_HASH_MB, TranspositionTable},
};

pub(crate) const DEFAULT_THREADS: usize = 1;
pub(crate) const MAX_THREADS: usize = 64;

#[derive(Clone)]
pub(crate) struct SearchRequest {
    pub(crate) limits: SearchLimits,
    pub(crate) soft_deadline: Option<Instant>,
    pub(crate) hard_deadline: Option<Instant>,
    pub(crate) stop_flag: Option<Arc<AtomicBool>>,
}

struct WorkerJob {
    position: Position,
    limits: SearchLimits,
    tt: Arc<TranspositionTable>,
    nnue: Option<Arc<NnueService>>,
    tablebases: Option<Arc<TablebaseService>>,
    control: SearchControl,
    done_sender: Sender<()>,
}

struct HelperSearchSpec<'a> {
    helper_count: usize,
    position: &'a Position,
    limits: SearchLimits,
    tt: Arc<TranspositionTable>,
    nnue: Option<Arc<NnueService>>,
    tablebases: Option<Arc<TablebaseService>>,
    stop_flag: Option<Arc<AtomicBool>>,
    helper_stop_flag: Arc<AtomicBool>,
    soft_deadline: Option<Instant>,
    hard_deadline: Option<Instant>,
}

enum WorkerCommand {
    Search(Box<WorkerJob>),
    Shutdown,
}

struct WorkerHandle {
    sender: Sender<WorkerCommand>,
    join_handle: Option<JoinHandle<()>>,
}

struct WorkerPool {
    workers: Vec<WorkerHandle>,
    active_helpers: Arc<AtomicUsize>,
}

impl WorkerPool {
    fn new() -> Self {
        Self {
            workers: Vec::new(),
            active_helpers: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn ensure_capacity(&mut self, helper_count: usize) {
        while self.workers.len() < helper_count {
            let worker_index = self.workers.len() + 1;
            self.workers.push(spawn_worker(worker_index));
        }
    }

    fn start_helpers(&mut self, spec: HelperSearchSpec<'_>) -> Receiver<()> {
        self.ensure_capacity(spec.helper_count);
        self.active_helpers
            .store(spec.helper_count, Ordering::Relaxed);
        let (done_sender, done_receiver) = mpsc::channel();

        for worker_index in 0..spec.helper_count {
            let command = WorkerCommand::Search(Box::new(WorkerJob {
                position: spec.position.clone(),
                limits: spec.limits,
                tt: Arc::clone(&spec.tt),
                nnue: spec.nnue.clone(),
                tablebases: spec.tablebases.clone(),
                control: SearchControl {
                    stop_flag: spec.stop_flag.clone(),
                    helper_stop_flag: Some(Arc::clone(&spec.helper_stop_flag)),
                    soft_deadline: spec.soft_deadline,
                    hard_deadline: spec.hard_deadline,
                    role: SearchThreadRole::Helper(worker_index + 1),
                },
                done_sender: done_sender.clone(),
            }));

            self.workers[worker_index]
                .sender
                .send(command)
                .expect("SMP worker must accept search command");
        }

        drop(done_sender);
        done_receiver
    }

    fn finish_helpers(&self, helper_count: usize, done_receiver: Receiver<()>) {
        for _ in 0..helper_count {
            done_receiver
                .recv()
                .expect("SMP worker must report completion");
        }
        self.active_helpers.store(0, Ordering::Relaxed);
    }

    #[cfg(any(test, debug_assertions))]
    fn worker_count(&self) -> usize {
        self.workers.len()
    }

    #[cfg(any(test, debug_assertions))]
    fn active_helper_count(&self) -> usize {
        self.active_helpers.load(Ordering::Relaxed)
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

pub(crate) struct UciSearchService {
    hash_mb: usize,
    threads: usize,
    syzygy_path: String,
    eval_file: String,
    tt: Arc<TranspositionTable>,
    nnue: Option<Arc<NnueService>>,
    tablebases: Option<Arc<TablebaseService>>,
    workers: WorkerPool,
}

impl UciSearchService {
    pub(crate) fn new() -> Self {
        Self {
            hash_mb: DEFAULT_HASH_MB,
            threads: DEFAULT_THREADS,
            syzygy_path: String::new(),
            eval_file: String::new(),
            tt: Arc::new(TranspositionTable::new_mb(DEFAULT_HASH_MB)),
            nnue: None,
            tablebases: None,
            workers: WorkerPool::new(),
        }
    }

    pub(crate) fn hash_mb(&self) -> usize {
        self.hash_mb
    }

    pub(crate) fn threads(&self) -> usize {
        self.threads
    }

    pub(crate) fn syzygy_path(&self) -> &str {
        &self.syzygy_path
    }

    pub(crate) fn eval_file(&self) -> &str {
        &self.eval_file
    }

    pub(crate) fn set_threads(&mut self, threads: usize) {
        self.threads = threads.clamp(1, MAX_THREADS);
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
        self.syzygy_path = path.to_owned();
        self.tablebases = Some(tablebases);
        Ok(())
    }

    pub(crate) fn set_eval_file(&mut self, path: &str) -> Result<(), String> {
        let path = path.trim();
        if path.is_empty() {
            self.eval_file.clear();
            self.nnue = None;
            return Ok(());
        }

        if path == self.eval_file {
            return Ok(());
        }

        let nnue = NnueService::open_eval_file(path)?;
        self.eval_file = path.to_owned();
        self.nnue = Some(nnue);
        Ok(())
    }

    pub(crate) fn resize_hash(&mut self, hash_mb: usize) {
        let hash_mb = hash_mb.max(1);
        self.hash_mb = hash_mb;
        self.tt = Arc::new(TranspositionTable::new_mb(hash_mb));
    }

    pub(crate) fn clear_hash(&mut self) {
        self.tt.clear();
    }

    pub(crate) fn search(
        &mut self,
        position: &mut Position,
        request: SearchRequest,
    ) -> SearchResult {
        let limits = request.limits.with_hash_mb(self.hash_mb);
        let effective_threads = self.effective_threads(limits.tt_enabled);
        if effective_threads <= 1 {
            return root::search_with_control(
                position,
                limits,
                limits.tt_enabled.then(|| Arc::clone(&self.tt)),
                self.nnue.clone(),
                self.tablebases.clone(),
                SearchControl {
                    stop_flag: request.stop_flag,
                    helper_stop_flag: None,
                    soft_deadline: request.soft_deadline,
                    hard_deadline: request.hard_deadline,
                    role: SearchThreadRole::Main,
                },
            );
        }

        let helper_count = effective_threads - 1;
        let helper_stop_flag = Arc::new(AtomicBool::new(false));
        let done_receiver = self.workers.start_helpers(HelperSearchSpec {
            helper_count,
            position,
            limits,
            tt: Arc::clone(&self.tt),
            nnue: self.nnue.clone(),
            tablebases: self.tablebases.clone(),
            stop_flag: request.stop_flag.clone(),
            helper_stop_flag: Arc::clone(&helper_stop_flag),
            soft_deadline: request.soft_deadline,
            hard_deadline: request.hard_deadline,
        });

        let result = root::search_with_control(
            position,
            limits,
            Some(Arc::clone(&self.tt)),
            self.nnue.clone(),
            self.tablebases.clone(),
            SearchControl {
                stop_flag: request.stop_flag,
                helper_stop_flag: Some(Arc::clone(&helper_stop_flag)),
                soft_deadline: request.soft_deadline,
                hard_deadline: request.hard_deadline,
                role: SearchThreadRole::Main,
            },
        );

        helper_stop_flag.store(true, Ordering::Relaxed);
        self.workers.finish_helpers(helper_count, done_receiver);
        result
    }

    fn effective_threads(&self, tt_enabled: bool) -> usize {
        if !tt_enabled {
            return 1;
        }

        self.threads
            .min(runtime_thread_capacity())
            .clamp(1, MAX_THREADS)
    }

    #[cfg(any(test, debug_assertions))]
    pub(crate) fn debug_tt_entry_count(&self) -> usize {
        self.tt.debug_entry_count()
    }

    #[cfg(any(test, debug_assertions))]
    pub(crate) fn debug_worker_count(&self) -> usize {
        self.workers.worker_count()
    }

    #[cfg(any(test, debug_assertions))]
    pub(crate) fn debug_active_helper_count(&self) -> usize {
        self.workers.active_helper_count()
    }

    #[cfg(any(test, debug_assertions))]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn debug_nnue_is_enabled(&self) -> bool {
        self.nnue.is_some()
    }

    #[cfg(any(test, debug_assertions))]
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
        self.syzygy_path = path.to_owned();
        self.tablebases = Some(tablebases);
    }

    #[cfg(test)]
    pub(crate) fn debug_install_nnue(&mut self, path: &str, nnue: Arc<NnueService>) {
        self.eval_file = path.to_owned();
        self.nnue = Some(nnue);
    }
}

impl Default for UciSearchService {
    fn default() -> Self {
        Self::new()
    }
}

fn runtime_thread_capacity() -> usize {
    thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(MAX_THREADS)
        .clamp(1, MAX_THREADS)
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
                let mut position = job.position;
                let _ = root::search_with_control(
                    &mut position,
                    job.limits,
                    Some(job.tt),
                    job.nnue,
                    job.tablebases,
                    job.control,
                );
                let _ = job.done_sender.send(());
            }
            WorkerCommand::Shutdown => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::SearchLimits;
    use crate::search::nnue::{NnueService, tiny_test_evalfile_path};
    use crate::search::tablebase::{MockTablebaseBackend, TablebaseService, WdlOutcome};
    use std::{sync::Arc, time::Instant};

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

    #[test]
    fn worker_pool_scales_and_helpers_return_to_idle() {
        let mut service = UciSearchService::new();
        service.set_threads(4);
        let mut position = Position::startpos();

        let result = service.search(
            &mut position,
            SearchRequest {
                limits: SearchLimits::new(2),
                soft_deadline: None,
                hard_deadline: None,
                stop_flag: None,
            },
        );

        assert!(result.best_move.is_some());
        assert_eq!(service.debug_active_helper_count(), 0);
        assert!(service.debug_worker_count() >= service.effective_threads(true) - 1);
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
            },
        );

        assert_eq!(service.debug_worker_count(), worker_count);
        assert_eq!(service.debug_active_helper_count(), 0);
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
                },
            );
            let candidate_result = candidate.search(
                &mut candidate_position,
                SearchRequest {
                    limits: SearchLimits::new(5),
                    soft_deadline: None,
                    hard_deadline: None,
                    stop_flag: None,
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
            },
        );

        assert!(result.best_move.is_some());
        assert_eq!(result.nodes, 0);
        assert_eq!(position.to_fen(), before);
        assert_eq!(service.debug_active_helper_count(), 0);
    }
}
