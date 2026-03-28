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
    root::{self, SearchControl, SearchThreadRole},
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
    control: SearchControl,
    done_sender: Sender<()>,
}

struct HelperSearchSpec<'a> {
    helper_count: usize,
    position: &'a Position,
    limits: SearchLimits,
    tt: Arc<TranspositionTable>,
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
    tt: Arc<TranspositionTable>,
    workers: WorkerPool,
}

impl UciSearchService {
    pub(crate) fn new() -> Self {
        Self {
            hash_mb: DEFAULT_HASH_MB,
            threads: DEFAULT_THREADS,
            tt: Arc::new(TranspositionTable::new_mb(DEFAULT_HASH_MB)),
            workers: WorkerPool::new(),
        }
    }

    pub(crate) fn hash_mb(&self) -> usize {
        self.hash_mb
    }

    pub(crate) fn threads(&self) -> usize {
        self.threads
    }

    pub(crate) fn set_threads(&mut self, threads: usize) {
        self.threads = threads.clamp(1, MAX_THREADS);
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
            stop_flag: request.stop_flag.clone(),
            helper_stop_flag: Arc::clone(&helper_stop_flag),
            soft_deadline: request.soft_deadline,
            hard_deadline: request.hard_deadline,
        });

        let result = root::search_with_control(
            position,
            limits,
            Some(Arc::clone(&self.tt)),
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
                let _ =
                    root::search_with_control(&mut position, job.limits, Some(job.tt), job.control);
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
}
