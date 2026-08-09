use std::{
    sync::{
        Arc, Condvar, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Instant,
};

use crate::core::{Color, Move, MoveList, PieceType, Position, Score, movelist::MAX_MOVES, see};

use super::{
    eval,
    limits::{SearchHeuristics, SearchLimits},
    movepicker::MovePicker,
    nnue::{NnueSearchState, NnueService},
    qsearch,
    tablebase::{self, TablebaseService},
    tt::{self, Bound, TtHit, TtStore},
};

#[cfg(feature = "spsa-tuning")]
use super::parameters::SearchParameters;

#[cfg(feature = "spsa-tuning")]
macro_rules! search_parameter {
    ($context:expr, $field:ident, $default:expr) => {
        $context.parameters.$field
    };
}

#[cfg(not(feature = "spsa-tuning"))]
macro_rules! search_parameter {
    ($context:expr, $field:ident, $default:expr) => {
        $default
    };
}

pub(crate) const MAX_PLY: usize = 128;
pub(crate) const INF: i32 = 32_000;
pub(crate) const MATE_SCORE: i32 = 30_000;
const MATE_THRESHOLD: i32 = MATE_SCORE - MAX_PLY as i32;
#[cfg_attr(feature = "spsa-tuning", allow(dead_code))]
const ASPIRATION_DELTA: i32 = 36;
#[cfg_attr(feature = "spsa-tuning", allow(dead_code))]
const NULL_MOVE_STATIC_MARGIN: i32 = 32;
const HISTORY_MAX: i32 = 16_384;
const PIECE_TYPE_COUNT: usize = 6;
const CORRECTION_HISTORY_SIZE: usize = 8_192;
const CORRECTION_HISTORY_MASK: usize = CORRECTION_HISTORY_SIZE - 1;
const CORRECTION_HISTORY_LIMIT: i32 = 16;
const SINGULAR_MIN_DEPTH: usize = 8;
const SINGULAR_TT_DEPTH_SLACK: usize = 3;
#[cfg(any(test, debug_assertions, feature = "internal-testing"))]
const MULTI_CUT_MIN_DEPTH: usize = 7;
#[cfg(any(test, debug_assertions, feature = "internal-testing"))]
const MULTI_CUT_MOVE_LIMIT: usize = 6;
#[cfg(any(test, debug_assertions, feature = "internal-testing"))]
const MULTI_CUT_REQUIRED_CUTOFFS: usize = 3;
#[cfg(any(test, debug_assertions, feature = "internal-testing"))]
const MULTI_CUT_REDUCTION: usize = 3;

type ContinuationHistory = [[[[[i16; 64]; PIECE_TYPE_COUNT]; 64]; PIECE_TYPE_COUNT]; 2];
type CaptureHistory = [[[[i16; PIECE_TYPE_COUNT]; 64]; PIECE_TYPE_COUNT]; 2];
pub(crate) type InfoReporter<'a> = Option<Box<dyn FnMut(&str) + 'a>>;

struct CorrectionHistory {
    pawn: Box<[i16]>,
    non_pawn: Box<[i16]>,
}

impl CorrectionHistory {
    fn new() -> Self {
        Self {
            pawn: vec![0; 2 * CORRECTION_HISTORY_SIZE].into_boxed_slice(),
            non_pawn: vec![0; 2 * 2 * CORRECTION_HISTORY_SIZE].into_boxed_slice(),
        }
    }

    fn correction(&self, position: &Position) -> i32 {
        let side = position.side_to_move().index();
        let (pawn_key, non_pawn_keys) = correction_history_keys(position);
        let pawn = self.pawn[side * CORRECTION_HISTORY_SIZE + pawn_key] as i32;
        let white_non_pawn = self.non_pawn
            [(side * 2 + Color::White.index()) * CORRECTION_HISTORY_SIZE + non_pawn_keys[0]]
            as i32;
        let black_non_pawn = self.non_pawn
            [(side * 2 + Color::Black.index()) * CORRECTION_HISTORY_SIZE + non_pawn_keys[1]]
            as i32;
        (pawn + white_non_pawn + black_non_pawn) / 3
    }

    fn update(&mut self, position: &Position, bonus: i32) {
        let side = position.side_to_move().index();
        let (pawn_key, non_pawn_keys) = correction_history_keys(position);
        update_correction_entry(
            &mut self.pawn[side * CORRECTION_HISTORY_SIZE + pawn_key],
            bonus,
        );
        for color in Color::ALL {
            let index =
                (side * 2 + color.index()) * CORRECTION_HISTORY_SIZE + non_pawn_keys[color.index()];
            update_correction_entry(&mut self.non_pawn[index], bonus);
        }
    }
}

fn update_correction_entry(entry: &mut i16, bonus: i32) {
    let current = i32::from(*entry);
    let bounded_bonus = bonus.clamp(-2, 2);
    let gravity = current * bounded_bonus.abs() / CORRECTION_HISTORY_LIMIT;
    *entry = (current + bounded_bonus - gravity)
        .clamp(-CORRECTION_HISTORY_LIMIT, CORRECTION_HISTORY_LIMIT) as i16;
}

fn correction_history_keys(position: &Position) -> (usize, [usize; 2]) {
    let (pawn_key, non_pawn_keys) = position.correction_history_keys();
    (
        (pawn_key as usize) & CORRECTION_HISTORY_MASK,
        [
            (non_pawn_keys[0] as usize) & CORRECTION_HISTORY_MASK,
            (non_pawn_keys[1] as usize) & CORRECTION_HISTORY_MASK,
        ],
    )
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SearchStats {
    pub nodes: u64,
    pub elapsed_ms: u128,
    pub tt_hits: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SearchResult {
    pub best_move: Option<Move>,
    pub score: Score,
    pub depth: u8,
    pub seldepth: u8,
    pub nodes: u64,
    pub pv: Vec<Move>,
    pub info_lines: Vec<String>,
    pub tt_hits: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum SearchThreadRole {
    #[default]
    Main,
    Helper(usize),
}

/// Coordinates a young-brothers-wait root split while keeping the main thread
/// authoritative for the exact score, PV, time management, and UCI output.
pub(crate) struct RootSplitCoordinator {
    state: Mutex<RootSplitState>,
    wake: Condvar,
}

struct RootSplitState {
    released_depth: usize,
    main_alpha: [i32; MAX_PLY],
    cancelled: bool,
}

impl RootSplitCoordinator {
    pub(crate) const fn new() -> Self {
        Self {
            state: Mutex::new(RootSplitState {
                released_depth: 0,
                main_alpha: [-INF; MAX_PLY],
                cancelled: false,
            }),
            wake: Condvar::new(),
        }
    }

    fn release_siblings(&self, depth: usize, alpha: i32) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.main_alpha[depth] = alpha;
        state.released_depth = state.released_depth.max(depth);
        drop(state);
        self.wake.notify_all();
    }

    fn wait_for_siblings(&self, depth: usize) -> Option<i32> {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        while state.released_depth < depth && !state.cancelled {
            state = self
                .wake
                .wait(state)
                .unwrap_or_else(|error| error.into_inner());
        }
        (!state.cancelled).then_some(state.main_alpha[depth])
    }

    pub(crate) fn cancel(&self) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        state.cancelled = true;
        drop(state);
        self.wake.notify_all();
    }
}

impl SearchThreadRole {
    fn is_main(self) -> bool {
        matches!(self, Self::Main)
    }

    fn helper_index(self) -> Option<usize> {
        match self {
            Self::Main => None,
            Self::Helper(index) => Some(index),
        }
    }
}

#[derive(Clone, Default)]
pub(crate) struct SearchControl {
    pub(crate) stop_flag: Option<Arc<AtomicBool>>,
    pub(crate) helper_stop_flag: Option<Arc<AtomicBool>>,
    pub(crate) soft_deadline: Option<Instant>,
    pub(crate) hard_deadline: Option<Instant>,
    pub(crate) ponder_state: Option<Arc<PonderState>>,
    pub(crate) node_budget: Option<Arc<NodeBudget>>,
    pub(crate) role: SearchThreadRole,
    pub(crate) root_moves: Option<Vec<Move>>,
    pub(crate) root_split: Option<Arc<RootSplitCoordinator>>,
}

/// Runtime state for a UCI ponder search.
///
/// Deadlines supplied with `go ponder` describe the budget *after* `ponderhit`.
/// Until the hit arrives, wall-clock deadlines are suspended, while explicit
/// `stop` and `quit` requests remain immediately effective.
pub(crate) struct PonderState {
    timing: Mutex<PonderTiming>,
    wake: Condvar,
}

struct PonderTiming {
    started: Instant,
    hit_at: Option<Instant>,
    cancelled: bool,
}

impl PonderState {
    pub(crate) fn new(started: Instant) -> Self {
        Self {
            timing: Mutex::new(PonderTiming {
                started,
                hit_at: None,
                cancelled: false,
            }),
            wake: Condvar::new(),
        }
    }

    pub(crate) fn arm(&self, started: Instant) {
        self.timing
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .started = started;
    }

    pub(crate) fn hit(&self, hit_at: Instant) {
        let mut timing = self
            .timing
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if timing.hit_at.is_none() {
            timing.hit_at = Some(hit_at);
        }
        drop(timing);
        self.wake.notify_all();
    }

    pub(crate) fn cancel(&self) {
        let mut timing = self
            .timing
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        timing.cancelled = true;
        drop(timing);
        self.wake.notify_all();
    }

    pub(crate) fn wait_until_released(&self, stop_flag: Option<&AtomicBool>) {
        let mut timing = self
            .timing
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        while timing.hit_at.is_none()
            && !timing.cancelled
            && !stop_flag.is_some_and(|flag| flag.load(Ordering::Relaxed))
        {
            let (next_timing, _) = self
                .wake
                .wait_timeout(timing, std::time::Duration::from_millis(10))
                .unwrap_or_else(|error| error.into_inner());
            timing = next_timing;
        }
    }

    fn hit_at(&self) -> Option<Instant> {
        self.timing
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .hit_at
    }

    fn cancelled(&self) -> bool {
        self.timing
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .cancelled
    }

    fn adjust_deadline(&self, deadline: Instant) -> Option<Instant> {
        let timing = self
            .timing
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        timing
            .hit_at?
            .checked_add(deadline.saturating_duration_since(timing.started))
    }
}

#[derive(Clone, Default)]
pub(crate) struct SearchResources {
    pub(crate) tt: Option<Arc<tt::TranspositionTable>>,
    pub(crate) nnue: Option<Arc<NnueService>>,
    pub(crate) tablebases: Option<Arc<TablebaseService>>,
    pub(crate) classical_weights: Option<eval::ClassicalEvalWeights>,
}

impl SearchControl {
    fn can_interrupt(&self) -> bool {
        self.stop_flag.is_some()
            || self.helper_stop_flag.is_some()
            || self.soft_deadline.is_some()
            || self.hard_deadline.is_some()
            || self.node_budget.is_some()
    }
}

/// A precise aggregate node budget shared by the main search and every helper.
///
/// Node-limited searches are uncommon enough that one relaxed atomic operation per
/// node is preferable to chunking: chunk reservations can overshoot small limits or
/// strand a large part of the budget in helpers that finish early.
pub(crate) struct NodeBudget {
    limit: u64,
    consumed: AtomicU64,
}

impl NodeBudget {
    pub(crate) const fn new(limit: u64) -> Self {
        Self {
            limit,
            consumed: AtomicU64::new(0),
        }
    }

    fn try_consume(&self) -> bool {
        self.consumed
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |consumed| {
                (consumed < self.limit).then_some(consumed + 1)
            })
            .is_ok()
    }

    fn exhausted(&self) -> bool {
        self.consumed.load(Ordering::Relaxed) >= self.limit
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct IterationStability {
    previous_move: Option<Move>,
    previous_score: i32,
    stable_iterations: u8,
    score_delta: i32,
    has_previous: bool,
    comparisons: u8,
}

impl IterationStability {
    fn record(&mut self, best_move: Option<Move>, score: i32) {
        if self.has_previous {
            self.comparisons = self.comparisons.saturating_add(1);
            self.score_delta = score.saturating_sub(self.previous_score).abs();
            if best_move == self.previous_move && self.score_delta <= 20 {
                self.stable_iterations = self.stable_iterations.saturating_add(1);
            } else {
                self.stable_iterations = 0;
            }
        }
        self.previous_move = best_move;
        self.previous_score = score;
        self.has_previous = true;
    }

    #[cfg_attr(feature = "spsa-tuning", allow(dead_code))]
    fn soft_budget_factor(self) -> f64 {
        if self.comparisons == 0 {
            return 1.0;
        }
        if self.stable_iterations >= 3 {
            return 0.70;
        }
        if self.stable_iterations == 2 {
            return 0.82;
        }
        if self.stable_iterations == 1 {
            return 0.95;
        }
        if self.score_delta >= 80 {
            return 1.45;
        }
        1.25
    }

    #[cfg(feature = "spsa-tuning")]
    fn soft_budget_factor_with_parameters(self, parameters: SearchParameters) -> f64 {
        if self.comparisons == 0 {
            return 1.0;
        }
        if self.stable_iterations >= 3 {
            return f64::from(parameters.time_stable3_pct) / 100.0;
        }
        if self.stable_iterations == 2 {
            return f64::from(parameters.time_stable2_pct) / 100.0;
        }
        if self.stable_iterations == 1 {
            return f64::from(parameters.time_stable1_pct) / 100.0;
        }
        if self.score_delta >= parameters.time_score_swing_cp {
            return f64::from(parameters.time_score_swing_pct) / 100.0;
        }
        f64::from(parameters.time_unstable_pct) / 100.0
    }
}

pub(crate) struct SearchContext<'a> {
    started: Instant,
    pub(crate) nodes: u64,
    pub(crate) seldepth: usize,
    tt_hits: u64,
    pv_table: [[Move; MAX_PLY]; MAX_PLY],
    pub(crate) pv_length: [usize; MAX_PLY],
    previous_iteration_pv: [Move; MAX_PLY],
    previous_iteration_pv_length: usize,
    previous_moves: [Move; MAX_PLY],
    killer_moves: [[Move; 2]; MAX_PLY],
    quiet_history: [[[i16; 64]; 64]; 2],
    continuation_history: Box<ContinuationHistory>,
    capture_history: Box<CaptureHistory>,
    correction_history: Option<CorrectionHistory>,
    static_evals: [i32; MAX_PLY],
    static_eval_valid: [bool; MAX_PLY],
    excluded_moves: Option<Box<[Move; MAX_PLY]>>,
    classical_weights: Option<eval::ClassicalEvalWeights>,
    heuristics: SearchHeuristics,
    #[cfg(feature = "spsa-tuning")]
    parameters: SearchParameters,
    control: SearchControl,
    tt: Option<Arc<tt::TranspositionTable>>,
    nnue: Option<NnueSearchState>,
    tablebases: Option<Arc<TablebaseService>>,
    info_reporter: InfoReporter<'a>,
    debug_counters: SearchDebugCounters,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct SearchDebugCounters {
    lmr_reductions: u32,
    lmr_researches: u32,
    pvs_scout_searches: u32,
    pvs_full_researches: u32,
    reverse_futility_prunes: u32,
    futility_prunes: u32,
    late_move_prunes: u32,
    see_prunes: u32,
    history_prunes: u32,
    internal_iterative_reductions: u32,
    probcut_attempts: u32,
    probcut_prunes: u32,
    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    multi_cut_attempts: u32,
    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    multi_cut_probes: u32,
    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    multi_cut_prunes: u32,
    null_move_verifications: u32,
    correction_history_lookups: u32,
    correction_history_updates: u32,
    singular_verifications: u32,
    singular_extensions: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SearchNodeState {
    pub(crate) is_pv: bool,
    pub(crate) null_move_allowed: bool,
    pub(crate) cut_node: bool,
}

impl SearchNodeState {
    const fn new(is_pv: bool) -> Self {
        Self {
            is_pv,
            null_move_allowed: true,
            cut_node: false,
        }
    }

    const fn cut() -> Self {
        Self {
            is_pv: false,
            null_move_allowed: true,
            cut_node: true,
        }
    }

    const fn after_null_move() -> Self {
        Self {
            is_pv: false,
            null_move_allowed: false,
            cut_node: true,
        }
    }
}

impl Default for SearchNodeState {
    fn default() -> Self {
        Self::new(false)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LmrCandidate {
    depth: usize,
    is_pv: bool,
    in_check: bool,
    mv: Move,
    gives_check: bool,
    is_hash_move: bool,
    quiets_searched: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ForwardPruneCandidate {
    node_state: SearchNodeState,
    depth: usize,
    alpha: i32,
    static_eval: i32,
    in_check: bool,
    mv: Move,
    gives_check: bool,
    is_hash_move: bool,
    has_searched_move: bool,
    quiets_searched: usize,
}

#[cfg(test)]
impl ForwardPruneCandidate {
    fn quiet(depth: usize, alpha: i32, static_eval: i32, mv: Move) -> Self {
        Self {
            node_state: SearchNodeState::new(false),
            depth,
            alpha,
            static_eval,
            in_check: false,
            mv,
            gives_check: false,
            is_hash_move: false,
            has_searched_move: true,
            quiets_searched: 0,
        }
    }
}

#[derive(Clone, Copy)]
struct TtStoreInput {
    key: u64,
    depth: u8,
    ply: usize,
    best_move: Move,
    static_eval: i16,
    score: i32,
    bound: Bound,
}

#[derive(Clone, Copy)]
struct CorrectionUpdate {
    depth: usize,
    in_check: bool,
    best_move: Move,
    static_eval: i32,
    score: i32,
    bound: Bound,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SingularProbe {
    excluded_move: Move,
    depth: usize,
    beta: i32,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct MoveOrderHints {
    pub(crate) ply: usize,
    pub(crate) quiescence_only: bool,
    pub(crate) pv_move: Option<Move>,
    pub(crate) tt_move: Option<Move>,
}

pub fn search(position: &mut Position, limits: SearchLimits) -> SearchResult {
    search_with_control(
        position,
        limits,
        SearchResources::default(),
        SearchControl::default(),
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn search_with_control<'a>(
    position: &mut Position,
    limits: SearchLimits,
    resources: SearchResources,
    control: SearchControl,
    info_reporter: InfoReporter<'a>,
) -> SearchResult {
    SearchContext::with_tt(limits, resources, control, info_reporter).run(position, limits)
}

enum RootSearchOutcome {
    Complete(Option<Move>, i32),
    Aborted(Option<Move>),
}

impl<'a> SearchContext<'a> {
    #[cfg(test)]
    pub(crate) fn new(limits: SearchLimits) -> Self {
        Self::with_tt(
            limits,
            SearchResources::default(),
            SearchControl::default(),
            None,
        )
    }

    fn with_tt(
        limits: SearchLimits,
        resources: SearchResources,
        control: SearchControl,
        info_reporter: InfoReporter<'a>,
    ) -> Self {
        let mut control = control;
        if control.node_budget.is_none()
            && let Some(node_limit) = limits.node_limit
        {
            control.node_budget = Some(Arc::new(NodeBudget::new(node_limit)));
        }
        Self {
            started: Instant::now(),
            nodes: 0,
            seldepth: 0,
            tt_hits: 0,
            pv_table: [[Move::NONE; MAX_PLY]; MAX_PLY],
            pv_length: [0; MAX_PLY],
            previous_iteration_pv: [Move::NONE; MAX_PLY],
            previous_iteration_pv_length: 0,
            previous_moves: [Move::NONE; MAX_PLY],
            killer_moves: [[Move::NONE; 2]; MAX_PLY],
            quiet_history: [[[0; 64]; 64]; 2],
            continuation_history: Box::new(
                [[[[[0; 64]; PIECE_TYPE_COUNT]; 64]; PIECE_TYPE_COUNT]; 2],
            ),
            capture_history: Box::new([[[[0; PIECE_TYPE_COUNT]; 64]; PIECE_TYPE_COUNT]; 2]),
            correction_history: limits
                .heuristics
                .correction_history
                .then(CorrectionHistory::new),
            static_evals: [0; MAX_PLY],
            static_eval_valid: [false; MAX_PLY],
            excluded_moves: limits
                .heuristics
                .singular_extensions
                .then(|| Box::new([Move::NONE; MAX_PLY])),
            classical_weights: resources.classical_weights,
            heuristics: limits.heuristics,
            #[cfg(feature = "spsa-tuning")]
            parameters: limits.parameters,
            control,
            tt: resources.tt.or_else(|| {
                limits
                    .tt_enabled
                    .then(|| Arc::new(tt::TranspositionTable::new_mb(limits.hash_mb)))
            }),
            nnue: resources.nnue.map(NnueSearchState::new),
            tablebases: resources.tablebases,
            info_reporter,
            debug_counters: SearchDebugCounters::default(),
        }
    }

    fn run(&mut self, position: &mut Position, limits: SearchLimits) -> SearchResult {
        match (self.tablebases.is_some(), self.nnue.is_some()) {
            (false, false) => self.run_core::<false, false>(position, limits),
            (false, true) => self.run_core::<false, true>(position, limits),
            (true, false) => self.run_core::<true, false>(position, limits),
            (true, true) => self.run_core::<true, true>(position, limits),
        }
    }

    fn run_core<const USE_TABLEBASES: bool, const USE_NNUE: bool>(
        &mut self,
        position: &mut Position,
        limits: SearchLimits,
    ) -> SearchResult {
        // A claimable draw at the root is already available before making a
        // move. Do not let a tablebase result or a mating continuation after
        // the claim boundary override it. Checkmate is deliberately excluded
        // by `is_draw`, so terminal mates still receive their mate score.
        if is_draw(position) {
            return SearchResult {
                best_move: position.select_placeholder_bestmove(),
                score: Score(0),
                depth: 0,
                seldepth: 0,
                nodes: 0,
                pv: Vec::new(),
                info_lines: Vec::new(),
                tt_hits: 0,
            };
        }

        let depth_limit = limits.depth.max(1).min((MAX_PLY - 1) as u8);
        let mut best_move = None;
        let mut best_score = 0i32;
        let mut best_pv = Vec::new();
        let mut completed_depth = 0u8;
        let mut info_lines = Vec::new();
        let mut stability = IterationStability::default();
        let mut last_iteration_elapsed = None;
        let fallback_best_move = self
            .control
            .can_interrupt()
            .then(|| position.select_placeholder_bestmove())
            .flatten();

        if USE_TABLEBASES && let Some(root_probe) = self.try_root_tablebase_result(position) {
            return root_probe;
        }

        if USE_NNUE {
            self.prepare_nnue(position);
        }

        for depth in 1..=depth_limit {
            if self.hard_stop_requested()
                || (completed_depth > 0
                    && self.adaptive_soft_stop_requested(stability, last_iteration_elapsed))
            {
                break;
            }
            let iteration_started = Instant::now();
            if self.control.role.is_main()
                && let Some(tt) = self.tt.as_ref()
            {
                tt.new_generation();
            }
            let depth_result = if self.heuristics.aspiration_windows
                && depth > 1
                && best_score.abs() < MATE_THRESHOLD
            {
                self.search_root_with_aspiration_core::<USE_TABLEBASES, USE_NNUE>(
                    position,
                    depth as usize,
                    best_score,
                )
            } else {
                self.search_root_core::<USE_TABLEBASES, USE_NNUE>(
                    position,
                    depth as usize,
                    -INF,
                    INF,
                )
            };

            let (depth_best_move, depth_score) = match depth_result {
                RootSearchOutcome::Complete(best_move, score) => (best_move, score),
                RootSearchOutcome::Aborted(partial_best_move) => {
                    if completed_depth == 0 {
                        best_move = partial_best_move.or(fallback_best_move);
                    }
                    break;
                }
            };

            best_move = depth_best_move;
            best_score = depth_score;
            completed_depth = depth;
            stability.record(best_move, best_score);
            last_iteration_elapsed = Some(iteration_started.elapsed());

            let pv = self.collect_pv(0);
            best_pv = pv.clone();
            self.capture_completed_pv(&pv);
            if self.control.role.is_main() {
                let info_line = format_info_line(
                    depth,
                    self.seldepth.min(u8::MAX as usize) as u8,
                    depth_score,
                    self.nodes,
                    self.started.elapsed().as_millis(),
                    self.tt_hits,
                    &pv,
                );
                self.report_info_line(&info_line);
                info_lines.push(info_line);
            }
        }

        if completed_depth == 0 && best_move.is_none() {
            best_move = fallback_best_move;
        }

        SearchResult {
            best_move,
            score: Score(if completed_depth == 0 && best_move.is_none() {
                terminal_score(position, 0)
            } else {
                best_score
            }),
            depth: completed_depth,
            seldepth: self.seldepth.min(u8::MAX as usize) as u8,
            nodes: self.nodes,
            pv: best_pv,
            info_lines,
            tt_hits: self.tt_hits,
        }
    }

    fn try_root_tablebase_result(&mut self, position: &Position) -> Option<SearchResult> {
        if is_draw(position) {
            return None;
        }

        if !self.control.role.is_main() {
            return None;
        }

        let tablebases = self.tablebases.as_ref()?;
        if !tablebases.supports_root(position) {
            return None;
        }

        let mut legal_moves = MoveList::new();
        let mut probe_position = position.clone();
        probe_position.generate_legal_moves(&mut legal_moves);
        self.apply_root_move_filter(&mut legal_moves);
        if legal_moves.is_empty() {
            return None;
        }

        let root_probe = tablebases
            .probe_root(&probe_position, &legal_moves)
            .ok()??;
        let score = tablebase::score_from_wdl(root_probe.wdl, 0);
        let pv = vec![root_probe.best_move];
        let info_lines = if self.control.role.is_main() {
            let info_line = format_info_line(
                0,
                0,
                score,
                self.nodes,
                self.started.elapsed().as_millis(),
                self.tt_hits,
                &pv,
            );
            self.report_info_line(&info_line);
            vec![info_line]
        } else {
            Vec::new()
        };

        Some(SearchResult {
            best_move: Some(root_probe.best_move),
            score: Score(score),
            depth: 0,
            seldepth: 0,
            nodes: self.nodes,
            pv,
            info_lines,
            tt_hits: self.tt_hits,
        })
    }

    fn report_info_line(&mut self, line: &str) {
        if let Some(reporter) = self.info_reporter.as_mut() {
            reporter(line);
        }
    }

    fn search_root_with_aspiration_core<const USE_TABLEBASES: bool, const USE_NNUE: bool>(
        &mut self,
        position: &mut Position,
        depth: usize,
        guess: i32,
    ) -> RootSearchOutcome {
        let mut delta = search_parameter!(self, aspiration_delta, ASPIRATION_DELTA);
        let mut alpha = (guess - delta).max(-INF);
        let mut beta = (guess + delta).min(INF);
        loop {
            if self.hard_stop_requested() {
                return RootSearchOutcome::Aborted(None);
            }
            let RootSearchOutcome::Complete(best_move, score) =
                self.search_root_core::<USE_TABLEBASES, USE_NNUE>(position, depth, alpha, beta)
            else {
                return RootSearchOutcome::Aborted(None);
            };
            if score <= alpha && alpha > -INF {
                delta = (delta * 2).min(INF / 2);
                alpha = (alpha - delta).max(-INF);
                continue;
            }
            if score >= beta && beta < INF {
                delta = (delta * 2).min(INF / 2);
                beta = (beta + delta).min(INF);
                continue;
            }
            return RootSearchOutcome::Complete(best_move, score);
        }
    }

    fn search_root_core<const USE_TABLEBASES: bool, const USE_NNUE: bool>(
        &mut self,
        position: &mut Position,
        depth: usize,
        mut alpha: i32,
        beta: i32,
    ) -> RootSearchOutcome {
        self.pv_length[0] = 0;
        let pv_move_hint = self.previous_pv_move(0);
        let tt_move_hint = self
            .probe_tt(position.search_key())
            .and_then(|hit| (!hit.best_move.is_none()).then_some(hit.best_move));

        let mut legal_moves = MoveList::new();
        position.generate_legal_moves(&mut legal_moves);
        self.apply_root_move_filter(&mut legal_moves);
        if legal_moves.is_empty() {
            return RootSearchOutcome::Complete(None, terminal_score(position, 0));
        }
        let pv_move_hint = validated_move_hint(&legal_moves, pv_move_hint);
        let tt_move_hint = validated_tt_move_hint(&legal_moves, tt_move_hint);
        let ordering_hints = MoveOrderHints {
            ply: 0,
            quiescence_only: false,
            pv_move: pv_move_hint,
            tt_move: tt_move_hint,
        };

        if let Some(helper_index) = self.control.role.helper_index() {
            if let Some(root_split) = self.control.root_split.as_ref() {
                let Some(main_alpha) = root_split.wait_for_siblings(depth) else {
                    return RootSearchOutcome::Aborted(None);
                };
                alpha = alpha.max(main_alpha).min(beta.saturating_sub(1));
            }
            let ordered_moves =
                self.helper_root_order(position, &legal_moves, ordering_hints, helper_index);
            return self.search_root_for_helper_core::<USE_TABLEBASES, USE_NNUE>(
                position,
                depth,
                alpha,
                beta,
                &ordered_moves,
            );
        }

        let mut best_move = None;

        let mut move_picker = MovePicker::new(self, position, &legal_moves, ordering_hints);
        while let Some(mv) = move_picker.next() {
            if self.hard_stop_requested() {
                return RootSearchOutcome::Aborted(best_move);
            }
            let undo = self
                .make_search_move::<USE_NNUE>(position, mv)
                .expect("root move must be legal during search");
            self.set_previous_move(1, mv);
            let score_result = if best_move.is_none() {
                self.alpha_beta_core::<USE_TABLEBASES, USE_NNUE>(
                    position,
                    depth.saturating_sub(1),
                    1,
                    -beta,
                    -alpha,
                    SearchNodeState::new(true),
                )
                .map(|score| -score)
            } else {
                self.debug_counters.pvs_scout_searches += 1;
                let scout_beta = -alpha;
                let scout_alpha = scout_beta.saturating_sub(1);
                let Some(score) = self.alpha_beta_core::<USE_TABLEBASES, USE_NNUE>(
                    position,
                    depth.saturating_sub(1),
                    1,
                    scout_alpha,
                    scout_beta,
                    SearchNodeState::cut(),
                ) else {
                    self.set_previous_move(1, Move::NONE);
                    self.unmake_search_move::<USE_NNUE>(position, mv, undo);
                    return RootSearchOutcome::Aborted(best_move);
                };
                let score = -score;
                if score > alpha {
                    self.debug_counters.pvs_full_researches += 1;
                    self.alpha_beta_core::<USE_TABLEBASES, USE_NNUE>(
                        position,
                        depth.saturating_sub(1),
                        1,
                        -beta,
                        -alpha,
                        SearchNodeState::new(true),
                    )
                    .map(|score| -score)
                } else {
                    Some(score)
                }
            };
            let Some(score) = score_result else {
                self.set_previous_move(1, Move::NONE);
                self.unmake_search_move::<USE_NNUE>(position, mv, undo);
                return RootSearchOutcome::Aborted(best_move);
            };
            self.set_previous_move(1, Move::NONE);
            self.unmake_search_move::<USE_NNUE>(position, mv, undo);

            if score > alpha || best_move.is_none() {
                alpha = score;
                best_move = Some(mv);
                self.update_pv(0, mv);
            }

            if self.control.role.is_main()
                && let Some(root_split) = self.control.root_split.as_ref()
            {
                root_split.release_siblings(depth, alpha);
            }
        }

        RootSearchOutcome::Complete(best_move, alpha)
    }

    fn search_root_for_helper_core<const USE_TABLEBASES: bool, const USE_NNUE: bool>(
        &mut self,
        position: &mut Position,
        depth: usize,
        mut alpha: i32,
        beta: i32,
        ordered_moves: &[Move],
    ) -> RootSearchOutcome {
        let mut best_move = None;

        for &mv in ordered_moves {
            if self.hard_stop_requested() {
                return RootSearchOutcome::Aborted(best_move);
            }
            let undo = self
                .make_search_move::<USE_NNUE>(position, mv)
                .expect("helper root move must be legal during search");
            self.set_previous_move(1, mv);
            let score_result = if best_move.is_none() {
                self.alpha_beta_core::<USE_TABLEBASES, USE_NNUE>(
                    position,
                    depth.saturating_sub(1),
                    1,
                    -beta,
                    -alpha,
                    SearchNodeState::new(true),
                )
                .map(|score| -score)
            } else {
                self.debug_counters.pvs_scout_searches += 1;
                let scout_beta = -alpha;
                let scout_alpha = scout_beta.saturating_sub(1);
                let Some(score) = self.alpha_beta_core::<USE_TABLEBASES, USE_NNUE>(
                    position,
                    depth.saturating_sub(1),
                    1,
                    scout_alpha,
                    scout_beta,
                    SearchNodeState::cut(),
                ) else {
                    self.set_previous_move(1, Move::NONE);
                    self.unmake_search_move::<USE_NNUE>(position, mv, undo);
                    return RootSearchOutcome::Aborted(best_move);
                };
                let score = -score;
                if score > alpha {
                    self.debug_counters.pvs_full_researches += 1;
                    self.alpha_beta_core::<USE_TABLEBASES, USE_NNUE>(
                        position,
                        depth.saturating_sub(1),
                        1,
                        -beta,
                        -alpha,
                        SearchNodeState::new(true),
                    )
                    .map(|score| -score)
                } else {
                    Some(score)
                }
            };
            let Some(score) = score_result else {
                self.set_previous_move(1, Move::NONE);
                self.unmake_search_move::<USE_NNUE>(position, mv, undo);
                return RootSearchOutcome::Aborted(best_move);
            };
            self.set_previous_move(1, Move::NONE);
            self.unmake_search_move::<USE_NNUE>(position, mv, undo);

            if best_move.is_none() {
                best_move = Some(mv);
                self.update_pv(0, mv);
            }
            if score > alpha {
                alpha = score;
                best_move = Some(mv);
                self.update_pv(0, mv);
            }
        }

        RootSearchOutcome::Complete(best_move, alpha)
    }

    fn apply_root_move_filter(&self, legal_moves: &mut MoveList) {
        let Some(root_moves) = self.control.root_moves.as_deref() else {
            return;
        };

        let mut filtered = MoveList::new();
        for mv in legal_moves.as_slice().iter().copied() {
            if root_moves.contains(&mv) {
                filtered.push(mv);
            }
        }
        *legal_moves = filtered;
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn alpha_beta(
        &mut self,
        position: &mut Position,
        depth: usize,
        ply: usize,
        alpha: i32,
        beta: i32,
        node_state: SearchNodeState,
    ) -> Option<i32> {
        match (self.tablebases.is_some(), self.nnue.is_some()) {
            (false, false) => {
                self.alpha_beta_core::<false, false>(position, depth, ply, alpha, beta, node_state)
            }
            (false, true) => {
                self.alpha_beta_core::<false, true>(position, depth, ply, alpha, beta, node_state)
            }
            (true, false) => {
                self.alpha_beta_core::<true, false>(position, depth, ply, alpha, beta, node_state)
            }
            (true, true) => {
                self.alpha_beta_core::<true, true>(position, depth, ply, alpha, beta, node_state)
            }
        }
    }

    fn alpha_beta_core<const USE_TABLEBASES: bool, const USE_NNUE: bool>(
        &mut self,
        position: &mut Position,
        mut depth: usize,
        ply: usize,
        mut alpha: i32,
        mut beta: i32,
        node_state: SearchNodeState,
    ) -> Option<i32> {
        if !self.count_node() {
            return None;
        }
        self.seldepth = self.seldepth.max(ply);
        if self.nodes & 1023 == 0 && self.hard_stop_requested() {
            return None;
        }

        if ply >= MAX_PLY - 1 {
            self.clear_pv(ply);
            return Some(self.evaluate_position::<USE_NNUE>(position));
        }
        self.clear_pv(ply);
        let excluded_move = self.excluded_move(ply);
        let is_exclusion = excluded_move.is_some();

        if is_draw(position) {
            return Some(0);
        }

        if !is_exclusion
            && USE_TABLEBASES
            && let Some(tablebase_score) = self.try_non_root_tablebase_score(position, ply)
        {
            return Some(tablebase_score);
        }

        let (mate_alpha, mate_beta) = mate_distance_bounds(ply);
        alpha = alpha.max(mate_alpha);
        beta = beta.min(mate_beta);
        if alpha >= beta {
            return Some(alpha);
        }

        if depth == 0 {
            return qsearch::qsearch_from_main::<USE_NNUE>(self, position, ply, alpha, beta);
        }

        let tt_key = position.search_key();
        let alpha_start = alpha;

        let tt_hit = self.probe_tt(tt_key);
        if !is_exclusion
            && let Some(hit) = tt_hit
            && let Some(cutoff) = tt_cutoff_score(hit, depth, ply, alpha, beta)
        {
            return Some(cutoff);
        }

        let in_check = position.is_in_check(position.side_to_move());
        let raw_static_eval = if in_check {
            // Static evaluation cannot be used for stand-pat or forward pruning
            // while in check. Preserve a neutral TT payload and avoid an expensive
            // NNUE propagation that would otherwise be discarded.
            0
        } else if self.heuristics.tt_static_eval {
            tt_hit
                .map(|hit| hit.eval as i32)
                .unwrap_or_else(|| self.evaluate_position::<USE_NNUE>(position))
        } else {
            self.evaluate_position::<USE_NNUE>(position)
        };
        let static_eval = self.correct_static_eval(position, raw_static_eval, in_check);
        self.static_evals[ply] = static_eval;
        self.static_eval_valid[ply] = !in_check;
        let improving =
            ply >= 2 && self.static_eval_valid[ply - 2] && static_eval > self.static_evals[ply - 2];

        let pv_move_hint = node_state
            .is_pv
            .then(|| self.previous_pv_move(ply))
            .flatten();
        let tt_move_hint = (!is_exclusion)
            .then(|| tt_hit.and_then(|hit| (!hit.best_move.is_none()).then_some(hit.best_move)))
            .flatten();

        if self.heuristics.internal_iterative_reduction
            && !node_state.is_pv
            && !is_exclusion
            && !in_check
            && depth >= 7
            && tt_move_hint.is_none()
        {
            depth -= 1;
            self.debug_counters.internal_iterative_reductions += 1;
        }

        let mut legal_moves = MoveList::new();
        position.generate_legal_moves(&mut legal_moves);
        if legal_moves.is_empty() {
            return Some(terminal_score(position, ply));
        }
        let pv_move_hint = validated_move_hint(&legal_moves, pv_move_hint);
        let tt_move_hint = validated_tt_move_hint(&legal_moves, tt_move_hint);

        let singular_move = if let Some(probe) = singular_probe(
            self.heuristics,
            excluded_move,
            depth,
            ply,
            tt_hit,
            tt_move_hint,
        ) {
            self.debug_counters.singular_verifications =
                self.debug_counters.singular_verifications.saturating_add(1);
            self.set_excluded_move(ply, probe.excluded_move);
            let verification_score = self.alpha_beta_core::<USE_TABLEBASES, USE_NNUE>(
                position,
                probe.depth,
                ply,
                probe.beta.saturating_sub(1),
                probe.beta,
                SearchNodeState::after_null_move(),
            );
            self.set_excluded_move(ply, Move::NONE);
            let verification_score = verification_score?;
            self.clear_pv(ply);
            if verification_score < probe.beta {
                self.debug_counters.singular_extensions =
                    self.debug_counters.singular_extensions.saturating_add(1);
                Some(probe.excluded_move)
            } else {
                None
            }
        } else {
            None
        };

        let ordering_hints = MoveOrderHints {
            ply,
            quiescence_only: false,
            pv_move: pv_move_hint,
            tt_move: tt_move_hint,
        };
        let mut best_move = Move::NONE;
        let mut searched_moves = 0usize;
        let mut quiets_searched = 0usize;
        let mut searched_quiets = [Move::NONE; MAX_MOVES];
        let mut searched_quiet_count = 0usize;
        let mut searched_captures = [Move::NONE; MAX_MOVES];
        let mut searched_capture_count = 0usize;

        if !is_exclusion
            && null_move_is_eligible_with_parameters(
                self.heuristics,
                position,
                node_state,
                depth,
                beta,
                static_eval,
                in_check,
                search_parameter!(self, null_static_margin, NULL_MOVE_STATIC_MARGIN),
            )
            && let Ok(null_undo) = position.make_null_move()
        {
            self.set_previous_move(ply + 1, Move::NONE);
            let reduction = null_move_reduction_with_parameters(
                depth,
                static_eval,
                beta,
                search_parameter!(self, null_base_reduction, 2_i32) as usize,
                search_parameter!(self, null_depth_divisor, 6_i32) as usize,
                search_parameter!(self, null_eval_divisor, 256),
            );
            let null_beta = (-beta).saturating_add(1).min(INF);
            let null_score = self.alpha_beta_core::<USE_TABLEBASES, USE_NNUE>(
                position,
                depth - 1 - reduction,
                ply + 1,
                -beta,
                null_beta,
                SearchNodeState::after_null_move(),
            );
            position.unmake_null_move(null_undo);
            self.set_previous_move(ply + 1, Move::NONE);

            if let Some(score) = null_score {
                let null_score = -score;
                if null_score >= beta {
                    let verified_score = if null_move_requires_verification_with_parameters(
                        depth,
                        search_parameter!(self, null_verify_depth, 10_i32) as usize,
                    ) {
                        self.debug_counters.null_move_verifications += 1;
                        self.alpha_beta_core::<USE_TABLEBASES, USE_NNUE>(
                            position,
                            depth - reduction,
                            ply,
                            beta.saturating_sub(1).max(-INF),
                            beta,
                            SearchNodeState::after_null_move(),
                        )
                    } else {
                        Some(null_score)
                    };

                    let verified_score = verified_score?;
                    if verified_score >= beta {
                        self.store_tt(TtStoreInput {
                            key: tt_key,
                            depth: depth.min(u8::MAX as usize) as u8,
                            ply,
                            best_move: Move::NONE,
                            static_eval: raw_static_eval.clamp(i16::MIN as i32, i16::MAX as i32)
                                as i16,
                            score: verified_score,
                            bound: Bound::Lower,
                        });
                        return Some(verified_score);
                    }
                }
            } else {
                return None;
            }
        }

        if !is_exclusion
            && reverse_futility_is_eligible_with_parameters(
                self.heuristics,
                position,
                node_state,
                depth,
                beta,
                static_eval,
                in_check,
                search_parameter!(self, reverse_futility_slope, 140),
            )
        {
            self.debug_counters.reverse_futility_prunes += 1;
            self.store_tt(TtStoreInput {
                key: tt_key,
                depth: depth.min(u8::MAX as usize) as u8,
                ply,
                best_move: Move::NONE,
                static_eval: raw_static_eval.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                score: beta,
                bound: Bound::Lower,
            });
            return Some(beta);
        }

        // Multi-Cut is intentionally isolated behind a default-off heuristic seam. At a cut
        // node, several independent reduced-depth fail-highs are evidence that at least one
        // full-depth move will also fail high. This is probabilistic forward pruning, so keep
        // the guards strict, suppress recursive Multi-Cut/null probing in child trials, and
        // publish only a reduced-depth TT bound.
        #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
        if !is_exclusion
            && multi_cut_is_eligible(
                self.heuristics,
                position,
                node_state,
                depth,
                beta,
                static_eval,
                in_check,
            )
        {
            self.debug_counters.multi_cut_attempts =
                self.debug_counters.multi_cut_attempts.saturating_add(1);
            let mut cutoffs = 0usize;
            let mut probed = 0usize;
            let mut candidates =
                MovePicker::new(self, position, &legal_moves, ordering_hints).ordered();
            candidates.truncate(MULTI_CUT_MOVE_LIMIT);
            candidates
                .retain(|mv| !mv.is_capture() || mv.is_promotion() || position.see(*mv).0 >= 0);
            let candidate_count = candidates.len();

            for mv in candidates {
                probed += 1;
                self.debug_counters.multi_cut_probes =
                    self.debug_counters.multi_cut_probes.saturating_add(1);
                let undo = self
                    .make_search_move::<USE_NNUE>(position, mv)
                    .expect("multi-cut candidate must be legal");
                self.set_previous_move(ply + 1, mv);
                let child_score = self.alpha_beta_core::<USE_TABLEBASES, USE_NNUE>(
                    position,
                    depth - 1 - MULTI_CUT_REDUCTION,
                    ply + 1,
                    -beta,
                    (-beta).saturating_add(1),
                    SearchNodeState::after_null_move(),
                );
                self.set_previous_move(ply + 1, Move::NONE);
                self.unmake_search_move::<USE_NNUE>(position, mv, undo);
                let score = -child_score?;
                if score >= beta {
                    cutoffs += 1;
                    if cutoffs >= MULTI_CUT_REQUIRED_CUTOFFS {
                        self.debug_counters.multi_cut_prunes =
                            self.debug_counters.multi_cut_prunes.saturating_add(1);
                        self.store_tt(TtStoreInput {
                            key: tt_key,
                            depth: depth
                                .saturating_sub(MULTI_CUT_REDUCTION + 1)
                                .min(u8::MAX as usize) as u8,
                            ply,
                            best_move: mv,
                            static_eval: raw_static_eval.clamp(i16::MIN as i32, i16::MAX as i32)
                                as i16,
                            score: beta,
                            bound: Bound::Lower,
                        });
                        return Some(beta);
                    }
                }

                // Once too few candidates remain to reach the required consensus, stop paying
                // for speculative probes and continue with the normal full-depth search.
                if cutoffs + (candidate_count - probed) < MULTI_CUT_REQUIRED_CUTOFFS {
                    break;
                }
            }
        }

        if !is_exclusion
            && probcut_is_eligible(
                self.heuristics,
                node_state,
                depth,
                beta,
                static_eval,
                in_check,
                search_parameter!(self, probcut_base, 180),
                search_parameter!(self, probcut_slope, 5),
                search_parameter!(self, probcut_static_offset, 80),
            )
        {
            self.debug_counters.probcut_attempts += 1;
            let probcut_beta = beta
                + probcut_margin(
                    depth,
                    search_parameter!(self, probcut_base, 180),
                    search_parameter!(self, probcut_slope, 5),
                );
            for mv in legal_moves.iter().copied() {
                if !mv.is_capture() || mv.is_promotion() {
                    continue;
                }
                if position.see(mv).0 < 0 {
                    continue;
                }
                let undo = self
                    .make_search_move::<USE_NNUE>(position, mv)
                    .expect("probcut move must be legal");
                self.set_previous_move(ply + 1, mv);
                let score = self.alpha_beta_core::<USE_TABLEBASES, USE_NNUE>(
                    position,
                    depth.saturating_sub(4),
                    ply + 1,
                    -probcut_beta,
                    (-probcut_beta).saturating_add(1),
                    SearchNodeState::cut(),
                );
                self.set_previous_move(ply + 1, Move::NONE);
                self.unmake_search_move::<USE_NNUE>(position, mv, undo);
                let score = score?;
                let score = -score;
                if score >= probcut_beta {
                    self.debug_counters.probcut_prunes += 1;
                    self.store_tt(TtStoreInput {
                        key: tt_key,
                        depth: depth.saturating_sub(3).min(u8::MAX as usize) as u8,
                        ply,
                        best_move: mv,
                        static_eval: raw_static_eval.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                        score,
                        bound: Bound::Lower,
                    });
                    return Some(score);
                }
            }
        }

        let mut move_picker = MovePicker::new(self, position, &legal_moves, ordering_hints);
        while let Some(mv) = move_picker.next() {
            if excluded_move == Some(mv) {
                continue;
            }
            let is_quiet = !mv.is_capture() && !mv.is_promotion();
            let history_score = if is_quiet {
                self.quiet_history_score(position, mv, ply)
            } else {
                self.capture_history_score(position, mv)
            };
            let see_score = if !is_quiet {
                position.see(mv).0 as i32
            } else {
                0
            };
            if is_quiet {
                quiets_searched += 1;
            }

            let gives_check = position.gives_check(mv);
            let child_is_pv = node_state.is_pv && best_move.is_none();
            let is_hash_move = tt_move_hint == Some(mv);
            let has_searched_move = searched_moves > 0;
            let forward_prune_candidate = ForwardPruneCandidate {
                node_state,
                depth,
                alpha,
                static_eval,
                in_check,
                mv,
                gives_check,
                is_hash_move,
                has_searched_move,
                quiets_searched,
            };

            if !is_exclusion
                && futility_pruning_is_eligible_with_parameters(
                    self.heuristics,
                    forward_prune_candidate,
                    search_parameter!(self, futility_base, 90),
                    search_parameter!(self, futility_slope, 120),
                )
            {
                self.debug_counters.futility_prunes += 1;
                continue;
            }

            if !is_exclusion
                && late_move_pruning_is_eligible_with_parameters(
                    self.heuristics,
                    forward_prune_candidate,
                    search_parameter!(self, futility_base, 90),
                    search_parameter!(self, futility_slope, 120),
                    search_parameter!(self, late_move_base, 3_i32) as usize,
                    search_parameter!(self, late_move_slope, 3_i32) as usize,
                )
            {
                self.debug_counters.late_move_prunes += 1;
                continue;
            }

            if !is_exclusion
                && see_pruning_is_eligible(
                    self.heuristics,
                    node_state,
                    depth,
                    in_check,
                    mv,
                    gives_check,
                    is_hash_move,
                    searched_moves,
                    see_score,
                    search_parameter!(self, see_margin, 70),
                )
            {
                self.debug_counters.see_prunes += 1;
                continue;
            }

            if !is_exclusion
                && history_pruning_is_eligible(
                    self.heuristics,
                    node_state,
                    depth,
                    in_check,
                    is_quiet,
                    gives_check,
                    is_hash_move,
                    quiets_searched,
                    history_score,
                    search_parameter!(self, history_prune_threshold, 2_000),
                )
            {
                self.debug_counters.history_prunes += 1;
                continue;
            }

            let undo = self
                .make_search_move::<USE_NNUE>(position, mv)
                .expect("searched move must be legal");
            self.previous_moves[ply + 1] = mv;
            let child_depth = depth - 1 + usize::from(singular_move == Some(mv));

            let score_result = if !is_exclusion
                && lmr_is_eligible(
                    self.heuristics,
                    LmrCandidate {
                        depth,
                        is_pv: child_is_pv,
                        in_check,
                        mv,
                        gives_check,
                        is_hash_move,
                        quiets_searched,
                    },
                ) {
                self.debug_counters.lmr_reductions += 1;
                let reduction = if self.heuristics.contextual_lmr {
                    contextual_lmr_reduction_with_parameters(
                        depth,
                        quiets_searched,
                        improving,
                        node_state.cut_node,
                        history_score,
                        search_parameter!(self, lmr_divisor_pct, 150),
                    )
                } else {
                    lmr_reduction_with_parameters(
                        depth,
                        quiets_searched,
                        search_parameter!(self, lmr_divisor_pct, 150),
                    )
                };
                let scout_beta = (-alpha).min(INF);
                let scout_alpha = scout_beta.saturating_sub(1);
                let Some(reduced_score) = self.alpha_beta_core::<USE_TABLEBASES, USE_NNUE>(
                    position,
                    child_depth.saturating_sub(reduction),
                    ply + 1,
                    scout_alpha,
                    scout_beta,
                    SearchNodeState::cut(),
                ) else {
                    self.unmake_search_move::<USE_NNUE>(position, mv, undo);
                    return None;
                };
                let reduced_score = -reduced_score;
                if lmr_requires_full_research(reduced_score, alpha) {
                    self.debug_counters.lmr_researches += 1;
                    let Some(full_depth_score) = self.alpha_beta_core::<USE_TABLEBASES, USE_NNUE>(
                        position,
                        child_depth,
                        ply + 1,
                        scout_alpha,
                        scout_beta,
                        SearchNodeState::cut(),
                    ) else {
                        self.previous_moves[ply + 1] = Move::NONE;
                        self.unmake_search_move::<USE_NNUE>(position, mv, undo);
                        return None;
                    };
                    let full_depth_score = -full_depth_score;
                    if node_state.is_pv && full_depth_score > alpha {
                        self.debug_counters.pvs_full_researches += 1;
                        self.alpha_beta_core::<USE_TABLEBASES, USE_NNUE>(
                            position,
                            child_depth,
                            ply + 1,
                            -beta,
                            -alpha,
                            SearchNodeState::new(true),
                        )
                        .map(|score| -score)
                    } else {
                        Some(full_depth_score)
                    }
                } else {
                    Some(reduced_score)
                }
            } else if best_move.is_none() {
                self.alpha_beta_core::<USE_TABLEBASES, USE_NNUE>(
                    position,
                    child_depth,
                    ply + 1,
                    -beta,
                    -alpha,
                    SearchNodeState::new(child_is_pv),
                )
                .map(|score| -score)
            } else {
                self.debug_counters.pvs_scout_searches += 1;
                let scout_beta = -alpha;
                let scout_alpha = scout_beta.saturating_sub(1);
                let Some(score) = self.alpha_beta_core::<USE_TABLEBASES, USE_NNUE>(
                    position,
                    child_depth,
                    ply + 1,
                    scout_alpha,
                    scout_beta,
                    SearchNodeState::cut(),
                ) else {
                    self.previous_moves[ply + 1] = Move::NONE;
                    self.unmake_search_move::<USE_NNUE>(position, mv, undo);
                    return None;
                };
                let score = -score;
                if score > alpha {
                    self.debug_counters.pvs_full_researches += 1;
                    self.alpha_beta_core::<USE_TABLEBASES, USE_NNUE>(
                        position,
                        child_depth,
                        ply + 1,
                        -beta,
                        -alpha,
                        SearchNodeState::new(node_state.is_pv),
                    )
                    .map(|score| -score)
                } else {
                    Some(score)
                }
            };

            let Some(score) = score_result else {
                self.previous_moves[ply + 1] = Move::NONE;
                self.unmake_search_move::<USE_NNUE>(position, mv, undo);
                return None;
            };
            searched_moves += 1;
            if is_quiet {
                searched_quiets[searched_quiet_count] = mv;
                searched_quiet_count += 1;
            } else if mv.is_capture() {
                searched_captures[searched_capture_count] = mv;
                searched_capture_count += 1;
            }
            self.previous_moves[ply + 1] = Move::NONE;
            self.unmake_search_move::<USE_NNUE>(position, mv, undo);

            if score > alpha {
                alpha = score;
                best_move = mv;
                self.update_pv(ply, mv);
                if alpha >= beta {
                    if !is_exclusion && is_quiet {
                        self.record_killer(ply, mv);
                        self.record_quiet_cutoff(position, mv, ply, depth);
                        for failed in searched_quiets[..searched_quiet_count.saturating_sub(1)]
                            .iter()
                            .copied()
                        {
                            self.record_quiet_malus(position, failed, ply, depth);
                        }
                        for failed in searched_captures[..searched_capture_count].iter().copied() {
                            self.record_capture_history(position, failed, depth, false);
                        }
                    } else if !is_exclusion && mv.is_capture() {
                        self.record_capture_history(position, mv, depth, true);
                        for failed in searched_captures[..searched_capture_count.saturating_sub(1)]
                            .iter()
                            .copied()
                        {
                            self.record_capture_history(position, failed, depth, false);
                        }
                    }
                    break;
                }
            }
        }

        let bound = if alpha <= alpha_start {
            Bound::Upper
        } else if alpha >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        if !is_exclusion {
            self.update_correction_history(
                position,
                CorrectionUpdate {
                    depth,
                    in_check,
                    best_move,
                    static_eval,
                    score: alpha,
                    bound,
                },
            );
            self.store_tt(TtStoreInput {
                key: tt_key,
                depth: depth.min(u8::MAX as usize) as u8,
                ply,
                best_move,
                static_eval: raw_static_eval.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                score: alpha,
                bound,
            });
        }
        Some(alpha)
    }

    fn try_non_root_tablebase_score(&self, position: &Position, ply: usize) -> Option<i32> {
        let tablebases = self.tablebases.as_ref()?;
        if !tablebases.supports_non_root(position) {
            return None;
        }

        let mut legal_moves = MoveList::new();
        let mut probe_position = position.clone();
        probe_position.generate_legal_moves(&mut legal_moves);
        if legal_moves.is_empty() {
            return Some(terminal_score(position, ply));
        }

        tablebases
            .probe_wdl(&probe_position)
            .ok()
            .flatten()
            .map(|outcome| tablebase::score_from_wdl(outcome, ply))
    }

    pub(crate) fn update_pv(&mut self, ply: usize, mv: Move) {
        self.pv_table[ply][ply] = mv;
        let next_len = self.pv_length[ply + 1];
        for index in 0..next_len.saturating_sub(ply + 1) {
            self.pv_table[ply][ply + 1 + index] = self.pv_table[ply + 1][ply + 1 + index];
        }
        self.pv_length[ply] = next_len.max(ply + 1);
    }

    pub(crate) fn clear_pv(&mut self, ply: usize) {
        self.pv_length[ply] = ply;
    }

    fn collect_pv(&self, ply: usize) -> Vec<Move> {
        let end = self.pv_length[ply];
        if end <= ply {
            return Vec::new();
        }
        self.pv_table[ply][ply..end].to_vec()
    }

    fn capture_completed_pv(&mut self, pv: &[Move]) {
        self.previous_iteration_pv.fill(Move::NONE);
        self.previous_iteration_pv_length = pv.len();
        for (index, mv) in pv.iter().copied().enumerate() {
            self.previous_iteration_pv[index] = mv;
        }
    }

    pub(crate) fn previous_pv_move(&self, ply: usize) -> Option<Move> {
        if !self.heuristics.pv_move_ordering || ply >= self.previous_iteration_pv_length {
            return None;
        }

        for index in 0..ply {
            if self.previous_moves[index + 1] != self.previous_iteration_pv[index] {
                return None;
            }
        }

        let mv = self.previous_iteration_pv[ply];
        (!mv.is_none()).then_some(mv)
    }

    pub(crate) fn score_move(&self, position: &Position, mv: Move, hints: MoveOrderHints) -> i32 {
        if self.heuristics.pv_move_ordering && hints.pv_move == Some(mv) {
            return 500_000;
        }

        if hints.tt_move == Some(mv) {
            return 400_000;
        }

        if mv.is_capture() {
            return self.capture_order_score(position, mv);
        }

        if mv.is_promotion() {
            return 150_000
                + promotion_score(mv.promotion().expect("promotion flag must encode piece"));
        }

        if hints.quiescence_only {
            return i32::MIN / 2;
        }

        if self.heuristics.killer_moves && self.killer_moves[hints.ply][0] == mv {
            return 140_000;
        }
        if self.heuristics.killer_moves && self.killer_moves[hints.ply][1] == mv {
            return 130_000;
        }

        let mut quiet_score = quiet_shape_bonus(position, mv);
        quiet_score += self.quiet_history_score(position, mv, hints.ply);
        quiet_score
    }

    fn quiet_history_score(&self, position: &Position, mv: Move, ply: usize) -> i32 {
        let mut score = 0;
        if self.heuristics.quiet_history {
            score += self.history_score(position, mv);
        }
        if self.heuristics.continuation_history {
            score += self.continuation_score(position, mv, ply);
        }
        score
    }

    fn capture_order_score(&self, position: &Position, mv: Move) -> i32 {
        let see_score = position.see(mv).0 as i32;
        let history_score = self.capture_history_score(position, mv);
        if !self.heuristics.capture_buckets {
            return 200_000 + see_score + history_score;
        }

        if see_score > 0 {
            320_000 + see_score + history_score
        } else if see_score == 0 {
            260_000 + history_score
        } else {
            40_000 + see_score + history_score
        }
    }

    fn capture_history_score(&self, position: &Position, mv: Move) -> i32 {
        let Some((color, attacker, captured)) = capture_context(position, mv) else {
            return 0;
        };
        self.capture_history[color][attacker][mv.to().index()][captured] as i32
    }

    fn history_score(&self, position: &Position, mv: Move) -> i32 {
        let color = position.side_to_move().index();
        self.quiet_history[color][mv.from().index()][mv.to().index()] as i32
    }

    fn continuation_score(&self, position: &Position, mv: Move, ply: usize) -> i32 {
        let Some((prev_piece, prev_to)) = self.previous_move_context(position, ply) else {
            return 0;
        };
        let Some(piece) = position.piece_at(mv.from()) else {
            return 0;
        };
        let color = position.side_to_move().index();
        self.continuation_history[color][prev_piece][prev_to.index()][piece.piece_type().index()]
            [mv.to().index()] as i32
    }

    fn previous_move_context(
        &self,
        position: &Position,
        ply: usize,
    ) -> Option<(usize, crate::core::Square)> {
        if ply == 0 {
            return None;
        }

        let prev = self.previous_moves[ply];
        if prev.is_none() {
            return None;
        }
        let piece = position.piece_at(prev.to())?;
        Some((piece.piece_type().index(), prev.to()))
    }

    fn record_quiet_cutoff(&mut self, position: &Position, mv: Move, ply: usize, depth: usize) {
        if !self.heuristics.quiet_history && !self.heuristics.continuation_history {
            return;
        }

        let bonus = history_bonus(depth, search_parameter!(self, history_bonus_scale, 32));
        self.update_quiet_history(position, mv, ply, bonus);
    }

    fn record_quiet_malus(&mut self, position: &Position, mv: Move, ply: usize, depth: usize) {
        if !self.heuristics.history_maluses {
            return;
        }
        let malus = -history_bonus(depth, search_parameter!(self, history_bonus_scale, 32));
        self.update_quiet_history(position, mv, ply, malus);
    }

    fn update_quiet_history(&mut self, position: &Position, mv: Move, ply: usize, bonus: i32) {
        if self.heuristics.quiet_history {
            let color = position.side_to_move().index();
            let entry = &mut self.quiet_history[color][mv.from().index()][mv.to().index()];
            update_history_entry(entry, bonus);
        }

        if !self.heuristics.continuation_history {
            return;
        }

        let Some((prev_piece, prev_to)) = self.previous_move_context(position, ply) else {
            return;
        };
        let Some(piece) = position.piece_at(mv.from()) else {
            return;
        };
        let color = position.side_to_move().index();
        let entry = &mut self.continuation_history[color][prev_piece][prev_to.index()]
            [piece.piece_type().index()][mv.to().index()];
        update_history_entry(entry, bonus);
    }

    fn record_capture_history(
        &mut self,
        position: &Position,
        mv: Move,
        depth: usize,
        success: bool,
    ) {
        if !self.heuristics.capture_history {
            return;
        }
        let Some((color, attacker, captured)) = capture_context(position, mv) else {
            return;
        };
        let bonus = if success {
            history_bonus(depth, search_parameter!(self, history_bonus_scale, 32))
        } else {
            -history_bonus(depth, search_parameter!(self, history_bonus_scale, 32))
        };
        let entry = &mut self.capture_history[color][attacker][mv.to().index()][captured];
        update_history_entry(entry, bonus);
    }

    fn record_killer(&mut self, ply: usize, mv: Move) {
        if !self.heuristics.killer_moves || ply >= MAX_PLY {
            return;
        }

        if self.killer_moves[ply][0] == mv {
            return;
        }

        self.killer_moves[ply][1] = self.killer_moves[ply][0];
        self.killer_moves[ply][0] = mv;
    }

    pub(crate) fn probe_tt(&mut self, key: u64) -> Option<TtHit> {
        let hit = self.tt.as_ref().and_then(|tt| tt.probe(key));
        if hit.is_some() {
            self.tt_hits += 1;
        }
        hit
    }

    pub(crate) fn qsearch_tt_enabled(&self) -> bool {
        self.heuristics.qsearch_tt
    }

    fn store_tt(&mut self, input: TtStoreInput) {
        let Some(tt) = self.tt.as_ref() else {
            return;
        };

        tt.store(
            input.key,
            TtStore {
                best_move: input.best_move,
                score: tt::normalize_score_for_store(input.score, input.ply),
                eval: input.static_eval,
                depth: input.depth,
                bound: input.bound,
            },
        );
    }

    pub(crate) fn store_qsearch_tt(
        &mut self,
        key: u64,
        ply: usize,
        best_move: Move,
        static_eval: i32,
        score: i32,
        bound: Bound,
    ) {
        if !self.heuristics.qsearch_tt {
            return;
        }
        self.store_tt(TtStoreInput {
            key,
            depth: 0,
            ply,
            best_move,
            static_eval: static_eval.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            score,
            bound,
        });
    }

    fn prepare_nnue(&mut self, position: &Position) {
        self.nnue
            .as_mut()
            .expect("NNUE preparation requires an active NNUE service")
            .reset(position);
    }

    fn correct_static_eval(
        &mut self,
        position: &Position,
        raw_static_eval: i32,
        in_check: bool,
    ) -> i32 {
        if !self.heuristics.correction_history || in_check {
            return raw_static_eval;
        }
        self.debug_counters.correction_history_lookups = self
            .debug_counters
            .correction_history_lookups
            .saturating_add(1);
        raw_static_eval.saturating_add(
            self.correction_history
                .as_ref()
                .expect("enabled correction history must be allocated")
                .correction(position),
        )
    }

    fn update_correction_history(&mut self, position: &Position, update: CorrectionUpdate) {
        if !correction_history_update_is_eligible(self.heuristics, update) {
            return;
        }
        let error = update.score.saturating_sub(update.static_eval);
        let bonus = (error.saturating_mul(update.depth.min(16) as i32) / 128).clamp(-2, 2);
        if bonus == 0 {
            return;
        }
        self.correction_history
            .as_mut()
            .expect("enabled correction history must be allocated")
            .update(position, bonus);
        self.debug_counters.correction_history_updates = self
            .debug_counters
            .correction_history_updates
            .saturating_add(1);
    }

    pub(crate) fn evaluate_position<const USE_NNUE: bool>(&self, position: &Position) -> i32 {
        if USE_NNUE {
            self.nnue
                .as_ref()
                .expect("NNUE evaluation requires an active NNUE service")
                .evaluate(position)
                .0
        } else {
            self.classical_weights.as_ref().map_or_else(
                || eval::evaluate(position).0,
                |weights| eval::evaluate_with_weights(position, weights).0,
            )
        }
    }

    pub(crate) fn make_search_move<const USE_NNUE: bool>(
        &mut self,
        position: &mut Position,
        mv: Move,
    ) -> Result<crate::core::UndoState, crate::core::MoveError> {
        let undo = position.make_generated_move(mv)?;
        if USE_NNUE {
            self.nnue
                .as_mut()
                .expect("NNUE move application requires an active NNUE service")
                .push_child(position, mv, undo);
        }
        Ok(undo)
    }

    pub(crate) fn unmake_search_move<const USE_NNUE: bool>(
        &mut self,
        position: &mut Position,
        mv: Move,
        undo: crate::core::UndoState,
    ) {
        if USE_NNUE {
            self.nnue
                .as_mut()
                .expect("NNUE move restoration requires an active NNUE service")
                .pop();
        }
        position.unmake_move(mv, undo);
    }

    pub(crate) fn set_previous_move(&mut self, ply: usize, mv: Move) {
        if ply < MAX_PLY {
            self.previous_moves[ply] = mv;
        }
    }

    fn excluded_move(&self, ply: usize) -> Option<Move> {
        self.excluded_moves
            .as_ref()
            .map(|moves| moves[ply])
            .filter(|mv| !mv.is_none())
    }

    fn set_excluded_move(&mut self, ply: usize, mv: Move) {
        self.excluded_moves
            .as_mut()
            .expect("singular exclusion storage must be allocated")[ply] = mv;
    }

    pub(crate) fn hard_stop_requested(&self) -> bool {
        self.control
            .stop_flag
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
            || self
                .control
                .helper_stop_flag
                .as_ref()
                .is_some_and(|flag| flag.load(Ordering::Relaxed))
            || self
                .effective_deadline(self.control.hard_deadline)
                .is_some_and(|deadline| Instant::now() >= deadline)
            || self
                .control
                .ponder_state
                .as_ref()
                .is_some_and(|ponder| ponder.cancelled())
            || self
                .control
                .node_budget
                .as_ref()
                .is_some_and(|budget| budget.exhausted())
    }

    pub(crate) fn count_node(&mut self) -> bool {
        if self
            .control
            .node_budget
            .as_ref()
            .is_some_and(|budget| !budget.try_consume())
        {
            return false;
        }
        self.nodes = self.nodes.saturating_add(1);
        true
    }

    fn adaptive_soft_stop_requested(
        &self,
        stability: IterationStability,
        last_iteration_elapsed: Option<std::time::Duration>,
    ) -> bool {
        let Some(soft_deadline) = self.effective_deadline(self.control.soft_deadline) else {
            return false;
        };
        let hard_deadline = self.effective_deadline(self.control.hard_deadline);
        let has_extension_window = hard_deadline.is_some_and(|hard| hard > soft_deadline);
        let now = Instant::now();
        if !has_extension_window {
            return now >= soft_deadline;
        }

        let timing_started = self
            .control
            .ponder_state
            .as_ref()
            .and_then(|ponder| ponder.hit_at())
            .unwrap_or(self.started);
        let nominal_budget = soft_deadline.saturating_duration_since(timing_started);
        #[cfg(not(feature = "spsa-tuning"))]
        let soft_budget_factor = stability.soft_budget_factor();
        #[cfg(feature = "spsa-tuning")]
        let soft_budget_factor = stability.soft_budget_factor_with_parameters(self.parameters);
        let adaptive_budget = nominal_budget.mul_f64(soft_budget_factor);
        let adaptive_deadline = timing_started
            .checked_add(adaptive_budget)
            .unwrap_or(soft_deadline);
        let effective_deadline =
            hard_deadline.map_or(adaptive_deadline, |hard| adaptive_deadline.min(hard));
        if now >= effective_deadline {
            return true;
        }

        // Fixed movetime intentionally consumes its full allocation. Clock searches have a
        // distinct soft/hard window, so avoid beginning an iteration that is unlikely to finish
        // inside the adaptive target. Two times the previous iteration is deliberately modest:
        // iterative-deepening cost is noisy because TT reuse and aspiration re-searches can make
        // neighboring depths non-monotonic.
        last_iteration_elapsed.is_some_and(|elapsed| {
            elapsed.mul_f64(2.0) >= effective_deadline.saturating_duration_since(now)
        })
    }

    fn effective_deadline(&self, deadline: Option<Instant>) -> Option<Instant> {
        let deadline = deadline?;
        match self.control.ponder_state.as_ref() {
            Some(ponder) => ponder.adjust_deadline(deadline),
            None => Some(deadline),
        }
    }

    #[cfg(test)]
    fn debug_counters(&self) -> SearchDebugCounters {
        self.debug_counters
    }
}

fn lmr_is_eligible(heuristics: SearchHeuristics, candidate: LmrCandidate) -> bool {
    heuristics.late_move_reductions
        && !candidate.is_pv
        && !candidate.in_check
        && candidate.depth >= 4
        && !candidate.mv.is_capture()
        && !candidate.mv.is_promotion()
        && !candidate.gives_check
        && !candidate.is_hash_move
        && candidate.quiets_searched > 2
}

fn singular_probe(
    heuristics: SearchHeuristics,
    excluded_move: Option<Move>,
    depth: usize,
    ply: usize,
    tt_hit: Option<TtHit>,
    legal_tt_move: Option<Move>,
) -> Option<SingularProbe> {
    if !heuristics.singular_extensions
        || ply == 0
        || excluded_move.is_some()
        || depth < SINGULAR_MIN_DEPTH
    {
        return None;
    }

    let hit = tt_hit?;
    let excluded_move = legal_tt_move?;
    if hit.best_move != excluded_move
        || !matches!(hit.bound, Bound::Exact | Bound::Lower)
        || (hit.depth as usize).saturating_add(SINGULAR_TT_DEPTH_SLACK) < depth
    {
        return None;
    }

    let tt_score = tt::denormalize_score_from_tt(hit.score, ply);
    // Mate and Syzygy scores carry proof semantics that an approximate exclusion
    // search must never reinterpret as ordinary evaluator evidence.
    if tt_score.abs() >= tablebase::TABLEBASE_SCORE_BAND {
        return None;
    }

    Some(SingularProbe {
        excluded_move,
        depth: (depth - 1) / 2,
        beta: tt_score.saturating_sub(2 * depth as i32),
    })
}

fn lmr_reduction_with_parameters(depth: usize, quiets_searched: usize, divisor_pct: i32) -> usize {
    #[cfg(not(feature = "spsa-tuning"))]
    let _ = divisor_pct;
    #[cfg(not(feature = "spsa-tuning"))]
    static REDUCTIONS: OnceLock<[[u8; MAX_MOVES + 1]; MAX_PLY]> = OnceLock::new();
    #[cfg(feature = "spsa-tuning")]
    static REDUCTIONS: [OnceLock<Box<[[u8; MAX_MOVES + 1]; MAX_PLY]>>; 221] =
        [const { OnceLock::new() }; 221];
    #[cfg(not(feature = "spsa-tuning"))]
    let reductions = REDUCTIONS.get_or_init(|| build_lmr_reductions(150));
    #[cfg(feature = "spsa-tuning")]
    let reductions = REDUCTIONS[(divisor_pct - 80).clamp(0, 220) as usize]
        .get_or_init(|| Box::new(build_lmr_reductions(divisor_pct)));
    reductions[depth.min(MAX_PLY - 1)][quiets_searched.min(MAX_MOVES)] as usize
}

fn build_lmr_reductions(divisor_pct: i32) -> [[u8; MAX_MOVES + 1]; MAX_PLY] {
    let mut table = [[0; MAX_MOVES + 1]; MAX_PLY];
    for (depth, row) in table.iter_mut().enumerate().skip(1) {
        for (move_count, reduction) in row.iter_mut().enumerate().skip(1) {
            let divisor = f64::from(divisor_pct) / 100.0;
            let calculated = ((depth as f64).ln() * (move_count as f64).ln() / divisor)
                .floor()
                .max(1.0) as usize;
            *reduction = calculated.min(depth.saturating_sub(1)) as u8;
        }
    }
    table
}

fn contextual_lmr_reduction_with_parameters(
    depth: usize,
    quiets_searched: usize,
    improving: bool,
    cut_node: bool,
    history_score: i32,
    divisor_pct: i32,
) -> usize {
    let mut reduction = lmr_reduction_with_parameters(depth, quiets_searched, divisor_pct) as i32;
    if cut_node {
        reduction += 1;
    }
    if improving {
        reduction -= 1;
    }
    if history_score < -4_000 {
        reduction += 1;
    } else if history_score > 4_000 {
        reduction -= 1;
    }
    reduction.clamp(1, depth.saturating_sub(1) as i32) as usize
}

fn lmr_requires_full_research(reduced_score: i32, alpha: i32) -> bool {
    reduced_score > alpha
}

fn correction_history_update_is_eligible(
    heuristics: SearchHeuristics,
    update: CorrectionUpdate,
) -> bool {
    heuristics.correction_history
        && !update.in_check
        && update.depth > 0
        && update.score.abs() < MATE_THRESHOLD
        && update.bound != Bound::Exact
        && (update.best_move.is_none()
            || (!update.best_move.is_capture() && !update.best_move.is_promotion()))
}

#[allow(clippy::too_many_arguments)]
fn reverse_futility_is_eligible_with_parameters(
    heuristics: SearchHeuristics,
    position: &Position,
    node_state: SearchNodeState,
    depth: usize,
    beta: i32,
    static_eval: i32,
    in_check: bool,
    margin_slope: i32,
) -> bool {
    heuristics.reverse_futility_pruning
        && !node_state.is_pv
        && !in_check
        && depth <= 2
        && beta > -MATE_THRESHOLD
        && beta < MATE_THRESHOLD
        && static_eval >= beta + reverse_futility_margin(depth, margin_slope)
        && position.has_non_pawn_material(position.side_to_move())
}

fn reverse_futility_margin(depth: usize, slope: i32) -> i32 {
    slope * depth as i32
}

#[allow(clippy::too_many_arguments)]
fn probcut_is_eligible(
    heuristics: SearchHeuristics,
    node_state: SearchNodeState,
    depth: usize,
    beta: i32,
    static_eval: i32,
    in_check: bool,
    margin_base: i32,
    margin_slope: i32,
    static_offset: i32,
) -> bool {
    heuristics.probcut
        && !node_state.is_pv
        && !in_check
        && depth >= 5
        && beta > -MATE_THRESHOLD
        && beta < MATE_THRESHOLD - probcut_margin(depth, margin_base, margin_slope)
        && static_eval >= beta - static_offset
}

#[cfg(any(test, debug_assertions, feature = "internal-testing"))]
fn multi_cut_is_eligible(
    heuristics: SearchHeuristics,
    position: &Position,
    node_state: SearchNodeState,
    depth: usize,
    beta: i32,
    static_eval: i32,
    in_check: bool,
) -> bool {
    heuristics.multi_cut
        && node_state.cut_node
        && node_state.null_move_allowed
        && !node_state.is_pv
        && !in_check
        && depth >= MULTI_CUT_MIN_DEPTH
        && beta > -MATE_THRESHOLD
        && beta < MATE_THRESHOLD
        && static_eval >= beta - 120
        && position.has_non_pawn_material(position.side_to_move())
}

fn probcut_margin(depth: usize, base: i32, slope: i32) -> i32 {
    (base - slope * depth.min(12) as i32).max(1)
}

fn futility_pruning_is_eligible_with_parameters(
    heuristics: SearchHeuristics,
    candidate: ForwardPruneCandidate,
    margin_base: i32,
    margin_slope: i32,
) -> bool {
    heuristics.futility_pruning
        && !candidate.node_state.is_pv
        && !candidate.in_check
        && candidate.depth <= 2
        && !candidate.mv.is_capture()
        && !candidate.mv.is_promotion()
        && !candidate.gives_check
        && !candidate.is_hash_move
        && candidate.has_searched_move
        && candidate.static_eval + futility_margin(candidate.depth, margin_base, margin_slope)
            <= candidate.alpha
}

fn late_move_pruning_is_eligible_with_parameters(
    heuristics: SearchHeuristics,
    candidate: ForwardPruneCandidate,
    margin_base: i32,
    margin_slope: i32,
    threshold_base: usize,
    threshold_slope: usize,
) -> bool {
    heuristics.late_move_pruning
        && !candidate.node_state.is_pv
        && !candidate.in_check
        && candidate.depth <= 2
        && !candidate.mv.is_capture()
        && !candidate.mv.is_promotion()
        && !candidate.gives_check
        && !candidate.is_hash_move
        && candidate.has_searched_move
        && candidate.static_eval + futility_margin(candidate.depth, margin_base, margin_slope)
            <= candidate.alpha
        && candidate.quiets_searched
            > late_move_pruning_threshold(candidate.depth, threshold_base, threshold_slope)
}

fn futility_margin(depth: usize, base: i32, slope: i32) -> i32 {
    base + slope * depth as i32
}

fn late_move_pruning_threshold(depth: usize, base: usize, slope: usize) -> usize {
    base + depth * slope
}

#[allow(clippy::too_many_arguments)]
fn see_pruning_is_eligible(
    heuristics: SearchHeuristics,
    node_state: SearchNodeState,
    depth: usize,
    in_check: bool,
    mv: Move,
    gives_check: bool,
    is_hash_move: bool,
    searched_moves: usize,
    see_score: i32,
    margin: i32,
) -> bool {
    heuristics.see_pruning
        && !node_state.is_pv
        && !in_check
        && depth <= 4
        && mv.is_capture()
        && !mv.is_promotion()
        && !gives_check
        && !is_hash_move
        && searched_moves > 0
        && see_score < -(margin * depth as i32)
}

#[allow(clippy::too_many_arguments)]
fn history_pruning_is_eligible(
    heuristics: SearchHeuristics,
    node_state: SearchNodeState,
    depth: usize,
    in_check: bool,
    is_quiet: bool,
    gives_check: bool,
    is_hash_move: bool,
    quiets_searched: usize,
    history_score: i32,
    threshold: i32,
) -> bool {
    heuristics.history_pruning
        && !node_state.is_pv
        && !in_check
        && depth <= 3
        && is_quiet
        && !gives_check
        && !is_hash_move
        && quiets_searched > 3 + depth * 2
        && history_score < -threshold
}

#[allow(clippy::too_many_arguments)]
fn null_move_is_eligible_with_parameters(
    heuristics: SearchHeuristics,
    position: &Position,
    node_state: SearchNodeState,
    depth: usize,
    beta: i32,
    static_eval: i32,
    in_check: bool,
    static_margin: i32,
) -> bool {
    heuristics.null_move_pruning
        && node_state.null_move_allowed
        && !node_state.is_pv
        && !in_check
        && depth >= 4
        && beta > -MATE_THRESHOLD
        && beta < MATE_THRESHOLD
        && static_eval >= beta + static_margin
        && position.has_non_pawn_material(position.side_to_move())
}

fn null_move_reduction_with_parameters(
    depth: usize,
    static_eval: i32,
    beta: i32,
    base_reduction: usize,
    depth_divisor: usize,
    eval_divisor: i32,
) -> usize {
    let depth_component = depth / depth_divisor;
    let eval_component = static_eval
        .saturating_sub(beta)
        .max(0)
        .div_euclid(eval_divisor)
        .min(2) as usize;
    (base_reduction + depth_component + eval_component).min(depth.saturating_sub(1))
}

const fn null_move_requires_verification_with_parameters(
    depth: usize,
    verification_depth: usize,
) -> bool {
    depth >= verification_depth
}

// Default-value wrappers keep the focused heuristic tests readable and make
// default equivalence explicit. Live searches use the parameterized variants.
#[cfg(test)]
fn lmr_reduction(depth: usize, quiets_searched: usize) -> usize {
    lmr_reduction_with_parameters(depth, quiets_searched, 150)
}

#[cfg(test)]
fn contextual_lmr_reduction(
    depth: usize,
    quiets_searched: usize,
    improving: bool,
    cut_node: bool,
    history_score: i32,
) -> usize {
    contextual_lmr_reduction_with_parameters(
        depth,
        quiets_searched,
        improving,
        cut_node,
        history_score,
        150,
    )
}

#[cfg(test)]
fn reverse_futility_is_eligible(
    heuristics: SearchHeuristics,
    position: &Position,
    node_state: SearchNodeState,
    depth: usize,
    beta: i32,
    static_eval: i32,
    in_check: bool,
) -> bool {
    reverse_futility_is_eligible_with_parameters(
        heuristics,
        position,
        node_state,
        depth,
        beta,
        static_eval,
        in_check,
        140,
    )
}

#[cfg(test)]
fn futility_pruning_is_eligible(
    heuristics: SearchHeuristics,
    candidate: ForwardPruneCandidate,
) -> bool {
    futility_pruning_is_eligible_with_parameters(heuristics, candidate, 90, 120)
}

#[cfg(test)]
fn late_move_pruning_is_eligible(
    heuristics: SearchHeuristics,
    candidate: ForwardPruneCandidate,
) -> bool {
    late_move_pruning_is_eligible_with_parameters(heuristics, candidate, 90, 120, 3, 3)
}

#[cfg(test)]
fn null_move_is_eligible(
    heuristics: SearchHeuristics,
    position: &Position,
    node_state: SearchNodeState,
    depth: usize,
    beta: i32,
    static_eval: i32,
    in_check: bool,
) -> bool {
    null_move_is_eligible_with_parameters(
        heuristics,
        position,
        node_state,
        depth,
        beta,
        static_eval,
        in_check,
        NULL_MOVE_STATIC_MARGIN,
    )
}

#[cfg(test)]
fn null_move_reduction(depth: usize, static_eval: i32, beta: i32) -> usize {
    null_move_reduction_with_parameters(depth, static_eval, beta, 2, 6, 256)
}

#[cfg(test)]
const fn null_move_requires_verification(depth: usize) -> bool {
    null_move_requires_verification_with_parameters(depth, 10)
}

impl SearchContext<'_> {
    fn helper_root_order(
        &self,
        position: &Position,
        legal_moves: &MoveList,
        hints: MoveOrderHints,
        helper_index: usize,
    ) -> Vec<Move> {
        let mut ordered = MovePicker::new(self, position, legal_moves, hints).ordered();

        let hinted_prefix = ordered
            .iter()
            .take_while(|mv| hints.pv_move == Some(**mv) || hints.tt_move == Some(**mv))
            .count();

        let tail = &mut ordered[hinted_prefix..];
        if !tail.is_empty() {
            tail.rotate_left(helper_index % tail.len());
        }
        ordered
    }
}

pub(crate) fn is_draw(position: &Position) -> bool {
    if position.is_draw_by_repetition() || position.is_insufficient_material() {
        return true;
    }

    if !position.is_draw_by_fifty_move() {
        return false;
    }

    // Checkmate ends the game before a fifty-move claim can be made. Most
    // fifty-move positions can take the cheap path; only a checked side needs
    // legal-evasion generation to distinguish mate from a claimable draw.
    if !position.is_in_check(position.side_to_move()) {
        return true;
    }

    let mut probe = position.clone();
    let mut legal_moves = MoveList::new();
    probe.generate_legal_moves(&mut legal_moves);
    !legal_moves.is_empty()
}

pub(crate) fn terminal_score(position: &Position, ply: usize) -> i32 {
    if position.is_in_check(position.side_to_move()) {
        -mate_score(ply)
    } else {
        0
    }
}

pub(crate) fn mate_score(ply: usize) -> i32 {
    MATE_SCORE - ply as i32
}

pub(crate) fn mate_distance_bounds(ply: usize) -> (i32, i32) {
    (-mate_score(ply), mate_score(ply + 1))
}

pub(crate) fn is_quiescence_move(mv: Move, position: &Position) -> bool {
    mv.is_capture() || mv.is_promotion() || position.is_in_check(position.side_to_move())
}

fn quiet_shape_bonus(position: &Position, mv: Move) -> i32 {
    let _ = position;
    let _ = mv;
    0
}

fn promotion_score(piece_type: crate::core::PieceType) -> i32 {
    see::promotion_gain(piece_type).0 as i32
}

fn history_bonus(depth: usize, scale: i32) -> i32 {
    ((depth * depth) as i32 * scale).clamp(scale, HISTORY_MAX / 2)
}

fn update_history_entry(entry: &mut i16, bonus: i32) {
    let bonus = bonus.clamp(-HISTORY_MAX, HISTORY_MAX);
    let current = *entry as i32;
    let updated = current + bonus - current * bonus.abs() / HISTORY_MAX;
    *entry = updated.clamp(-HISTORY_MAX, HISTORY_MAX) as i16;
}

fn capture_context(position: &Position, mv: Move) -> Option<(usize, usize, usize)> {
    if !mv.is_capture() {
        return None;
    }
    let attacker = position.piece_at(mv.from())?.piece_type().index();
    let captured = if mv.is_en_passant() {
        PieceType::Pawn.index()
    } else {
        position.piece_at(mv.to())?.piece_type().index()
    };
    Some((position.side_to_move().index(), attacker, captured))
}

fn format_info_line(
    depth: u8,
    seldepth: u8,
    score: i32,
    nodes: u64,
    elapsed_ms: u128,
    tt_hits: u64,
    pv: &[Move],
) -> String {
    let pv_text = pv
        .iter()
        .map(|mv| mv.to_string())
        .collect::<Vec<_>>()
        .join(" ");

    let score_text = if score.abs() >= MATE_THRESHOLD {
        if score > 0 {
            format!("score mate {}", (MATE_SCORE - score + 1) / 2)
        } else {
            format!("score mate -{}", (MATE_SCORE + score + 1) / 2)
        }
    } else {
        format!("score cp {score}")
    };
    let nps = ((nodes as u128) * 1_000)
        .checked_div(elapsed_ms)
        .unwrap_or_else(|| nodes.saturating_mul(1_000) as u128) as u64;

    if pv_text.is_empty() {
        format!(
            "info depth {depth} seldepth {seldepth} {score_text} nodes {nodes} nps {nps} time {elapsed_ms} tthits {tt_hits}"
        )
    } else {
        format!(
            "info depth {depth} seldepth {seldepth} {score_text} nodes {nodes} nps {nps} time {elapsed_ms} tthits {tt_hits} pv {pv_text}"
        )
    }
}

pub(crate) fn tt_cutoff_score(
    hit: TtHit,
    depth: usize,
    ply: usize,
    alpha: i32,
    beta: i32,
) -> Option<i32> {
    if hit.depth < depth.min(u8::MAX as usize) as u8 {
        return None;
    }

    let score = tt::denormalize_score_from_tt(hit.score, ply);
    match hit.bound {
        Bound::Exact => Some(score),
        Bound::Lower if score >= beta => Some(score),
        Bound::Upper if score <= alpha => Some(score),
        Bound::Lower | Bound::Upper => None,
    }
}

pub(crate) fn validated_move_hint(legal_moves: &MoveList, move_hint: Option<Move>) -> Option<Move> {
    let move_hint = move_hint?;
    legal_moves
        .as_slice()
        .iter()
        .copied()
        .find(|mv| *mv == move_hint)
}

fn validated_tt_move_hint(legal_moves: &MoveList, tt_move_hint: Option<Move>) -> Option<Move> {
    validated_move_hint(legal_moves, tt_move_hint)
}

#[cfg(test)]
mod tests {
    use super::{
        Bound, CorrectionUpdate, ForwardPruneCandidate, IterationStability, LmrCandidate,
        MULTI_CUT_MIN_DEPTH, MULTI_CUT_REQUIRED_CUTOFFS, Move, MoveList, MoveOrderHints,
        PonderState, Position, SearchContext, SearchHeuristics, SearchLimits, SearchNodeState,
        SearchResources, contextual_lmr_reduction, correction_history_keys,
        correction_history_update_is_eligible, futility_pruning_is_eligible,
        late_move_pruning_is_eligible, lmr_is_eligible, lmr_reduction, lmr_requires_full_research,
        mate_distance_bounds, multi_cut_is_eligible, null_move_is_eligible, null_move_reduction,
        null_move_requires_verification, reverse_futility_is_eligible, singular_probe,
        tt_cutoff_score, update_history_entry, validated_tt_move_hint,
    };
    use crate::core::{ParsedMove, Square, chess_move::FLAG_CAPTURE};
    use crate::search::tablebase::{self, MockTablebaseBackend, TablebaseService, WdlOutcome};
    use crate::search::tt::{TtHit, normalize_score_for_store};
    use std::{sync::Arc, time::Instant};

    fn square(text: &str) -> Square {
        Square::from_coord_text(text).expect("test square must parse")
    }

    #[test]
    fn invalid_tt_move_hint_is_ignored_safely() {
        let mut position = Position::startpos();
        let mut legal_moves = MoveList::new();
        position.generate_legal_moves(&mut legal_moves);

        let invalid = Move::new(Square::A1, Square::A8);
        assert_eq!(validated_tt_move_hint(&legal_moves, Some(invalid)), None);
    }

    #[test]
    fn valid_tt_move_hint_is_reused_for_ordering() {
        let mut position = Position::startpos();
        let parsed = ParsedMove::parse("e2e4").expect("parse must succeed");
        let mut legal_moves = MoveList::new();
        position.generate_legal_moves(&mut legal_moves);
        let valid = legal_moves
            .as_slice()
            .iter()
            .copied()
            .find(|mv| mv.matches_parsed(parsed))
            .expect("legal move must exist");

        assert_eq!(
            validated_tt_move_hint(&legal_moves, Some(valid)),
            Some(valid)
        );
    }

    #[test]
    fn correction_history_keys_separate_pawn_and_each_sides_non_pawn_structure() {
        let start = Position::startpos();
        let start_keys = correction_history_keys(&start);

        let mut pawn_move = start.clone();
        pawn_move
            .apply_uci_move("e2e4")
            .expect("pawn move must be legal");
        let pawn_move_keys = correction_history_keys(&pawn_move);
        assert_ne!(pawn_move_keys.0, start_keys.0);
        assert_eq!(pawn_move_keys.1, start_keys.1);

        let mut knight_move = start.clone();
        knight_move
            .apply_uci_move("g1f3")
            .expect("knight move must be legal");
        let knight_move_keys = correction_history_keys(&knight_move);
        assert_eq!(knight_move_keys.0, start_keys.0);
        assert_ne!(knight_move_keys.1[0], start_keys.1[0]);
        assert_eq!(knight_move_keys.1[1], start_keys.1[1]);
    }

    #[test]
    fn correction_history_toggle_is_an_isolated_static_eval_seam() {
        let position = Position::startpos();
        let heuristics = SearchHeuristics::phase9_default().with_correction_history(true);
        let mut enabled = SearchContext::new(SearchLimits::new(2).with_heuristics(heuristics));
        enabled
            .correction_history
            .as_mut()
            .expect("enabled correction history must be allocated")
            .update(&position, 64);
        assert_eq!(enabled.correct_static_eval(&position, 100, false), 102);
        assert_eq!(enabled.debug_counters().correction_history_lookups, 1);

        let mut disabled = SearchContext::new(SearchLimits::new(2));
        assert!(disabled.correction_history.is_none());
        assert_eq!(disabled.correct_static_eval(&position, 100, false), 100);
        assert_eq!(disabled.debug_counters().correction_history_lookups, 0);
    }

    #[test]
    fn correction_history_updates_only_quiet_fail_high_and_fail_low_nodes() {
        let quiet = Move::new(square("e2"), square("e4"));
        let capture = Move::new(square("e4"), square("d5")).with_flags(FLAG_CAPTURE);
        let base = CorrectionUpdate {
            depth: 5,
            in_check: false,
            best_move: quiet,
            static_eval: 0,
            score: 100,
            bound: Bound::Lower,
        };
        let heuristics = SearchHeuristics::phase9_default().with_correction_history(true);
        assert!(correction_history_update_is_eligible(heuristics, base));
        assert!(correction_history_update_is_eligible(
            heuristics,
            CorrectionUpdate {
                best_move: Move::NONE,
                bound: Bound::Upper,
                ..base
            }
        ));
        for rejected in [
            CorrectionUpdate {
                best_move: capture,
                ..base
            },
            CorrectionUpdate {
                bound: Bound::Exact,
                ..base
            },
            CorrectionUpdate {
                in_check: true,
                ..base
            },
            CorrectionUpdate {
                score: super::MATE_THRESHOLD,
                ..base
            },
        ] {
            assert!(!correction_history_update_is_eligible(heuristics, rejected));
        }
    }

    #[test]
    fn corrected_static_eval_never_contaminates_the_tt_eval_payload() {
        let position = Position::startpos();
        let raw_eval = crate::search::eval::evaluate(&position).0;
        let heuristics = SearchHeuristics::phase8_baseline().with_correction_history(true);
        let mut context = SearchContext::new(SearchLimits::new(2).with_heuristics(heuristics));
        context
            .correction_history
            .as_mut()
            .expect("enabled correction history must be allocated")
            .update(&position, 64);
        assert_eq!(
            context.correct_static_eval(&position, raw_eval, false),
            raw_eval + 2
        );

        let mut searched = position.clone();
        context
            .alpha_beta(
                &mut searched,
                1,
                0,
                -super::INF,
                super::INF,
                SearchNodeState::new(true),
            )
            .expect("search must complete");
        let hit = context
            .probe_tt(position.search_key())
            .expect("completed node must be stored");
        assert_eq!(i32::from(hit.eval), raw_eval);
    }

    #[test]
    fn quiet_search_bounds_train_correction_history_in_live_search() {
        let heuristics = SearchHeuristics::phase8_baseline().with_correction_history(true);
        let mut context = SearchContext::new(
            SearchLimits::new(3)
                .without_tt()
                .with_heuristics(heuristics),
        );
        let mut position = Position::startpos();
        context
            .alpha_beta(&mut position, 2, 0, -501, -500, SearchNodeState::cut())
            .expect("quiet fail-high search must complete");

        let counters = context.debug_counters();
        assert!(counters.correction_history_lookups > 0);
        assert!(counters.correction_history_updates > 0);
    }

    #[test]
    fn tt_cutoff_semantics_follow_key_depth_and_bound_rules() {
        let exact = TtHit {
            key_tag: 1,
            best_move: Move::NONE,
            score: normalize_score_for_store(32, 3),
            eval: 0,
            depth: 6,
            bound: Bound::Exact,
            generation: 1,
        };
        assert_eq!(tt_cutoff_score(exact, 4, 3, -10, 10), Some(32));

        let lower = TtHit {
            bound: Bound::Lower,
            score: normalize_score_for_store(120, 2),
            ..exact
        };
        assert_eq!(tt_cutoff_score(lower, 4, 2, -20, 100), Some(120));
        assert_eq!(tt_cutoff_score(lower, 7, 2, -20, 100), None);

        let upper = TtHit {
            bound: Bound::Upper,
            score: normalize_score_for_store(-80, 2),
            ..exact
        };
        assert_eq!(tt_cutoff_score(upper, 4, 2, -70, 50), Some(-80));
        assert_eq!(tt_cutoff_score(upper, 4, 2, -90, 50), None);
    }

    #[test]
    fn singular_probe_requires_the_full_safe_tt_contract() {
        let tt_move = Move::new(square("e2"), square("e4"));
        let hit = TtHit {
            key_tag: 1,
            best_move: tt_move,
            score: normalize_score_for_store(240, 3),
            eval: 0,
            depth: 8,
            bound: Bound::Lower,
            generation: 1,
        };
        let enabled = SearchHeuristics::phase9_default().with_singular_extensions(true);
        let expected = super::SingularProbe {
            excluded_move: tt_move,
            depth: 3,
            beta: 224,
        };
        assert_eq!(
            singular_probe(enabled, None, 8, 3, Some(hit), Some(tt_move)),
            Some(expected)
        );

        let disabled = SearchHeuristics::phase9_default();
        assert!(singular_probe(disabled, None, 8, 3, Some(hit), Some(tt_move)).is_none());
        assert!(singular_probe(enabled, None, 8, 0, Some(hit), Some(tt_move)).is_none());
        assert!(singular_probe(enabled, None, 7, 3, Some(hit), Some(tt_move)).is_none());
        assert!(singular_probe(enabled, None, 8, 3, Some(hit), None).is_none());
        assert!(
            singular_probe(
                enabled,
                None,
                8,
                3,
                Some(hit),
                Some(Move::new(square("d2"), square("d4"))),
            )
            .is_none()
        );
        assert!(singular_probe(enabled, Some(tt_move), 8, 3, Some(hit), Some(tt_move),).is_none());
        assert!(
            singular_probe(
                enabled,
                None,
                12,
                3,
                Some(TtHit { depth: 8, ..hit }),
                Some(tt_move),
            )
            .is_none()
        );
        assert!(
            singular_probe(
                enabled,
                None,
                8,
                3,
                Some(TtHit {
                    bound: Bound::Upper,
                    ..hit
                }),
                Some(tt_move),
            )
            .is_none()
        );

        for protected_score in [
            crate::search::tablebase::TABLEBASE_SCORE_BAND,
            -crate::search::tablebase::TABLEBASE_SCORE_BAND,
            super::MATE_THRESHOLD,
            -super::MATE_THRESHOLD,
        ] {
            assert!(
                singular_probe(
                    enabled,
                    None,
                    8,
                    3,
                    Some(TtHit {
                        score: normalize_score_for_store(protected_score, 3),
                        ..hit
                    }),
                    Some(tt_move),
                )
                .is_none(),
                "protected score {protected_score} must not drive singular search"
            );
        }
    }

    #[test]
    fn forced_line_tt_evidence_triggers_one_ply_singular_extension() {
        let heuristics = SearchHeuristics::phase9_default().with_singular_extensions(true);
        let mut expected = None;
        for _ in 0..2 {
            let mut context = SearchContext::new(
                SearchLimits::new(9)
                    .with_hash_mb(1)
                    .with_heuristics(heuristics),
            );
            let mut position = Position::from_fen("8/8/8/8/8/4k3/7P/6RK w - - 0 1")
                .expect("forced-line FEN must parse");
            let mut legal_moves = MoveList::new();
            position.generate_legal_moves(&mut legal_moves);
            let tt_move = legal_moves
                .iter()
                .copied()
                .find(|mv| mv.to_string() == "h2h3")
                .expect("test TT move must be legal");
            context.store_tt(super::TtStoreInput {
                key: position.search_key(),
                depth: 7,
                ply: 1,
                best_move: tt_move,
                static_eval: 0,
                score: 1_200,
                bound: Bound::Exact,
            });

            let score = context
                .alpha_beta(
                    &mut position,
                    8,
                    1,
                    -super::INF,
                    super::INF,
                    SearchNodeState::new(true),
                )
                .expect("singular forced-line search must complete");
            let counters = context.debug_counters();
            assert!(counters.singular_verifications > 0);
            assert!(counters.singular_extensions > 0);
            assert!(counters.singular_extensions <= counters.singular_verifications);
            let observed = (score, context.nodes, counters);
            if let Some(expected) = expected {
                assert_eq!(
                    observed, expected,
                    "T1 singular search must be deterministic"
                );
            } else {
                expected = Some(observed);
            }
        }
    }

    #[test]
    fn exclusion_search_never_reuses_or_overwrites_the_current_position_tt_entry() {
        let heuristics = SearchHeuristics::phase9_default().with_singular_extensions(true);
        let mut context = SearchContext::new(
            SearchLimits::new(8)
                .with_hash_mb(1)
                .with_heuristics(heuristics),
        );
        let mut position = Position::from_fen("8/8/8/8/8/4k3/7P/6RK w - - 0 1")
            .expect("exclusion-search FEN must parse");
        let mut legal_moves = MoveList::new();
        position.generate_legal_moves(&mut legal_moves);
        let excluded = legal_moves
            .iter()
            .copied()
            .find(|mv| mv.to_string() == "h2h3")
            .expect("excluded move must be legal");
        let key = position.search_key();
        context.store_tt(super::TtStoreInput {
            key,
            depth: 7,
            ply: 1,
            best_move: excluded,
            static_eval: 37,
            score: 1_200,
            bound: Bound::Exact,
        });
        let before = context.probe_tt(key).expect("seed TT entry must exist");

        context.set_excluded_move(1, excluded);
        let result = context.alpha_beta(
            &mut position,
            3,
            1,
            1_183,
            1_184,
            SearchNodeState::after_null_move(),
        );
        context.set_excluded_move(1, Move::NONE);
        result.expect("exclusion search must complete");

        assert_eq!(
            context.probe_tt(key),
            Some(before),
            "the incomplete move set must not publish a bound for the full position"
        );
    }

    #[test]
    fn search_context_can_be_constructed_with_and_without_tt() {
        let _with_tt = SearchContext::new(SearchLimits::new(3));
        let _without_tt = SearchContext::new(SearchLimits::new(3).without_tt());
    }

    #[test]
    fn pv_move_hint_outranks_tt_move_hint() {
        let mut position = Position::startpos();
        let mut legal_moves = MoveList::new();
        position.generate_legal_moves(&mut legal_moves);
        let pv_move = legal_moves
            .iter()
            .copied()
            .find(|mv| mv.matches_parsed(ParsedMove::parse("e2e4").expect("parse must succeed")))
            .expect("pv move must exist");
        let tt_move = legal_moves
            .iter()
            .copied()
            .find(|mv| mv.matches_parsed(ParsedMove::parse("d2d4").expect("parse must succeed")))
            .expect("tt move must exist");

        let context = SearchContext::new(SearchLimits::new(3));
        let pv_score = context.score_move(
            &position,
            pv_move,
            MoveOrderHints {
                ply: 0,
                quiescence_only: false,
                pv_move: Some(pv_move),
                tt_move: Some(tt_move),
            },
        );
        let tt_score = context.score_move(
            &position,
            tt_move,
            MoveOrderHints {
                ply: 0,
                quiescence_only: false,
                pv_move: Some(pv_move),
                tt_move: Some(tt_move),
            },
        );

        assert!(pv_score > tt_score);
    }

    #[test]
    fn killer_and_history_quiets_outrank_plain_quiets() {
        let mut position = Position::startpos();
        let mut legal_moves = MoveList::new();
        position.generate_legal_moves(&mut legal_moves);
        let killer = legal_moves
            .iter()
            .copied()
            .find(|mv| mv.matches_parsed(ParsedMove::parse("g1f3").expect("parse must succeed")))
            .expect("killer move must exist");
        let history = legal_moves
            .iter()
            .copied()
            .find(|mv| mv.matches_parsed(ParsedMove::parse("b1c3").expect("parse must succeed")))
            .expect("history move must exist");
        let plain = legal_moves
            .iter()
            .copied()
            .find(|mv| mv.matches_parsed(ParsedMove::parse("a2a3").expect("parse must succeed")))
            .expect("plain move must exist");

        let mut context = SearchContext::new(SearchLimits::new(3));
        context.killer_moves[0][0] = killer;
        context.quiet_history[position.side_to_move().index()][history.from().index()]
            [history.to().index()] = 4_000;

        let hints = MoveOrderHints {
            ply: 0,
            quiescence_only: false,
            pv_move: None,
            tt_move: None,
        };
        let killer_score = context.score_move(&position, killer, hints);
        let history_score = context.score_move(&position, history, hints);
        let plain_score = context.score_move(&position, plain, hints);

        assert!(killer_score > history_score);
        assert!(history_score > plain_score);
    }

    #[test]
    fn continuation_history_boosts_matching_reply() {
        let mut position = Position::startpos();
        let mut initial_moves = MoveList::new();
        position.generate_legal_moves(&mut initial_moves);
        let previous_move = initial_moves
            .iter()
            .copied()
            .find(|mv| mv.matches_parsed(ParsedMove::parse("e2e4").expect("parse must succeed")))
            .expect("previous move must exist");
        position
            .make_move(previous_move)
            .expect("previous move must be legal");

        let mut legal_moves = MoveList::new();
        position.generate_legal_moves(&mut legal_moves);
        let boosted = legal_moves
            .iter()
            .copied()
            .find(|mv| mv.matches_parsed(ParsedMove::parse("g8f6").expect("parse must succeed")))
            .expect("boosted move must exist");
        let plain = legal_moves
            .iter()
            .copied()
            .find(|mv| mv.matches_parsed(ParsedMove::parse("a7a6").expect("parse must succeed")))
            .expect("plain move must exist");

        let mut context = SearchContext::new(SearchLimits::new(3));
        context.previous_moves[1] = previous_move;
        let moved_piece = position
            .piece_at(boosted.from())
            .expect("boosted move piece must exist");
        let entry = &mut context.continuation_history[position.side_to_move().index()]
            [crate::core::PieceType::Pawn.index()][square("e4").index()]
            [moved_piece.piece_type().index()][boosted.to().index()];
        *entry = 4_000;

        let hints = MoveOrderHints {
            ply: 1,
            quiescence_only: false,
            pv_move: None,
            tt_move: None,
        };
        let boosted_score = context.score_move(&position, boosted, hints);
        let plain_score = context.score_move(&position, plain, hints);

        assert!(boosted_score > plain_score);
    }

    #[test]
    fn previous_iteration_pv_hint_extends_below_root_when_prefix_matches() {
        let mut root = Position::startpos();
        let mut root_moves = MoveList::new();
        root.generate_legal_moves(&mut root_moves);
        let e2e4 = root_moves
            .iter()
            .copied()
            .find(|mv| mv.matches_parsed(ParsedMove::parse("e2e4").expect("parse must succeed")))
            .expect("e2e4 must exist");

        root.make_move(e2e4).expect("e2e4 must be legal");
        let mut after_e4_moves = MoveList::new();
        root.generate_legal_moves(&mut after_e4_moves);
        let e7e5 = after_e4_moves
            .iter()
            .copied()
            .find(|mv| mv.matches_parsed(ParsedMove::parse("e7e5").expect("parse must succeed")))
            .expect("e7e5 must exist");

        root.make_move(e7e5).expect("e7e5 must be legal");
        let mut after_e4e5_moves = MoveList::new();
        root.generate_legal_moves(&mut after_e4e5_moves);
        let g1f3 = after_e4e5_moves
            .iter()
            .copied()
            .find(|mv| mv.matches_parsed(ParsedMove::parse("g1f3").expect("parse must succeed")))
            .expect("g1f3 must exist");

        let mut context = SearchContext::new(SearchLimits::new(4));
        context.previous_iteration_pv[0] = e2e4;
        context.previous_iteration_pv[1] = e7e5;
        context.previous_iteration_pv[2] = g1f3;
        context.previous_iteration_pv_length = 3;

        assert_eq!(context.previous_pv_move(0), Some(e2e4));

        context.previous_moves[1] = e2e4;
        assert_eq!(context.previous_pv_move(1), Some(e7e5));

        context.previous_moves[2] = e7e5;
        assert_eq!(context.previous_pv_move(2), Some(g1f3));
    }

    #[test]
    fn previous_iteration_pv_hint_is_withheld_on_prefix_mismatch() {
        let mut position = Position::startpos();
        let mut legal_moves = MoveList::new();
        position.generate_legal_moves(&mut legal_moves);
        let e2e4 = legal_moves
            .iter()
            .copied()
            .find(|mv| mv.matches_parsed(ParsedMove::parse("e2e4").expect("parse must succeed")))
            .expect("e2e4 must exist");
        let d2d4 = legal_moves
            .iter()
            .copied()
            .find(|mv| mv.matches_parsed(ParsedMove::parse("d2d4").expect("parse must succeed")))
            .expect("d2d4 must exist");
        let e2e3 = legal_moves
            .iter()
            .copied()
            .find(|mv| mv.matches_parsed(ParsedMove::parse("e2e3").expect("parse must succeed")))
            .expect("placeholder move must exist");

        let mut context = SearchContext::new(SearchLimits::new(4));
        context.previous_iteration_pv[0] = e2e4;
        context.previous_iteration_pv[1] = e2e3;
        context.previous_iteration_pv_length = 2;
        context.previous_moves[1] = d2d4;

        assert_eq!(context.previous_pv_move(1), None);
    }

    #[test]
    fn capture_buckets_prefer_non_losing_captures() {
        let position = Position::from_fen("4k3/8/8/5r1q/3N4/8/4p3/4K3 w - - 0 1")
            .expect("FEN parse must succeed");
        let winning = Move::new(square("d4"), square("f5")).with_flags(FLAG_CAPTURE);
        let losing = Move::new(square("d4"), square("e2")).with_flags(FLAG_CAPTURE);

        let context = SearchContext::new(SearchLimits::new(3));
        assert!(
            context.capture_order_score(&position, winning)
                > context.capture_order_score(&position, losing)
        );
    }

    #[test]
    fn lmr_eligibility_respects_locked_guards() {
        let quiet = Move::new(square("a2"), square("a3"));
        let capture = quiet.with_flags(FLAG_CAPTURE);
        let promotion =
            Move::new(square("a7"), square("a8")).with_promotion(crate::core::PieceType::Queen);

        assert!(lmr_is_eligible(
            SearchHeuristics::phase8_baseline().with_late_move_reductions(true),
            LmrCandidate {
                depth: 5,
                is_pv: false,
                in_check: false,
                mv: quiet,
                gives_check: false,
                is_hash_move: false,
                quiets_searched: 4,
            },
        ));
        assert!(!lmr_is_eligible(
            SearchHeuristics::phase8_baseline().with_late_move_reductions(true),
            LmrCandidate {
                depth: 3,
                is_pv: false,
                in_check: false,
                mv: quiet,
                gives_check: false,
                is_hash_move: false,
                quiets_searched: 4,
            },
        ));
        assert!(!lmr_is_eligible(
            SearchHeuristics::phase8_baseline().with_late_move_reductions(true),
            LmrCandidate {
                depth: 4,
                is_pv: true,
                in_check: false,
                mv: quiet,
                gives_check: false,
                is_hash_move: false,
                quiets_searched: 4,
            },
        ));
        assert!(!lmr_is_eligible(
            SearchHeuristics::phase8_baseline().with_late_move_reductions(true),
            LmrCandidate {
                depth: 4,
                is_pv: false,
                in_check: true,
                mv: quiet,
                gives_check: false,
                is_hash_move: false,
                quiets_searched: 4,
            },
        ));
        assert!(!lmr_is_eligible(
            SearchHeuristics::phase8_baseline().with_late_move_reductions(true),
            LmrCandidate {
                depth: 4,
                is_pv: false,
                in_check: false,
                mv: capture,
                gives_check: false,
                is_hash_move: false,
                quiets_searched: 4,
            },
        ));
        assert!(!lmr_is_eligible(
            SearchHeuristics::phase8_baseline().with_late_move_reductions(true),
            LmrCandidate {
                depth: 4,
                is_pv: false,
                in_check: false,
                mv: promotion,
                gives_check: false,
                is_hash_move: false,
                quiets_searched: 4,
            },
        ));
        assert!(!lmr_is_eligible(
            SearchHeuristics::phase8_baseline().with_late_move_reductions(true),
            LmrCandidate {
                depth: 4,
                is_pv: false,
                in_check: false,
                mv: quiet,
                gives_check: true,
                is_hash_move: false,
                quiets_searched: 4,
            },
        ));
        assert!(!lmr_is_eligible(
            SearchHeuristics::phase8_baseline().with_late_move_reductions(true),
            LmrCandidate {
                depth: 4,
                is_pv: false,
                in_check: false,
                mv: quiet,
                gives_check: false,
                is_hash_move: true,
                quiets_searched: 4,
            },
        ));
        assert!(!lmr_is_eligible(
            SearchHeuristics::phase8_baseline().with_late_move_reductions(true),
            LmrCandidate {
                depth: 4,
                is_pv: false,
                in_check: false,
                mv: quiet,
                gives_check: false,
                is_hash_move: false,
                quiets_searched: 2,
            },
        ));
    }

    #[test]
    fn lmr_reduction_scales_for_deeper_late_quiets() {
        assert_eq!(lmr_reduction(4, 3), 1);
        assert_eq!(lmr_reduction(7, 5), 2);
        assert_eq!(lmr_reduction(10, 9), 3);
    }

    #[test]
    fn lmr_reduces_late_quiets_and_researches_on_alpha_improvement() {
        let mut position = Position::startpos();
        let limits = SearchLimits::new(7)
            .with_heuristics(SearchHeuristics::phase8_baseline().with_late_move_reductions(true));
        let mut context = SearchContext::new(limits);

        let _ = context
            .alpha_beta(&mut position, 6, 1, -20, 20, SearchNodeState::new(false))
            .expect("search must complete");

        assert!(context.debug_counters().lmr_reductions > 0);
        assert!(context.debug_counters().lmr_researches > 0);
    }

    #[test]
    fn pvs_uses_scout_windows_and_full_researches() {
        let mut position = Position::startpos();
        let mut context = SearchContext::new(SearchLimits::new(5));

        let _ = context
            .alpha_beta(&mut position, 4, 1, -20, 20, SearchNodeState::new(true))
            .expect("search must complete");

        assert!(context.debug_counters().pvs_scout_searches > 0);
        assert!(context.debug_counters().pvs_full_researches > 0);
    }

    #[test]
    fn lmr_alpha_raise_requires_full_research() {
        assert!(!lmr_requires_full_research(20, 20));
        assert!(lmr_requires_full_research(21, 20));
    }

    #[test]
    fn reverse_futility_pruning_respects_core_guards() {
        let position = Position::startpos();
        let heuristics = SearchHeuristics::phase8_baseline().with_reverse_futility_pruning(true);

        assert!(reverse_futility_is_eligible(
            heuristics,
            &position,
            SearchNodeState::new(false),
            2,
            32,
            320,
            false,
        ));
        assert!(!reverse_futility_is_eligible(
            heuristics,
            &position,
            SearchNodeState::new(true),
            2,
            32,
            256,
            false,
        ));
        assert!(!reverse_futility_is_eligible(
            heuristics,
            &position,
            SearchNodeState::new(false),
            4,
            32,
            256,
            false,
        ));
        assert!(!reverse_futility_is_eligible(
            heuristics,
            &position,
            SearchNodeState::new(false),
            2,
            32,
            100,
            false,
        ));
        assert!(!reverse_futility_is_eligible(
            heuristics,
            &position,
            SearchNodeState::new(false),
            2,
            32,
            256,
            true,
        ));
    }

    #[test]
    fn futility_pruning_respects_core_guards() {
        let heuristics = SearchHeuristics::phase8_baseline().with_futility_pruning(true);
        let quiet = Move::new(square("a2"), square("a3"));
        let capture = quiet.with_flags(FLAG_CAPTURE);

        assert!(futility_pruning_is_eligible(
            heuristics,
            ForwardPruneCandidate::quiet(2, 400, 0, quiet),
        ));
        assert!(!futility_pruning_is_eligible(
            heuristics,
            ForwardPruneCandidate {
                node_state: SearchNodeState::new(true),
                ..ForwardPruneCandidate::quiet(2, 300, 0, quiet)
            },
        ));
        assert!(!futility_pruning_is_eligible(
            heuristics,
            ForwardPruneCandidate::quiet(4, 600, 0, quiet),
        ));
        assert!(!futility_pruning_is_eligible(
            heuristics,
            ForwardPruneCandidate::quiet(2, 300, 0, capture),
        ));
        assert!(!futility_pruning_is_eligible(
            heuristics,
            ForwardPruneCandidate {
                has_searched_move: false,
                ..ForwardPruneCandidate::quiet(2, 300, 0, quiet)
            },
        ));
    }

    #[test]
    fn late_move_pruning_respects_core_guards() {
        let heuristics = SearchHeuristics::phase8_baseline().with_late_move_pruning(true);
        let quiet = Move::new(square("a2"), square("a3"));

        assert!(late_move_pruning_is_eligible(
            heuristics,
            ForwardPruneCandidate {
                quiets_searched: 17,
                ..ForwardPruneCandidate::quiet(2, 400, 0, quiet)
            },
        ));
        assert!(!late_move_pruning_is_eligible(
            heuristics,
            ForwardPruneCandidate {
                node_state: SearchNodeState::new(true),
                quiets_searched: 16,
                ..ForwardPruneCandidate::quiet(2, 400, 0, quiet)
            },
        ));
        assert!(!late_move_pruning_is_eligible(
            heuristics,
            ForwardPruneCandidate {
                quiets_searched: 16,
                ..ForwardPruneCandidate::quiet(3, 400, 0, quiet)
            },
        ));
        assert!(!late_move_pruning_is_eligible(
            heuristics,
            ForwardPruneCandidate {
                quiets_searched: 8,
                ..ForwardPruneCandidate::quiet(2, 300, 0, quiet)
            },
        ));
        assert!(!late_move_pruning_is_eligible(
            heuristics,
            ForwardPruneCandidate {
                is_hash_move: true,
                quiets_searched: 16,
                ..ForwardPruneCandidate::quiet(2, 400, 0, quiet)
            },
        ));
    }

    #[test]
    fn null_move_pruning_respects_core_guards() {
        let mut position = Position::startpos();
        let heuristics = SearchHeuristics::phase8_baseline().with_null_move_pruning(true);
        let eval = 64;

        assert!(null_move_is_eligible(
            heuristics,
            &position,
            SearchNodeState::new(false),
            4,
            32,
            eval,
            false,
        ));
        assert!(!null_move_is_eligible(
            heuristics,
            &position,
            SearchNodeState::new(true),
            4,
            32,
            eval,
            false,
        ));
        assert!(!null_move_is_eligible(
            heuristics,
            &position,
            SearchNodeState::after_null_move(),
            4,
            32,
            eval,
            false,
        ));
        assert!(!null_move_is_eligible(
            heuristics,
            &position,
            SearchNodeState::new(false),
            2,
            32,
            eval,
            false,
        ));
        assert!(!null_move_is_eligible(
            heuristics,
            &position,
            SearchNodeState::new(false),
            4,
            32,
            16,
            false,
        ));

        position =
            Position::from_fen("8/8/8/8/3k4/8/4p3/3K4 b - - 0 1").expect("FEN parse must succeed");
        assert!(!null_move_is_eligible(
            heuristics,
            &position,
            SearchNodeState::new(false),
            4,
            32,
            eval,
            false,
        ));
    }

    #[test]
    fn null_move_reduction_grows_with_depth() {
        assert_eq!(null_move_reduction(4, 100, 100), 2);
        assert_eq!(null_move_reduction(6, 100, 100), 3);
        assert_eq!(null_move_reduction(7, 100, 100), 3);
        assert_eq!(null_move_reduction(7, 356, 100), 4);
        assert_eq!(null_move_reduction(7, 612, 100), 5);
        assert_eq!(null_move_reduction(12, 100, 100), 4);
    }

    #[test]
    fn multi_cut_respects_cut_node_depth_recursion_and_material_guards() {
        let position = Position::startpos();
        let heuristics = SearchHeuristics::phase9_default().with_multi_cut(true);

        assert!(multi_cut_is_eligible(
            heuristics,
            &position,
            SearchNodeState::cut(),
            MULTI_CUT_MIN_DEPTH,
            20,
            0,
            false,
        ));
        assert!(!multi_cut_is_eligible(
            SearchHeuristics::phase9_default(),
            &position,
            SearchNodeState::cut(),
            MULTI_CUT_MIN_DEPTH,
            20,
            0,
            false,
        ));
        assert!(!multi_cut_is_eligible(
            heuristics,
            &position,
            SearchNodeState::new(true),
            MULTI_CUT_MIN_DEPTH,
            20,
            0,
            false,
        ));
        assert!(!multi_cut_is_eligible(
            heuristics,
            &position,
            SearchNodeState::after_null_move(),
            MULTI_CUT_MIN_DEPTH,
            20,
            0,
            false,
        ));
        assert!(!multi_cut_is_eligible(
            heuristics,
            &position,
            SearchNodeState::cut(),
            MULTI_CUT_MIN_DEPTH - 1,
            20,
            0,
            false,
        ));
        assert!(!multi_cut_is_eligible(
            heuristics,
            &position,
            SearchNodeState::cut(),
            MULTI_CUT_MIN_DEPTH,
            20,
            0,
            true,
        ));

        let pawn_ending = Position::from_fen("8/8/8/8/3k4/8/4p3/3K4 b - - 0 1")
            .expect("pawn-ending FEN must parse");
        assert!(!multi_cut_is_eligible(
            heuristics,
            &pawn_ending,
            SearchNodeState::cut(),
            MULTI_CUT_MIN_DEPTH,
            -500,
            -500,
            false,
        ));
    }

    #[test]
    fn multi_cut_isolated_seam_prunes_deterministically_and_restores_position() {
        let heuristics = SearchHeuristics::phase8_baseline().with_multi_cut(true);
        let mut expected = None;

        for _ in 0..2 {
            let limits = SearchLimits::new(8)
                .without_tt()
                .with_heuristics(heuristics);
            let mut context = SearchContext::new(limits);
            let mut position = Position::startpos();
            let before = position.to_fen();
            let score = context
                .alpha_beta(
                    &mut position,
                    MULTI_CUT_MIN_DEPTH,
                    1,
                    -1_001,
                    -1_000,
                    SearchNodeState::cut(),
                )
                .expect("multi-cut search must complete");
            let counters = context.debug_counters();

            assert_eq!(position.to_fen(), before);
            assert!(counters.multi_cut_attempts > 0);
            assert!(counters.multi_cut_probes >= MULTI_CUT_REQUIRED_CUTOFFS as u32);
            assert!(counters.multi_cut_prunes > 0);
            let observed = (score, context.nodes, counters);
            if let Some(expected) = expected {
                assert_eq!(observed, expected, "T1 Multi-Cut must be deterministic");
            } else {
                expected = Some(observed);
            }
        }
    }

    #[test]
    fn deep_null_move_cutoffs_require_verification() {
        assert!(!null_move_requires_verification(9));
        assert!(null_move_requires_verification(10));
        assert!(null_move_requires_verification(16));
    }

    #[test]
    fn reverse_futility_pruning_triggers_in_search() {
        let mut position =
            Position::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1").expect("FEN parse must succeed");
        let limits = SearchLimits::new(3).with_heuristics(
            SearchHeuristics::phase8_baseline().with_reverse_futility_pruning(true),
        );
        let mut context = SearchContext::new(limits);

        let _ = context
            .alpha_beta(&mut position, 2, 1, -50, 50, SearchNodeState::new(false))
            .expect("search must complete");

        assert!(context.debug_counters().reverse_futility_prunes > 0);
    }

    #[test]
    fn futility_pruning_triggers_in_search() {
        let mut position = Position::startpos();
        let limits = SearchLimits::new(3)
            .with_heuristics(SearchHeuristics::phase8_baseline().with_futility_pruning(true));
        let mut context = SearchContext::new(limits);

        let _ = context
            .alpha_beta(&mut position, 2, 1, 400, 420, SearchNodeState::new(false))
            .expect("search must complete");

        assert!(context.debug_counters().futility_prunes > 0);
    }

    #[test]
    fn late_move_pruning_triggers_in_search() {
        let mut position = Position::startpos();
        let limits = SearchLimits::new(4)
            .with_heuristics(SearchHeuristics::phase8_baseline().with_late_move_pruning(true));
        let mut context = SearchContext::new(limits);

        let _ = context
            .alpha_beta(&mut position, 3, 1, 400, 420, SearchNodeState::new(false))
            .expect("search must complete");

        assert!(context.debug_counters().late_move_prunes > 0);
    }

    #[test]
    fn tablebase_non_root_wdl_substitution_uses_dedicated_score_band() {
        let fen = "8/8/8/8/8/3Q4/2K5/k7 w - - 0 1";
        let tablebases = TablebaseService::from_backend_for_tests(
            "/mock",
            Arc::new(MockTablebaseBackend::new().with_wdl_probe(fen, WdlOutcome::Win)),
        );
        let mut position = Position::from_fen(fen).expect("FEN parse must succeed");
        let mut context = SearchContext::with_tt(
            SearchLimits::new(2),
            SearchResources {
                tablebases: Some(tablebases),
                ..SearchResources::default()
            },
            super::SearchControl::default(),
            None,
        );

        let score = context
            .alpha_beta(
                &mut position,
                2,
                3,
                -super::INF,
                super::INF,
                SearchNodeState::new(false),
            )
            .expect("tablebase substitution must return a score");

        assert_eq!(score, tablebase::score_from_wdl(WdlOutcome::Win, 3));
    }

    #[test]
    fn tablebase_probe_does_not_override_direct_fifty_move_draw() {
        let fen = "8/8/8/8/8/3Q4/2K5/k7 w - - 100 1";
        let tablebases = TablebaseService::from_backend_for_tests(
            "/mock",
            Arc::new(MockTablebaseBackend::new().with_wdl_probe(fen, WdlOutcome::Win)),
        );
        let mut position = Position::from_fen(fen).expect("FEN parse must succeed");
        let result = super::search_with_control(
            &mut position,
            SearchLimits::new(2),
            SearchResources {
                tablebases: Some(tablebases),
                ..SearchResources::default()
            },
            super::SearchControl::default(),
            None,
        );

        assert_eq!(result.score.0, 0);
    }

    #[test]
    fn mate_distance_bounds_tighten_monotonically_with_ply() {
        assert_eq!(
            mate_distance_bounds(0),
            (-super::MATE_SCORE, super::MATE_SCORE - 1)
        );
        assert_eq!(
            mate_distance_bounds(7),
            (-super::MATE_SCORE + 7, super::MATE_SCORE - 8)
        );
    }

    #[test]
    fn contextual_lmr_uses_node_and_history_evidence() {
        let base = contextual_lmr_reduction(10, 12, false, false, 0);
        assert!(contextual_lmr_reduction(10, 12, false, true, -5_000) > base);
        assert!(contextual_lmr_reduction(10, 12, true, false, 5_000) < base);
    }

    #[test]
    fn bounded_history_update_learns_successes_and_failures() {
        let mut entry = 0i16;
        update_history_entry(&mut entry, 1_000);
        assert!(entry > 0);
        update_history_entry(&mut entry, -2_000);
        assert!(entry < 0);
        for _ in 0..100 {
            update_history_entry(&mut entry, super::HISTORY_MAX);
        }
        assert_eq!(entry, super::HISTORY_MAX as i16);
    }

    #[test]
    fn internal_iterative_reduction_triggers_without_a_hash_move() {
        let mut position = Position::startpos();
        let mut context = SearchContext::new(SearchLimits::new(8));
        let _ = context
            .alpha_beta(&mut position, 7, 1, -50, 50, SearchNodeState::new(false))
            .expect("search must complete");
        assert!(context.debug_counters().internal_iterative_reductions > 0);
    }

    #[test]
    fn completed_search_reports_selective_depth() {
        let mut position = Position::startpos();
        let result = super::search(&mut position, SearchLimits::new(3));
        assert!(result.seldepth >= result.depth);
        assert!(
            result
                .info_lines
                .iter()
                .all(|line| line.contains(" seldepth "))
        );
    }

    #[test]
    fn node_limit_is_exact_and_preserves_a_legal_fallback() {
        let mut position = Position::startpos();
        let before = position.to_fen();
        let result = super::search(
            &mut position,
            SearchLimits::new(127).with_node_limit(Some(257)),
        );
        assert_eq!(result.nodes, 257);
        assert!(result.best_move.is_some());
        assert_eq!(position.to_fen(), before);

        let result = super::search(
            &mut position,
            SearchLimits::new(127).with_node_limit(Some(0)),
        );
        assert_eq!(result.nodes, 0);
        assert!(result.best_move.is_some());
        assert_eq!(position.to_fen(), before);
    }

    #[test]
    fn iteration_stability_adapts_soft_budget_without_overreacting() {
        let e2e4 = Move::new(square("e2"), square("e4"));
        let d2d4 = Move::new(square("d2"), square("d4"));
        let mut stability = IterationStability::default();
        assert_eq!(stability.soft_budget_factor(), 1.0);
        stability.record(Some(e2e4), 12);
        assert_eq!(stability.soft_budget_factor(), 1.0);
        stability.record(Some(e2e4), 18);
        assert_eq!(stability.soft_budget_factor(), 0.95);
        stability.record(Some(e2e4), 20);
        assert_eq!(stability.soft_budget_factor(), 0.82);
        stability.record(Some(e2e4), 19);
        assert_eq!(stability.soft_budget_factor(), 0.70);

        stability.record(Some(d2d4), 22);
        assert_eq!(stability.soft_budget_factor(), 1.25);
        stability.record(Some(e2e4), 150);
        assert_eq!(stability.soft_budget_factor(), 1.45);
    }

    #[test]
    fn clock_search_uses_iteration_cost_but_fixed_movetime_does_not_stop_early() {
        let now = Instant::now();
        let stable_move = Move::new(square("e2"), square("e4"));
        let mut stable = IterationStability::default();
        for score in [10, 11, 12, 13] {
            stable.record(Some(stable_move), score);
        }
        assert_eq!(stable.soft_budget_factor(), 0.70);

        let mut fixed = SearchContext::new(SearchLimits::new(10));
        fixed.started = now;
        fixed.control.soft_deadline = Some(now + std::time::Duration::from_secs(10));
        fixed.control.hard_deadline = fixed.control.soft_deadline;
        assert!(
            !fixed.adaptive_soft_stop_requested(stable, Some(std::time::Duration::from_secs(6)),)
        );

        let mut clocked = SearchContext::new(SearchLimits::new(10));
        clocked.started = now;
        clocked.control.soft_deadline = Some(now + std::time::Duration::from_secs(10));
        clocked.control.hard_deadline = Some(now + std::time::Duration::from_secs(15));
        assert!(clocked.adaptive_soft_stop_requested(
            IterationStability::default(),
            Some(std::time::Duration::from_secs(6)),
        ));
    }

    #[test]
    fn ponder_suspends_deadlines_until_hit_then_starts_the_full_budget() {
        let ponder_started = Instant::now();
        let soft_budget = std::time::Duration::from_secs(2);
        let hard_budget = std::time::Duration::from_secs(3);
        let ponder = Arc::new(PonderState::new(ponder_started));
        let mut context = SearchContext::new(SearchLimits::new(10));
        context.control.soft_deadline = Some(ponder_started + soft_budget);
        context.control.hard_deadline = Some(ponder_started + hard_budget);
        context.control.ponder_state = Some(Arc::clone(&ponder));

        assert_eq!(
            context.effective_deadline(context.control.soft_deadline),
            None
        );
        assert_eq!(
            context.effective_deadline(context.control.hard_deadline),
            None
        );
        assert!(!context.hard_stop_requested());

        let hit_at = ponder_started + std::time::Duration::from_secs(10);
        ponder.hit(hit_at);
        assert_eq!(
            context.effective_deadline(context.control.soft_deadline),
            Some(hit_at + soft_budget)
        );
        assert_eq!(
            context.effective_deadline(context.control.hard_deadline),
            Some(hit_at + hard_budget)
        );
    }

    #[test]
    #[ignore = "manual before/after report for modern search selectivity"]
    fn modern_search_profile_report() {
        let fens = [
            crate::core::STARTPOS_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 2 3",
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        ];
        let baseline = SearchHeuristics::phase9_default().with_modern_search(false);
        let mut qsearch_tt = baseline;
        qsearch_tt.qsearch_tt = true;
        let mut tt_static_eval = baseline;
        tt_static_eval.tt_static_eval = true;
        let mut capture_history = baseline;
        capture_history.capture_history = true;
        let mut iir = baseline;
        iir.internal_iterative_reduction = true;
        let mut see_pruning = baseline;
        see_pruning.see_pruning = true;
        let mut history_pruning = baseline;
        history_pruning.history_pruning = true;
        let mut probcut = baseline;
        probcut.probcut = true;
        let mut history_maluses = baseline;
        history_maluses.history_maluses = true;
        let mut contextual_lmr = baseline;
        contextual_lmr.contextual_lmr = true;
        let profiles = [
            ("baseline", baseline),
            ("qsearch_tt", qsearch_tt),
            ("tt_static_eval", tt_static_eval),
            ("capture_history", capture_history),
            ("iir", iir),
            ("see_pruning", see_pruning),
            ("history_pruning", history_pruning),
            ("probcut", probcut),
            ("history_maluses", history_maluses),
            ("contextual_lmr", contextual_lmr),
            ("all", SearchHeuristics::phase9_default()),
        ];
        for (name, heuristics) in profiles {
            let started = Instant::now();
            let mut nodes = 0u64;
            let mut depth_sum = 0u64;
            for fen in fens {
                let mut position = Position::from_fen(fen).expect("bench FEN must parse");
                let limits = SearchLimits::new(7).with_heuristics(heuristics);
                let result = super::search(&mut position, limits);
                nodes += result.nodes;
                depth_sum += result.seldepth as u64;
            }
            println!(
                "modern_search {name}: nodes {nodes} seldepth_sum {depth_sum} time_ms {}",
                started.elapsed().as_millis()
            );
        }
    }

    #[test]
    #[ignore = "manual isolated singular-extension A/B profile report"]
    fn singular_extension_profile_report() {
        let fens = [
            crate::core::STARTPOS_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 2 3",
            "2kr3r/ppp2ppp/2n1bn2/2b1p3/4P3/2NP1N2/PPP2PPP/R1B2RK1 b - - 0 9",
        ];
        for (name, heuristics) in [
            ("singular_off", SearchHeuristics::phase9_default()),
            (
                "singular_on",
                SearchHeuristics::phase9_default().with_singular_extensions(true),
            ),
        ] {
            let started = Instant::now();
            let mut nodes = 0u64;
            let mut checksum = 0u64;
            let mut verifications = 0u64;
            let mut extensions = 0u64;
            for fen in fens {
                let mut position = Position::from_fen(fen).expect("bench FEN must parse");
                let limits = SearchLimits::new(9).with_heuristics(heuristics);
                let mut context = SearchContext::new(limits);
                let result = context.run(&mut position, limits);
                nodes = nodes.saturating_add(result.nodes);
                checksum = checksum.rotate_left(11)
                    ^ result.nodes
                    ^ (result.score.0 as i64 as u64)
                    ^ result.best_move.map_or(0, |mv| mv.raw() as u64);
                let counters = context.debug_counters();
                verifications =
                    verifications.saturating_add(u64::from(counters.singular_verifications));
                extensions = extensions.saturating_add(u64::from(counters.singular_extensions));
                println!(
                    "{name}: fen {fen} best {:?} score {} nodes {} verifications {} extensions {}",
                    result.best_move,
                    result.score.0,
                    result.nodes,
                    counters.singular_verifications,
                    counters.singular_extensions,
                );
            }
            println!(
                "{name}: nodes {nodes} checksum {checksum:016x} verifications {verifications} extensions {extensions} time_ms {}",
                started.elapsed().as_millis()
            );
        }
    }
}
