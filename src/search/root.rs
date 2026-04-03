use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use crate::core::{Move, MoveList, Position, Score, see};

use super::{
    eval,
    limits::{SearchHeuristics, SearchLimits},
    movepicker::MovePicker,
    nnue::{NnueSearchState, NnueService},
    qsearch,
    tablebase::{self, TablebaseService},
    tt::{self, Bound, TtHit, TtStore},
};

pub(crate) const MAX_PLY: usize = 128;
pub(crate) const INF: i32 = 32_000;
pub(crate) const MATE_SCORE: i32 = 30_000;
const MATE_THRESHOLD: i32 = MATE_SCORE - MAX_PLY as i32;
const ASPIRATION_DELTA: i32 = 32;
const HISTORY_MAX: i32 = 16_384;
const PIECE_TYPE_COUNT: usize = 6;

type ContinuationHistory = [[[[[i16; 64]; PIECE_TYPE_COUNT]; 64]; PIECE_TYPE_COUNT]; 2];

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
    pub(crate) role: SearchThreadRole,
}

impl SearchControl {
    fn can_interrupt(&self) -> bool {
        self.stop_flag.is_some()
            || self.helper_stop_flag.is_some()
            || self.soft_deadline.is_some()
            || self.hard_deadline.is_some()
    }
}

pub(crate) struct SearchContext {
    started: Instant,
    pub(crate) nodes: u64,
    tt_hits: u64,
    pv_table: [[Move; MAX_PLY]; MAX_PLY],
    pub(crate) pv_length: [usize; MAX_PLY],
    previous_iteration_pv: [Move; MAX_PLY],
    previous_iteration_pv_length: usize,
    previous_moves: [Move; MAX_PLY],
    killer_moves: [[Move; 2]; MAX_PLY],
    quiet_history: [[[i16; 64]; 64]; 2],
    continuation_history: Box<ContinuationHistory>,
    heuristics: SearchHeuristics,
    control: SearchControl,
    tt: Option<Arc<tt::TranspositionTable>>,
    nnue: Option<NnueSearchState>,
    tablebases: Option<Arc<TablebaseService>>,
    debug_counters: SearchDebugCounters,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct SearchDebugCounters {
    lmr_reductions: u32,
    lmr_researches: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SearchNodeState {
    pub(crate) is_pv: bool,
    pub(crate) null_move_allowed: bool,
}

impl SearchNodeState {
    const fn new(is_pv: bool) -> Self {
        Self {
            is_pv,
            null_move_allowed: true,
        }
    }

    const fn after_null_move() -> Self {
        Self {
            is_pv: false,
            null_move_allowed: false,
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

#[derive(Clone, Copy, Default)]
pub(crate) struct MoveOrderHints {
    pub(crate) ply: usize,
    pub(crate) quiescence_only: bool,
    pub(crate) pv_move: Option<Move>,
    pub(crate) tt_move: Option<Move>,
}

pub fn search(position: &mut Position, limits: SearchLimits) -> SearchResult {
    search_with_control(position, limits, None, None, None, SearchControl::default())
}

pub(crate) fn search_with_control(
    position: &mut Position,
    limits: SearchLimits,
    tt: Option<Arc<tt::TranspositionTable>>,
    nnue: Option<Arc<NnueService>>,
    tablebases: Option<Arc<TablebaseService>>,
    control: SearchControl,
) -> SearchResult {
    SearchContext::with_tt(limits, tt, nnue, tablebases, control).run(position, limits)
}

enum RootSearchOutcome {
    Complete(Option<Move>, i32),
    Aborted(Option<Move>),
}

impl SearchContext {
    #[cfg(test)]
    pub(crate) fn new(limits: SearchLimits) -> Self {
        Self::with_tt(limits, None, None, None, SearchControl::default())
    }

    fn with_tt(
        limits: SearchLimits,
        tt: Option<Arc<tt::TranspositionTable>>,
        nnue: Option<Arc<NnueService>>,
        tablebases: Option<Arc<TablebaseService>>,
        control: SearchControl,
    ) -> Self {
        Self {
            started: Instant::now(),
            nodes: 0,
            tt_hits: 0,
            pv_table: [[Move::NONE; MAX_PLY]; MAX_PLY],
            pv_length: [0; MAX_PLY],
            previous_iteration_pv: [Move::NONE; MAX_PLY],
            previous_iteration_pv_length: 0,
            previous_moves: [Move::NONE; MAX_PLY],
            killer_moves: [[Move::NONE; 2]; MAX_PLY],
            quiet_history: [[[0; 64]; 64]; 2],
            continuation_history: Box::new([[[[[0; 64]; PIECE_TYPE_COUNT]; 64]; PIECE_TYPE_COUNT]; 2]),
            heuristics: limits.heuristics,
            control,
            tt: tt.or_else(|| {
                limits
                    .tt_enabled
                    .then(|| Arc::new(tt::TranspositionTable::new_mb(limits.hash_mb)))
            }),
            nnue: nnue.map(NnueSearchState::new),
            tablebases,
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
        let depth_limit = limits.depth.max(1).min((MAX_PLY - 1) as u8);
        let mut best_move = None;
        let mut best_score = 0i32;
        let mut best_pv = Vec::new();
        let mut completed_depth = 0u8;
        let mut info_lines = Vec::new();
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
            if self.hard_stop_requested() || (completed_depth > 0 && self.soft_stop_requested()) {
                break;
            }
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

            let pv = self.collect_pv(0);
            best_pv = pv.clone();
            self.capture_completed_pv(&pv);
            if self.control.role.is_main() {
                info_lines.push(format_info_line(
                    depth,
                    depth_score,
                    self.nodes,
                    self.started.elapsed().as_millis(),
                    self.tt_hits,
                    &pv,
                ));
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
        if legal_moves.is_empty() {
            return None;
        }

        let root_probe = tablebases
            .probe_root(&probe_position, &legal_moves)
            .ok()??;
        let score = tablebase::score_from_wdl(root_probe.wdl, 0);
        let pv = vec![root_probe.best_move];
        let info_lines = if self.control.role.is_main() {
            vec![format_info_line(
                0,
                score,
                self.nodes,
                self.started.elapsed().as_millis(),
                self.tt_hits,
                &pv,
            )]
        } else {
            Vec::new()
        };

        Some(SearchResult {
            best_move: Some(root_probe.best_move),
            score: Score(score),
            depth: 0,
            nodes: self.nodes,
            pv,
            info_lines,
            tt_hits: self.tt_hits,
        })
    }

    fn search_root_with_aspiration_core<const USE_TABLEBASES: bool, const USE_NNUE: bool>(
        &mut self,
        position: &mut Position,
        depth: usize,
        guess: i32,
    ) -> RootSearchOutcome {
        let mut delta = ASPIRATION_DELTA;
        loop {
            if self.hard_stop_requested() {
                return RootSearchOutcome::Aborted(None);
            }
            let alpha = (guess - delta).max(-INF);
            let beta = (guess + delta).min(INF);
            let RootSearchOutcome::Complete(best_move, score) =
                self.search_root_core::<USE_TABLEBASES, USE_NNUE>(position, depth, alpha, beta)
            else {
                return RootSearchOutcome::Aborted(None);
            };
            if score <= alpha && alpha > -INF {
                delta = (delta * 2).min(INF / 2);
                continue;
            }
            if score >= beta && beta < INF {
                delta = (delta * 2).min(INF / 2);
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

        for mv in MovePicker::new(self, position, &legal_moves, ordering_hints).ordered() {
            if self.hard_stop_requested() {
                return RootSearchOutcome::Aborted(best_move);
            }
            let undo = self
                .make_search_move::<USE_NNUE>(position, mv)
                .expect("root move must be legal during search");
            let child_is_pv = best_move.is_none();
            let Some(score) = self.alpha_beta_core::<USE_TABLEBASES, USE_NNUE>(
                position,
                depth.saturating_sub(1),
                1,
                -beta,
                -alpha,
                SearchNodeState::new(child_is_pv),
            ) else {
                self.unmake_search_move::<USE_NNUE>(position, mv, undo);
                return RootSearchOutcome::Aborted(best_move);
            };
            self.unmake_search_move::<USE_NNUE>(position, mv, undo);
            let score = -score;

            if score > alpha || best_move.is_none() {
                alpha = score;
                best_move = Some(mv);
                self.update_pv(0, mv);
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
            let child_is_pv = best_move.is_none();
            let Some(score) = self.alpha_beta_core::<USE_TABLEBASES, USE_NNUE>(
                position,
                depth.saturating_sub(1),
                1,
                -beta,
                -alpha,
                SearchNodeState::new(child_is_pv),
            ) else {
                self.unmake_search_move::<USE_NNUE>(position, mv, undo);
                return RootSearchOutcome::Aborted(best_move);
            };
            self.unmake_search_move::<USE_NNUE>(position, mv, undo);
            let score = -score;

            if score > alpha || best_move.is_none() {
                alpha = score;
                best_move = Some(mv);
                self.update_pv(0, mv);
            }
        }

        RootSearchOutcome::Complete(best_move, alpha)
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
        depth: usize,
        ply: usize,
        mut alpha: i32,
        beta: i32,
        node_state: SearchNodeState,
    ) -> Option<i32> {
        self.nodes += 1;
        if self.nodes & 1023 == 0 && self.hard_stop_requested() {
            return None;
        }

        if ply >= MAX_PLY - 1 {
            return Some(self.evaluate_position::<USE_NNUE>(position));
        }

        if is_draw(position) {
            return Some(0);
        }

        if USE_TABLEBASES
            && let Some(tablebase_score) = self.try_non_root_tablebase_score(position, ply)
        {
            return Some(tablebase_score);
        }

        let tt_key = position.search_key();
        let alpha_start = alpha;
        let static_eval = self.evaluate_position::<USE_NNUE>(position);

        let tt_hit = self.probe_tt(tt_key);
        if let Some(hit) = tt_hit
            && let Some(cutoff) = tt_cutoff_score(hit, depth, ply, alpha, beta)
        {
            return Some(cutoff);
        }

        if depth == 0 {
            return qsearch::qsearch::<USE_NNUE>(self, position, ply, alpha, beta);
        }

        let in_check = position.is_in_check(position.side_to_move());
        self.pv_length[ply] = 0;
        let pv_move_hint = self.previous_pv_move(ply);
        let tt_move_hint =
            tt_hit.and_then(|hit| (!hit.best_move.is_none()).then_some(hit.best_move));

        let mut legal_moves = MoveList::new();
        position.generate_legal_moves(&mut legal_moves);
        if legal_moves.is_empty() {
            return Some(terminal_score(position, ply));
        }
        let pv_move_hint = validated_move_hint(&legal_moves, pv_move_hint);
        let tt_move_hint = validated_tt_move_hint(&legal_moves, tt_move_hint);

        let ordering_hints = MoveOrderHints {
            ply,
            quiescence_only: false,
            pv_move: pv_move_hint,
            tt_move: tt_move_hint,
        };
        let mut best_move = Move::NONE;
        let mut quiets_searched = 0usize;

        if null_move_is_eligible(
            self.heuristics,
            position,
            node_state,
            depth,
            beta,
            static_eval,
            in_check,
        ) && let Ok(null_undo) = position.make_null_move()
        {
            self.set_previous_move(ply + 1, Move::NONE);
            let reduction = null_move_reduction(depth);
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
                if -score >= beta {
                    self.store_tt(TtStoreInput {
                        key: tt_key,
                        depth: depth.min(u8::MAX as usize) as u8,
                        ply,
                        best_move: Move::NONE,
                        static_eval: static_eval.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                        score: beta,
                        bound: Bound::Lower,
                    });
                    return Some(beta);
                }
            } else {
                return None;
            }
        }

        for mv in MovePicker::new(self, position, &legal_moves, ordering_hints).ordered() {
            let is_quiet = !mv.is_capture() && !mv.is_promotion();
            if is_quiet {
                quiets_searched += 1;
            }

            let undo = self
                .make_search_move::<USE_NNUE>(position, mv)
                .expect("searched move must be legal");
            self.previous_moves[ply + 1] = mv;
            let gives_check = position.is_in_check(position.side_to_move());
            let child_is_pv = node_state.is_pv && best_move.is_none();
            let is_hash_move = tt_move_hint == Some(mv);

            let score_result = if lmr_is_eligible(
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
                let Some(reduced_score) = self.alpha_beta_core::<USE_TABLEBASES, USE_NNUE>(
                    position,
                    depth - 2,
                    ply + 1,
                    -beta,
                    -alpha,
                    SearchNodeState::new(false),
                ) else {
                    self.unmake_search_move::<USE_NNUE>(position, mv, undo);
                    return None;
                };
                let reduced_score = -reduced_score;
                if lmr_requires_full_research(reduced_score, alpha) {
                    self.debug_counters.lmr_researches += 1;
                    self.alpha_beta_core::<USE_TABLEBASES, USE_NNUE>(
                        position,
                        depth - 1,
                        ply + 1,
                        -beta,
                        -alpha,
                        SearchNodeState::new(false),
                    )
                    .map(|score| -score)
                } else {
                    Some(reduced_score)
                }
            } else {
                self.alpha_beta_core::<USE_TABLEBASES, USE_NNUE>(
                    position,
                    depth - 1,
                    ply + 1,
                    -beta,
                    -alpha,
                    SearchNodeState::new(child_is_pv),
                )
                .map(|score| -score)
            };

            let Some(score) = score_result else {
                self.previous_moves[ply + 1] = Move::NONE;
                self.unmake_search_move::<USE_NNUE>(position, mv, undo);
                return None;
            };
            self.previous_moves[ply + 1] = Move::NONE;
            self.unmake_search_move::<USE_NNUE>(position, mv, undo);

            if score > alpha {
                alpha = score;
                best_move = mv;
                self.update_pv(ply, mv);
                if alpha >= beta {
                    if !mv.is_capture() && !mv.is_promotion() {
                        self.record_killer(ply, mv);
                        self.record_quiet_cutoff(position, mv, ply, depth);
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
        self.store_tt(TtStoreInput {
            key: tt_key,
            depth: depth.min(u8::MAX as usize) as u8,
            ply,
            best_move,
            static_eval: static_eval.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            score: alpha,
            bound,
        });
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
        if !self.heuristics.pv_move_ordering || ply != 0 || ply >= self.previous_iteration_pv_length
        {
            return None;
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
        if self.heuristics.quiet_history {
            quiet_score += self.history_score(position, mv);
        }
        if self.heuristics.continuation_history {
            quiet_score += self.continuation_score(position, mv, hints.ply);
        }
        quiet_score
    }

    fn capture_order_score(&self, position: &Position, mv: Move) -> i32 {
        let see_score = position.see(mv).0 as i32;
        if !self.heuristics.capture_buckets {
            return 200_000 + see_score;
        }

        if see_score > 0 {
            320_000 + see_score
        } else if see_score == 0 {
            260_000
        } else {
            40_000 + see_score
        }
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

        let bonus = ((depth * depth) as i32 * 32).clamp(32, HISTORY_MAX);
        if self.heuristics.quiet_history {
            let color = position.side_to_move().index();
            let entry = &mut self.quiet_history[color][mv.from().index()][mv.to().index()];
            let current = *entry as i32;
            let updated = current + bonus - current * bonus / HISTORY_MAX;
            *entry = updated.clamp(-HISTORY_MAX, HISTORY_MAX) as i16;
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
        let current = *entry as i32;
        let updated = current + bonus - current * bonus / HISTORY_MAX;
        *entry = updated.clamp(-HISTORY_MAX, HISTORY_MAX) as i16;
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

    fn prepare_nnue(&mut self, position: &Position) {
        self.nnue
            .as_mut()
            .expect("NNUE preparation requires an active NNUE service")
            .reset(position);
    }

    pub(crate) fn evaluate_position<const USE_NNUE: bool>(&self, position: &Position) -> i32 {
        if USE_NNUE {
            self.nnue
                .as_ref()
                .expect("NNUE evaluation requires an active NNUE service")
                .evaluate(position)
                .0
        } else {
            eval::evaluate(position).0
        }
    }

    pub(crate) fn make_search_move<const USE_NNUE: bool>(
        &mut self,
        position: &mut Position,
        mv: Move,
    ) -> Result<crate::core::UndoState, crate::core::MoveError> {
        let undo = position.make_move(mv)?;
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
                .control
                .hard_deadline
                .is_some_and(|deadline| Instant::now() >= deadline)
    }

    fn soft_stop_requested(&self) -> bool {
        self.control
            .soft_deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
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
        && candidate.quiets_searched > 3
}

fn lmr_requires_full_research(reduced_score: i32, alpha: i32) -> bool {
    reduced_score > alpha
}

fn null_move_is_eligible(
    heuristics: SearchHeuristics,
    position: &Position,
    node_state: SearchNodeState,
    depth: usize,
    beta: i32,
    static_eval: i32,
    in_check: bool,
) -> bool {
    heuristics.null_move_pruning
        && node_state.null_move_allowed
        && !node_state.is_pv
        && !in_check
        && depth >= 3
        && beta > -MATE_THRESHOLD
        && beta < MATE_THRESHOLD
        && static_eval >= beta
        && position.has_non_pawn_material(position.side_to_move())
}

fn null_move_reduction(depth: usize) -> usize {
    if depth >= 7 { 3 } else { 2 }
}

impl SearchContext {
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
    position.is_draw_by_repetition()
        || position.is_draw_by_fifty_move()
        || position.is_insufficient_material()
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

pub(crate) fn is_quiescence_move(mv: Move, position: &Position) -> bool {
    mv.is_capture() || mv.is_promotion() || position.is_in_check(position.side_to_move())
}

fn quiet_shape_bonus(position: &Position, mv: Move) -> i32 {
    let Some(piece) = position.piece_at(mv.from()) else {
        return 0;
    };

    let quiet_bonus = match piece.piece_type() {
        crate::core::PieceType::Pawn => 5,
        crate::core::PieceType::Knight => 10,
        crate::core::PieceType::Bishop => 9,
        crate::core::PieceType::Rook => 4,
        crate::core::PieceType::Queen => 2,
        crate::core::PieceType::King => {
            if mv.is_castle() {
                30
            } else {
                0
            }
        }
    };

    quiet_bonus + square_progress_bonus(piece.color(), mv.to())
}

fn promotion_score(piece_type: crate::core::PieceType) -> i32 {
    see::promotion_gain(piece_type).0 as i32
}

fn square_progress_bonus(color: crate::core::Color, square: crate::core::Square) -> i32 {
    match color {
        crate::core::Color::White => square.rank() as i32,
        crate::core::Color::Black => 7 - square.rank() as i32,
    }
}

fn format_info_line(
    depth: u8,
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

    if pv_text.is_empty() {
        format!("info depth {depth} {score_text} nodes {nodes} time {elapsed_ms} tthits {tt_hits}")
    } else {
        format!(
            "info depth {depth} {score_text} nodes {nodes} time {elapsed_ms} tthits {tt_hits} pv {pv_text}"
        )
    }
}

fn tt_cutoff_score(hit: TtHit, depth: usize, ply: usize, alpha: i32, beta: i32) -> Option<i32> {
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
        Bound, LmrCandidate, Move, MoveList, MoveOrderHints, Position, SearchContext,
        SearchHeuristics, SearchLimits, SearchNodeState, lmr_is_eligible,
        lmr_requires_full_research, null_move_is_eligible, null_move_reduction,
        tt_cutoff_score, validated_tt_move_hint,
    };
    use crate::core::{ParsedMove, Square, chess_move::FLAG_CAPTURE};
    use crate::search::tablebase::{self, MockTablebaseBackend, TablebaseService, WdlOutcome};
    use crate::search::tt::{TtHit, normalize_score_for_store};
    use std::sync::Arc;

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
                depth: 4,
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
                quiets_searched: 3,
            },
        ));
    }

    #[test]
    fn lmr_reduces_late_quiets_and_researches_on_alpha_improvement() {
        let mut position = Position::startpos();
        let limits = SearchLimits::new(5)
            .with_heuristics(SearchHeuristics::phase8_baseline().with_late_move_reductions(true));
        let mut context = SearchContext::new(limits);

        let _ = context
            .alpha_beta(
                &mut position,
                4,
                1,
                -20,
                20,
                SearchNodeState::new(false),
            )
            .expect("search must complete");

        assert!(context.debug_counters().lmr_reductions > 0);
        assert!(context.debug_counters().lmr_researches > 0);
    }

    #[test]
    fn lmr_alpha_raise_requires_full_research() {
        assert!(!lmr_requires_full_research(20, 20));
        assert!(lmr_requires_full_research(21, 20));
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

        position = Position::from_fen("8/8/8/8/3k4/8/4p3/3K4 b - - 0 1")
            .expect("FEN parse must succeed");
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
        assert_eq!(null_move_reduction(3), 2);
        assert_eq!(null_move_reduction(6), 2);
        assert_eq!(null_move_reduction(7), 3);
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
            None,
            None,
            Some(tablebases),
            super::SearchControl::default(),
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
            None,
            None,
            Some(tablebases),
            super::SearchControl::default(),
        );

        assert_eq!(result.score.0, 0);
    }
}
