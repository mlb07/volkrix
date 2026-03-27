use std::time::Instant;

use crate::core::{Move, MoveList, Position, Score, see};

use super::{
    eval,
    limits::{SearchHeuristics, SearchLimits},
    qsearch,
    tt::{self, Bound, TtHit, TtStore},
};

pub(crate) const MAX_PLY: usize = 128;
pub(crate) const INF: i32 = 32_000;
pub(crate) const MATE_SCORE: i32 = 30_000;
const MATE_THRESHOLD: i32 = MATE_SCORE - MAX_PLY as i32;
const ASPIRATION_DELTA: i32 = 32;
const HISTORY_MAX: i32 = 16_384;

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

pub(crate) struct SearchContext {
    started: Instant,
    pub(crate) nodes: u64,
    tt_hits: u64,
    pv_table: [[Move; MAX_PLY]; MAX_PLY],
    pub(crate) pv_length: [usize; MAX_PLY],
    previous_iteration_pv: [Move; MAX_PLY],
    previous_iteration_pv_length: usize,
    killer_moves: [[Move; 2]; MAX_PLY],
    quiet_history: [[[i16; 64]; 64]; 2],
    heuristics: SearchHeuristics,
    tt: Option<tt::TranspositionTable>,
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
    SearchContext::new(limits).run(position, limits)
}

impl SearchContext {
    pub(crate) fn new(limits: SearchLimits) -> Self {
        Self {
            started: Instant::now(),
            nodes: 0,
            tt_hits: 0,
            pv_table: [[Move::NONE; MAX_PLY]; MAX_PLY],
            pv_length: [0; MAX_PLY],
            previous_iteration_pv: [Move::NONE; MAX_PLY],
            previous_iteration_pv_length: 0,
            killer_moves: [[Move::NONE; 2]; MAX_PLY],
            quiet_history: [[[0; 64]; 64]; 2],
            heuristics: limits.heuristics,
            tt: limits
                .tt_enabled
                .then(|| tt::TranspositionTable::new_mb(limits.hash_mb)),
        }
    }

    fn run(&mut self, position: &mut Position, limits: SearchLimits) -> SearchResult {
        let depth_limit = limits.depth.max(1).min((MAX_PLY - 1) as u8);
        let mut best_move = None;
        let mut best_score = 0i32;
        let mut info_lines = Vec::new();

        for depth in 1..=depth_limit {
            if let Some(tt) = self.tt.as_mut() {
                tt.new_generation();
            }
            let (depth_best_move, depth_score) = if self.heuristics.aspiration_windows
                && depth > 1
                && best_score.abs() < MATE_THRESHOLD
            {
                self.search_root_with_aspiration(position, depth as usize, best_score)
            } else {
                self.search_root(position, depth as usize, -INF, INF)
            };
            best_move = depth_best_move;
            best_score = depth_score;

            let pv = self.collect_pv(0);
            self.capture_completed_pv(&pv);
            info_lines.push(format_info_line(
                depth,
                depth_score,
                self.nodes,
                self.started.elapsed().as_millis(),
                self.tt_hits,
                &pv,
            ));
        }

        SearchResult {
            best_move,
            score: Score(best_score),
            depth: depth_limit,
            nodes: self.nodes,
            pv: self.collect_pv(0),
            info_lines,
            tt_hits: self.tt_hits,
        }
    }

    fn search_root_with_aspiration(
        &mut self,
        position: &mut Position,
        depth: usize,
        guess: i32,
    ) -> (Option<Move>, i32) {
        let mut delta = ASPIRATION_DELTA;
        loop {
            let alpha = (guess - delta).max(-INF);
            let beta = (guess + delta).min(INF);
            let (best_move, score) = self.search_root(position, depth, alpha, beta);
            if score <= alpha && alpha > -INF {
                delta = (delta * 2).min(INF / 2);
                continue;
            }
            if score >= beta && beta < INF {
                delta = (delta * 2).min(INF / 2);
                continue;
            }
            return (best_move, score);
        }
    }

    fn search_root(
        &mut self,
        position: &mut Position,
        depth: usize,
        mut alpha: i32,
        beta: i32,
    ) -> (Option<Move>, i32) {
        self.pv_length[0] = 0;
        let pv_move_hint = self.previous_pv_move(0);
        let tt_move_hint = self
            .probe_tt(position.search_key())
            .and_then(|hit| (!hit.best_move.is_none()).then_some(hit.best_move));

        let mut legal_moves = MoveList::new();
        position.generate_legal_moves(&mut legal_moves);
        if legal_moves.is_empty() {
            return (None, terminal_score(position, 0));
        }
        let pv_move_hint = validated_move_hint(&legal_moves, pv_move_hint);
        let tt_move_hint = validated_tt_move_hint(&legal_moves, tt_move_hint);
        let ordering_hints = MoveOrderHints {
            ply: 0,
            quiescence_only: false,
            pv_move: pv_move_hint,
            tt_move: tt_move_hint,
        };

        let mut best_move = None;

        for index in 0..legal_moves.len() {
            self.pick_next_move(position, &mut legal_moves, index, ordering_hints);
            let mv = legal_moves.get(index);
            let undo = position
                .make_move(mv)
                .expect("root move must be legal during search");
            let score = -self.alpha_beta(position, depth.saturating_sub(1), 1, -beta, -alpha);
            position.unmake_move(mv, undo);

            if score > alpha || best_move.is_none() {
                alpha = score;
                best_move = Some(mv);
                self.update_pv(0, mv);
            }
        }

        (best_move, alpha)
    }

    pub(crate) fn alpha_beta(
        &mut self,
        position: &mut Position,
        depth: usize,
        ply: usize,
        mut alpha: i32,
        beta: i32,
    ) -> i32 {
        self.nodes += 1;

        if ply >= MAX_PLY - 1 {
            return eval::evaluate(position).0;
        }

        if is_draw(position) {
            return 0;
        }

        let tt_key = position.search_key();
        let alpha_start = alpha;
        let static_eval = eval::evaluate(position).0;

        let tt_hit = self.probe_tt(tt_key);
        if let Some(hit) = tt_hit
            && let Some(cutoff) = tt_cutoff_score(hit, depth, ply, alpha, beta)
        {
            return cutoff;
        }

        if depth == 0 {
            return qsearch::qsearch(self, position, ply, alpha, beta);
        }

        self.pv_length[ply] = 0;
        let pv_move_hint = self.previous_pv_move(ply);
        let tt_move_hint =
            tt_hit.and_then(|hit| (!hit.best_move.is_none()).then_some(hit.best_move));

        let mut legal_moves = MoveList::new();
        position.generate_legal_moves(&mut legal_moves);
        if legal_moves.is_empty() {
            return terminal_score(position, ply);
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

        for index in 0..legal_moves.len() {
            self.pick_next_move(position, &mut legal_moves, index, ordering_hints);
            let mv = legal_moves.get(index);
            let undo = position.make_move(mv).expect("searched move must be legal");
            let score = -self.alpha_beta(position, depth - 1, ply + 1, -beta, -alpha);
            position.unmake_move(mv, undo);

            if score > alpha {
                alpha = score;
                best_move = mv;
                self.update_pv(ply, mv);
                if alpha >= beta {
                    if !mv.is_capture() && !mv.is_promotion() {
                        self.record_killer(ply, mv);
                        self.record_quiet_cutoff(position, mv, depth);
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
            static_eval: static_eval as i16,
            score: alpha,
            bound,
        });
        alpha
    }

    pub(crate) fn pick_next_move(
        &self,
        position: &Position,
        moves: &mut MoveList,
        start: usize,
        hints: MoveOrderHints,
    ) {
        let mut best_index = start;
        let mut best_score = i32::MIN;
        for index in start..moves.len() {
            let mv = moves.get(index);
            if hints.quiescence_only && !is_quiescence_move(mv, position) {
                continue;
            }
            let score = self.score_move(position, mv, hints);
            if score > best_score {
                best_score = score;
                best_index = index;
            }
        }
        moves.swap(start, best_index);
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

    fn score_move(&self, position: &Position, mv: Move, hints: MoveOrderHints) -> i32 {
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

    fn record_quiet_cutoff(&mut self, position: &Position, mv: Move, depth: usize) {
        if !self.heuristics.quiet_history {
            return;
        }

        let bonus = ((depth * depth) as i32 * 32).clamp(32, HISTORY_MAX);
        let color = position.side_to_move().index();
        let entry = &mut self.quiet_history[color][mv.from().index()][mv.to().index()];
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
        let Some(tt) = self.tt.as_mut() else {
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
        Bound, Move, MoveList, MoveOrderHints, Position, SearchContext, SearchLimits,
        tt_cutoff_score, validated_tt_move_hint,
    };
    use crate::core::{ParsedMove, Square, chess_move::FLAG_CAPTURE};
    use crate::search::tt::{TtHit, normalize_score_for_store};

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
}
