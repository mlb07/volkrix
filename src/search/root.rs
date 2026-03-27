use std::time::Instant;

use crate::core::{Move, MoveList, Position, Score, see};

use super::{
    eval,
    limits::SearchLimits,
    qsearch,
    tt::{self, Bound, TtHit, TtStore},
};

pub(crate) const MAX_PLY: usize = 128;
pub(crate) const INF: i32 = 32_000;
pub(crate) const MATE_SCORE: i32 = 30_000;
const MATE_THRESHOLD: i32 = MATE_SCORE - MAX_PLY as i32;

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
            let (depth_best_move, depth_score) = self.search_root(position, depth as usize);
            best_move = depth_best_move;
            best_score = depth_score;

            let pv = self.collect_pv(0);
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

    fn search_root(&mut self, position: &mut Position, depth: usize) -> (Option<Move>, i32) {
        self.pv_length[0] = 0;
        let tt_move_hint = self
            .probe_tt(position.search_key())
            .and_then(|hit| (!hit.best_move.is_none()).then_some(hit.best_move));

        let mut legal_moves = MoveList::new();
        position.generate_legal_moves(&mut legal_moves);
        if legal_moves.is_empty() {
            return (None, terminal_score(position, 0));
        }
        let tt_move_hint = validated_tt_move_hint(&legal_moves, tt_move_hint);

        let mut best_move = None;
        let mut alpha = -INF;
        let beta = INF;

        for index in 0..legal_moves.len() {
            self.pick_next_move(position, &mut legal_moves, index, false, tt_move_hint);
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
        let tt_move_hint =
            tt_hit.and_then(|hit| (!hit.best_move.is_none()).then_some(hit.best_move));

        let mut legal_moves = MoveList::new();
        position.generate_legal_moves(&mut legal_moves);
        if legal_moves.is_empty() {
            return terminal_score(position, ply);
        }
        let tt_move_hint = validated_tt_move_hint(&legal_moves, tt_move_hint);
        let mut best_move = Move::NONE;

        for index in 0..legal_moves.len() {
            self.pick_next_move(position, &mut legal_moves, index, false, tt_move_hint);
            let mv = legal_moves.get(index);
            let undo = position.make_move(mv).expect("searched move must be legal");
            let score = -self.alpha_beta(position, depth - 1, ply + 1, -beta, -alpha);
            position.unmake_move(mv, undo);

            if score > alpha {
                alpha = score;
                best_move = mv;
                self.update_pv(ply, mv);
                if alpha >= beta {
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
        quiescence_only: bool,
        tt_move_hint: Option<Move>,
    ) {
        let mut best_index = start;
        let mut best_score = i32::MIN;
        for index in start..moves.len() {
            let mv = moves.get(index);
            if quiescence_only && !is_quiescence_move(mv, position) {
                continue;
            }
            let score = score_move(position, mv, quiescence_only, tt_move_hint);
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

    fn probe_tt(&mut self, key: u64) -> Option<TtHit> {
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

fn score_move(
    position: &Position,
    mv: Move,
    quiescence_only: bool,
    tt_move_hint: Option<Move>,
) -> i32 {
    if tt_move_hint == Some(mv) {
        return 400_000;
    }

    if mv.is_capture() {
        return 200_000 + position.see(mv).0 as i32;
    }

    if mv.is_promotion() {
        return 150_000 + mv.promotion().map_or(0, promotion_score);
    }

    if quiescence_only {
        return i32::MIN / 2;
    }

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

fn validated_tt_move_hint(legal_moves: &MoveList, tt_move_hint: Option<Move>) -> Option<Move> {
    let tt_move_hint = tt_move_hint?;
    legal_moves
        .as_slice()
        .iter()
        .copied()
        .find(|mv| *mv == tt_move_hint)
}

#[cfg(test)]
mod tests {
    use super::{
        Bound, Move, MoveList, Position, SearchContext, SearchLimits, tt_cutoff_score,
        validated_tt_move_hint,
    };
    use crate::core::{ParsedMove, Square};
    use crate::search::tt::{TtHit, normalize_score_for_store};

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
}
