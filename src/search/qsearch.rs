use crate::core::{Move, PieceType, Position, see};

use super::movepicker::MovePicker;
use super::root::{
    MAX_PLY, MoveOrderHints, SearchContext, is_draw, is_quiescence_move, terminal_score,
    tt_cutoff_score,
};
use crate::search::tt::Bound;

const DELTA_PRUNE_MARGIN: i32 = 200;

pub(crate) fn qsearch<const USE_NNUE: bool>(
    context: &mut SearchContext,
    position: &mut Position,
    ply: usize,
    mut alpha: i32,
    beta: i32,
) -> Option<i32> {
    context.nodes += 1;
    if context.nodes & 1023 == 0 && context.hard_stop_requested() {
        return None;
    }

    if ply >= MAX_PLY - 1 {
        return Some(context.evaluate_position::<USE_NNUE>(position));
    }

    if is_draw(position) {
        return Some(0);
    }

    let tt_key = position.search_key();
    let alpha_start = alpha;
    let mut best_move = Move::NONE;
    let mut static_eval = 0i32;
    let tt_hit = context.probe_tt(tt_key);
    if let Some(hit) = tt_hit
        && let Some(cutoff) = tt_cutoff_score(hit, 0, ply, alpha, beta)
    {
        return Some(cutoff);
    }

    let in_check = position.is_in_check(position.side_to_move());
    if !in_check {
        let evaluated = context.evaluate_position::<USE_NNUE>(position);
        static_eval = evaluated;
        if evaluated >= beta {
            context.store_qsearch_tt(
                tt_key,
                ply,
                Move::NONE,
                static_eval,
                evaluated,
                Bound::Lower,
            );
            return Some(beta);
        }
        if evaluated > alpha {
            alpha = evaluated;
        }
    }

    context.pv_length[ply] = 0;

    let tt_move_hint = tt_hit.and_then(|hit| (!hit.best_move.is_none()).then_some(hit.best_move));
    let ordering_hints = MoveOrderHints {
        ply,
        quiescence_only: !in_check,
        pv_move: None,
        tt_move: tt_move_hint,
    };
    let mut picker = MovePicker::new(context, position, ordering_hints);
    if picker.is_empty() {
        let terminal = terminal_score(position, ply);
        context.store_qsearch_tt(tt_key, ply, Move::NONE, static_eval, terminal, Bound::Exact);
        return Some(terminal);
    }

    while let Some(mv) = picker.next() {
        if !in_check && !is_quiescence_move(mv, position) {
            continue;
        }
        if !in_check && see_prunes_move(position, mv) {
            continue;
        }
        if !in_check && delta_prunes_move(position, mv, static_eval, alpha) {
            continue;
        }

        let undo = context
            .make_search_move::<USE_NNUE>(position, mv)
            .expect("quiescence move must be legal");
        let Some(score) = qsearch::<USE_NNUE>(context, position, ply + 1, -beta, -alpha) else {
            context.unmake_search_move::<USE_NNUE>(position, mv, undo);
            return None;
        };
        context.unmake_search_move::<USE_NNUE>(position, mv, undo);
        let score = -score;

        if score > alpha {
            alpha = score;
            best_move = mv;
            context.update_pv(ply, mv);
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
    context.store_qsearch_tt(tt_key, ply, best_move, static_eval, alpha, bound);
    Some(alpha)
}

fn see_prunes_move(position: &Position, mv: Move) -> bool {
    mv.is_capture() && !mv.is_promotion() && position.see(mv).0 < 0
}

fn delta_prunes_move(position: &Position, mv: Move, static_eval: i32, alpha: i32) -> bool {
    static_eval + qsearch_move_gain(position, mv) + DELTA_PRUNE_MARGIN < alpha
}

fn qsearch_move_gain(position: &Position, mv: Move) -> i32 {
    capture_gain(position, mv) + promotion_gain(mv)
}

fn capture_gain(position: &Position, mv: Move) -> i32 {
    capture_victim_type(position, mv)
        .map(|piece_type| see::piece_value(piece_type).0 as i32)
        .unwrap_or(0)
}

fn promotion_gain(mv: Move) -> i32 {
    mv.promotion()
        .map(|piece_type| see::promotion_gain(piece_type).0 as i32)
        .unwrap_or(0)
}

fn capture_victim_type(position: &Position, mv: Move) -> Option<PieceType> {
    if mv.is_en_passant() {
        Some(PieceType::Pawn)
    } else {
        position.piece_at(mv.to()).map(|piece| piece.piece_type())
    }
}

#[cfg(test)]
mod tests {
    use super::{capture_gain, delta_prunes_move, promotion_gain};
    use crate::core::{Move, MoveList, ParsedMove, Position};

    #[test]
    fn delta_pruning_skips_small_gains_but_keeps_large_captures() {
        let mut position = Position::from_fen("4k3/8/8/2p1q3/3P4/8/8/7K w - - 0 1")
            .expect("FEN parse must succeed");
        let mut legal_moves = MoveList::new();
        position.generate_legal_moves(&mut legal_moves);

        let queen_capture = legal_moves
            .iter()
            .copied()
            .find(|mv| mv.matches_parsed(ParsedMove::parse("d4e5").expect("parse must succeed")))
            .expect("queen capture must exist");
        let pawn_capture = legal_moves
            .iter()
            .copied()
            .find(|mv| mv.matches_parsed(ParsedMove::parse("d4c5").expect("parse must succeed")))
            .expect("pawn capture must exist");

        assert!(!delta_prunes_move(&position, queen_capture, 0, 700));
        assert!(delta_prunes_move(&position, pawn_capture, 0, 700));
    }

    #[test]
    fn gain_estimates_account_for_captures_and_promotions() {
        let position =
            Position::from_fen("4k3/6P1/8/8/8/8/8/4K3 w - - 0 1").expect("FEN parse must succeed");
        let promotion = ParsedMove::parse("g7g8q").expect("parse must succeed");
        let promotion = Move::new(promotion.from(), promotion.to()).with_promotion(
            promotion
                .promotion()
                .expect("parsed promotion move must carry promotion piece"),
        );

        let capture_position =
            Position::from_fen("4k3/8/8/4q3/3P4/8/8/7K w - - 0 1").expect("FEN parse must succeed");
        let mut legal_moves = MoveList::new();
        let mut probe = capture_position.clone();
        probe.generate_legal_moves(&mut legal_moves);
        let capture = legal_moves
            .iter()
            .copied()
            .find(|mv| mv.matches_parsed(ParsedMove::parse("d4e5").expect("parse must succeed")))
            .expect("capture must exist");

        assert_eq!(promotion_gain(promotion), 800);
        assert_eq!(capture_gain(&capture_position, capture), 900);
        assert_eq!(capture_gain(&position, promotion), 0);
    }
}
