use crate::core::{Move, MoveList, PieceType, Position, see};

use super::movepicker::MovePicker;
use super::root::{MAX_PLY, MoveOrderHints, SearchContext, is_draw, terminal_score};

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
        context.clear_pv(ply);
        return Some(context.evaluate_position::<USE_NNUE>(position));
    }
    context.clear_pv(ply);

    if is_draw(position) {
        return Some(0);
    }

    let in_check = position.is_in_check(position.side_to_move());
    let mut stand_pat = None;
    if !in_check {
        let eval = context.evaluate_position::<USE_NNUE>(position);
        stand_pat = Some(eval);
        if eval >= beta {
            return Some(beta);
        }
        if eval > alpha {
            alpha = eval;
        }
    }

    let mut legal_moves = MoveList::new();
    if in_check {
        position.generate_legal_moves(&mut legal_moves);
    } else {
        position.generate_legal_noisy_moves(&mut legal_moves);
    }
    if legal_moves.is_empty() {
        return if in_check {
            Some(terminal_score(position, ply))
        } else {
            Some(alpha)
        };
    }
    let ordering_hints = MoveOrderHints {
        ply,
        quiescence_only: !in_check,
        pv_move: None,
        tt_move: None,
    };

    let mut move_picker = MovePicker::new(context, position, &legal_moves, ordering_hints);
    while let Some(mv) = move_picker.next() {
        if !in_check && delta_pruning_is_eligible(position, mv, stand_pat.unwrap_or(alpha), alpha) {
            continue;
        }

        if !in_check && mv.is_capture() && !mv.is_promotion() && position.see(mv).0 < 0 {
            continue;
        }

        let undo = context
            .make_search_move::<USE_NNUE>(position, mv)
            .expect("quiescence move must be legal");
        context.set_previous_move(ply + 1, mv);
        let Some(score) = qsearch::<USE_NNUE>(context, position, ply + 1, -beta, -alpha) else {
            context.set_previous_move(ply + 1, crate::core::Move::NONE);
            context.unmake_search_move::<USE_NNUE>(position, mv, undo);
            return None;
        };
        context.set_previous_move(ply + 1, crate::core::Move::NONE);
        context.unmake_search_move::<USE_NNUE>(position, mv, undo);
        let score = -score;

        if score > alpha {
            alpha = score;
            context.update_pv(ply, mv);
            if alpha >= beta {
                break;
            }
        }
    }

    Some(alpha)
}

fn delta_pruning_is_eligible(position: &Position, mv: Move, stand_pat: i32, alpha: i32) -> bool {
    !mv.is_promotion()
        && stand_pat + noisy_move_gain(position, mv) + delta_pruning_margin() <= alpha
}

fn noisy_move_gain(position: &Position, mv: Move) -> i32 {
    if mv.is_en_passant() {
        return piece_value(PieceType::Pawn);
    }
    let capture_gain = position
        .piece_at(mv.to())
        .map_or(0, |piece| piece_value(piece.piece_type()));
    capture_gain
        + mv.promotion()
            .map_or(0, |piece_type| see::promotion_gain(piece_type).0 as i32)
}

fn piece_value(piece_type: PieceType) -> i32 {
    see::piece_value(piece_type).0 as i32
}

fn delta_pruning_margin() -> i32 {
    180
}
