use crate::core::{MoveList, Position};

use super::{
    eval,
    root::{MAX_PLY, MoveOrderHints, SearchContext, is_draw, is_quiescence_move, terminal_score},
};

pub(crate) fn qsearch(
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
        return Some(eval::evaluate(position).0);
    }

    if is_draw(position) {
        return Some(0);
    }

    let in_check = position.is_in_check(position.side_to_move());
    if !in_check {
        let stand_pat = eval::evaluate(position).0;
        if stand_pat >= beta {
            return Some(beta);
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }
    }

    context.pv_length[ply] = 0;

    let mut legal_moves = MoveList::new();
    position.generate_legal_moves(&mut legal_moves);
    if legal_moves.is_empty() {
        return Some(terminal_score(position, ply));
    }
    let ordering_hints = MoveOrderHints {
        ply,
        quiescence_only: !in_check,
        pv_move: None,
        tt_move: None,
    };

    for index in 0..legal_moves.len() {
        context.pick_next_move(position, &mut legal_moves, index, ordering_hints);
        let mv = legal_moves.get(index);
        if !in_check && !is_quiescence_move(mv, position) {
            continue;
        }

        let undo = position
            .make_move(mv)
            .expect("quiescence move must be legal");
        let Some(score) = qsearch(context, position, ply + 1, -beta, -alpha) else {
            position.unmake_move(mv, undo);
            return None;
        };
        position.unmake_move(mv, undo);
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
