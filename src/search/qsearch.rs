use crate::core::{MoveList, Position};

use super::{
    eval,
    root::{MAX_PLY, SearchContext, is_draw, is_quiescence_move, terminal_score},
};

pub(crate) fn qsearch(
    context: &mut SearchContext,
    position: &mut Position,
    ply: usize,
    mut alpha: i32,
    beta: i32,
) -> i32 {
    context.nodes += 1;

    if ply >= MAX_PLY - 1 {
        return eval::evaluate(position).0;
    }

    if is_draw(position) {
        return 0;
    }

    let in_check = position.is_in_check(position.side_to_move());
    if !in_check {
        let stand_pat = eval::evaluate(position).0;
        if stand_pat >= beta {
            return beta;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }
    }

    context.pv_length[ply] = 0;

    let mut legal_moves = MoveList::new();
    position.generate_legal_moves(&mut legal_moves);
    if legal_moves.is_empty() {
        return terminal_score(position, ply);
    }

    for index in 0..legal_moves.len() {
        context.pick_next_move(position, &mut legal_moves, index, !in_check, None);
        let mv = legal_moves.get(index);
        if !in_check && !is_quiescence_move(mv, position) {
            continue;
        }

        let undo = position
            .make_move(mv)
            .expect("quiescence move must be legal");
        let score = -qsearch(context, position, ply + 1, -beta, -alpha);
        position.unmake_move(mv, undo);

        if score > alpha {
            alpha = score;
            context.update_pv(ply, mv);
            if alpha >= beta {
                break;
            }
        }
    }

    alpha
}
