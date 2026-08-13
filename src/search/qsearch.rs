use crate::core::{Move, MoveList, PieceType, Position, see};

use super::movepicker::MovePicker;
use super::{
    root::{
        MAX_PLY, MoveOrderHints, SearchContext, is_draw, mate_distance_bounds, terminal_score,
        tt_cutoff_score, validated_move_hint,
    },
    tt::Bound,
};

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn qsearch<const USE_NNUE: bool>(
    context: &mut SearchContext,
    position: &mut Position,
    ply: usize,
    alpha: i32,
    beta: i32,
) -> Option<i32> {
    qsearch_core::<USE_NNUE>(context, position, ply, alpha, beta, true)
}

pub(crate) fn qsearch_from_main<const USE_NNUE: bool>(
    context: &mut SearchContext,
    position: &mut Position,
    ply: usize,
    alpha: i32,
    beta: i32,
) -> Option<i32> {
    qsearch_core::<USE_NNUE>(context, position, ply, alpha, beta, false)
}

fn qsearch_core<const USE_NNUE: bool>(
    context: &mut SearchContext,
    position: &mut Position,
    ply: usize,
    mut alpha: i32,
    beta: i32,
    count_entry: bool,
) -> Option<i32> {
    if count_entry && !context.count_node() {
        return None;
    }
    context.seldepth = context.seldepth.max(ply);
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

    let (mate_alpha, mate_beta) = mate_distance_bounds(ply);
    alpha = alpha.max(mate_alpha);
    let beta = beta.min(mate_beta);
    if alpha >= beta {
        return Some(alpha);
    }

    let alpha_start = alpha;
    let tt_key = position.search_key();
    let tt_hit = if context.qsearch_tt_enabled() {
        context.probe_tt(tt_key)
    } else {
        None
    };
    if let Some(hit) = tt_hit
        && let Some(cutoff) = tt_cutoff_score(hit, 0, ply, alpha, beta)
    {
        return Some(cutoff);
    }

    let in_check = position.is_in_check(position.side_to_move());
    let static_eval = if in_check {
        0
    } else {
        tt_hit
            .map(|hit| hit.eval as i32)
            .unwrap_or_else(|| context.evaluate_position::<USE_NNUE>(position))
    };
    let mut stand_pat = None;
    if !in_check {
        stand_pat = Some(static_eval);
        if static_eval >= beta {
            context.store_qsearch_tt(
                tt_key,
                ply,
                Move::NONE,
                static_eval,
                static_eval,
                Bound::Lower,
            );
            return Some(static_eval);
        }
        if static_eval > alpha {
            alpha = static_eval;
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
    let tt_move = validated_move_hint(
        &legal_moves,
        tt_hit.and_then(|hit| (!hit.best_move.is_none()).then_some(hit.best_move)),
    );
    let ordering_hints = MoveOrderHints {
        ply,
        quiescence_only: !in_check,
        pv_move: None,
        tt_move,
    };

    let mut best_move = Move::NONE;
    let mut best_score = alpha;
    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    let mut searched_captures = [Move::NONE; crate::core::movelist::MAX_MOVES];
    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    let mut searched_capture_count = 0usize;
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
        context.set_previous_move_from_position(position, ply + 1, mv);
        let Some(score) = qsearch_core::<USE_NNUE>(context, position, ply + 1, -beta, -alpha, true)
        else {
            context.set_previous_move(ply + 1, crate::core::Move::NONE);
            context.unmake_search_move::<USE_NNUE>(position, mv, undo);
            return None;
        };
        context.set_previous_move(ply + 1, crate::core::Move::NONE);
        context.unmake_search_move::<USE_NNUE>(position, mv, undo);
        let score = -score;

        #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
        if mv.is_capture() {
            debug_assert!(searched_capture_count < searched_captures.len());
            searched_captures[searched_capture_count] = mv;
            searched_capture_count += 1;
        }

        if score > alpha {
            alpha = score;
            best_score = score;
            best_move = mv;
            context.update_pv(ply, mv);
            if alpha >= beta {
                #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
                if mv.is_capture() && context.qsearch_capture_history_enabled() {
                    context.record_qsearch_capture_history(position, mv, true);
                    for failed in searched_captures[..searched_capture_count.saturating_sub(1)]
                        .iter()
                        .copied()
                    {
                        context.record_qsearch_capture_history(position, failed, false);
                    }
                }
                break;
            }
        }
    }

    let bound = if best_score <= alpha_start {
        Bound::Upper
    } else if best_score >= beta {
        Bound::Lower
    } else {
        Bound::Exact
    };
    context.store_qsearch_tt(tt_key, ply, best_move, static_eval, best_score, bound);
    Some(best_score)
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

#[cfg(test)]
mod tests {
    use super::qsearch;
    use crate::{
        core::{ParsedMove, Position},
        search::{SearchLimits, limits::SearchHeuristics, root::SearchContext},
    };

    #[test]
    fn recursive_quiescence_reuses_an_exact_tt_result() {
        let mut position = Position::startpos();
        let mut context = SearchContext::new(SearchLimits::new(1));
        let first = qsearch::<false>(&mut context, &mut position, 1, -100, 100)
            .expect("first quiescence search must complete");

        context.nodes = 0;
        let second = qsearch::<false>(&mut context, &mut position, 1, -100, 100)
            .expect("second quiescence search must complete");

        assert_eq!(second, first);
        assert_eq!(context.nodes, 1, "the TT hit must avoid move expansion");
    }

    #[test]
    fn capture_history_candidate_trains_a_qsearch_cutoff() {
        let mut position = Position::from_fen("6k1/8/8/5r2/3N4/8/8/6K1 w - - 0 1")
            .expect("FEN parse must succeed");
        let mut legal_moves = crate::core::MoveList::new();
        position.generate_legal_noisy_moves(&mut legal_moves);
        let capture = legal_moves
            .iter()
            .copied()
            .find(|mv| mv.matches_parsed(ParsedMove::parse("d4f5").expect("parse must succeed")))
            .expect("winning capture must exist");
        let before = position.to_fen();
        let heuristics = SearchHeuristics::phase9_default().with_capture_history_experiment(true);
        let mut context = SearchContext::new(
            SearchLimits::new(1)
                .without_tt()
                .with_heuristics(heuristics),
        );

        qsearch::<false>(&mut context, &mut position, 1, -500, -100)
            .expect("quiescence search must complete");
        assert_eq!(position.to_fen(), before);
        assert!(context.debug_capture_history_score(&position, capture) > 0);
    }
}
