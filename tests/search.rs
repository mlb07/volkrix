use volkrix::{
    core::{Position, PositionStatus},
    search::{SearchLimits, evaluate, search},
};

fn bestmove(position: &mut Position, depth: u8) -> String {
    search(position, SearchLimits::new(depth))
        .best_move
        .map(|mv| mv.to_string())
        .unwrap_or_else(|| "0000".to_owned())
}

#[test]
fn mate_in_one_is_found() {
    let mut position =
        Position::from_fen("7k/6Q1/6K1/8/8/8/8/8 w - - 0 1").expect("FEN parse must succeed");
    let result = search(&mut position, SearchLimits::new(1));
    let best_move = result.best_move.expect("mate in one must yield a move");

    position
        .make_move(best_move)
        .map(|undo| {
            assert_eq!(position.status(), PositionStatus::Checkmate);
            position.unmake_move(best_move, undo);
        })
        .expect("best move must be legal");
}

#[test]
fn forced_mate_is_preferred_over_material_gain() {
    let mut position =
        Position::from_fen("7k/8/6K1/8/8/8/8/r6Q w - - 0 1").expect("FEN parse must succeed");
    let result = search(&mut position, SearchLimits::new(1));
    let best_move = result.best_move.expect("search must return a move");

    assert_ne!(best_move.to_string(), "h1a1");
    position
        .make_move(best_move)
        .map(|undo| {
            assert_eq!(position.status(), PositionStatus::Checkmate);
            position.unmake_move(best_move, undo);
        })
        .expect("best move must be legal");
}

#[test]
fn stalemate_is_recognized_correctly() {
    let mut position =
        Position::from_fen("7k/5Q2/7K/8/8/8/8/8 b - - 0 1").expect("FEN parse must succeed");
    let result = search(&mut position, SearchLimits::new(1));

    assert!(result.best_move.is_none());
    assert_eq!(result.score.0, 0);
}

#[test]
fn stalemate_is_avoided_when_a_winning_move_exists() {
    let mut position =
        Position::from_fen("k7/8/1QK5/8/8/8/8/8 w - - 0 1").expect("FEN parse must succeed");

    assert_eq!(bestmove(&mut position, 1), "b6b7");
}

#[test]
fn repetition_is_handled_as_draw_inside_search() {
    let mut position = Position::startpos();
    for mv in [
        "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8",
    ] {
        position
            .apply_uci_move(mv)
            .expect("cycle move must be legal");
    }

    let result = search(&mut position, SearchLimits::new(2));
    assert_eq!(result.score.0, 0);
}

#[test]
fn fifty_move_is_handled_as_draw_inside_search() {
    let mut position =
        Position::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 100 1").expect("FEN parse must succeed");

    let result = search(&mut position, SearchLimits::new(2));
    assert_eq!(result.score.0, 0);
}

#[test]
fn insufficient_material_is_handled_as_draw_inside_search() {
    let mut position =
        Position::from_fen("4k3/8/8/8/8/8/8/3BK3 w - - 0 1").expect("FEN parse must succeed");

    let result = search(&mut position, SearchLimits::new(2));
    assert_eq!(result.score.0, 0);
}

#[test]
fn bestmove_is_legal_and_deterministic() {
    let mut first = Position::startpos();
    let mut second = Position::startpos();

    let first_move = bestmove(&mut first, 3);
    let second_move = bestmove(&mut second, 3);

    assert_eq!(first_move, second_move);
    Position::startpos()
        .apply_uci_move(&first_move)
        .expect("searched bestmove must be legal");
}

#[test]
fn search_leaves_root_position_unchanged() {
    let mut position =
        Position::from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 2 3")
            .expect("FEN parse must succeed");
    let before = position.to_fen();
    let before_key = position.zobrist_key();
    let before_history = position.debug_repetition_history_snapshot();

    let _ = search(&mut position, SearchLimits::new(3));

    let after_history = position.debug_repetition_history_snapshot();
    assert_eq!(position.to_fen(), before);
    assert_eq!(position.zobrist_key(), before_key);
    assert_eq!(after_history.len(), before_history.len());
    assert_eq!(after_history, before_history);
    position.validate().expect("position must still validate");
}

#[test]
fn search_leaves_root_position_unchanged_with_tt_disabled() {
    let mut position =
        Position::from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 2 3")
            .expect("FEN parse must succeed");
    let before = position.to_fen();
    let before_key = position.zobrist_key();
    let before_history = position.debug_repetition_history_snapshot();

    let _ = search(&mut position, SearchLimits::new(3).without_tt());

    let after_history = position.debug_repetition_history_snapshot();
    assert_eq!(position.to_fen(), before);
    assert_eq!(position.zobrist_key(), before_key);
    assert_eq!(after_history, before_history);
    position.validate().expect("position must still validate");
}

#[test]
fn tt_enabled_and_disabled_search_agree_on_curated_positions() {
    let cases = [
        ("7k/5Q2/7K/8/8/8/8/8 b - - 0 1", 2),
        ("4k3/8/8/8/8/8/8/3BK3 w - - 0 1", 2),
        ("k7/8/1QK5/8/8/8/8/8 w - - 0 1", 2),
        ("3qk3/8/8/3r4/8/8/8/3QK3 w - - 0 1", 2),
    ];

    for (fen, depth) in cases {
        let mut with_tt = Position::from_fen(fen).expect("FEN parse must succeed");
        let mut without_tt = Position::from_fen(fen).expect("FEN parse must succeed");

        let tt_result = search(&mut with_tt, SearchLimits::new(depth));
        let no_tt_result = search(&mut without_tt, SearchLimits::new(depth).without_tt());

        assert_eq!(tt_result.score, no_tt_result.score, "{fen}");
        match (tt_result.best_move, no_tt_result.best_move) {
            (Some(tt_move), Some(no_tt_move)) => {
                with_tt
                    .apply_uci_move(&tt_move.to_string())
                    .expect("TT bestmove must be legal");
                without_tt
                    .apply_uci_move(&no_tt_move.to_string())
                    .expect("non-TT bestmove must be legal");
            }
            (None, None) => {}
            other => panic!("TT and non-TT move availability mismatch for {fen}: {other:?}"),
        }
    }
}

#[test]
fn exact_bestmove_regressions_use_unique_root_positions() {
    let cases = [("k7/8/1QK5/8/8/8/8/8 w - - 0 1", 2, "b6b7")];

    for (fen, depth, expected_move) in cases {
        let mut with_tt = Position::from_fen(fen).expect("FEN parse must succeed");
        let mut without_tt = Position::from_fen(fen).expect("FEN parse must succeed");

        assert_eq!(bestmove(&mut with_tt, depth), expected_move);
        assert_eq!(
            search(&mut without_tt, SearchLimits::new(depth).without_tt())
                .best_move
                .map(|mv| mv.to_string())
                .unwrap_or_else(|| "0000".to_owned()),
            expected_move
        );
    }
}

#[test]
fn quiescence_avoids_simple_horizon_blunders() {
    let mut position =
        Position::from_fen("3qk3/8/8/3r4/8/8/8/3QK3 w - - 0 1").expect("FEN parse must succeed");

    assert_ne!(bestmove(&mut position, 1), "d1d5");
}

#[test]
fn evaluation_score_sign_is_sane() {
    let white_better =
        Position::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1").expect("FEN parse must succeed");
    let black_better =
        Position::from_fen("r3k3/8/8/8/8/8/8/4K3 w - - 0 1").expect("FEN parse must succeed");

    assert!(evaluate(&white_better).0 > 0);
    assert!(evaluate(&black_better).0 < 0);
}

#[test]
fn mirrored_positions_evaluate_sensibly() {
    let white =
        Position::from_fen("4k3/8/8/3P4/8/8/8/4K3 w - - 0 1").expect("FEN parse must succeed");
    let black =
        Position::from_fen("4k3/8/8/8/3p4/8/8/4K3 w - - 0 1").expect("FEN parse must succeed");
    let symmetric = Position::startpos();

    assert_eq!(evaluate(&white).0, -evaluate(&black).0);
    assert_eq!(evaluate(&symmetric).0, 0);
}
