use volkrix::core::{Position, PositionStatus};

#[test]
fn threefold_repetition_is_detected_from_actual_move_cycles() {
    let mut position = Position::startpos();
    for mv in [
        "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8",
    ] {
        position
            .apply_uci_move(mv)
            .expect("cycle move must be legal");
    }

    assert!(position.is_draw_by_repetition());
    assert_eq!(position.status(), PositionStatus::DrawByRepetition);
}

#[test]
fn fifty_move_helper_uses_halfmove_threshold() {
    let mut position =
        Position::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 100 1").expect("FEN parse must succeed");

    assert!(position.is_draw_by_fifty_move());
    assert_eq!(position.status(), PositionStatus::DrawByFiftyMove);
}

#[test]
fn insufficient_material_helper_matches_phase_three_cases() {
    let cases = [
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/8/3NK3 w - - 0 1",
        "4k3/8/8/8/8/8/8/3BK3 w - - 0 1",
        "2b1k3/8/8/8/8/8/8/3BK3 w - - 0 1",
    ];

    for fen in cases {
        let position = Position::from_fen(fen).expect("FEN parse must succeed");
        assert!(position.is_insufficient_material(), "{fen}");
    }
}

#[test]
fn checkmate_and_stalemate_status_are_reported() {
    let mut checkmate =
        Position::from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1").expect("FEN parse must succeed");
    let mut stalemate =
        Position::from_fen("7k/5Q2/7K/8/8/8/8/8 b - - 0 1").expect("FEN parse must succeed");

    assert_eq!(checkmate.status(), PositionStatus::Checkmate);
    assert_eq!(stalemate.status(), PositionStatus::Stalemate);
}
