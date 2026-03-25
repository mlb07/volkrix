use volkrix::core::{Position, STARTPOS_FEN};

#[test]
fn start_position_round_trips() {
    let position = Position::startpos();
    assert_eq!(
        position.to_fen(),
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    );
}

#[test]
fn fen_round_trip_preserves_fields() {
    let fen = "r3k2r/pppq1ppp/2npbn2/4p3/4P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 4 9";
    let position = Position::from_fen(fen).expect("FEN parse must succeed");
    assert_eq!(position.to_fen(), fen);
}

#[test]
fn invalid_fen_is_rejected() {
    let error = Position::from_fen("bad fen").expect_err("FEN parse must fail");
    assert!(error.to_string().contains("6 fields"));
    assert_ne!(STARTPOS_FEN, "bad fen");
}
