use volkrix::core::{Piece, Position, Square};

#[test]
fn standard_move_sequence_updates_position() {
    let mut position = Position::startpos();
    for mv in ["e2e4", "c7c5", "g1f3", "d7d6", "f1b5", "b8c6", "e1g1"] {
        position
            .apply_uci_move(mv)
            .unwrap_or_else(|error| panic!("move {mv} must be legal: {error}"));
    }

    assert_eq!(
        position.piece_at(Square::from_coord_text("g1").expect("square parse must succeed")),
        Some(Piece::WhiteKing)
    );
    assert_eq!(
        position.piece_at(Square::from_coord_text("f1").expect("square parse must succeed")),
        Some(Piece::WhiteRook)
    );
    position.validate().expect("position must remain valid");
}

#[test]
fn en_passant_sequence_is_supported() {
    let mut position = Position::startpos();
    for mv in ["e2e4", "a7a6", "e4e5", "d7d5", "e5d6"] {
        position
            .apply_uci_move(mv)
            .unwrap_or_else(|error| panic!("move {mv} must be legal: {error}"));
    }

    assert_eq!(
        position.piece_at(Square::from_coord_text("d6").expect("square parse must succeed")),
        Some(Piece::WhitePawn)
    );
    assert_eq!(
        position.piece_at(Square::from_coord_text("d5").expect("square parse must succeed")),
        None
    );
}

#[test]
fn promotion_from_fen_is_supported() {
    let mut position =
        Position::from_fen("7k/P7/8/8/8/8/7p/K7 w - - 0 1").expect("FEN parse must succeed");
    position
        .apply_uci_move("a7a8q")
        .expect("promotion move must be legal");

    assert_eq!(
        position.piece_at(Square::from_coord_text("a8").expect("square parse must succeed")),
        Some(Piece::WhiteQueen)
    );
}

#[test]
fn illegal_move_does_not_mutate_position() {
    let mut position = Position::startpos();
    let original_fen = position.to_fen();

    let result = position.apply_uci_move("e2e5");
    assert!(result.is_err());
    assert_eq!(position.to_fen(), original_fen);
}
