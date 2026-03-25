use volkrix::core::{CastlingRights, Move, PieceType, Square};

#[test]
fn square_round_trip_works() {
    let square = Square::from_coord_text("e4").expect("square parse must succeed");
    assert_eq!(square.to_string(), "e4");
}

#[test]
fn move_round_trip_works() {
    let from = Square::from_coord_text("a7").expect("square parse must succeed");
    let to = Square::from_coord_text("a8").expect("square parse must succeed");
    let mv = Move::new(from, to).with_promotion(PieceType::Queen);
    assert_eq!(mv.to_string(), "a7a8q");
}

#[test]
fn castling_rights_render_to_fen() {
    let mut rights = CastlingRights::NONE;
    rights.insert(CastlingRights::WHITE_KINGSIDE);
    rights.insert(CastlingRights::BLACK_QUEENSIDE);
    assert_eq!(rights.to_fen(), "Kq");
}
