use volkrix::core::{MoveList, Position, Square};

#[test]
fn pinned_knight_has_no_legal_moves() {
    let mut position =
        Position::from_fen("k3r3/8/8/8/8/8/4N3/4K3 w - - 0 1").expect("FEN parse must succeed");
    let mut moves = MoveList::new();
    let knight_square = Square::from_coord_text("e2").expect("square parse must succeed");

    position.generate_legal_moves(&mut moves);

    assert!(
        moves.as_slice().iter().all(|mv| mv.from() != knight_square),
        "pinned knight moves must be filtered"
    );
}

#[test]
fn double_check_only_allows_king_moves() {
    let mut position =
        Position::from_fen("k3r3/8/8/8/1b6/8/8/4K3 w - - 0 1").expect("FEN parse must succeed");
    let mut moves = MoveList::new();
    let king_square = Square::from_coord_text("e1").expect("square parse must succeed");

    position.generate_legal_moves(&mut moves);

    assert!(
        moves.as_slice().iter().all(|mv| mv.from() == king_square),
        "double check must only allow king moves"
    );
}

#[test]
fn illegal_en_passant_discovered_check_is_filtered() {
    let mut position =
        Position::from_fen("k7/8/8/4KPpr/8/8/8/8 w - g6 0 1").expect("FEN parse must succeed");
    let mut moves = MoveList::new();

    position.generate_legal_moves(&mut moves);

    assert!(
        moves.as_slice().iter().all(|mv| mv.to_string() != "f5g6"),
        "en passant that exposes rook attack on the king must be illegal"
    );
}

#[test]
fn double_pawn_push_can_block_slider_check() {
    let mut position =
        Position::from_fen("rnbqkbnr/ppp1pppp/3p4/8/Q7/2P5/PP1PPPPP/RNB1KBNR b KQkq - 0 2")
            .expect("FEN parse must succeed");
    let mut moves = MoveList::new();

    position.generate_legal_moves(&mut moves);

    assert!(
        moves.as_slice().iter().any(|mv| mv.to_string() == "b7b5"),
        "double pawn pushes that block a slider check must remain legal evasions"
    );
}
