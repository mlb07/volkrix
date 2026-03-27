use volkrix::core::{Move, MoveList, ParsedMove, Position, STARTPOS_FEN};

fn legal_move(position: &mut Position, uci: &str) -> Move {
    let parsed = ParsedMove::parse(uci).expect("UCI move parse must succeed");
    let mut legal_moves = MoveList::new();
    position.generate_legal_moves(&mut legal_moves);
    legal_moves
        .as_slice()
        .iter()
        .copied()
        .find(|mv| mv.matches_parsed(parsed))
        .unwrap_or_else(|| panic!("expected legal move {uci}"))
}

#[test]
fn equivalent_positions_share_the_same_zobrist_key() {
    let startpos = Position::startpos();
    let from_fen = Position::from_fen(STARTPOS_FEN).expect("FEN parse must succeed");

    assert_eq!(startpos.zobrist_key(), from_fen.zobrist_key());
}

#[test]
fn non_capturable_en_passant_square_does_not_change_zobrist_key() {
    let without_ep =
        Position::from_fen("4k3/8/8/8/4P3/8/8/4K3 b - - 0 1").expect("FEN parse must succeed");
    let with_ep =
        Position::from_fen("4k3/8/8/8/4P3/8/8/4K3 b - e3 0 1").expect("FEN parse must succeed");

    assert_eq!(without_ep.zobrist_key(), with_ep.zobrist_key());
}

#[test]
fn capturable_en_passant_square_changes_zobrist_key() {
    let without_ep =
        Position::from_fen("4k3/8/8/8/3pP3/8/8/4K3 b - - 0 1").expect("FEN parse must succeed");
    let with_ep =
        Position::from_fen("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1").expect("FEN parse must succeed");

    assert_ne!(without_ep.zobrist_key(), with_ep.zobrist_key());
}

#[test]
fn zobrist_restores_exactly_across_sensitive_move_classes() {
    let cases = [
        (STARTPOS_FEN, "e2e4"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
        ("4k3/8/8/8/2p5/3B4/8/4K3 w - - 0 1", "d3c4"),
        ("7k/P7/8/8/8/8/8/K7 w - - 0 1", "a7a8q"),
        ("4k2r/6P1/8/8/8/8/8/4K3 w - - 0 1", "g7h8q"),
    ];

    for (fen, uci) in cases {
        let mut position = Position::from_fen(fen).expect("FEN parse must succeed");
        let key_before = position.zobrist_key();
        let fen_before = position.to_fen();

        let mv = legal_move(&mut position, uci);
        let undo = position.make_move(mv).expect("move must be legal");
        position.unmake_move(mv, undo);

        assert_eq!(
            position.zobrist_key(),
            key_before,
            "key must restore for {uci}"
        );
        assert_eq!(position.to_fen(), fen_before, "FEN must restore for {uci}");
    }
}
