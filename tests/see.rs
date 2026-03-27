use volkrix::core::{Move, MoveList, ParsedMove, Position, Value};

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
fn quiet_non_promotion_returns_zero() {
    let mut position = Position::startpos();
    let mv = legal_move(&mut position, "g1f3");

    assert_eq!(position.see(mv), Value(0));
}

#[test]
fn quiet_promotion_returns_promotion_gain_over_a_pawn() {
    let mut position =
        Position::from_fen("7k/P7/8/8/8/8/8/K7 w - - 0 1").expect("FEN parse must succeed");
    let mv = legal_move(&mut position, "a7a8q");

    assert_eq!(position.see(mv), Value(800));
}

#[test]
fn simple_undefended_capture_returns_captured_value() {
    let mut position =
        Position::from_fen("4k3/8/8/8/8/8/4r3/3QK3 w - - 0 1").expect("FEN parse must succeed");
    let mv = legal_move(&mut position, "d1e2");

    assert_eq!(position.see(mv), Value(500));
}

#[test]
fn defended_exchange_can_be_negative() {
    let mut position =
        Position::from_fen("4k3/8/8/8/2p5/3n4/4B3/4K3 w - - 0 1").expect("FEN parse must succeed");
    let mv = legal_move(&mut position, "e2d3");

    assert_eq!(position.see(mv), Value(-10));
}

#[test]
fn xray_recapture_is_accounted_for() {
    let mut position =
        Position::from_fen("3qk3/8/8/3r4/8/8/3R4/4K3 w - - 0 1").expect("FEN parse must succeed");
    let mv = legal_move(&mut position, "d2d5");

    assert_eq!(position.see(mv), Value(0));
}

#[test]
fn en_passant_see_uses_the_captured_pawn_value() {
    let mut position =
        Position::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1").expect("FEN parse must succeed");
    let mv = legal_move(&mut position, "e5d6");

    assert_eq!(position.see(mv), Value(100));
}

#[test]
fn capture_promotion_adds_capture_value_and_promotion_gain() {
    let mut position =
        Position::from_fen("4k2r/6P1/8/8/8/8/8/4K3 w - - 0 1").expect("FEN parse must succeed");
    let mv = legal_move(&mut position, "g7h8q");

    assert_eq!(position.see(mv), Value(1300));
}
