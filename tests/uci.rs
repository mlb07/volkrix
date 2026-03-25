use volkrix::{core::Position, uci::UciEngine};

#[test]
fn uci_handshake_returns_required_lines() {
    let mut engine = UciEngine::new();
    let response = engine.handle_line("uci");
    assert!(response.lines.iter().any(|line| line == "uciok"));
    assert!(
        response
            .lines
            .iter()
            .any(|line| line.starts_with("id name Volkrix"))
    );
}

#[test]
fn invalid_position_command_does_not_corrupt_state() {
    let mut engine = UciEngine::new();
    let original_fen = engine.position().to_fen();

    let response = engine.handle_line("position startpos moves e2e5");
    assert!(
        response
            .lines
            .iter()
            .any(|line| line.contains("illegal move"))
    );
    assert_eq!(engine.position().to_fen(), original_fen);
}

#[test]
fn go_depth_returns_a_legal_move() {
    let mut engine = UciEngine::new();
    let response = engine.handle_line("go depth 1");
    let bestmove_line = response
        .lines
        .iter()
        .find(|line| line.starts_with("bestmove "))
        .expect("bestmove line must exist");
    let bestmove = bestmove_line
        .strip_prefix("bestmove ")
        .expect("bestmove line must contain prefix");
    assert_ne!(bestmove, "0000");

    let mut position = Position::startpos();
    position
        .apply_uci_move(bestmove)
        .expect("bestmove must be legal");
}
