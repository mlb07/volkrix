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
    assert!(
        response
            .lines
            .iter()
            .any(|line| { line == "option name Hash type spin default 16 min 1 max 512" })
    );
    assert!(
        response
            .lines
            .iter()
            .any(|line| line == "option name Clear Hash type button")
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

#[test]
fn setoption_hash_updates_persistent_hash_size() {
    let mut engine = UciEngine::new();
    assert_eq!(engine.debug_hash_mb(), 16);

    let response = engine.handle_line("setoption name Hash value 32");
    assert!(response.lines.is_empty());
    assert_eq!(engine.debug_hash_mb(), 32);
}

#[test]
fn setoption_hash_rejects_bad_values() {
    let mut engine = UciEngine::new();

    let malformed = engine.handle_line("setoption name Hash value nope");
    assert!(
        malformed
            .lines
            .iter()
            .any(|line| line.contains("invalid setoption name Hash value value 'nope'"))
    );
    assert_eq!(engine.debug_hash_mb(), 16);

    let out_of_range = engine.handle_line("setoption name Hash value 1024");
    assert!(
        out_of_range
            .lines
            .iter()
            .any(|line| line.contains("Hash value must be between 1 and 512"))
    );
    assert_eq!(engine.debug_hash_mb(), 16);
}

#[test]
fn clear_hash_resets_tt_without_corrupting_position_state() {
    let mut engine = UciEngine::new();
    engine.handle_line("position startpos moves e2e4 e7e5");
    let before = engine.position().to_fen();

    let search = engine.handle_line("go depth 2");
    assert!(
        search
            .lines
            .iter()
            .any(|line| line.starts_with("bestmove "))
    );
    assert!(engine.debug_tt_entry_count() > 0);

    let clear = engine.handle_line("setoption name Clear Hash");
    assert!(clear.lines.is_empty());
    assert_eq!(engine.debug_tt_entry_count(), 0);
    assert_eq!(engine.position().to_fen(), before);
}

#[test]
fn go_movetime_returns_a_legal_move() {
    let mut engine = UciEngine::new();
    let response = engine.handle_line("go movetime 10");
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

#[test]
fn clocked_go_returns_a_legal_move() {
    let mut engine = UciEngine::new();
    let response = engine.handle_line("go wtime 1000 btime 1000 winc 100 binc 100 movestogo 10");
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
