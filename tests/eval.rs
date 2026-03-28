#![cfg(any(debug_assertions, feature = "internal-testing"))]

use volkrix::{
    core::Position,
    search::{eval::debug_evaluate_breakdown, evaluate},
};

#[test]
fn tapered_phase_shifts_between_opening_middlegame_and_endgame() {
    let opening = Position::startpos();
    let middlegame =
        Position::from_fen("r3k2r/ppp2ppp/2n1bn2/3p4/3P4/2N1BN2/PPP2PPP/R3K2R w KQkq - 0 1")
            .expect("FEN parse must succeed");
    let endgame =
        Position::from_fen("4k3/8/8/3P4/8/8/8/4K3 w - - 0 1").expect("FEN parse must succeed");

    let opening_breakdown = debug_evaluate_breakdown(&opening);
    let middlegame_breakdown = debug_evaluate_breakdown(&middlegame);
    let endgame_breakdown = debug_evaluate_breakdown(&endgame);

    assert!(opening_breakdown.phase > middlegame_breakdown.phase);
    assert!(middlegame_breakdown.phase > endgame_breakdown.phase);
}

#[test]
fn eval_is_deterministic_and_does_not_mutate_position_state() {
    let position =
        Position::from_fen("r2q1rk1/ppp2ppp/2npbn2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8")
            .expect("FEN parse must succeed");
    let before_fen = position.to_fen();
    let before_zobrist = position.zobrist_key();
    let before_search_key = position.debug_search_key();
    let before_history = position.debug_repetition_history_snapshot();

    let first = evaluate(&position);
    let second = evaluate(&position);

    assert_eq!(first, second);
    assert_eq!(position.to_fen(), before_fen);
    assert_eq!(position.zobrist_key(), before_zobrist);
    assert_eq!(position.debug_search_key(), before_search_key);
    assert_eq!(position.debug_repetition_history_snapshot(), before_history);
    position.validate().expect("position must remain valid");
}

#[test]
fn bishop_pair_bonus_is_present_only_with_two_bishops() {
    let with_pair =
        Position::from_fen("4k3/8/8/8/8/8/3BB3/4K3 w - - 0 1").expect("FEN parse must succeed");
    let without_pair =
        Position::from_fen("4k3/8/8/8/8/8/4B3/4K3 w - - 0 1").expect("FEN parse must succeed");

    let with_pair_breakdown = debug_evaluate_breakdown(&with_pair);
    let without_pair_breakdown = debug_evaluate_breakdown(&without_pair);

    assert!(with_pair_breakdown.bishop_pair > 0);
    assert_eq!(without_pair_breakdown.bishop_pair, 0);
}

#[test]
fn passed_pawn_bonus_grows_with_advance() {
    let advanced =
        Position::from_fen("4k3/8/3P4/8/8/8/8/4K3 w - - 0 1").expect("FEN parse must succeed");
    let less_advanced =
        Position::from_fen("4k3/8/8/8/3P4/8/8/4K3 w - - 0 1").expect("FEN parse must succeed");

    let advanced_breakdown = debug_evaluate_breakdown(&advanced);
    let less_advanced_breakdown = debug_evaluate_breakdown(&less_advanced);

    assert!(advanced_breakdown.passed_pawns > less_advanced_breakdown.passed_pawns);
}

#[test]
fn isolated_and_doubled_pawns_are_penalized() {
    let healthy =
        Position::from_fen("4k3/8/8/8/8/3P4/2P1P3/4K3 w - - 0 1").expect("FEN parse must succeed");
    let damaged =
        Position::from_fen("4k3/8/8/8/8/2P5/P1P5/4K3 w - - 0 1").expect("FEN parse must succeed");

    let healthy_breakdown = debug_evaluate_breakdown(&healthy);
    let damaged_breakdown = debug_evaluate_breakdown(&damaged);

    assert!(healthy_breakdown.pawn_structure > damaged_breakdown.pawn_structure);
}

#[test]
fn rook_file_terms_distinguish_open_semi_open_and_blocked_files() {
    let open =
        Position::from_fen("4k3/8/8/8/8/8/8/3RK3 w - - 0 1").expect("FEN parse must succeed");
    let semi_open =
        Position::from_fen("4k3/3p4/8/8/8/8/8/3RK3 w - - 0 1").expect("FEN parse must succeed");
    let blocked =
        Position::from_fen("4k3/3p4/8/8/8/8/3P4/3RK3 w - - 0 1").expect("FEN parse must succeed");

    let open_breakdown = debug_evaluate_breakdown(&open);
    let semi_open_breakdown = debug_evaluate_breakdown(&semi_open);
    let blocked_breakdown = debug_evaluate_breakdown(&blocked);

    assert!(open_breakdown.rook_placement > semi_open_breakdown.rook_placement);
    assert!(semi_open_breakdown.rook_placement > blocked_breakdown.rook_placement);
}

#[test]
fn king_safety_prefers_a_sheltered_king() {
    let sheltered =
        Position::from_fen("4k3/8/8/8/8/8/5PPP/6K1 w - - 0 1").expect("FEN parse must succeed");
    let exposed =
        Position::from_fen("4k3/8/8/8/8/8/PPP5/6K1 w - - 0 1").expect("FEN parse must succeed");

    let sheltered_breakdown = debug_evaluate_breakdown(&sheltered);
    let exposed_breakdown = debug_evaluate_breakdown(&exposed);

    assert!(sheltered_breakdown.king_safety > exposed_breakdown.king_safety);
}

#[test]
fn mobility_prefers_freer_pieces_over_cramped_placement() {
    let free =
        Position::from_fen("4k3/8/8/8/2BN4/3Q4/8/3RK3 w - - 0 1").expect("FEN parse must succeed");
    let cramped =
        Position::from_fen("4k3/8/8/8/8/8/8/RNBQK3 w - - 0 1").expect("FEN parse must succeed");

    let free_breakdown = debug_evaluate_breakdown(&free);
    let cramped_breakdown = debug_evaluate_breakdown(&cramped);

    assert!(free_breakdown.mobility > cramped_breakdown.mobility);
}

#[test]
fn basic_threat_terms_reward_piece_pressure() {
    let active =
        Position::from_fen("4k3/8/5n2/4P3/8/8/8/4K3 w - - 0 1").expect("FEN parse must succeed");
    let quiet =
        Position::from_fen("4k3/7n/8/4P3/8/8/8/4K3 w - - 0 1").expect("FEN parse must succeed");

    let active_breakdown = debug_evaluate_breakdown(&active);
    let quiet_breakdown = debug_evaluate_breakdown(&quiet);

    assert!(active_breakdown.threats > quiet_breakdown.threats);
}
