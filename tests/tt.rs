use volkrix::core::Position;
use volkrix::search::{BenchConfig, SearchLimits, run_bench, search};

#[test]
fn bench_is_reproducible_with_tt_enabled() {
    let first = run_bench(BenchConfig::new(4).with_hash_mb(1));
    let second = run_bench(BenchConfig::new(4).with_hash_mb(1));

    assert_eq!(first.depth, second.depth);
    assert_eq!(first.positions, second.positions);
    assert_eq!(first.total_nodes, second.total_nodes);
    assert_eq!(first.checksum, second.checksum);
}

#[test]
fn bench_is_reproducible_with_tt_disabled() {
    let first = run_bench(BenchConfig::new(4).with_hash_mb(1).without_tt());
    let second = run_bench(BenchConfig::new(4).with_hash_mb(1).without_tt());

    assert_eq!(first.total_nodes, second.total_nodes);
    assert_eq!(first.checksum, second.checksum);
}

#[test]
fn tt_on_and_tt_off_return_same_unique_bestmove_on_curated_position() {
    let fen = "k7/8/1QK5/8/8/8/8/8 w - - 0 1";
    let mut tt_position = Position::from_fen(fen).expect("FEN parse must succeed");
    let mut no_tt_position = Position::from_fen(fen).expect("FEN parse must succeed");

    let tt = search(&mut tt_position, SearchLimits::new(2));
    let no_tt = search(&mut no_tt_position, SearchLimits::new(2).without_tt());

    assert_eq!(tt.score, no_tt.score);
    assert_eq!(
        tt.best_move.map(|mv| mv.to_string()),
        Some("b6b7".to_owned())
    );
    assert_eq!(
        no_tt.best_move.map(|mv| mv.to_string()),
        Some("b6b7".to_owned())
    );
}
