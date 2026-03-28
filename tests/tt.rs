#![cfg(any(debug_assertions, feature = "internal-testing"))]

use volkrix::core::Position;
use volkrix::search::{
    BenchConfig, SearchLimits,
    internal::{
        HeuristicProfile, run_profile_bench, run_threaded_profile_bench,
        run_threaded_timed_profile_bench,
    },
    run_bench, search,
};

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

#[test]
#[ignore = "manual benchmark profile report for Phase 9 heuristics"]
fn phase_nine_heuristic_profile_report() {
    let profiles = [
        ("phase8_baseline", HeuristicProfile::Phase8Baseline),
        ("lmr_only", HeuristicProfile::LmrOnly),
        ("phase9_default", HeuristicProfile::Phase9Default),
    ];

    for (name, profile) in profiles {
        let result = run_profile_bench(5, profile);
        println!(
            "profile {name}: nodes {} checksum {:016x} time_ms {}",
            result.total_nodes, result.checksum, result.elapsed_ms
        );
    }
}

#[test]
#[ignore = "manual benchmark profile report for Phase 10 SMP threads"]
fn phase_ten_smp_profile_report() {
    let fixed_depth = [
        ("phase9_baseline_threads1", 1usize),
        ("phase10_default_threads1", 1usize),
        ("phase10_default_threads2", 2usize),
        ("phase10_default_threads4", 4usize),
    ];
    for (name, threads) in fixed_depth {
        let result = run_threaded_profile_bench(5, HeuristicProfile::Phase9Default, threads);
        println!(
            "fixed_depth {name}: threads {threads} nodes {} checksum {:016x} time_ms {} nps {}",
            result.total_nodes,
            result.checksum,
            result.elapsed_ms,
            result.nps()
        );
    }

    let fixed_time = [
        ("phase9_baseline_threads1", 1usize),
        ("phase10_default_threads1", 1usize),
        ("phase10_default_threads2", 2usize),
        ("phase10_default_threads4", 4usize),
    ];
    for (name, threads) in fixed_time {
        let result = run_threaded_timed_profile_bench(50, HeuristicProfile::Phase9Default, threads);
        println!(
            "fixed_time {name}: threads {threads} depth_sum {} nodes {} checksum {:016x} time_ms {}",
            result.total_completed_depth, result.total_nodes, result.checksum, result.elapsed_ms
        );
    }
}

#[test]
fn phase8_baseline_matches_documented_phase8_bench_signature() {
    let result = run_profile_bench(5, HeuristicProfile::Phase8Baseline);
    assert_eq!(result.total_nodes, 541_650);
    assert_eq!(result.checksum, 0x244a_715d_e801_bc83);
}

#[test]
fn phase9_profile_benches_remain_reproducible() {
    let phase_eight_first = run_profile_bench(5, HeuristicProfile::Phase8Baseline);
    let phase_eight_second = run_profile_bench(5, HeuristicProfile::Phase8Baseline);
    assert_eq!(
        phase_eight_first.total_nodes,
        phase_eight_second.total_nodes
    );
    assert_eq!(phase_eight_first.checksum, phase_eight_second.checksum);

    let phase_nine_first = run_profile_bench(5, HeuristicProfile::Phase9Default);
    let phase_nine_second = run_profile_bench(5, HeuristicProfile::Phase9Default);
    assert_eq!(phase_nine_first.total_nodes, phase_nine_second.total_nodes);
    assert_eq!(phase_nine_first.checksum, phase_nine_second.checksum);
}

#[test]
fn phase9_default_now_matches_lmr_only_profile() {
    let lmr_only = run_profile_bench(5, HeuristicProfile::LmrOnly);
    let phase_nine = run_profile_bench(5, HeuristicProfile::Phase9Default);

    assert_eq!(lmr_only.total_nodes, phase_nine.total_nodes);
    assert_eq!(lmr_only.checksum, phase_nine.checksum);
}

#[test]
fn phase10_threads_one_matches_retained_phase9_signature() {
    let result = run_threaded_profile_bench(5, HeuristicProfile::Phase9Default, 1);
    assert_eq!(result.total_nodes, 505_147);
    assert_eq!(result.checksum, 0x244a_71a6_5613_ec7f);
}

#[test]
fn phase10_threads_one_remains_reproducible() {
    let first = run_threaded_profile_bench(5, HeuristicProfile::Phase9Default, 1);
    let second = run_threaded_profile_bench(5, HeuristicProfile::Phase9Default, 1);

    assert_eq!(first.total_nodes, second.total_nodes);
    assert_eq!(first.checksum, second.checksum);
}
