#![cfg(any(debug_assertions, feature = "internal-testing"))]

use volkrix::core::Position;
use volkrix::search::{
    BenchConfig, SearchLimits,
    internal::{
        HeuristicProfile, SmpProfile, run_profile_bench, run_profile_bench_with_eval_file,
        run_profile_position_with_eval_file, run_smp_profile_bench, run_smp_timed_profile_bench,
        run_threaded_profile_bench, run_threaded_timed_profile_bench, run_threaded_tiny_nnue_bench,
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
#[ignore = "manual benchmark parity report for all SMP policies"]
fn smp_policy_benchmark_parity_report() {
    for threads in [2usize, 3, 4] {
        for strategy in [
            SmpProfile::Lazy,
            SmpProfile::RootSplit,
            SmpProfile::Diversified,
            SmpProfile::Adaptive,
        ] {
            let result =
                run_smp_profile_bench(6, HeuristicProfile::Phase9Default, threads, strategy);
            println!(
                "smp_ab strategy {strategy:?} threads {threads} nodes {} checksum {:016x} time_ms {} nps {}",
                result.total_nodes,
                result.checksum,
                result.elapsed_ms,
                result.nps()
            );
        }
    }

    for threads in [2usize, 3, 4] {
        for strategy in [
            SmpProfile::Lazy,
            SmpProfile::RootSplit,
            SmpProfile::Diversified,
            SmpProfile::Adaptive,
        ] {
            let result = run_smp_timed_profile_bench(
                100,
                HeuristicProfile::Phase9Default,
                threads,
                strategy,
            );
            println!(
                "smp_ab_timed strategy {strategy:?} threads {threads} depth_sum {} nodes {} checksum {:016x} time_ms {}",
                result.total_completed_depth,
                result.total_nodes,
                result.checksum,
                result.elapsed_ms,
            );
        }
    }
}

#[test]
#[ignore = "manual isolated Multi-Cut A/B profile; strength still requires paired SPRT"]
fn multi_cut_ab_profile_report() {
    for profile in [
        HeuristicProfile::Phase9Default,
        HeuristicProfile::MultiCutEnabled,
    ] {
        let result = run_profile_bench(8, profile);
        println!(
            "multi_cut_ab profile {profile:?} nodes {} checksum {:016x} time_ms {} nps {}",
            result.total_nodes,
            result.checksum,
            result.elapsed_ms,
            result.nps(),
        );
    }
}

#[test]
#[ignore = "manual isolated razoring A/B profile; strength still requires paired SPRT"]
fn razoring_ab_profile_report() {
    for profile in [
        HeuristicProfile::Phase9Default,
        HeuristicProfile::RazoringEnabled,
    ] {
        let result = run_profile_bench(9, profile);
        println!(
            "razoring_ab profile {profile:?} nodes {} checksum {:016x} time_ms {} nps {}",
            result.total_nodes,
            result.checksum,
            result.elapsed_ms,
            result.nps(),
        );
    }
}

#[test]
#[ignore = "manual NNUE razoring A/B profile; set VOLKRIX_PROFILE_EVALFILE"]
fn razoring_nnue_ab_profile_report() {
    let eval_file = std::env::var("VOLKRIX_PROFILE_EVALFILE")
        .expect("VOLKRIX_PROFILE_EVALFILE must identify the frozen NNUE under test");
    for profile in [
        HeuristicProfile::Phase9Default,
        HeuristicProfile::RazoringEnabled,
    ] {
        let result = run_profile_bench_with_eval_file(8, profile, &eval_file);
        println!(
            "razoring_nnue_ab profile {profile:?} nodes {} checksum {:016x} time_ms {} nps {} evaluator {}",
            result.total_nodes,
            result.checksum,
            result.elapsed_ms,
            result.nps(),
            result.evaluator,
        );
    }

    let fens = [
        volkrix::core::STARTPOS_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 2 3",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    ];
    for (index, fen) in fens.iter().enumerate() {
        for profile in [
            HeuristicProfile::Phase9Default,
            HeuristicProfile::RazoringEnabled,
        ] {
            let result = run_profile_position_with_eval_file(fen, 8, profile, &eval_file);
            println!(
                "razoring_nnue_position {} profile {profile:?} bestmove {:?} score {} nodes {}",
                index + 1,
                result.best_move,
                result.score.0,
                result.nodes,
            );
        }
    }
}

#[test]
#[ignore = "manual NNUE capture-LMR A/B profile; set VOLKRIX_PROFILE_EVALFILE"]
fn capture_lmr_nnue_ab_profile_report() {
    let eval_file = std::env::var("VOLKRIX_PROFILE_EVALFILE")
        .expect("VOLKRIX_PROFILE_EVALFILE must identify the frozen NNUE under test");
    let depth = std::env::var("VOLKRIX_PROFILE_DEPTH")
        .map(|value| {
            value
                .parse::<u8>()
                .expect("VOLKRIX_PROFILE_DEPTH must be u8")
        })
        .unwrap_or(9);
    let profiles = [
        HeuristicProfile::Phase9Default,
        HeuristicProfile::CaptureLmrEnabled,
    ];

    for profile in profiles {
        let first = run_profile_bench_with_eval_file(depth, profile, &eval_file);
        let second = run_profile_bench_with_eval_file(depth, profile, &eval_file);
        assert_eq!(first.total_nodes, second.total_nodes);
        assert_eq!(first.checksum, second.checksum);
        println!(
            "capture_lmr_nnue_ab depth {depth} profile {profile:?} nodes {} checksum {:016x} time_ms {} nps {} evaluator {}",
            first.total_nodes,
            first.checksum,
            first.elapsed_ms,
            first.nps(),
            first.evaluator,
        );
    }

    let fens = [
        volkrix::core::STARTPOS_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r2q1rk1/ppp2ppp/2npbn2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8",
        "2kr3r/ppp2ppp/2n1bn2/2b1p3/4P3/2NP1N2/PPP2PPP/R1B2RK1 b - - 0 9",
    ];
    for (index, fen) in fens.iter().enumerate() {
        for profile in profiles {
            let result = run_profile_position_with_eval_file(fen, depth, profile, &eval_file);
            println!(
                "capture_lmr_nnue_position {} depth {depth} profile {profile:?} bestmove {} score {} nodes {}",
                index + 1,
                result
                    .best_move
                    .map_or_else(|| "none".to_owned(), |mv| mv.to_string()),
                result.score.0,
                result.nodes,
            );
        }
    }
}

#[test]
#[ignore = "manual isolated search-ordering A/B profiles; strength requires paired testing"]
fn search_ordering_candidate_ab_profile_report() {
    let eval_file = std::env::var("VOLKRIX_PROFILE_EVALFILE").ok();
    let depth = std::env::var("VOLKRIX_PROFILE_DEPTH")
        .map(|value| {
            value
                .parse::<u8>()
                .expect("VOLKRIX_PROFILE_DEPTH must be u8")
        })
        .unwrap_or(6);
    let candidates = [
        HeuristicProfile::OrderedProbCutEnabled,
        HeuristicProfile::CaptureHistoryEnabled,
        HeuristicProfile::MultiPlyContinuationEnabled,
        HeuristicProfile::ContextualLmrEnabled,
    ];
    for candidate in candidates {
        for profile in [HeuristicProfile::Phase9Default, candidate] {
            let first = eval_file.as_ref().map_or_else(
                || run_profile_bench(depth, profile),
                |path| run_profile_bench_with_eval_file(depth, profile, path),
            );
            let second = eval_file.as_ref().map_or_else(
                || run_profile_bench(depth, profile),
                |path| run_profile_bench_with_eval_file(depth, profile, path),
            );
            assert_eq!(first.total_nodes, second.total_nodes);
            assert_eq!(first.checksum, second.checksum);
            println!(
                "search_ordering_ab depth {depth} candidate {candidate:?} profile {profile:?} nodes {} checksum {:016x} time_ms {} nps {} evaluator {}",
                first.total_nodes,
                first.checksum,
                first.elapsed_ms,
                first.nps(),
                first.evaluator,
            );
        }
    }
}

#[test]
#[ignore = "manual no-tablebase profile report for Phase 11 disabled-path preservation"]
fn phase_eleven_no_tablebase_profile_report() {
    let baseline = run_threaded_profile_bench(5, HeuristicProfile::Phase9Default, 1);
    println!(
        "phase10_baseline_syzygy_empty_threads1: nodes {} checksum {:016x} time_ms {} nps {}",
        baseline.total_nodes,
        baseline.checksum,
        baseline.elapsed_ms,
        baseline.nps()
    );

    let current = run_threaded_profile_bench(5, HeuristicProfile::Phase9Default, 1);
    println!(
        "phase11_default_syzygy_empty_threads1: nodes {} checksum {:016x} time_ms {} nps {}",
        current.total_nodes,
        current.checksum,
        current.elapsed_ms,
        current.nps()
    );
}

#[test]
fn phase8_baseline_remains_reproducible() {
    let first = run_profile_bench(5, HeuristicProfile::Phase8Baseline);
    let second = run_profile_bench(5, HeuristicProfile::Phase8Baseline);
    assert_eq!(first.total_nodes, second.total_nodes);
    assert_eq!(first.checksum, second.checksum);
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
fn correction_history_on_and_off_profiles_are_each_reproducible() {
    for profile in [
        HeuristicProfile::Phase9Default,
        HeuristicProfile::CorrectionHistoryEnabled,
    ] {
        let first = run_profile_bench(5, profile);
        let second = run_profile_bench(5, profile);
        assert_eq!(first.total_nodes, second.total_nodes);
        assert_eq!(first.checksum, second.checksum);
    }
}

#[test]
#[ignore = "manual isolated correction-history A/B profile report"]
fn correction_history_profile_report() {
    for (name, profile) in [
        ("correction_history_off", HeuristicProfile::Phase9Default),
        (
            "correction_history_on",
            HeuristicProfile::CorrectionHistoryEnabled,
        ),
    ] {
        let result = run_profile_bench(7, profile);
        println!(
            "{name}: nodes {} checksum {:016x} time_ms {} nps {}",
            result.total_nodes,
            result.checksum,
            result.elapsed_ms,
            result.nps()
        );
    }
}

#[test]
fn lmr_only_remains_reproducible() {
    let first = run_profile_bench(5, HeuristicProfile::LmrOnly);
    let second = run_profile_bench(5, HeuristicProfile::LmrOnly);
    assert_eq!(first.total_nodes, second.total_nodes);
    assert_eq!(first.checksum, second.checksum);
}

#[test]
fn single_thread_service_path_matches_direct_search_path() {
    let direct = run_profile_bench(5, HeuristicProfile::Phase9Default);
    let service = run_threaded_profile_bench(5, HeuristicProfile::Phase9Default, 1);
    assert_eq!(direct.total_nodes, service.total_nodes);
    assert_eq!(direct.checksum, service.checksum);
}

#[test]
fn phase10_threads_one_remains_reproducible() {
    let first = run_threaded_profile_bench(5, HeuristicProfile::Phase9Default, 1);
    let second = run_threaded_profile_bench(5, HeuristicProfile::Phase9Default, 1);

    assert_eq!(first.total_nodes, second.total_nodes);
    assert_eq!(first.checksum, second.checksum);
}

#[test]
fn phase11_syzygy_empty_threads_one_remains_reproducible() {
    let first = run_threaded_profile_bench(5, HeuristicProfile::Phase9Default, 1);
    let second = run_threaded_profile_bench(5, HeuristicProfile::Phase9Default, 1);

    assert_eq!(first.total_nodes, second.total_nodes);
    assert_eq!(first.checksum, second.checksum);
}

#[test]
#[ignore = "manual benchmark profile report for Phase 12 NNUE integration"]
fn phase_twelve_nnue_profile_report() {
    let baseline = run_threaded_profile_bench(5, HeuristicProfile::Phase9Default, 1);
    println!(
        "phase11_baseline_evalfile_empty_threads1: nodes {} checksum {:016x} time_ms {} nps {}",
        baseline.total_nodes,
        baseline.checksum,
        baseline.elapsed_ms,
        baseline.nps()
    );

    let current = run_threaded_profile_bench(5, HeuristicProfile::Phase9Default, 1);
    println!(
        "phase12_default_evalfile_empty_threads1: nodes {} checksum {:016x} time_ms {} nps {}",
        current.total_nodes,
        current.checksum,
        current.elapsed_ms,
        current.nps()
    );

    let tiny_threads_one = run_threaded_tiny_nnue_bench(5, 1);
    println!(
        "phase12_tiny_nnue_threads1: nodes {} checksum {:016x} time_ms {} nps {}",
        tiny_threads_one.total_nodes,
        tiny_threads_one.checksum,
        tiny_threads_one.elapsed_ms,
        tiny_threads_one.nps()
    );

    let tiny_threads_two = run_threaded_tiny_nnue_bench(5, 2);
    println!(
        "phase12_tiny_nnue_threads2: nodes {} checksum {:016x} time_ms {} nps {}",
        tiny_threads_two.total_nodes,
        tiny_threads_two.checksum,
        tiny_threads_two.elapsed_ms,
        tiny_threads_two.nps()
    );
}

#[test]
fn phase12_evalfile_empty_threads_one_remains_reproducible() {
    let first = run_threaded_profile_bench(5, HeuristicProfile::Phase9Default, 1);
    let second = run_threaded_profile_bench(5, HeuristicProfile::Phase9Default, 1);

    assert_eq!(first.total_nodes, second.total_nodes);
    assert_eq!(first.checksum, second.checksum);
}
