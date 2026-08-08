use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use volkrix::{core::Position, search::internal::phase12_test_evalfile_path, uci::UciEngine};

const STOCKFISH_18_NETWORK_FILE: &str = "nn-c288c895ea92.nnue";
static TEMP_DIRECTORY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new(label: &str) -> Self {
        let sequence = TEMP_DIRECTORY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("volkrix-{label}-{}-{sequence}", std::process::id()));
        fs::create_dir(&path).expect("test directory must be created");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }

    fn executable_path(&self) -> PathBuf {
        self.0.join("volkrix")
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("test directory must be removed");
    }
}

fn assert_info_pvs_are_legal(response_lines: &[String], root_position: &Position) {
    for line in response_lines {
        let Some((_, pv_text)) = line.split_once(" pv ") else {
            continue;
        };

        let mut position = root_position.clone();
        for move_text in pv_text.split_whitespace() {
            position
                .apply_uci_move(move_text)
                .unwrap_or_else(|error| panic!("illegal PV move {move_text} in '{line}': {error}"));
        }
    }
}

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
            .any(|line| line == "id author Monty Bognar")
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
    assert!(
        response
            .lines
            .iter()
            .any(|line| line == "option name Threads type spin default 1 min 1 max 64")
    );
    assert!(
        response.lines.iter().any(|line| {
            line == "option name Move Overhead type spin default 10 min 0 max 5000"
        })
    );
    assert!(
        response
            .lines
            .iter()
            .any(|line| line == "option name Ponder type check default false")
    );
    assert!(
        response
            .lines
            .iter()
            .any(|line| line == "option name SyzygyPath type string default")
    );
    assert!(
        response
            .lines
            .iter()
            .any(|line| { line == "option name SyzygyProbeLimit type spin default 7 min 0 max 7" })
    );
    assert!(
        response
            .lines
            .iter()
            .any(|line| line == "option name Syzygy50MoveRule type check default true")
    );
    assert!(
        response
            .lines
            .iter()
            .any(|line| line == "option name EvalFile type string default")
    );
    assert!(
        response
            .lines
            .iter()
            .any(|line| line == "option name SmallEvalFile type string default")
    );
    assert!(response.lines.iter().any(|line| {
        line == "option name DualEvalPolicy type combo default off var off var small-fallback"
    }));
    assert!(response.lines.iter().any(|line| {
        line == "option name DualEvalThreshold type spin default 200 min 0 max 2000"
    }));
}

#[test]
fn automatic_evalfile_discovery_prefers_the_explicit_environment_path() {
    let directory = TestDirectory::new("eval-discovery-env");
    let sibling = directory.path().join(STOCKFISH_18_NETWORK_FILE);
    fs::write(&sibling, b"not a usable network").expect("sibling fixture must be written");
    let explicit = phase12_test_evalfile_path();

    let mut engine = UciEngine::debug_new_with_eval_discovery(
        Some(&explicit),
        Some(&directory.executable_path()),
    );

    assert_eq!(engine.debug_eval_file(), explicit);
    let handshake = engine.handle_line("uci");
    assert!(
        handshake.lines.iter().any(|line| {
            line == &format!("option name EvalFile type string default {explicit}")
        })
    );
    assert!(
        handshake
            .lines
            .iter()
            .all(|line| !line.contains("warning:"))
    );
}

#[test]
fn corrupt_automatic_sibling_falls_back_to_classical_with_uci_diagnostic() {
    let directory = TestDirectory::new("eval-discovery-corrupt");
    let sibling = directory.path().join(STOCKFISH_18_NETWORK_FILE);
    fs::write(&sibling, b"not a usable network").expect("sibling fixture must be written");

    let mut engine =
        UciEngine::debug_new_with_eval_discovery(None, Some(&directory.executable_path()));

    assert_eq!(engine.debug_eval_file(), "");
    let handshake = engine.handle_line("uci");
    assert!(
        handshake
            .lines
            .iter()
            .any(|line| line == "option name EvalFile type string default")
    );
    assert!(handshake.lines.iter().any(|line| {
        line.starts_with("info string warning: automatic EvalFile")
            && line.contains("using classical evaluation")
    }));
    assert_eq!(handshake.lines.last().map(String::as_str), Some("uciok"));
}

#[test]
fn missing_automatic_sibling_silently_keeps_classical_evaluation() {
    let directory = TestDirectory::new("eval-discovery-missing");
    let mut engine =
        UciEngine::debug_new_with_eval_discovery(None, Some(&directory.executable_path()));

    assert_eq!(engine.debug_eval_file(), "");
    let handshake = engine.handle_line("uci");
    assert!(
        handshake
            .lines
            .iter()
            .all(|line| !line.contains("warning:"))
    );
}

#[test]
fn explicit_empty_evalfile_disables_an_automatically_loaded_network() {
    let explicit = phase12_test_evalfile_path();
    let directory = TestDirectory::new("eval-discovery-disable");
    let mut engine = UciEngine::debug_new_with_eval_discovery(
        Some(&explicit),
        Some(&directory.executable_path()),
    );
    assert_eq!(engine.debug_eval_file(), explicit);

    assert!(
        engine
            .handle_line("setoption name EvalFile value")
            .lines
            .is_empty()
    );
    assert_eq!(engine.debug_eval_file(), "");
    assert!(
        engine
            .handle_line("uci")
            .lines
            .iter()
            .any(|line| { line == "option name EvalFile type string default" })
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
    assert_info_pvs_are_legal(&response.lines, &Position::startpos());
    assert!(
        response
            .lines
            .iter()
            .any(|line| line.starts_with("info depth 1 ") && line.contains(" nps "))
    );
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
fn go_nodes_returns_a_legal_move_and_accepts_combined_depth_cap() {
    let mut engine = UciEngine::new();
    for command in ["go nodes 257", "go depth 10 nodes 257"] {
        let response = engine.handle_line(command);
        assert!(
            response
                .lines
                .iter()
                .all(|line| !line.starts_with("info string error:")),
            "'{command}' must be accepted"
        );
        let bestmove = response
            .lines
            .iter()
            .find_map(|line| line.strip_prefix("bestmove "))
            .expect("node-limited search must return bestmove");
        assert_ne!(bestmove, "0000");
        let mut position = Position::startpos();
        position
            .apply_uci_move(bestmove)
            .expect("node-limited bestmove must be legal");
    }
}

#[test]
fn single_thread_fixed_node_search_is_reproducible_from_clean_state() {
    fn stable_result(command: &str) -> (String, Vec<String>) {
        let mut engine = UciEngine::new();
        assert!(
            engine
                .handle_line("setoption name Threads value 1")
                .lines
                .is_empty()
        );
        let response = engine.handle_line(command);
        let bestmove = response
            .lines
            .iter()
            .find_map(|line| line.strip_prefix("bestmove "))
            .expect("fixed-node search must return bestmove")
            .to_owned();
        let final_info = response
            .lines
            .iter()
            .rfind(|line| line.starts_with("info depth "))
            .expect("fixed-node search must report info");
        let fields = final_info.split_whitespace().collect::<Vec<_>>();
        let mut stable_fields = Vec::new();
        let mut index = 0usize;
        while index < fields.len() {
            if matches!(fields[index], "nps" | "time") {
                index += 2;
            } else {
                stable_fields.push(fields[index].to_owned());
                index += 1;
            }
        }
        (bestmove, stable_fields)
    }

    let first = stable_result("go nodes 4001");
    let second = stable_result("go nodes 4001");
    assert_eq!(first, second);
    assert!(second.1.iter().any(|field| field == "nodes"));
}

#[test]
fn go_nodes_validates_its_value_and_mate_remains_unadvertised() {
    let mut engine = UciEngine::new();
    for command in [
        "go nodes",
        "go nodes 0",
        "go nodes nope",
        "go nodes 1 nodes 2",
        "go mate 3",
    ] {
        let response = engine.handle_line(command);
        assert!(
            response
                .lines
                .iter()
                .any(|line| line.starts_with("info string error:")),
            "'{command}' must be rejected"
        );
    }
}

#[test]
fn debug_command_is_accepted_as_uci_noop() {
    let mut engine = UciEngine::new();
    assert!(engine.handle_line("debug on").lines.is_empty());
    assert!(engine.handle_line("debug off").lines.is_empty());

    let response = engine.handle_line("debug maybe");
    assert!(
        response
            .lines
            .iter()
            .any(|line| line.contains("unsupported debug argument 'maybe'"))
    );
}

#[test]
fn go_depth_reports_only_legal_pv_lines() {
    let mut engine = UciEngine::new();
    let position_command = "position startpos moves e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1";
    let root_position = {
        let mut position = Position::startpos();
        for move_text in position_command
            .strip_prefix("position startpos moves ")
            .expect("test command must contain moves")
            .split_whitespace()
        {
            position
                .apply_uci_move(move_text)
                .expect("test move sequence must be legal");
        }
        position
    };

    assert!(engine.handle_line(position_command).lines.is_empty());
    let first = engine.handle_line("go depth 5");
    assert_info_pvs_are_legal(&first.lines, &root_position);
    let second = engine.handle_line("go depth 5");
    assert_info_pvs_are_legal(&second.lines, &root_position);
}

#[test]
fn go_searchmoves_restricts_root_candidates() {
    let mut engine = UciEngine::new();
    let response = engine.handle_line("go searchmoves e2e4 depth 2");
    assert_info_pvs_are_legal(&response.lines, &Position::startpos());
    assert!(response.lines.iter().any(|line| line == "bestmove e2e4"));
}

#[test]
fn go_searchmoves_accepts_options_after_move_list() {
    let mut engine = UciEngine::new();
    let response = engine.handle_line("go searchmoves e2e4 d2d4 depth 1");
    assert_info_pvs_are_legal(&response.lines, &Position::startpos());
    let bestmove_line = response
        .lines
        .iter()
        .find(|line| line.starts_with("bestmove "))
        .expect("bestmove line must exist");
    let bestmove = bestmove_line
        .strip_prefix("bestmove ")
        .expect("bestmove line must contain prefix");
    assert!(matches!(bestmove, "e2e4" | "d2d4"));
}

#[test]
fn go_searchmoves_rejects_illegal_root_moves_without_searching() {
    let mut engine = UciEngine::new();
    let response = engine.handle_line("go searchmoves e2e5 depth 1");
    assert!(
        response
            .lines
            .iter()
            .any(|line| line.contains("illegal go searchmoves move 'e2e5'"))
    );
    assert!(
        response
            .lines
            .iter()
            .all(|line| !line.starts_with("bestmove "))
    );
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
fn setoption_threads_updates_configured_thread_count() {
    let mut engine = UciEngine::new();
    assert_eq!(engine.debug_threads(), 1);

    let response = engine.handle_line("setoption name Threads value 4");
    assert!(response.lines.is_empty());
    assert_eq!(engine.debug_threads(), 4);
    let handshake = engine.handle_line("uci");
    assert!(
        handshake
            .lines
            .iter()
            .any(|line| line == "option name Threads type spin default 1 min 1 max 64")
    );
}

#[test]
fn setoption_threads_rejects_bad_values() {
    let mut engine = UciEngine::new();

    let malformed = engine.handle_line("setoption name Threads value nope");
    assert!(
        malformed
            .lines
            .iter()
            .any(|line| line.contains("invalid setoption name Threads value value 'nope'"))
    );
    assert_eq!(engine.debug_threads(), 1);

    let zero = engine.handle_line("setoption name Threads value 0");
    assert!(
        zero.lines
            .iter()
            .any(|line| line.contains("Threads value must be between 1 and 64"))
    );
    assert_eq!(engine.debug_threads(), 1);

    let too_high = engine.handle_line("setoption name Threads value 65");
    assert!(
        too_high
            .lines
            .iter()
            .any(|line| line.contains("Threads value must be between 1 and 64"))
    );
    assert_eq!(engine.debug_threads(), 1);
}

#[test]
fn setoption_move_overhead_updates_and_validates_configuration() {
    let mut engine = UciEngine::new();
    assert_eq!(engine.debug_move_overhead_ms(), 10);

    let accepted = engine.handle_line("setoption name Move Overhead value 75");
    assert!(accepted.lines.is_empty());
    assert_eq!(engine.debug_move_overhead_ms(), 75);

    for command in [
        "setoption name Move Overhead value nope",
        "setoption name Move Overhead value 5001",
        "setoption name Move Overhead",
        "setoption name Move Overhead value 1 2",
    ] {
        let rejected = engine.handle_line(command);
        assert!(
            rejected
                .lines
                .iter()
                .any(|line| line.starts_with("info string error:")),
            "'{command}' must be rejected"
        );
        assert_eq!(engine.debug_move_overhead_ms(), 75);
    }
}

#[test]
fn setoption_ponder_is_advertised_and_validated() {
    let mut engine = UciEngine::new();
    assert!(!engine.debug_ponder_enabled());
    assert!(
        engine
            .handle_line("setoption name Ponder value true")
            .lines
            .is_empty()
    );
    assert!(engine.debug_ponder_enabled());
    assert!(
        engine
            .handle_line("setoption name Ponder value false")
            .lines
            .is_empty()
    );
    assert!(!engine.debug_ponder_enabled());

    for command in [
        "setoption name Ponder",
        "setoption name Ponder value maybe",
        "setoption name Ponder value true false",
    ] {
        let response = engine.handle_line(command);
        assert!(
            response
                .lines
                .iter()
                .any(|line| line.starts_with("info string error:")),
            "'{command}' must be rejected"
        );
        assert!(!engine.debug_ponder_enabled());
    }
}

#[test]
fn go_rejects_ambiguous_or_incomplete_limit_arguments() {
    let mut engine = UciEngine::new();
    for command in [
        "go depth 2 depth 3",
        "go movetime 10 winc 1",
        "go infinite movestogo 10",
        "go winc 0",
        "go wtime 1000 btime 1000 movestogo 0",
        "go searchmoves e2e4 searchmoves d2d4",
    ] {
        let response = engine.handle_line(command);
        assert!(
            response
                .lines
                .iter()
                .any(|line| line.starts_with("info string error:")),
            "'{command}' must be rejected"
        );
        assert!(
            response
                .lines
                .iter()
                .all(|line| !line.starts_with("bestmove "))
        );
    }
}

#[test]
fn syzygypath_defaults_to_empty_and_rejects_unusable_paths() {
    let mut engine = UciEngine::new();
    assert_eq!(engine.debug_syzygy_path(), "");

    let disable = engine.handle_line("setoption name SyzygyPath value");
    assert_eq!(disable.lines, vec!["info string syzygy disabled"]);
    assert_eq!(engine.debug_syzygy_path(), "");

    let rejected = engine.handle_line("setoption name SyzygyPath value /tmp/syzygy");
    assert!(
        rejected
            .lines
            .iter()
            .any(|line| line.contains("did not load any supported Syzygy tablebase files"))
    );
    assert_eq!(engine.debug_syzygy_path(), "");
}

#[test]
fn syzygy_probe_controls_validate_and_update_without_loaded_tables() {
    let mut engine = UciEngine::new();
    assert_eq!(engine.debug_syzygy_probe_limit(), 7);
    assert!(engine.debug_syzygy_50_move_rule());

    assert!(
        engine
            .handle_line("setoption name SyzygyProbeLimit value 5")
            .lines
            .is_empty()
    );
    assert_eq!(engine.debug_syzygy_probe_limit(), 5);
    assert!(
        engine
            .handle_line("setoption name Syzygy50MoveRule value false")
            .lines
            .is_empty()
    );
    assert!(!engine.debug_syzygy_50_move_rule());

    for command in [
        "setoption name SyzygyProbeLimit value 8",
        "setoption name SyzygyProbeLimit value -1",
        "setoption name SyzygyProbeLimit value 5 extra",
        "setoption name Syzygy50MoveRule value maybe",
        "setoption name Syzygy50MoveRule value true extra",
    ] {
        let response = engine.handle_line(command);
        assert!(
            response
                .lines
                .iter()
                .any(|line| line.starts_with("info string error:")),
            "'{command}' must be rejected"
        );
    }

    assert_eq!(engine.debug_syzygy_probe_limit(), 5);
    assert!(!engine.debug_syzygy_50_move_rule());
}

#[test]
fn evalfile_defaults_to_empty_and_rejects_unusable_paths() {
    let mut engine = UciEngine::new();
    assert_eq!(engine.debug_eval_file(), "");

    let disable = engine.handle_line("setoption name EvalFile value");
    assert!(disable.lines.is_empty());
    assert_eq!(engine.debug_eval_file(), "");

    let invalid_path = "/tmp/definitely-missing.volknnue";
    let rejected = engine.handle_line(&format!("setoption name EvalFile value {invalid_path}"));
    assert!(
        rejected
            .lines
            .iter()
            .any(|line| line.contains("failed to read EvalFile"))
    );
    assert_eq!(engine.debug_eval_file(), "");

    let eval_file = phase12_test_evalfile_path();
    let loaded = engine.handle_line(&format!("setoption name EvalFile value {eval_file}"));
    assert!(loaded.lines.is_empty());
    assert_eq!(engine.debug_eval_file(), eval_file);

    let still_loaded = engine.handle_line(&format!("setoption name EvalFile value {invalid_path}"));
    assert!(
        still_loaded
            .lines
            .iter()
            .any(|line| line.contains("failed to read EvalFile"))
    );
    assert_eq!(engine.debug_eval_file(), eval_file);
}

#[test]
fn dual_eval_is_explicit_default_off_and_counts_selected_networks() {
    let mut engine = UciEngine::debug_new_with_eval_discovery(None, None);
    assert_eq!(engine.debug_small_eval_file(), "");
    assert_eq!(engine.debug_dual_eval_config(), ("off", 200));

    let premature = engine.handle_line("setoption name DualEvalPolicy value small-fallback");
    assert!(
        premature
            .lines
            .iter()
            .any(|line| { line.contains("requires both EvalFile and SmallEvalFile") })
    );
    assert_eq!(engine.debug_dual_eval_config().0, "off");

    let eval_file = phase12_test_evalfile_path();
    assert!(
        engine
            .handle_line(&format!("setoption name EvalFile value {eval_file}"))
            .lines
            .is_empty()
    );
    assert!(
        engine
            .handle_line(&format!("setoption name SmallEvalFile value {eval_file}"))
            .lines
            .is_empty()
    );
    assert_eq!(engine.debug_small_eval_file(), eval_file);

    let invalid =
        engine.handle_line("setoption name SmallEvalFile value /tmp/definitely-missing-small.nnue");
    assert!(
        invalid
            .lines
            .iter()
            .any(|line| line.contains("SmallEvalFile"))
    );
    assert_eq!(engine.debug_small_eval_file(), eval_file);

    assert!(
        engine
            .handle_line("setoption name DualEvalThreshold value 0")
            .lines
            .is_empty()
    );
    assert!(
        engine
            .handle_line("setoption name DualEvalPolicy value small-fallback")
            .lines
            .is_empty()
    );
    assert_eq!(engine.debug_dual_eval_config(), ("small-fallback", 0));
    let search = engine.handle_line("go depth 2");
    assert!(
        search
            .lines
            .iter()
            .any(|line| line.starts_with("bestmove "))
    );
    let (small_selected, big_fallbacks) = engine.debug_dual_eval_counters();
    assert!(small_selected > 0);
    assert_eq!(big_fallbacks, 0);

    assert!(
        engine
            .handle_line("setoption name SmallEvalFile value")
            .lines
            .is_empty()
    );
    assert_eq!(engine.debug_small_eval_file(), "");
    assert_eq!(engine.debug_dual_eval_config().0, "off");
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

#[test]
fn partial_clock_go_commands_return_legal_moves() {
    let cases = [
        ("position startpos", "go wtime 100"),
        ("position startpos moves e2e4", "go btime 100"),
        // A controller may send only one clock even when it is not labelled for the side to move.
        // Treat it as a conservative proxy instead of starting an unbounded search.
        ("position startpos", "go btime 100"),
    ];

    for (position_command, go_command) in cases {
        let mut engine = UciEngine::new();
        assert!(engine.handle_line(position_command).lines.is_empty());
        let root = engine.position().clone();
        let response = engine.handle_line(go_command);
        assert!(
            response
                .lines
                .iter()
                .all(|line| !line.starts_with("info string error:")),
            "'{go_command}' must be accepted: {:?}",
            response.lines
        );
        let bestmove = response
            .lines
            .iter()
            .find_map(|line| line.strip_prefix("bestmove "))
            .expect("partial-clock search must return bestmove");
        assert_ne!(bestmove, "0000");
        let mut checked = root;
        checked
            .apply_uci_move(bestmove)
            .expect("partial-clock bestmove must be legal");
    }
}

#[test]
fn threaded_go_depth_returns_a_legal_move_and_leaves_helpers_idle() {
    let mut engine = UciEngine::new();
    assert!(
        engine
            .handle_line("setoption name Threads value 2")
            .lines
            .is_empty()
    );

    let response = engine.handle_line("go depth 2");
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
        .expect("threaded bestmove must be legal");
    assert!(engine.debug_worker_count() >= 1);
    assert_eq!(engine.debug_active_helper_count(), 0);
}

#[test]
fn nnue_enabled_go_depth_returns_a_legal_move() {
    let mut engine = UciEngine::new();
    let eval_file = phase12_test_evalfile_path();
    assert!(
        engine
            .handle_line(&format!("setoption name EvalFile value {eval_file}"))
            .lines
            .is_empty()
    );

    let response = engine.handle_line("go depth 2");
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
        .expect("NNUE-enabled bestmove must be legal");
}
