use std::path::Path;

mod bullet_trainer;
mod engine_match;

use bullet_trainer::{BulletTrainingConfig, train_bullet, train_bullet_with_config};
use engine_match::compare_external_engines;
use volkrix::nnue_training::{
    CorpusExpansionConfig, LabelGenerationConfig, LabelMode, MatchConfig, MatchMode,
    PositionFilter, SelfPlayCorpusConfig, compare_candidate_vs_fallback, expand_fen_corpus,
    export_examples, export_examples_with_config, generate_selfplay_corpus,
    pack_checkpoint_to_volknnue, validate_volknnue,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("expand-fens") => {
            let mut flags = parse_flag_map(args)?;
            let input = take_required_flag(&mut flags, "--input")?;
            let output = take_required_flag(&mut flags, "--output")?;
            let config = CorpusExpansionConfig {
                max_plies: parse_optional_flag(&mut flags, "--max-plies")?.unwrap_or(3),
                branching: parse_optional_flag(&mut flags, "--branching")?.unwrap_or(4),
                max_positions: parse_optional_flag(&mut flags, "--max-positions")?.unwrap_or(1024),
            };
            ensure_no_unknown_flags(flags)?;
            let summary = expand_fen_corpus(Path::new(&input), Path::new(&output), config)?;
            println!(
                "expanded {} positions from {} seed lines (blank {}, invalid {}, terminal {})",
                summary.emitted_positions,
                summary.input_lines,
                summary.skipped_blank_lines,
                summary.skipped_invalid_positions,
                summary.skipped_terminal_positions
            );
            Ok(())
        }
        Some("selfplay-fens") => {
            let mut flags = parse_flag_map(args)?;
            let input = take_required_flag(&mut flags, "--input")?;
            let output = take_required_flag(&mut flags, "--output")?;
            let hash_mb = parse_optional_flag(&mut flags, "--hash-mb")?.unwrap_or(64usize);
            let max_plies = parse_optional_flag(&mut flags, "--max-plies")?.unwrap_or(80usize);
            let max_positions =
                parse_optional_flag(&mut flags, "--max-positions")?.unwrap_or(10_000usize);
            let mode = match (
                parse_optional_flag::<u8>(&mut flags, "--depth")?,
                parse_optional_flag::<u64>(&mut flags, "--movetime-ms")?,
            ) {
                (Some(depth), None) => MatchMode::FixedDepth(depth),
                (None, Some(movetime_ms)) => MatchMode::MoveTimeMs(movetime_ms),
                (None, None) => MatchMode::FixedDepth(2),
                (Some(_), Some(_)) => {
                    return Err("use either --depth or --movetime-ms, not both".to_owned());
                }
            };
            ensure_no_unknown_flags(flags)?;
            let summary = generate_selfplay_corpus(
                Path::new(&input),
                Path::new(&output),
                SelfPlayCorpusConfig {
                    hash_mb,
                    max_plies,
                    max_positions,
                    mode,
                },
            )?;
            println!(
                "generated {} self-play positions from {} seed lines (blank {}, invalid {}, terminal {}, games {})",
                summary.emitted_positions,
                summary.input_lines,
                summary.skipped_blank_lines,
                summary.skipped_invalid_positions,
                summary.skipped_terminal_positions,
                summary.completed_games
            );
            Ok(())
        }
        Some("export-examples") => {
            let mut flags = parse_flag_map(args)?;
            let input = take_required_flag(&mut flags, "--input")?;
            let output = take_required_flag(&mut flags, "--output")?;
            let default_config = LabelGenerationConfig::default();
            let config = LabelGenerationConfig {
                depth: parse_optional_flag(&mut flags, "--label-depth")?
                    .unwrap_or(default_config.depth),
                tt_enabled: parse_optional_bool_flag(&mut flags, "--tt")?
                    .unwrap_or(default_config.tt_enabled),
                hash_mb: parse_optional_flag(&mut flags, "--hash-mb")?
                    .unwrap_or(default_config.hash_mb),
                mode: parse_optional_label_mode_flag(&mut flags, "--label-mode")?
                    .unwrap_or(default_config.mode),
                workers: parse_optional_flag(&mut flags, "--workers")?
                    .unwrap_or(default_config.workers),
                position_filter: parse_optional_position_filter_flag(
                    &mut flags,
                    "--position-filter",
                )?
                .unwrap_or(default_config.position_filter),
                label_timeout_ms: parse_optional_flag(&mut flags, "--label-timeout-ms")?
                    .or(default_config.label_timeout_ms),
            };
            ensure_no_unknown_flags(flags)?;
            let summary = if config == default_config {
                export_examples(Path::new(&input), Path::new(&output))?
            } else {
                export_examples_with_config(Path::new(&input), Path::new(&output), config)?
            };
            println!(
                "exported {} examples from {} lines (blank {}, filtered {}, incomplete {}, tb-scope {})",
                summary.emitted_examples,
                summary.input_lines,
                summary.skipped_blank_lines,
                summary.skipped_position_filter_positions,
                summary.skipped_incomplete_search_positions,
                summary.skipped_tablebase_scope_positions
            );
            Ok(())
        }
        Some("pack-volknnue") => {
            let mut flags = parse_flag_map(args)?;
            let checkpoint_dir = take_required_flag(&mut flags, "--checkpoint-dir")?;
            let output = take_required_flag(&mut flags, "--output")?;
            ensure_no_unknown_flags(flags)?;
            let summary =
                pack_checkpoint_to_volknnue(Path::new(&checkpoint_dir), Path::new(&output))?;
            println!(
                "packed VOLKNNUE to '{}' with manifest '{}'",
                summary.output_path.display(),
                summary.manifest_path.display()
            );
            Ok(())
        }
        Some("train-bullet") => {
            let mut flags = parse_flag_map(args)?;
            let examples = take_required_flag(&mut flags, "--examples")?;
            let checkpoint_dir = take_required_flag(&mut flags, "--checkpoint-dir")?;
            let init_from_checkpoint_dir =
                parse_optional_flag::<String>(&mut flags, "--init-from-checkpoint-dir")?
                    .map(std::path::PathBuf::from);
            let default_config = BulletTrainingConfig::default();
            let end_superbatch = parse_optional_flag(&mut flags, "--superbatches")?
                .unwrap_or(default_config.end_superbatch);
            let custom_config = BulletTrainingConfig {
                batch_size: parse_optional_flag(&mut flags, "--batch-size")?
                    .unwrap_or(default_config.batch_size),
                start_superbatch: 1,
                end_superbatch,
                initial_lr: parse_optional_flag(&mut flags, "--initial-lr")?
                    .unwrap_or(default_config.initial_lr),
                final_lr: parse_optional_flag(&mut flags, "--final-lr")?
                    .unwrap_or(default_config.final_lr),
                save_rate: parse_optional_flag(&mut flags, "--save-rate")?
                    .unwrap_or(end_superbatch),
                trainer_threads: parse_optional_flag(&mut flags, "--trainer-threads")?
                    .unwrap_or(default_config.trainer_threads),
                loader_threads: parse_optional_flag(&mut flags, "--loader-threads")?
                    .unwrap_or(default_config.loader_threads),
                batch_queue_size: parse_optional_flag(&mut flags, "--batch-queue-size")?
                    .unwrap_or(default_config.batch_queue_size),
                eval_scale: parse_optional_flag(&mut flags, "--eval-scale")?
                    .unwrap_or(default_config.eval_scale),
                init_from_checkpoint_dir,
            };
            ensure_no_unknown_flags(flags)?;
            let summary = if custom_config == default_config {
                train_bullet(Path::new(&examples), Path::new(&checkpoint_dir))?
            } else {
                train_bullet_with_config(
                    Path::new(&examples),
                    Path::new(&checkpoint_dir),
                    custom_config,
                )?
            };
            println!(
                "trained Bullet checkpoint in '{}' (train {}, validation {})",
                summary.checkpoint_dir.display(),
                summary.train_examples,
                summary.validation_examples
            );
            println!(
                "wrote Bullet datasets '{}' and '{}'",
                summary.train_data_path.display(),
                summary.validation_data_path.display()
            );
            println!(
                "wrote Bullet metadata '{}' and raw Bullet checkpoints under '{}'",
                summary.metadata_path.display(),
                summary.bullet_checkpoint_dir.display()
            );
            if let Some(log_path) = &summary.training_log_path {
                println!("final Bullet log '{}'", log_path.display());
            }
            if let Some(loss) = summary.final_training_loss {
                println!("final Bullet running loss {}", loss);
            }
            Ok(())
        }
        Some("compare-fallback") => {
            let mut flags = parse_flag_map(args)?;
            let openings = take_required_flag(&mut flags, "--openings")?;
            let candidate = take_required_flag(&mut flags, "--candidate")?;
            let hash_mb = parse_optional_flag(&mut flags, "--hash-mb")?.unwrap_or(64usize);
            let max_plies = parse_optional_flag(&mut flags, "--max-plies")?.unwrap_or(160usize);
            let mode = match (
                parse_optional_flag::<u8>(&mut flags, "--depth")?,
                parse_optional_flag::<u64>(&mut flags, "--movetime-ms")?,
            ) {
                (Some(depth), None) => MatchMode::FixedDepth(depth),
                (None, Some(movetime_ms)) => MatchMode::MoveTimeMs(movetime_ms),
                (None, None) => MatchMode::MoveTimeMs(100),
                (Some(_), Some(_)) => {
                    return Err("use either --depth or --movetime-ms, not both".to_owned());
                }
            };
            ensure_no_unknown_flags(flags)?;
            let summary = compare_candidate_vs_fallback(
                Path::new(&openings),
                Path::new(&candidate),
                MatchConfig {
                    hash_mb,
                    max_plies,
                    mode,
                },
            )?;
            println!(
                "candidate vs fallback: {} games over {} openings => {}W {}D {}L",
                summary.games,
                summary.openings,
                summary.candidate_wins,
                summary.draws,
                summary.fallback_wins
            );
            for game in summary.game_summaries.iter().take(4) {
                println!(
                    "game candidate {:?} opening '{}' outcome {:?} plies {} candidate_score {:?} fallback_score {:?}",
                    game.candidate_color,
                    game.opening_fen,
                    game.outcome,
                    game.plies_played,
                    game.first_candidate_score_cp,
                    game.first_fallback_score_cp
                );
                if let Some(info) = &game.first_candidate_info_line {
                    println!("candidate info {}", info);
                }
                if let Some(info) = &game.first_fallback_info_line {
                    println!("fallback info {}", info);
                }
            }
            Ok(())
        }
        Some("compare-engines") => {
            let mut flags = parse_flag_map(args)?;
            let openings = take_required_flag(&mut flags, "--openings")?;
            let baseline = take_required_flag(&mut flags, "--baseline")?;
            let candidate = take_required_flag(&mut flags, "--candidate")?;
            let hash_mb = parse_optional_flag(&mut flags, "--hash-mb")?.unwrap_or(64usize);
            let max_plies = parse_optional_flag(&mut flags, "--max-plies")?.unwrap_or(160usize);
            let max_openings = parse_optional_flag(&mut flags, "--max-openings")?;
            let mode = match (
                parse_optional_flag::<u8>(&mut flags, "--depth")?,
                parse_optional_flag::<u64>(&mut flags, "--movetime-ms")?,
            ) {
                (Some(depth), None) => MatchMode::FixedDepth(depth),
                (None, Some(movetime_ms)) => MatchMode::MoveTimeMs(movetime_ms),
                (None, None) => MatchMode::MoveTimeMs(100),
                (Some(_), Some(_)) => {
                    return Err("use either --depth or --movetime-ms, not both".to_owned());
                }
            };
            ensure_no_unknown_flags(flags)?;
            let summary = compare_external_engines(
                Path::new(&openings),
                Path::new(&baseline),
                Path::new(&candidate),
                MatchConfig {
                    hash_mb,
                    max_plies,
                    mode,
                },
                max_openings,
            )?;
            let score = candidate_score_fraction(&summary);
            println!(
                "candidate vs baseline: {} games over {} openings => {}W {}D {}L score {:.1}%",
                summary.games,
                summary.openings,
                summary.candidate_wins,
                summary.draws,
                summary.fallback_wins,
                score * 100.0
            );
            if let Some(elo) = approximate_elo_from_score(score) {
                println!("approximate Elo difference from score rate: {elo:+.1}");
            }
            for game in summary.game_summaries.iter().take(4) {
                println!(
                    "game candidate {:?} opening '{}' outcome {:?} plies {} candidate_score {:?} baseline_score {:?}",
                    game.candidate_color,
                    game.opening_fen,
                    game.outcome,
                    game.plies_played,
                    game.first_candidate_score_cp,
                    game.first_fallback_score_cp
                );
                if let Some(info) = &game.first_candidate_info_line {
                    println!("candidate info {}", info);
                }
                if let Some(info) = &game.first_fallback_info_line {
                    println!("baseline info {}", info);
                }
            }
            Ok(())
        }
        Some("validate-volknnue") => {
            let mut flags = parse_flag_map(args)?;
            let evalfile = take_required_flag(&mut flags, "--evalfile")?;
            ensure_no_unknown_flags(flags)?;
            let summary = validate_volknnue(Path::new(&evalfile))?;
            println!("validated VOLKNNUE '{}'", summary.net_path.display());
            if let Some(manifest_path) = summary.manifest_path {
                println!("found sidecar manifest '{}'", manifest_path.display());
            }
            Ok(())
        }
        _ => Err(usage()),
    }
}

fn parse_flag_map(
    args: impl Iterator<Item = String>,
) -> Result<std::collections::BTreeMap<String, String>, String> {
    let mut flags = std::collections::BTreeMap::new();
    let mut iter = args.peekable();
    while let Some(flag) = iter.next() {
        if !flag.starts_with("--") {
            return Err(format!("expected --flag, found '{flag}'"));
        }
        let value = iter
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        if flags.insert(flag.clone(), value).is_some() {
            return Err(format!("duplicate flag '{flag}'"));
        }
    }
    Ok(flags)
}

fn take_required_flag(
    flags: &mut std::collections::BTreeMap<String, String>,
    flag: &str,
) -> Result<String, String> {
    flags.remove(flag).ok_or_else(|| format!("missing {flag}"))
}

fn parse_optional_flag<T: std::str::FromStr>(
    flags: &mut std::collections::BTreeMap<String, String>,
    flag: &str,
) -> Result<Option<T>, String>
where
    T::Err: std::fmt::Display,
{
    match flags.remove(flag) {
        Some(value) => value
            .parse::<T>()
            .map(Some)
            .map_err(|error| format!("invalid value for {flag}: {error}")),
        None => Ok(None),
    }
}

fn parse_optional_bool_flag(
    flags: &mut std::collections::BTreeMap<String, String>,
    flag: &str,
) -> Result<Option<bool>, String> {
    match flags.remove(flag) {
        Some(value) => match value.as_str() {
            "on" | "true" | "1" => Ok(Some(true)),
            "off" | "false" | "0" => Ok(Some(false)),
            _ => Err(format!("invalid value for {flag}: expected on/off")),
        },
        None => Ok(None),
    }
}

fn parse_optional_label_mode_flag(
    flags: &mut std::collections::BTreeMap<String, String>,
    flag: &str,
) -> Result<Option<LabelMode>, String> {
    match flags.remove(flag) {
        Some(value) => match value.as_str() {
            "search" => Ok(Some(LabelMode::Search)),
            "static" => Ok(Some(LabelMode::StaticEval)),
            _ => Err(format!("invalid value for {flag}: expected search|static")),
        },
        None => Ok(None),
    }
}

fn parse_optional_position_filter_flag(
    flags: &mut std::collections::BTreeMap<String, String>,
    flag: &str,
) -> Result<Option<PositionFilter>, String> {
    match flags.remove(flag) {
        Some(value) => match value.as_str() {
            "any" => Ok(Some(PositionFilter::Any)),
            "quiet" => Ok(Some(PositionFilter::Quiet)),
            _ => Err(format!("invalid value for {flag}: expected any|quiet")),
        },
        None => Ok(None),
    }
}

fn ensure_no_unknown_flags(
    flags: std::collections::BTreeMap<String, String>,
) -> Result<(), String> {
    if flags.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "unexpected extra flags: {}",
            flags.keys().cloned().collect::<Vec<_>>().join(", ")
        ))
    }
}

fn candidate_score_fraction(summary: &volkrix::nnue_training::MatchSummary) -> f64 {
    if summary.games == 0 {
        return 0.0;
    }

    (summary.candidate_wins as f64 + 0.5 * summary.draws as f64) / summary.games as f64
}

fn approximate_elo_from_score(score: f64) -> Option<f64> {
    if !(0.0..1.0).contains(&score) || score == 0.5 {
        return None;
    }

    Some(-400.0 * ((1.0 / score) - 1.0).log10())
}

fn usage() -> String {
    [
        "usage:",
        "  cargo run -p volkrix-nnue -- expand-fens --input <seed-fens.txt> --output <corpus.fens> [--max-plies N] [--branching N] [--max-positions N]",
        "  cargo run -p volkrix-nnue -- selfplay-fens --input <seed-fens.txt> --output <corpus.fens> [--depth N | --movetime-ms N] [--hash-mb N] [--max-plies N] [--max-positions N]",
        "  cargo run -p volkrix-nnue -- export-examples --input <fens.txt> --output <examples.txt> [--label-depth N] [--tt on|off] [--hash-mb N] [--label-mode search|static] [--workers N] [--position-filter any|quiet] [--label-timeout-ms N]",
        "  cargo run -p volkrix-nnue -- train-bullet --examples <examples.txt> --checkpoint-dir <dir> [--init-from-checkpoint-dir <prior-dir>] [--batch-size N] [--superbatches N] [--initial-lr F] [--final-lr F]",
        "  cargo run -p volkrix-nnue -- pack-volknnue --checkpoint-dir <dir> --output <net.volknnue>",
        "  cargo run -p volkrix-nnue -- compare-fallback --openings <fens.txt> --candidate <net.volknnue> [--movetime-ms N | --depth N] [--hash-mb N] [--max-plies N]",
        "  cargo run -p volkrix-nnue -- compare-engines --openings <fens.txt> --baseline <baseline-bin> --candidate <candidate-bin> [--movetime-ms N | --depth N] [--hash-mb N] [--max-plies N] [--max-openings N]",
        "  cargo run -p volkrix-nnue -- validate-volknnue --evalfile <net.volknnue>",
    ]
    .join("\n")
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    };

    use volkrix::SOURCE_COMMIT;
    use volkrix::nnue_training::{
        CHECKPOINT_HIDDEN_BIASES_FILE, CHECKPOINT_INPUT_WEIGHTS_FILE, CHECKPOINT_MAGIC,
        CHECKPOINT_OUTPUT_BIAS_FILE, CHECKPOINT_OUTPUT_WEIGHTS_FILE, CHECKPOINT_VERSION,
        CheckpointManifest, CorpusExpansionConfig, LABEL_DEPTH, LabelGenerationConfig, LabelMode,
        OUTPUT_SCALE, PackedNetManifest, PositionFilter, SPLIT_RULE_DESCRIPTION,
        SelfPlayCorpusConfig, TARGET_CLIP_CP, TOPOLOGY_FEATURE_COUNT, TOPOLOGY_HIDDEN_SIZE,
        TOPOLOGY_ID, TOPOLOGY_NAME, TOPOLOGY_OUTPUT_INPUTS, TRAINER_LOSS, TRAINER_OPTIMIZER,
        TRAINER_SEED, expand_fen_corpus, export_examples, export_examples_with_config,
        generate_selfplay_corpus, normalize_fen, pack_checkpoint_to_volknnue,
        position_is_quiet_for_supervision, read_examples, split_for_normalized_fen,
        validate_volknnue,
    };

    fn temp_path(label: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be monotonic enough for tests")
            .as_nanos();
        std::env::temp_dir().join(format!("volkrix-phase13-{label}-{suffix}"))
    }

    fn fixture_manifest(examples_path: &Path) -> volkrix::nnue_training::ExampleExportManifest {
        let normalized =
            normalize_fen("4k3/8/8/8/8/8/3Q4/4K3 w - - 0 1").expect("fixture FEN must normalize");
        let manifest = read_examples(examples_path)
            .expect("examples file must parse")
            .0;
        assert_eq!(
            split_for_normalized_fen(&normalized),
            volkrix::nnue_training::DatasetSplit::Train
        );
        manifest
    }

    fn write_f32_file(path: &Path, values: &[f32]) {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(path, bytes).expect("checkpoint tensor file must write");
    }

    #[test]
    fn export_examples_writes_deterministic_manifest_and_rows() {
        let input_path = temp_path("fixture-fens");
        let output_path = temp_path("fixture-examples");
        fs::write(
            &input_path,
            [
                "4k3/8/8/8/8/8/3Q4/4K3 w - - 0 1",
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "8/8/8/8/8/3Q4/2K5/k7 w - - 0 1",
            ]
            .join("\n"),
        )
        .expect("fixture corpus must write");

        let summary = export_examples(&input_path, &output_path).expect("export must succeed");
        assert_eq!(summary.input_lines, 3);
        assert_eq!(summary.emitted_examples, 1);
        assert_eq!(summary.skipped_tablebase_scope_positions, 2);
        assert_eq!(summary.skipped_position_filter_positions, 0);

        let (manifest, records, parsed_summary) =
            read_examples(&output_path).expect("examples file must parse");
        assert_eq!(manifest.source_engine_commit, SOURCE_COMMIT);
        assert_eq!(manifest.label_depth, LABEL_DEPTH);
        assert_eq!(manifest.target_clip_cp, TARGET_CLIP_CP);
        assert_eq!(records.len(), 1);
        assert_eq!(parsed_summary.emitted_examples, 1);
        assert_eq!(records[0].target_cp.abs() <= TARGET_CLIP_CP, true);

        let _ = fs::remove_file(input_path);
        let _ = fs::remove_file(output_path);
    }

    #[test]
    fn export_examples_rejects_malformed_fen_input() {
        let input_path = temp_path("bad-fens");
        let output_path = temp_path("bad-examples");
        fs::write(&input_path, "not a fen\n").expect("fixture corpus must write");

        let error = export_examples(&input_path, &output_path)
            .expect_err("malformed input must be rejected cleanly");
        assert!(error.contains("failed to parse FEN"));

        let _ = fs::remove_file(input_path);
        let _ = fs::remove_file(output_path);
    }

    #[test]
    fn export_examples_with_config_records_label_environment() {
        let input_path = temp_path("config-fens");
        let output_path = temp_path("config-examples");
        fs::write(
            &input_path,
            "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 2 3\n",
        )
        .expect("fixture corpus must write");

        export_examples_with_config(
            &input_path,
            &output_path,
            LabelGenerationConfig {
                depth: 4,
                tt_enabled: true,
                hash_mb: 32,
                mode: LabelMode::Search,
                workers: 1,
                position_filter: PositionFilter::Any,
                label_timeout_ms: Some(10_000),
            },
        )
        .expect("export with custom config must succeed");
        let (manifest, records, _) = read_examples(&output_path).expect("examples must parse");
        assert_eq!(manifest.label_depth, 4);
        assert!(manifest.tt_enabled);
        assert_eq!(manifest.hash_mb, 32);
        assert_eq!(manifest.label_mode, LabelMode::Search);
        assert_eq!(manifest.export_workers, 1);
        assert_eq!(manifest.position_filter, PositionFilter::Any);
        assert_eq!(manifest.label_timeout_ms, Some(10_000));
        assert_eq!(records.len(), 1);

        let _ = fs::remove_file(input_path);
        let _ = fs::remove_file(output_path);
    }

    #[test]
    fn parallel_search_export_matches_single_worker_output() {
        let input_path = temp_path("parallel-fens");
        let one_path = temp_path("parallel-one");
        let many_path = temp_path("parallel-many");
        fs::write(
            &input_path,
            [
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
                "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 2 3",
            ]
            .join("\n"),
        )
        .expect("fixture corpus must write");

        let base = LabelGenerationConfig {
            depth: 1,
            tt_enabled: true,
            hash_mb: 16,
            mode: LabelMode::Search,
            workers: 1,
            position_filter: PositionFilter::Any,
            label_timeout_ms: None,
        };
        export_examples_with_config(&input_path, &one_path, base)
            .expect("single-worker export must succeed");
        export_examples_with_config(
            &input_path,
            &many_path,
            LabelGenerationConfig { workers: 2, ..base },
        )
        .expect("parallel export must succeed");

        let (one_manifest, one_records, _) = read_examples(&one_path).expect("examples must parse");
        let (many_manifest, many_records, _) =
            read_examples(&many_path).expect("examples must parse");
        assert_eq!(one_records, many_records);
        assert_eq!(one_manifest.label_depth, many_manifest.label_depth);
        assert_eq!(one_manifest.hash_mb, many_manifest.hash_mb);
        assert_eq!(one_manifest.label_mode, many_manifest.label_mode);
        assert_eq!(one_manifest.tt_enabled, many_manifest.tt_enabled);
        assert_eq!(one_manifest.export_workers, 1);
        assert_eq!(many_manifest.export_workers, 2);
        assert_eq!(one_manifest.position_filter, PositionFilter::Any);
        assert_eq!(many_manifest.position_filter, PositionFilter::Any);

        let _ = fs::remove_file(input_path);
        let _ = fs::remove_file(one_path);
        let _ = fs::remove_file(many_path);
    }

    #[test]
    fn quiet_position_filter_is_explicit_and_deterministic() {
        let quiet = volkrix::core::Position::from_fen(volkrix::core::STARTPOS_FEN)
            .expect("startpos must parse");
        assert!(position_is_quiet_for_supervision(&quiet));

        let tactical = volkrix::core::Position::from_fen(
            "rnb1kbnr/pppp1ppp/8/8/3p4/3Q4/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
        )
        .expect("tactical FEN must parse");
        assert!(!position_is_quiet_for_supervision(&tactical));

        let in_check = volkrix::core::Position::from_fen("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")
            .expect("check FEN must parse");
        assert!(!position_is_quiet_for_supervision(&in_check));
    }

    #[test]
    fn quiet_filter_skips_tactical_positions_and_records_manifest() {
        let input_path = temp_path("quiet-filter-fens");
        let output_path = temp_path("quiet-filter-examples");
        fs::write(
            &input_path,
            [
                volkrix::core::STARTPOS_FEN,
                "rnb1kbnr/pppp1ppp/8/8/3p4/3Q4/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
            ]
            .join("\n"),
        )
        .expect("fixture corpus must write");

        let summary = export_examples_with_config(
            &input_path,
            &output_path,
            LabelGenerationConfig {
                depth: 1,
                tt_enabled: true,
                hash_mb: 16,
                mode: LabelMode::Search,
                workers: 1,
                position_filter: PositionFilter::Quiet,
                label_timeout_ms: None,
            },
        )
        .expect("quiet-filter export must succeed");
        let (manifest, records, _) = read_examples(&output_path).expect("examples must parse");
        assert_eq!(summary.emitted_examples, 1);
        assert_eq!(summary.skipped_position_filter_positions, 1);
        assert_eq!(summary.skipped_incomplete_search_positions, 0);
        assert_eq!(manifest.position_filter, PositionFilter::Quiet);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].normalized_fen, volkrix::core::STARTPOS_FEN);

        let _ = fs::remove_file(input_path);
        let _ = fs::remove_file(output_path);
    }

    #[test]
    fn normalized_fen_hash_split_is_deterministic_on_canonical_fen() {
        let spaced = "  rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR   w  KQkq  -  0  1  ";
        let normalized = normalize_fen(spaced).expect("FEN must normalize");
        assert_eq!(normalized, volkrix::core::STARTPOS_FEN);
        assert_eq!(
            split_for_normalized_fen(&normalized),
            split_for_normalized_fen(volkrix::core::STARTPOS_FEN)
        );
    }

    #[test]
    fn expand_fens_is_deterministic_and_capped() {
        let input_path = temp_path("expand-seeds");
        let output_path = temp_path("expand-corpus");
        fs::write(
            &input_path,
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n",
        )
        .expect("seed corpus must write");

        let summary = expand_fen_corpus(
            &input_path,
            &output_path,
            CorpusExpansionConfig {
                max_plies: 2,
                branching: 2,
                max_positions: 5,
            },
        )
        .expect("expansion must succeed");
        let lines: Vec<String> = fs::read_to_string(&output_path)
            .expect("expanded corpus must read")
            .lines()
            .map(ToOwned::to_owned)
            .collect();
        assert_eq!(summary.emitted_positions, lines.len());
        assert!(lines.len() <= 5);
        assert_eq!(
            lines.first().map(String::as_str),
            Some(volkrix::core::STARTPOS_FEN)
        );

        let _ = fs::remove_file(input_path);
        let _ = fs::remove_file(output_path);
    }

    #[test]
    fn selfplay_fens_is_deterministic_and_capped() {
        let input_path = temp_path("selfplay-seeds");
        let output_path = temp_path("selfplay-corpus");
        fs::write(
            &input_path,
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n",
        )
        .expect("seed corpus must write");

        let summary = generate_selfplay_corpus(
            &input_path,
            &output_path,
            SelfPlayCorpusConfig {
                hash_mb: 16,
                max_plies: 4,
                max_positions: 5,
                mode: volkrix::nnue_training::MatchMode::FixedDepth(1),
            },
        )
        .expect("self-play generation must succeed");
        let lines: Vec<String> = fs::read_to_string(&output_path)
            .expect("self-play corpus must read")
            .lines()
            .map(ToOwned::to_owned)
            .collect();
        assert_eq!(summary.emitted_positions, lines.len());
        assert!(lines.len() <= 5);
        assert_eq!(
            lines.first().map(String::as_str),
            Some(volkrix::core::STARTPOS_FEN)
        );

        let _ = fs::remove_file(input_path);
        let _ = fs::remove_file(output_path);
    }

    #[test]
    fn pack_and_validate_round_trip_compatibility_checkpoint() {
        let input_path = temp_path("checkpoint-fens");
        let examples_path = temp_path("checkpoint-examples");
        fs::write(&input_path, "4k3/8/8/8/8/8/3Q4/4K3 w - - 0 1\n")
            .expect("fixture corpus must write");
        export_examples(&input_path, &examples_path).expect("export must succeed");
        let example_manifest = fixture_manifest(&examples_path);

        let checkpoint_dir = temp_path("checkpoint-dir");
        fs::create_dir_all(&checkpoint_dir).expect("checkpoint dir must create");
        let compatibility_topology_name = "HalfKP128x2";
        let manifest = CheckpointManifest {
            magic: CHECKPOINT_MAGIC.to_owned(),
            version: CHECKPOINT_VERSION,
            trainer_version: volkrix::nnue_training::TRAINER_VERSION.to_owned(),
            source_engine_commit: SOURCE_COMMIT.to_owned(),
            topology_name: compatibility_topology_name.to_owned(),
            topology_id: 1,
            feature_count: 40960,
            hidden_size: 128,
            output_inputs: 256,
            output_scale: OUTPUT_SCALE,
            seed: TRAINER_SEED,
            optimizer: TRAINER_OPTIMIZER.to_owned(),
            loss: TRAINER_LOSS.to_owned(),
            split_rule: SPLIT_RULE_DESCRIPTION.to_owned(),
            normalized_fen_rule: volkrix::nnue_training::NORMALIZED_FEN_RULE.to_owned(),
            examples_path: examples_path.display().to_string(),
            example_manifest,
            train_examples: 1,
            validation_examples: 0,
            epochs: 1,
            learning_rate: 0.001,
            weight_decay: 0.0001,
            init_from_mode: None,
            init_from_checkpoint_dir: None,
            init_from_checkpoint_trainer_version: None,
            init_from_checkpoint_epochs: None,
            init_from_checkpoint_examples_path: None,
            input_weights_file: CHECKPOINT_INPUT_WEIGHTS_FILE.to_owned(),
            hidden_biases_file: CHECKPOINT_HIDDEN_BIASES_FILE.to_owned(),
            output_weights_file: CHECKPOINT_OUTPUT_WEIGHTS_FILE.to_owned(),
            output_bias_file: CHECKPOINT_OUTPUT_BIAS_FILE.to_owned(),
        };
        fs::write(
            checkpoint_dir.join("manifest.json"),
            serde_json::to_string_pretty(&manifest).expect("checkpoint manifest must encode"),
        )
        .expect("checkpoint manifest must write");
        write_f32_file(
            &checkpoint_dir.join(CHECKPOINT_INPUT_WEIGHTS_FILE),
            &vec![0.0; 40960 * 128],
        );
        write_f32_file(
            &checkpoint_dir.join(CHECKPOINT_HIDDEN_BIASES_FILE),
            &vec![8.0; 128],
        );
        write_f32_file(
            &checkpoint_dir.join(CHECKPOINT_OUTPUT_WEIGHTS_FILE),
            &vec![1.0; 256],
        );
        write_f32_file(&checkpoint_dir.join(CHECKPOINT_OUTPUT_BIAS_FILE), &[0.0]);

        let output_path = temp_path("packed-net").with_extension("volknnue");
        let summary =
            pack_checkpoint_to_volknnue(&checkpoint_dir, &output_path).expect("pack must succeed");
        assert!(summary.output_path.exists());
        assert!(summary.manifest_path.exists());
        let packed_manifest: PackedNetManifest = serde_json::from_str(
            &fs::read_to_string(&summary.manifest_path).expect("pack manifest must read"),
        )
        .expect("pack manifest must parse");
        assert_eq!(packed_manifest.source_engine_commit, SOURCE_COMMIT);
        assert_eq!(
            packed_manifest.output_net,
            output_path.display().to_string()
        );
        assert_eq!(
            packed_manifest
                .checkpoint_manifest
                .example_manifest
                .source_engine_commit,
            SOURCE_COMMIT
        );
        assert_eq!(
            packed_manifest.checkpoint_manifest.examples_path,
            examples_path.display().to_string()
        );
        assert_eq!(
            packed_manifest.checkpoint_manifest.normalized_fen_rule,
            volkrix::nnue_training::NORMALIZED_FEN_RULE
        );

        let validation = validate_volknnue(&output_path).expect("validation must succeed");
        assert_eq!(validation.net_path, output_path);
        assert!(validation.manifest_path.is_some());

        let _ = fs::remove_file(input_path);
        let _ = fs::remove_file(examples_path);
        let _ = fs::remove_file(summary.output_path);
        let _ = fs::remove_file(summary.manifest_path);
        let _ = fs::remove_dir_all(checkpoint_dir);
    }

    #[test]
    fn pack_and_validate_round_trip_retained_production_checkpoint() {
        let input_path = temp_path("production-checkpoint-fens");
        let examples_path = temp_path("production-checkpoint-examples");
        fs::write(&input_path, "4k3/8/8/8/8/8/3Q4/4K3 w - - 0 1\n")
            .expect("fixture corpus must write");
        export_examples(&input_path, &examples_path).expect("export must succeed");
        let example_manifest = fixture_manifest(&examples_path);

        let checkpoint_dir = temp_path("production-checkpoint-dir");
        fs::create_dir_all(&checkpoint_dir).expect("checkpoint dir must create");
        let manifest = CheckpointManifest {
            magic: CHECKPOINT_MAGIC.to_owned(),
            version: CHECKPOINT_VERSION,
            trainer_version: volkrix::nnue_training::TRAINER_VERSION.to_owned(),
            source_engine_commit: SOURCE_COMMIT.to_owned(),
            topology_name: TOPOLOGY_NAME.to_owned(),
            topology_id: TOPOLOGY_ID,
            feature_count: TOPOLOGY_FEATURE_COUNT,
            hidden_size: TOPOLOGY_HIDDEN_SIZE,
            output_inputs: TOPOLOGY_OUTPUT_INPUTS,
            output_scale: OUTPUT_SCALE,
            seed: TRAINER_SEED,
            optimizer: TRAINER_OPTIMIZER.to_owned(),
            loss: TRAINER_LOSS.to_owned(),
            split_rule: SPLIT_RULE_DESCRIPTION.to_owned(),
            normalized_fen_rule: volkrix::nnue_training::NORMALIZED_FEN_RULE.to_owned(),
            examples_path: examples_path.display().to_string(),
            example_manifest,
            train_examples: 1,
            validation_examples: 0,
            epochs: 1,
            learning_rate: 0.001,
            weight_decay: 0.0001,
            init_from_mode: None,
            init_from_checkpoint_dir: None,
            init_from_checkpoint_trainer_version: None,
            init_from_checkpoint_epochs: None,
            init_from_checkpoint_examples_path: None,
            input_weights_file: CHECKPOINT_INPUT_WEIGHTS_FILE.to_owned(),
            hidden_biases_file: CHECKPOINT_HIDDEN_BIASES_FILE.to_owned(),
            output_weights_file: CHECKPOINT_OUTPUT_WEIGHTS_FILE.to_owned(),
            output_bias_file: CHECKPOINT_OUTPUT_BIAS_FILE.to_owned(),
        };
        fs::write(
            checkpoint_dir.join("manifest.json"),
            serde_json::to_string_pretty(&manifest).expect("checkpoint manifest must encode"),
        )
        .expect("checkpoint manifest must write");
        write_f32_file(
            &checkpoint_dir.join(CHECKPOINT_INPUT_WEIGHTS_FILE),
            &vec![0.0; TOPOLOGY_FEATURE_COUNT * TOPOLOGY_HIDDEN_SIZE],
        );
        write_f32_file(
            &checkpoint_dir.join(CHECKPOINT_HIDDEN_BIASES_FILE),
            &vec![8.0; TOPOLOGY_HIDDEN_SIZE],
        );
        write_f32_file(
            &checkpoint_dir.join(CHECKPOINT_OUTPUT_WEIGHTS_FILE),
            &vec![1.0; TOPOLOGY_OUTPUT_INPUTS],
        );
        write_f32_file(&checkpoint_dir.join(CHECKPOINT_OUTPUT_BIAS_FILE), &[0.0]);

        let output_path = temp_path("production-packed-net").with_extension("volknnue");
        let summary =
            pack_checkpoint_to_volknnue(&checkpoint_dir, &output_path).expect("pack must succeed");
        let packed_manifest: PackedNetManifest = serde_json::from_str(
            &fs::read_to_string(&summary.manifest_path).expect("pack manifest must read"),
        )
        .expect("pack manifest must parse");
        assert_eq!(packed_manifest.topology_name, TOPOLOGY_NAME);
        assert_eq!(packed_manifest.topology_id, TOPOLOGY_ID);
        assert_eq!(packed_manifest.hidden_size, TOPOLOGY_HIDDEN_SIZE);
        assert_eq!(packed_manifest.output_inputs, TOPOLOGY_OUTPUT_INPUTS);
        validate_volknnue(&output_path).expect("validation must succeed");

        let _ = fs::remove_file(input_path);
        let _ = fs::remove_file(examples_path);
        let _ = fs::remove_file(summary.output_path);
        let _ = fs::remove_file(summary.manifest_path);
        let _ = fs::remove_dir_all(checkpoint_dir);
    }

    #[test]
    fn pack_rejects_checkpoint_metadata_mismatch() {
        let checkpoint_dir = temp_path("bad-checkpoint");
        fs::create_dir_all(&checkpoint_dir).expect("checkpoint dir must create");

        let manifest = CheckpointManifest {
            magic: CHECKPOINT_MAGIC.to_owned(),
            version: CHECKPOINT_VERSION,
            trainer_version: volkrix::nnue_training::TRAINER_VERSION.to_owned(),
            source_engine_commit: SOURCE_COMMIT.to_owned(),
            topology_name: "HalfKP128x2".to_owned(),
            topology_id: 1,
            feature_count: 1,
            hidden_size: 128,
            output_inputs: 256,
            output_scale: OUTPUT_SCALE,
            seed: TRAINER_SEED,
            optimizer: TRAINER_OPTIMIZER.to_owned(),
            loss: TRAINER_LOSS.to_owned(),
            split_rule: SPLIT_RULE_DESCRIPTION.to_owned(),
            normalized_fen_rule: volkrix::nnue_training::NORMALIZED_FEN_RULE.to_owned(),
            examples_path: "/tmp/missing.examples".to_owned(),
            example_manifest: volkrix::nnue_training::ExampleExportManifest {
                magic: volkrix::nnue_training::EXAMPLES_MAGIC.to_owned(),
                version: 1,
                exporter_version: volkrix::nnue_training::EXPORTER_VERSION.to_owned(),
                source_engine_commit: SOURCE_COMMIT.to_owned(),
                topology_name: "HalfKP128x2".to_owned(),
                topology_id: 1,
                feature_count: 40960,
                hidden_size: 128,
                output_inputs: 256,
                label_depth: LABEL_DEPTH,
                target_clip_cp: TARGET_CLIP_CP,
                threads: 1,
                syzygy_path: String::new(),
                eval_file: String::new(),
                tt_enabled: false,
                hash_mb: volkrix::search::SearchLimits::new(LABEL_DEPTH).hash_mb,
                label_mode: LabelMode::Search,
                export_workers: 1,
                position_filter: PositionFilter::Any,
                label_timeout_ms: None,
                normalized_fen_rule: volkrix::nnue_training::NORMALIZED_FEN_RULE.to_owned(),
            },
            train_examples: 1,
            validation_examples: 0,
            epochs: 1,
            learning_rate: 0.001,
            weight_decay: 0.0001,
            init_from_mode: None,
            init_from_checkpoint_dir: None,
            init_from_checkpoint_trainer_version: None,
            init_from_checkpoint_epochs: None,
            init_from_checkpoint_examples_path: None,
            input_weights_file: CHECKPOINT_INPUT_WEIGHTS_FILE.to_owned(),
            hidden_biases_file: CHECKPOINT_HIDDEN_BIASES_FILE.to_owned(),
            output_weights_file: CHECKPOINT_OUTPUT_WEIGHTS_FILE.to_owned(),
            output_bias_file: CHECKPOINT_OUTPUT_BIAS_FILE.to_owned(),
        };
        fs::write(
            checkpoint_dir.join("manifest.json"),
            serde_json::to_string_pretty(&manifest).expect("checkpoint manifest must encode"),
        )
        .expect("checkpoint manifest must write");

        let output_path = temp_path("bad-packed-net").with_extension("volknnue");
        let error = pack_checkpoint_to_volknnue(&checkpoint_dir, &output_path)
            .expect_err("mismatched checkpoint metadata must be rejected");
        assert!(error.contains("feature count"));

        let _ = fs::remove_file(output_path);
        let _ = fs::remove_dir_all(checkpoint_dir);
    }
}
