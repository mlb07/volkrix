use std::path::Path;

use volkrix::nnue_training::{export_examples, pack_checkpoint_to_volknnue, validate_volknnue};

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("export-examples") => {
            let input = expect_flag_value(&mut args, "--input")?;
            let output = expect_flag_value(&mut args, "--output")?;
            ensure_no_extra_args(args)?;
            let summary = export_examples(Path::new(&input), Path::new(&output))?;
            println!(
                "exported {} examples from {} lines (blank {}, tb-scope {})",
                summary.emitted_examples,
                summary.input_lines,
                summary.skipped_blank_lines,
                summary.skipped_tablebase_scope_positions
            );
            Ok(())
        }
        Some("pack-volknnue") => {
            let checkpoint_dir = expect_flag_value(&mut args, "--checkpoint-dir")?;
            let output = expect_flag_value(&mut args, "--output")?;
            ensure_no_extra_args(args)?;
            let summary =
                pack_checkpoint_to_volknnue(Path::new(&checkpoint_dir), Path::new(&output))?;
            println!(
                "packed VOLKNNUE to '{}' with manifest '{}'",
                summary.output_path.display(),
                summary.manifest_path.display()
            );
            Ok(())
        }
        Some("validate-volknnue") => {
            let evalfile = expect_flag_value(&mut args, "--evalfile")?;
            ensure_no_extra_args(args)?;
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

fn expect_flag_value(
    args: &mut impl Iterator<Item = String>,
    expected_flag: &str,
) -> Result<String, String> {
    match args.next().as_deref() {
        Some(flag) if flag == expected_flag => args
            .next()
            .ok_or_else(|| format!("missing value for {expected_flag}")),
        Some(flag) => Err(format!("expected {expected_flag}, found {flag}")),
        None => Err(format!("missing {expected_flag}")),
    }
}

fn ensure_no_extra_args(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    if let Some(extra) = args.next() {
        Err(format!("unexpected extra argument '{extra}'"))
    } else {
        Ok(())
    }
}

fn usage() -> String {
    [
        "usage:",
        "  cargo run -p volkrix-nnue -- export-examples --input <fens.txt> --output <examples.txt>",
        "  cargo run -p volkrix-nnue -- pack-volknnue --checkpoint-dir <dir> --output <net.volknnue>",
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
        CheckpointManifest, LABEL_DEPTH, OUTPUT_SCALE, SPLIT_RULE_DESCRIPTION, TARGET_CLIP_CP,
        TOPOLOGY_NAME, TRAINER_LOSS, TRAINER_OPTIMIZER, TRAINER_SEED, export_examples,
        normalize_fen, pack_checkpoint_to_volknnue, read_examples, split_for_normalized_fen,
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
    fn pack_and_validate_round_trip_a_checkpoint() {
        let input_path = temp_path("checkpoint-fens");
        let examples_path = temp_path("checkpoint-examples");
        fs::write(&input_path, "4k3/8/8/8/8/8/3Q4/4K3 w - - 0 1\n")
            .expect("fixture corpus must write");
        export_examples(&input_path, &examples_path).expect("export must succeed");
        let example_manifest = fixture_manifest(&examples_path);

        let checkpoint_dir = temp_path("checkpoint-dir");
        fs::create_dir_all(&checkpoint_dir).expect("checkpoint dir must create");
        let manifest = CheckpointManifest {
            magic: CHECKPOINT_MAGIC.to_owned(),
            version: CHECKPOINT_VERSION,
            trainer_version: volkrix::nnue_training::TRAINER_VERSION.to_owned(),
            source_engine_commit: SOURCE_COMMIT.to_owned(),
            topology_name: TOPOLOGY_NAME.to_owned(),
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
    fn pack_rejects_checkpoint_metadata_mismatch() {
        let checkpoint_dir = temp_path("bad-checkpoint");
        fs::create_dir_all(&checkpoint_dir).expect("checkpoint dir must create");

        let manifest = CheckpointManifest {
            magic: CHECKPOINT_MAGIC.to_owned(),
            version: CHECKPOINT_VERSION,
            trainer_version: volkrix::nnue_training::TRAINER_VERSION.to_owned(),
            source_engine_commit: SOURCE_COMMIT.to_owned(),
            topology_name: TOPOLOGY_NAME.to_owned(),
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
                topology_name: TOPOLOGY_NAME.to_owned(),
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
                normalized_fen_rule: volkrix::nnue_training::NORMALIZED_FEN_RULE.to_owned(),
            },
            train_examples: 1,
            validation_examples: 0,
            epochs: 1,
            learning_rate: 0.001,
            weight_decay: 0.0001,
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
