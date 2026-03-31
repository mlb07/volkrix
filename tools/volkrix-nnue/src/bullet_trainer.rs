use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    slice,
};

use acyclib::{
    graph::{like::GraphLike, save::GraphWeights},
    trainer::schedule::TrainingSteps,
};
use bullet_lib::{
    game::inputs::SparseInputType,
    nn::optimiser::{AdamW, AdamWParams},
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, lr, wdl},
        settings::LocalSettings,
    },
    value::{
        ValueTrainerBuilder,
        loader::{
            CanBeDirectlySequentiallyLoaded, DirectSequentialDataLoader, GameResult,
            LoadableDataType,
        },
    },
};
use serde::{Deserialize, Serialize};
use volkrix::{
    SOURCE_COMMIT,
    nnue_training::{
        CHECKPOINT_HIDDEN_BIASES_FILE, CHECKPOINT_INPUT_WEIGHTS_FILE, CHECKPOINT_MAGIC,
        CHECKPOINT_OUTPUT_BIAS_FILE, CHECKPOINT_OUTPUT_WEIGHTS_FILE, CHECKPOINT_VERSION,
        CheckpointManifest, NORMALIZED_FEN_RULE, OUTPUT_SCALE, SPLIT_RULE_DESCRIPTION,
        TOPOLOGY_FEATURE_COUNT, TOPOLOGY_HIDDEN_SIZE, TOPOLOGY_ID, TOPOLOGY_NAME,
        TOPOLOGY_OUTPUT_INPUTS, TRAINER_SEED, read_checkpoint_manifest, read_examples,
        write_checkpoint_artifacts,
    },
};

pub const BULLET_GIT_REV: &str = "feab6443fc523c9d349427bca2d5bb3c04369420";
pub const BULLET_BACKEND: &str = "Bullet CPU";
pub const BULLET_TRAINER_VERSION: &str = "phase13-bullet-v1";
pub const BULLET_METADATA_FILE: &str = "bullet-training.json";
const BULLET_OUTPUT_DIR: &str = "bullet-checkpoints";
const BULLET_TRAIN_DATA_FILE: &str = "train.bulletdata";
const BULLET_VALIDATION_DATA_FILE: &str = "validation.bulletdata";
const MAX_FEATURES_PER_PERSPECTIVE: usize = 30;
const RESULT_DRAW: u8 = 1;
const QA: i16 = 255;
const TRAINING_EVAL_SCALE: f32 = 400.0;
const TRAINING_BATCH_SIZE: usize = 512;
const TRAINING_SUPERBATCHES: usize = 64;
const TRAINING_INITIAL_LR: f32 = 0.001;
const TRAINING_FINAL_LR: f32 = 0.0001;
const TRAINING_SAVE_RATE: usize = TRAINING_SUPERBATCHES;
const TRAINER_THREADS: usize = 1;
const LOADER_THREADS: usize = 1;
const BATCH_QUEUE_SIZE: usize = 8;
const ADAMW_DECAY: f32 = 0.01;
const ADAMW_BETA1: f32 = 0.9;
const ADAMW_BETA2: f32 = 0.999;
const HIDDEN_MIN_WEIGHT: f32 = -2.0;
const HIDDEN_MAX_WEIGHT: f32 = 2.0;
const OUTPUT_MIN_WEIGHT: f32 = -64.0;
const OUTPUT_MAX_WEIGHT: f32 = 64.0;

#[derive(Clone, Debug, PartialEq)]
pub struct BulletTrainingConfig {
    pub batch_size: usize,
    pub start_superbatch: usize,
    pub end_superbatch: usize,
    pub initial_lr: f32,
    pub final_lr: f32,
    pub save_rate: usize,
    pub trainer_threads: usize,
    pub loader_threads: usize,
    pub batch_queue_size: usize,
    pub eval_scale: f32,
    pub init_from_checkpoint_dir: Option<PathBuf>,
}

impl Default for BulletTrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: TRAINING_BATCH_SIZE,
            start_superbatch: 1,
            end_superbatch: TRAINING_SUPERBATCHES,
            initial_lr: TRAINING_INITIAL_LR,
            final_lr: TRAINING_FINAL_LR,
            save_rate: TRAINING_SAVE_RATE,
            trainer_threads: TRAINER_THREADS,
            loader_threads: LOADER_THREADS,
            batch_queue_size: BATCH_QUEUE_SIZE,
            eval_scale: TRAINING_EVAL_SCALE,
            init_from_checkpoint_dir: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BulletTrainingSummary {
    pub checkpoint_dir: PathBuf,
    pub bullet_checkpoint_dir: PathBuf,
    pub train_examples: usize,
    pub validation_examples: usize,
    pub train_data_path: PathBuf,
    pub validation_data_path: PathBuf,
    pub metadata_path: PathBuf,
    pub training_log_path: Option<PathBuf>,
    pub final_training_loss: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ResolvedInitCheckpoint {
    checkpoint_dir: PathBuf,
    bullet_checkpoint_path: PathBuf,
    parent_trainer_version: String,
    parent_epochs: u32,
    parent_examples_path: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BulletTrainingMetadata {
    trainer_backend: String,
    trainer_version: String,
    bullet_git_rev: String,
    source_engine_commit: String,
    topology_name: String,
    feature_count: usize,
    hidden_size: usize,
    output_inputs: usize,
    output_scale: i32,
    activation: String,
    eval_scale: f32,
    seed: u64,
    train_examples: usize,
    validation_examples: usize,
    batch_size: usize,
    batches_per_superbatch: usize,
    start_superbatch: usize,
    end_superbatch: usize,
    stage_superbatches: usize,
    initial_lr: f32,
    final_lr: f32,
    save_rate: usize,
    trainer_threads: usize,
    loader_threads: usize,
    batch_queue_size: usize,
    train_data_file: String,
    validation_data_file: String,
    init_from_mode: Option<String>,
    init_from_checkpoint_dir: Option<String>,
    init_from_bullet_checkpoint: Option<String>,
    init_from_checkpoint_trainer_version: Option<String>,
    init_from_checkpoint_epochs: Option<u32>,
    init_from_checkpoint_examples_path: Option<String>,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BulletExampleRecord {
    target_cp: i16,
    result_code: u8,
    feature_count: u8,
    reserved: [u8; 4],
    active_features: [u16; MAX_FEATURES_PER_PERSPECTIVE],
    passive_features: [u16; MAX_FEATURES_PER_PERSPECTIVE],
}

unsafe impl CanBeDirectlySequentiallyLoaded for BulletExampleRecord {}

impl Default for BulletExampleRecord {
    fn default() -> Self {
        Self {
            target_cp: 0,
            result_code: RESULT_DRAW,
            feature_count: 0,
            reserved: [0; 4],
            active_features: [0; MAX_FEATURES_PER_PERSPECTIVE],
            passive_features: [0; MAX_FEATURES_PER_PERSPECTIVE],
        }
    }
}

impl LoadableDataType for BulletExampleRecord {
    fn score(&self) -> i16 {
        self.target_cp
    }

    fn result(&self) -> GameResult {
        match self.result_code {
            0 => GameResult::Loss,
            2 => GameResult::Win,
            _ => GameResult::Draw,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct VolkrixHalfkpInputs;

impl SparseInputType for VolkrixHalfkpInputs {
    type RequiredDataType = BulletExampleRecord;

    fn num_inputs(&self) -> usize {
        TOPOLOGY_FEATURE_COUNT
    }

    fn max_active(&self) -> usize {
        MAX_FEATURES_PER_PERSPECTIVE
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let feature_count = usize::from(pos.feature_count).min(MAX_FEATURES_PER_PERSPECTIVE);
        for index in 0..feature_count {
            f(
                usize::from(pos.active_features[index]),
                usize::from(pos.passive_features[index]),
            );
        }
    }

    fn shorthand(&self) -> String {
        "halfkp40960".to_owned()
    }

    fn description(&self) -> String {
        "Volkrix HalfKP sparse inputs".to_owned()
    }
}

pub fn train_bullet(
    examples_path: &Path,
    checkpoint_dir: &Path,
) -> Result<BulletTrainingSummary, String> {
    train_bullet_with_config(
        examples_path,
        checkpoint_dir,
        BulletTrainingConfig::default(),
    )
}

pub fn train_bullet_with_config(
    examples_path: &Path,
    checkpoint_dir: &Path,
    config: BulletTrainingConfig,
) -> Result<BulletTrainingSummary, String> {
    let (example_manifest, records, _) = read_examples(examples_path)?;
    fs::create_dir_all(checkpoint_dir).map_err(|error| {
        format!(
            "failed to create checkpoint dir '{}': {error}",
            checkpoint_dir.display()
        )
    })?;

    let train_data_path = checkpoint_dir.join(BULLET_TRAIN_DATA_FILE);
    let validation_data_path = checkpoint_dir.join(BULLET_VALIDATION_DATA_FILE);
    let (train_examples, validation_examples) =
        write_bullet_datasets(&records, &train_data_path, &validation_data_path)?;
    if train_examples == 0 {
        return Err(
            "training split is empty; provide a larger or differently distributed corpus"
                .to_owned(),
        );
    }
    let init_checkpoint = config
        .init_from_checkpoint_dir
        .as_deref()
        .map(resolve_init_checkpoint)
        .transpose()?;

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .use_threads(config.trainer_threads)
        .inputs(VolkrixHalfkpInputs)
        .save_format(&saved_format())
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs| {
            let l0 = builder.new_affine("l0", TOPOLOGY_FEATURE_COUNT, TOPOLOGY_HIDDEN_SIZE);
            let l1 = builder.new_affine("l1", TOPOLOGY_OUTPUT_INPUTS, 1);

            let active_hidden = l0.forward(stm_inputs).crelu();
            let passive_hidden = l0.forward(ntm_inputs).crelu();
            let hidden_layer = active_hidden.concat(passive_hidden);
            l1.forward(hidden_layer)
        });

    let hidden_params = AdamWParams {
        decay: ADAMW_DECAY,
        beta1: ADAMW_BETA1,
        beta2: ADAMW_BETA2,
        min_weight: HIDDEN_MIN_WEIGHT,
        max_weight: HIDDEN_MAX_WEIGHT,
    };
    let output_params = AdamWParams {
        decay: ADAMW_DECAY,
        beta1: ADAMW_BETA1,
        beta2: ADAMW_BETA2,
        min_weight: OUTPUT_MIN_WEIGHT,
        max_weight: OUTPUT_MAX_WEIGHT,
    };
    trainer
        .optimiser
        .set_params_for_weight("l0w", hidden_params);
    trainer
        .optimiser
        .set_params_for_weight("l0b", hidden_params);
    trainer
        .optimiser
        .set_params_for_weight("l1w", output_params);
    trainer
        .optimiser
        .set_params_for_weight("l1b", output_params);

    let batches_per_superbatch = train_examples.div_ceil(config.batch_size).max(1);
    let stage_superbatches = config
        .end_superbatch
        .checked_sub(config.start_superbatch)
        .and_then(|delta| delta.checked_add(1))
        .ok_or_else(|| {
            format!(
                "invalid superbatch range {}..={}",
                config.start_superbatch, config.end_superbatch
            )
        })?;
    let schedule = TrainingSchedule {
        net_id: BULLET_TRAINER_VERSION.to_owned(),
        eval_scale: config.eval_scale,
        steps: TrainingSteps {
            batch_size: config.batch_size,
            batches_per_superbatch,
            start_superbatch: config.start_superbatch,
            end_superbatch: config.end_superbatch,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.0 },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr: config.initial_lr,
            final_lr: config.final_lr,
            final_superbatch: config.end_superbatch,
        },
        save_rate: config.save_rate,
    };

    let bullet_checkpoint_dir = checkpoint_dir.join(BULLET_OUTPUT_DIR);
    let bullet_checkpoint_dir_text = bullet_checkpoint_dir
        .to_str()
        .ok_or_else(|| {
            format!(
                "bullet checkpoint dir '{}' must be UTF-8",
                bullet_checkpoint_dir.display()
            )
        })?
        .to_owned();
    let settings = LocalSettings {
        threads: config.loader_threads,
        test_set: None,
        output_directory: &bullet_checkpoint_dir_text,
        batch_queue_size: config.batch_queue_size,
    };

    let train_data_path_text = train_data_path
        .to_str()
        .ok_or_else(|| {
            format!(
                "training data path '{}' must be UTF-8",
                train_data_path.display()
            )
        })?
        .to_owned();
    let dataloader = DirectSequentialDataLoader::new(&[&train_data_path_text]);
    if let Some(init) = &init_checkpoint {
        let bullet_weights_path = init
            .bullet_checkpoint_path
            .join("optimiser_state")
            .join("weights.bin");
        let bullet_weights_path_text = bullet_weights_path
            .to_str()
            .ok_or_else(|| {
                format!(
                    "Bullet weights path '{}' must be UTF-8",
                    bullet_weights_path.display()
                )
            })?
            .to_owned();
        trainer
            .optimiser
            .load_weights_from_file(&bullet_weights_path_text)
            .map_err(|error| {
                format!(
                    "failed to initialize Bullet weights from '{}': {error:?}",
                    bullet_weights_path.display()
                )
            })?;
    }
    trainer.run(&schedule, &settings, &dataloader);

    let graph_weights = GraphWeights::from(trainer.optimiser.graph.primary());
    let input_weights = scale_values(&graph_weights.get("l0w").values, f32::from(QA));
    let hidden_biases = scale_values(&graph_weights.get("l0b").values, f32::from(QA));
    let output_weights =
        scale_output_weights_for_runtime(&graph_weights.get("l1w").values, config.eval_scale);
    let output_bias =
        scale_output_bias_for_runtime(&graph_weights.get("l1b").values, config.eval_scale);

    let manifest = CheckpointManifest {
        magic: CHECKPOINT_MAGIC.to_owned(),
        version: CHECKPOINT_VERSION,
        trainer_version: format!("{BULLET_TRAINER_VERSION}+{BULLET_GIT_REV}"),
        source_engine_commit: SOURCE_COMMIT.to_owned(),
        topology_name: TOPOLOGY_NAME.to_owned(),
        topology_id: TOPOLOGY_ID,
        feature_count: TOPOLOGY_FEATURE_COUNT,
        hidden_size: TOPOLOGY_HIDDEN_SIZE,
        output_inputs: TOPOLOGY_OUTPUT_INPUTS,
        output_scale: OUTPUT_SCALE,
        seed: TRAINER_SEED,
        optimizer: format!("{BULLET_BACKEND} AdamW"),
        loss: "sigmoid-squared-error".to_owned(),
        split_rule: SPLIT_RULE_DESCRIPTION.to_owned(),
        normalized_fen_rule: NORMALIZED_FEN_RULE.to_owned(),
        examples_path: examples_path.display().to_string(),
        example_manifest,
        train_examples,
        validation_examples,
        epochs: config.end_superbatch as u32,
        learning_rate: config.initial_lr,
        weight_decay: ADAMW_DECAY,
        init_from_mode: init_checkpoint.as_ref().map(|_| "weights-only".to_owned()),
        input_weights_file: CHECKPOINT_INPUT_WEIGHTS_FILE.to_owned(),
        hidden_biases_file: CHECKPOINT_HIDDEN_BIASES_FILE.to_owned(),
        output_weights_file: CHECKPOINT_OUTPUT_WEIGHTS_FILE.to_owned(),
        output_bias_file: CHECKPOINT_OUTPUT_BIAS_FILE.to_owned(),
        init_from_checkpoint_dir: init_checkpoint
            .as_ref()
            .map(|init| init.checkpoint_dir.display().to_string()),
        init_from_checkpoint_trainer_version: init_checkpoint
            .as_ref()
            .map(|init| init.parent_trainer_version.clone()),
        init_from_checkpoint_epochs: init_checkpoint.as_ref().map(|init| init.parent_epochs),
        init_from_checkpoint_examples_path: init_checkpoint
            .as_ref()
            .map(|init| init.parent_examples_path.clone()),
    };
    write_checkpoint_artifacts(
        checkpoint_dir,
        &manifest,
        &input_weights,
        &hidden_biases,
        &output_weights,
        &output_bias,
    )?;

    let metadata = BulletTrainingMetadata {
        trainer_backend: BULLET_BACKEND.to_owned(),
        trainer_version: BULLET_TRAINER_VERSION.to_owned(),
        bullet_git_rev: BULLET_GIT_REV.to_owned(),
        source_engine_commit: SOURCE_COMMIT.to_owned(),
        topology_name: TOPOLOGY_NAME.to_owned(),
        feature_count: TOPOLOGY_FEATURE_COUNT,
        hidden_size: TOPOLOGY_HIDDEN_SIZE,
        output_inputs: TOPOLOGY_OUTPUT_INPUTS,
        output_scale: OUTPUT_SCALE,
        activation: "crelu (runtime equivalent after QA scaling)".to_owned(),
        eval_scale: config.eval_scale,
        seed: TRAINER_SEED,
        train_examples,
        validation_examples,
        batch_size: config.batch_size,
        batches_per_superbatch,
        start_superbatch: config.start_superbatch,
        end_superbatch: config.end_superbatch,
        stage_superbatches,
        initial_lr: config.initial_lr,
        final_lr: config.final_lr,
        save_rate: config.save_rate,
        trainer_threads: config.trainer_threads,
        loader_threads: config.loader_threads,
        batch_queue_size: config.batch_queue_size,
        train_data_file: train_data_path.display().to_string(),
        validation_data_file: validation_data_path.display().to_string(),
        init_from_mode: init_checkpoint.as_ref().map(|_| "weights-only".to_owned()),
        init_from_checkpoint_dir: init_checkpoint
            .as_ref()
            .map(|init| init.checkpoint_dir.display().to_string()),
        init_from_bullet_checkpoint: init_checkpoint
            .as_ref()
            .map(|init| init.bullet_checkpoint_path.display().to_string()),
        init_from_checkpoint_trainer_version: init_checkpoint
            .as_ref()
            .map(|init| init.parent_trainer_version.clone()),
        init_from_checkpoint_epochs: init_checkpoint.as_ref().map(|init| init.parent_epochs),
        init_from_checkpoint_examples_path: init_checkpoint
            .as_ref()
            .map(|init| init.parent_examples_path.clone()),
    };
    let metadata_path = checkpoint_dir.join(BULLET_METADATA_FILE);
    fs::write(
        &metadata_path,
        serde_json::to_string_pretty(&metadata)
            .map_err(|error| format!("failed to encode Bullet metadata JSON: {error}"))?,
    )
    .map_err(|error| {
        format!(
            "failed to write Bullet metadata '{}': {error}",
            metadata_path.display()
        )
    })?;

    let training_log_path = checkpoint_dir
        .join(BULLET_OUTPUT_DIR)
        .join(format!(
            "{BULLET_TRAINER_VERSION}-{}",
            config.end_superbatch
        ))
        .join("log.txt");
    let final_training_loss = if training_log_path.exists() {
        parse_final_training_loss(&training_log_path)?
    } else {
        None
    };

    Ok(BulletTrainingSummary {
        checkpoint_dir: checkpoint_dir.to_path_buf(),
        bullet_checkpoint_dir,
        train_examples,
        validation_examples,
        train_data_path,
        validation_data_path,
        metadata_path,
        training_log_path: training_log_path.exists().then_some(training_log_path),
        final_training_loss,
    })
}

fn resolve_init_checkpoint(checkpoint_dir: &Path) -> Result<ResolvedInitCheckpoint, String> {
    let parent_manifest = read_checkpoint_manifest(checkpoint_dir)?;
    let metadata_path = checkpoint_dir.join(BULLET_METADATA_FILE);
    let metadata_text = fs::read_to_string(&metadata_path).map_err(|error| {
        format!(
            "failed to read Bullet metadata '{}': {error}",
            metadata_path.display()
        )
    })?;
    let metadata: BulletTrainingMetadata =
        serde_json::from_str(&metadata_text).map_err(|error| {
            format!(
                "failed to parse Bullet metadata '{}': {error}",
                metadata_path.display()
            )
        })?;

    let bullet_checkpoint_path = resolve_bullet_checkpoint_path(
        &checkpoint_dir.join(BULLET_OUTPUT_DIR),
        &metadata.trainer_version,
        metadata.end_superbatch,
    )?;
    Ok(ResolvedInitCheckpoint {
        checkpoint_dir: checkpoint_dir.to_path_buf(),
        bullet_checkpoint_path,
        parent_trainer_version: parent_manifest.trainer_version,
        parent_epochs: parent_manifest.epochs,
        parent_examples_path: parent_manifest.examples_path,
    })
}

fn resolve_bullet_checkpoint_path(
    bullet_output_dir: &Path,
    trainer_version: &str,
    end_superbatch: usize,
) -> Result<PathBuf, String> {
    let expected = bullet_output_dir.join(format!("{trainer_version}-{end_superbatch}"));
    if expected.is_dir() {
        return Ok(expected);
    }

    let prefix = format!("{trainer_version}-");
    let mut candidates = Vec::new();
    let entries = fs::read_dir(bullet_output_dir).map_err(|error| {
        format!(
            "failed to read Bullet checkpoint dir '{}': {error}",
            bullet_output_dir.display()
        )
    })?;
    for entry in entries {
        let entry = entry.map_err(|error| {
            format!(
                "failed to read Bullet checkpoint entry in '{}': {error}",
                bullet_output_dir.display()
            )
        })?;
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_dir() {
            continue;
        }
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let Some(suffix) = name.strip_prefix(&prefix) else {
            continue;
        };
        let Ok(superbatch) = suffix.parse::<usize>() else {
            continue;
        };
        candidates.push((superbatch, entry.path()));
    }
    candidates.sort_by_key(|(superbatch, _)| *superbatch);
    candidates
        .pop()
        .map(|(_, path)| path)
        .ok_or_else(|| {
            format!(
                "failed to resolve Bullet checkpoint under '{}'; expected '{}' or at least one '{}*' directory",
                bullet_output_dir.display(),
                expected.display(),
                prefix
            )
        })
}

fn saved_format() -> [SavedFormat; 4] {
    [
        SavedFormat::id("l0w").round().quantise::<i16>(QA),
        SavedFormat::id("l0b").round().quantise::<i16>(QA),
        SavedFormat::id("l1w")
            .transform(|_, mut values| {
                for value in &mut values {
                    *value *= runtime_output_weight_scale(TRAINING_EVAL_SCALE);
                }
                values
            })
            .round()
            .quantise::<i16>(1),
        SavedFormat::id("l1b")
            .round()
            .quantise::<i32>(runtime_output_bias_scale(TRAINING_EVAL_SCALE) as i32),
    ]
}

fn scale_values(values: &[f32], scale: f32) -> Vec<f32> {
    values.iter().map(|value| value * scale).collect()
}

fn runtime_output_weight_scale(eval_scale: f32) -> f32 {
    eval_scale * OUTPUT_SCALE as f32 / f32::from(QA)
}

fn runtime_output_bias_scale(eval_scale: f32) -> f32 {
    eval_scale * OUTPUT_SCALE as f32
}

fn scale_output_weights_for_runtime(values: &[f32], eval_scale: f32) -> Vec<f32> {
    scale_values(values, runtime_output_weight_scale(eval_scale))
}

fn scale_output_bias_for_runtime(values: &[f32], eval_scale: f32) -> Vec<f32> {
    scale_values(values, runtime_output_bias_scale(eval_scale))
}

fn parse_final_training_loss(path: &Path) -> Result<Option<f32>, String> {
    let text = fs::read_to_string(path).map_err(|error| {
        format!(
            "failed to read Bullet training log '{}': {error}",
            path.display()
        )
    })?;
    let Some(last_line) = text.lines().last() else {
        return Ok(None);
    };
    let Some(loss_text) = last_line.rsplit(',').next() else {
        return Ok(None);
    };
    let loss = loss_text.parse::<f32>().map_err(|error| {
        format!(
            "failed to parse Bullet training loss '{}': {error}",
            loss_text
        )
    })?;
    Ok(Some(loss))
}

fn write_bullet_datasets(
    records: &[volkrix::nnue_training::ExampleRecord],
    train_path: &Path,
    validation_path: &Path,
) -> Result<(usize, usize), String> {
    let mut train_writer = BufWriter::new(File::create(train_path).map_err(|error| {
        format!(
            "failed to create Bullet training data '{}': {error}",
            train_path.display()
        )
    })?);
    let mut validation_writer = BufWriter::new(File::create(validation_path).map_err(|error| {
        format!(
            "failed to create Bullet validation data '{}': {error}",
            validation_path.display()
        )
    })?);

    let mut train_examples = 0usize;
    let mut validation_examples = 0usize;
    for record in records {
        let encoded = encode_record(record)?;
        let bytes = record_as_bytes(&encoded);
        match volkrix::nnue_training::split_for_normalized_fen(&record.normalized_fen) {
            volkrix::nnue_training::DatasetSplit::Train => {
                train_writer.write_all(bytes).map_err(|error| {
                    format!(
                        "failed to write Bullet training record '{}': {error}",
                        train_path.display()
                    )
                })?;
                train_examples += 1;
            }
            volkrix::nnue_training::DatasetSplit::Validation => {
                validation_writer.write_all(bytes).map_err(|error| {
                    format!(
                        "failed to write Bullet validation record '{}': {error}",
                        validation_path.display()
                    )
                })?;
                validation_examples += 1;
            }
        }
    }

    train_writer.flush().map_err(|error| {
        format!(
            "failed to flush Bullet training data '{}': {error}",
            train_path.display()
        )
    })?;
    validation_writer.flush().map_err(|error| {
        format!(
            "failed to flush Bullet validation data '{}': {error}",
            validation_path.display()
        )
    })?;
    Ok((train_examples, validation_examples))
}

fn encode_record(
    record: &volkrix::nnue_training::ExampleRecord,
) -> Result<BulletExampleRecord, String> {
    if record.active_features.len() != record.passive_features.len() {
        return Err(format!(
            "feature count mismatch for '{}': active {} passive {}",
            record.normalized_fen,
            record.active_features.len(),
            record.passive_features.len()
        ));
    }
    if record.active_features.len() > MAX_FEATURES_PER_PERSPECTIVE {
        return Err(format!(
            "feature count {} exceeded retained maximum {} for '{}'",
            record.active_features.len(),
            MAX_FEATURES_PER_PERSPECTIVE,
            record.normalized_fen
        ));
    }

    let mut encoded = BulletExampleRecord {
        target_cp: i16::try_from(record.target_cp).map_err(|_| {
            format!(
                "target {} exceeded Bullet record i16 range for '{}'",
                record.target_cp, record.normalized_fen
            )
        })?,
        result_code: RESULT_DRAW,
        feature_count: u8::try_from(record.active_features.len())
            .expect("retained feature count must fit in u8"),
        ..BulletExampleRecord::default()
    };
    for (index, feature) in record.active_features.iter().enumerate() {
        encoded.active_features[index] = *feature;
    }
    for (index, feature) in record.passive_features.iter().enumerate() {
        encoded.passive_features[index] = *feature;
    }
    Ok(encoded)
}

fn record_as_bytes(record: &BulletExampleRecord) -> &[u8] {
    // SAFETY: `BulletExampleRecord` is `repr(C) + Copy` and contains only plain integer fields.
    unsafe {
        slice::from_raw_parts(
            (record as *const BulletExampleRecord).cast::<u8>(),
            std::mem::size_of::<BulletExampleRecord>(),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    };

    use bullet_lib::{game::inputs::SparseInputType, value::loader::LoadableDataType};

    use super::{
        BULLET_METADATA_FILE, BULLET_OUTPUT_DIR, BULLET_TRAINER_VERSION, BulletTrainingConfig,
        resolve_bullet_checkpoint_path, train_bullet_with_config,
    };
    use volkrix::nnue_training::{
        CHECKPOINT_MANIFEST_FILE, ExampleRecord, export_examples, pack_checkpoint_to_volknnue,
        read_checkpoint_manifest, validate_volknnue,
    };

    fn temp_path(label: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be monotonic enough for tests")
            .as_nanos();
        std::env::temp_dir().join(format!("volkrix-bullet-{label}-{suffix}"))
    }

    #[test]
    fn bullet_training_smoke_produces_packable_checkpoint() {
        let input_path = temp_path("fixture-fens");
        let examples_path = temp_path("fixture-examples");
        fs::write(
            &input_path,
            [
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "r2q1rk1/ppp2ppp/2npbn2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8",
                "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 2 3",
                "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            ]
            .join("\n"),
        )
        .expect("fixture corpus must write");
        export_examples(&input_path, &examples_path).expect("export must succeed");

        let checkpoint_dir = temp_path("bullet-checkpoint-dir");
        let config = BulletTrainingConfig {
            batch_size: 4,
            start_superbatch: 1,
            end_superbatch: 1,
            initial_lr: 0.001,
            final_lr: 0.001,
            save_rate: 1,
            trainer_threads: 1,
            loader_threads: 1,
            batch_queue_size: 1,
            eval_scale: 400.0,
            init_from_checkpoint_dir: None,
        };
        let summary = train_bullet_with_config(&examples_path, &checkpoint_dir, config)
            .expect("Bullet training must succeed");
        assert!(summary.train_examples > 0);
        assert!(checkpoint_dir.join(CHECKPOINT_MANIFEST_FILE).exists());
        assert!(checkpoint_dir.join(BULLET_METADATA_FILE).exists());
        let checkpoint_manifest: volkrix::nnue_training::CheckpointManifest = serde_json::from_str(
            &fs::read_to_string(checkpoint_dir.join(CHECKPOINT_MANIFEST_FILE))
                .expect("checkpoint manifest must read"),
        )
        .expect("checkpoint manifest must parse");
        assert!(
            checkpoint_manifest
                .trainer_version
                .contains(super::BULLET_GIT_REV)
        );
        assert_eq!(checkpoint_manifest.train_examples, summary.train_examples);
        assert_eq!(
            checkpoint_manifest.validation_examples,
            summary.validation_examples
        );
        assert_eq!(
            checkpoint_manifest.example_manifest.source_engine_commit,
            volkrix::SOURCE_COMMIT
        );

        let bullet_metadata: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(checkpoint_dir.join(BULLET_METADATA_FILE))
                .expect("Bullet metadata must read"),
        )
        .expect("Bullet metadata must parse");
        assert_eq!(
            bullet_metadata["bullet_git_rev"].as_str(),
            Some(super::BULLET_GIT_REV)
        );
        assert_eq!(
            bullet_metadata["train_examples"].as_u64(),
            Some(summary.train_examples as u64)
        );

        let packed_net = checkpoint_dir.join("smoke.volknnue");
        pack_checkpoint_to_volknnue(&checkpoint_dir, &packed_net).expect("pack must succeed");
        validate_volknnue(&packed_net).expect("packed net must validate");

        cleanup(&input_path);
        cleanup(&examples_path);
        cleanup(&checkpoint_dir);
    }

    #[test]
    fn bullet_adapter_preserves_sparse_feature_pairs_and_target() {
        let record = ExampleRecord {
            fen: "dummy".to_owned(),
            normalized_fen: "dummy".to_owned(),
            side_to_move: volkrix::core::Color::White,
            raw_score_cp: 123,
            target_cp: 77,
            active_features: vec![7, 19, 42],
            passive_features: vec![8, 20, 43],
        };
        let encoded = super::encode_record(&record).expect("record must encode");
        assert_eq!(encoded.score(), 77);

        let mut mapped = Vec::new();
        super::VolkrixHalfkpInputs.map_features(&encoded, |active, passive| {
            mapped.push((active, passive));
        });
        assert_eq!(mapped, vec![(7, 8), (19, 20), (42, 43)]);
    }

    #[test]
    fn runtime_output_bridge_includes_eval_scale() {
        let scaled_weights = super::scale_output_weights_for_runtime(&[0.5, -0.25], 400.0);
        let scaled_bias = super::scale_output_bias_for_runtime(&[0.5], 400.0);

        let expected_weight_scale = 400.0 * 64.0 / 255.0;
        assert!((scaled_weights[0] - 0.5 * expected_weight_scale).abs() < 1e-6);
        assert!((scaled_weights[1] + 0.25 * expected_weight_scale).abs() < 1e-6);
        assert!((scaled_bias[0] - 0.5 * 400.0 * 64.0).abs() < 1e-6);
    }

    #[test]
    fn resolves_latest_bullet_checkpoint_when_expected_dir_is_missing() {
        let bullet_output_dir = temp_path("bullet-checkpoint-resolution");
        fs::create_dir_all(bullet_output_dir.join(format!("{BULLET_TRAINER_VERSION}-2")))
            .expect("checkpoint dir must create");
        fs::create_dir_all(bullet_output_dir.join(format!("{BULLET_TRAINER_VERSION}-5")))
            .expect("checkpoint dir must create");

        let resolved =
            resolve_bullet_checkpoint_path(&bullet_output_dir, BULLET_TRAINER_VERSION, 4)
                .expect("resolution must succeed");
        let expected_dir = format!("{BULLET_TRAINER_VERSION}-5");
        assert_eq!(
            resolved.file_name().and_then(|name| name.to_str()),
            Some(expected_dir.as_str())
        );

        cleanup(&bullet_output_dir);
    }

    #[test]
    fn resumed_training_records_parent_checkpoint_traceability() {
        let input_path = temp_path("resume-fens");
        let examples_path = temp_path("resume-examples");
        fs::write(
            &input_path,
            [
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "r2q1rk1/ppp2ppp/2npbn2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8",
                "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 2 3",
                "4k3/8/8/8/8/8/3Q4/4K3 w - - 0 1",
            ]
            .join("\n"),
        )
        .expect("fixture corpus must write");
        export_examples(&input_path, &examples_path).expect("export must succeed");

        let stage_one_dir = temp_path("resume-stage-one");
        let stage_one_summary = train_bullet_with_config(
            &examples_path,
            &stage_one_dir,
            BulletTrainingConfig {
                batch_size: 4,
                start_superbatch: 1,
                end_superbatch: 1,
                initial_lr: 0.001,
                final_lr: 0.001,
                save_rate: 1,
                trainer_threads: 1,
                loader_threads: 1,
                batch_queue_size: 1,
                eval_scale: 400.0,
                init_from_checkpoint_dir: None,
            },
        )
        .expect("stage one must train");
        assert!(
            stage_one_summary
                .bullet_checkpoint_dir
                .join(format!("{BULLET_TRAINER_VERSION}-1"))
                .is_dir()
        );

        let stage_two_dir = temp_path("resume-stage-two");
        let stage_two_summary = train_bullet_with_config(
            &examples_path,
            &stage_two_dir,
            BulletTrainingConfig {
                batch_size: 4,
                start_superbatch: 1,
                end_superbatch: 1,
                initial_lr: 0.0005,
                final_lr: 0.0005,
                save_rate: 1,
                trainer_threads: 1,
                loader_threads: 1,
                batch_queue_size: 1,
                eval_scale: 400.0,
                init_from_checkpoint_dir: Some(stage_one_dir.clone()),
            },
        )
        .expect("stage two must train");
        assert!(
            stage_two_summary
                .bullet_checkpoint_dir
                .join(format!("{BULLET_TRAINER_VERSION}-1"))
                .is_dir()
        );

        let stage_two_manifest =
            read_checkpoint_manifest(&stage_two_dir).expect("stage two manifest must parse");
        let stage_one_dir_text = stage_one_dir.display().to_string();
        let stage_one_bullet_checkpoint_text = stage_one_dir
            .join(BULLET_OUTPUT_DIR)
            .join(format!("{BULLET_TRAINER_VERSION}-1"))
            .display()
            .to_string();
        assert_eq!(
            stage_two_manifest.init_from_checkpoint_dir,
            Some(stage_one_dir_text.clone())
        );
        assert_eq!(
            stage_two_manifest.init_from_mode,
            Some("weights-only".to_owned())
        );
        assert_eq!(stage_two_manifest.init_from_checkpoint_epochs, Some(1));
        assert!(
            stage_two_manifest
                .init_from_checkpoint_trainer_version
                .as_deref()
                .is_some_and(|value| value.contains(BULLET_TRAINER_VERSION))
        );

        let stage_two_metadata: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(stage_two_dir.join(BULLET_METADATA_FILE))
                .expect("stage two metadata must read"),
        )
        .expect("stage two metadata must parse");
        assert_eq!(stage_two_metadata["start_superbatch"].as_u64(), Some(1));
        assert_eq!(stage_two_metadata["end_superbatch"].as_u64(), Some(1));
        assert_eq!(stage_two_metadata["stage_superbatches"].as_u64(), Some(1));
        assert_eq!(
            stage_two_metadata["init_from_mode"].as_str(),
            Some("weights-only")
        );
        assert_eq!(
            stage_two_metadata["init_from_checkpoint_dir"].as_str(),
            Some(stage_one_dir_text.as_str())
        );
        assert_eq!(
            stage_two_metadata["init_from_checkpoint_epochs"].as_u64(),
            Some(1)
        );
        assert_eq!(
            stage_two_metadata["init_from_bullet_checkpoint"].as_str(),
            Some(stage_one_bullet_checkpoint_text.as_str())
        );

        cleanup(&input_path);
        cleanup(&examples_path);
        cleanup(&stage_one_dir);
        cleanup(&stage_two_dir);
    }

    fn cleanup(path: &Path) {
        let _ = fs::remove_file(path);
        let _ = fs::remove_dir_all(path);
    }
}
