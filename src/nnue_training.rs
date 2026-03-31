use std::{
    collections::{HashSet, VecDeque},
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, mpsc},
    thread,
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};

use crate::{
    SOURCE_COMMIT,
    core::position::MoveGenStage,
    core::{Color, MoveList, Position, PositionStatus},
    search::{
        SearchLimits, eval,
        nnue::{
            NNUE_FEATURE_COUNT, RETAINED_PRODUCTION_TOPOLOGY, SparseFeaturePair, encode_volknnue,
            sparse_features_for_side_to_move, supported_halfkp_topology,
        },
        service::{SearchRequest, UciSearchService},
        tablebase::position_is_within_retained_scope,
    },
};

pub const EXAMPLES_MAGIC: &str = "VOLKRIX_EXAMPLES";
pub const EXAMPLES_VERSION: u32 = 1;
pub const CHECKPOINT_MAGIC: &str = "VOLKRIX_HALFKP_CHECKPOINT";
const LEGACY_CHECKPOINT_MAGIC: &str = "VOLKRIX_HALFKP128X2_CHECKPOINT";
pub const CHECKPOINT_VERSION: u32 = 1;
pub const PACK_MANIFEST_MAGIC: &str = "VOLKRIX_VOLKNNUE_PACK";
pub const PACK_MANIFEST_VERSION: u32 = 1;
pub const EXPORTER_VERSION: &str = "phase13-v1";
pub const TRAINER_VERSION: &str = "phase13-v1";
pub const PACKER_VERSION: &str = "phase13-v1";
pub const TOPOLOGY_NAME: &str = RETAINED_PRODUCTION_TOPOLOGY.name;
pub const TOPOLOGY_ID: u32 = RETAINED_PRODUCTION_TOPOLOGY.id;
pub const TOPOLOGY_FEATURE_COUNT: usize = NNUE_FEATURE_COUNT;
pub const TOPOLOGY_HIDDEN_SIZE: usize = RETAINED_PRODUCTION_TOPOLOGY.hidden_size;
pub const TOPOLOGY_OUTPUT_INPUTS: usize = RETAINED_PRODUCTION_TOPOLOGY.output_inputs();
pub const LABEL_DEPTH: u8 = 5;
pub const TARGET_CLIP_CP: i32 = 2_000;
pub const OUTPUT_SCALE: i32 = 64;
pub const TRAINER_SEED: u64 = 13;
pub const TRAINER_OPTIMIZER: &str = "AdamW";
pub const TRAINER_LOSS: &str = "SmoothL1Loss";
pub const NORMALIZED_FEN_RULE: &str =
    "normalized FEN is Position::from_fen(input)?.to_fen(), preserving all 6 canonical fields";
pub const SPLIT_RULE_DESCRIPTION: &str = "fnv1a64(normalized_fen_utf8) % 10 == 0 => validation";
const EXAMPLES_COLUMNS: &str =
    "fen\tnormalized_fen\tside_to_move\traw_score_cp\ttarget_cp\tactive_features\tpassive_features";
const EXAMPLES_TEMP_SUFFIX: &str = ".tmp";
pub const CHECKPOINT_MANIFEST_FILE: &str = "manifest.json";
pub const CHECKPOINT_INPUT_WEIGHTS_FILE: &str = "input_weights.f32le";
pub const CHECKPOINT_HIDDEN_BIASES_FILE: &str = "hidden_biases.f32le";
pub const CHECKPOINT_OUTPUT_WEIGHTS_FILE: &str = "output_weights.f32le";
pub const CHECKPOINT_OUTPUT_BIAS_FILE: &str = "output_bias.f32le";
const VALIDATION_MODULUS: u64 = 10;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LabelGenerationConfig {
    pub depth: u8,
    pub tt_enabled: bool,
    pub hash_mb: usize,
    pub mode: LabelMode,
    pub workers: usize,
    pub position_filter: PositionFilter,
    pub label_timeout_ms: Option<u64>,
}

impl Default for LabelGenerationConfig {
    fn default() -> Self {
        Self {
            depth: LABEL_DEPTH,
            tt_enabled: false,
            hash_mb: SearchLimits::new(LABEL_DEPTH).hash_mb,
            mode: LabelMode::Search,
            workers: 1,
            position_filter: PositionFilter::Any,
            label_timeout_ms: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum LabelMode {
    Search,
    StaticEval,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum PositionFilter {
    Any,
    Quiet,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CorpusExpansionConfig {
    pub max_plies: u8,
    pub branching: usize,
    pub max_positions: usize,
}

impl Default for CorpusExpansionConfig {
    fn default() -> Self {
        Self {
            max_plies: 3,
            branching: 4,
            max_positions: 1_024,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CorpusExpansionSummary {
    pub input_lines: usize,
    pub emitted_positions: usize,
    pub skipped_blank_lines: usize,
    pub skipped_invalid_positions: usize,
    pub skipped_terminal_positions: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SelfPlayCorpusConfig {
    pub hash_mb: usize,
    pub max_plies: usize,
    pub max_positions: usize,
    pub mode: MatchMode,
}

impl Default for SelfPlayCorpusConfig {
    fn default() -> Self {
        Self {
            hash_mb: 64,
            max_plies: 80,
            max_positions: 10_000,
            mode: MatchMode::FixedDepth(2),
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SelfPlayCorpusSummary {
    pub input_lines: usize,
    pub emitted_positions: usize,
    pub skipped_blank_lines: usize,
    pub skipped_invalid_positions: usize,
    pub skipped_terminal_positions: usize,
    pub completed_games: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DatasetSplit {
    Train,
    Validation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatchMode {
    FixedDepth(u8),
    MoveTimeMs(u64),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatchConfig {
    pub hash_mb: usize,
    pub max_plies: usize,
    pub mode: MatchMode,
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self {
            hash_mb: 64,
            max_plies: 160,
            mode: MatchMode::MoveTimeMs(100),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatchOutcome {
    CandidateWin,
    FallbackWin,
    Draw,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatchGameSummary {
    pub opening_fen: String,
    pub candidate_color: Color,
    pub outcome: MatchOutcome,
    pub terminal_status: PositionStatus,
    pub plies_played: usize,
    pub first_candidate_score_cp: Option<i32>,
    pub first_candidate_info_line: Option<String>,
    pub first_fallback_score_cp: Option<i32>,
    pub first_fallback_info_line: Option<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MatchSummary {
    pub openings: usize,
    pub games: usize,
    pub candidate_wins: usize,
    pub fallback_wins: usize,
    pub draws: usize,
    pub game_summaries: Vec<MatchGameSummary>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExampleExportManifest {
    pub magic: String,
    pub version: u32,
    pub exporter_version: String,
    pub source_engine_commit: String,
    pub topology_name: String,
    pub topology_id: u32,
    pub feature_count: usize,
    pub hidden_size: usize,
    pub output_inputs: usize,
    pub label_depth: u8,
    pub target_clip_cp: i32,
    pub threads: usize,
    pub syzygy_path: String,
    pub eval_file: String,
    pub tt_enabled: bool,
    pub hash_mb: usize,
    pub label_mode: LabelMode,
    pub export_workers: usize,
    pub position_filter: PositionFilter,
    pub label_timeout_ms: Option<u64>,
    pub normalized_fen_rule: String,
}

impl ExampleExportManifest {
    fn from_label_config(config: LabelGenerationConfig) -> Self {
        Self {
            magic: EXAMPLES_MAGIC.to_owned(),
            version: EXAMPLES_VERSION,
            exporter_version: EXPORTER_VERSION.to_owned(),
            source_engine_commit: SOURCE_COMMIT.to_owned(),
            topology_name: TOPOLOGY_NAME.to_owned(),
            topology_id: TOPOLOGY_ID,
            feature_count: TOPOLOGY_FEATURE_COUNT,
            hidden_size: TOPOLOGY_HIDDEN_SIZE,
            output_inputs: TOPOLOGY_OUTPUT_INPUTS,
            label_depth: config.depth,
            target_clip_cp: TARGET_CLIP_CP,
            threads: 1,
            syzygy_path: String::new(),
            eval_file: String::new(),
            tt_enabled: config.tt_enabled,
            hash_mb: config.hash_mb,
            label_mode: config.mode,
            export_workers: config.workers,
            position_filter: config.position_filter,
            label_timeout_ms: config.label_timeout_ms,
            normalized_fen_rule: NORMALIZED_FEN_RULE.to_owned(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExampleRecord {
    pub fen: String,
    pub normalized_fen: String,
    pub side_to_move: Color,
    pub raw_score_cp: i32,
    pub target_cp: i32,
    pub active_features: Vec<u16>,
    pub passive_features: Vec<u16>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExportSummary {
    pub input_lines: usize,
    pub emitted_examples: usize,
    pub skipped_blank_lines: usize,
    pub skipped_tablebase_scope_positions: usize,
    pub skipped_position_filter_positions: usize,
    pub skipped_incomplete_search_positions: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ExportDecision {
    Example(ExampleRecord),
    SkipTablebaseScope,
    SkipPositionFilter,
    SkipIncompleteSearch,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CheckpointManifest {
    pub magic: String,
    pub version: u32,
    pub trainer_version: String,
    pub source_engine_commit: String,
    pub topology_name: String,
    pub topology_id: u32,
    pub feature_count: usize,
    pub hidden_size: usize,
    pub output_inputs: usize,
    pub output_scale: i32,
    pub seed: u64,
    pub optimizer: String,
    pub loss: String,
    pub split_rule: String,
    pub normalized_fen_rule: String,
    pub examples_path: String,
    pub example_manifest: ExampleExportManifest,
    pub train_examples: usize,
    pub validation_examples: usize,
    pub epochs: u32,
    pub learning_rate: f32,
    pub weight_decay: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init_from_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init_from_checkpoint_dir: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init_from_checkpoint_trainer_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init_from_checkpoint_epochs: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init_from_checkpoint_examples_path: Option<String>,
    pub input_weights_file: String,
    pub hidden_biases_file: String,
    pub output_weights_file: String,
    pub output_bias_file: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PackedNetManifest {
    pub magic: String,
    pub version: u32,
    pub packer_version: String,
    pub source_engine_commit: String,
    pub output_net: String,
    pub topology_name: String,
    pub topology_id: u32,
    pub feature_count: usize,
    pub hidden_size: usize,
    pub output_inputs: usize,
    pub output_scale: i32,
    pub checkpoint_manifest: CheckpointManifest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackSummary {
    pub output_path: PathBuf,
    pub manifest_path: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidationSummary {
    pub net_path: PathBuf,
    pub manifest_path: Option<PathBuf>,
}

pub fn normalize_fen(input: &str) -> Result<String, String> {
    let position = Position::from_fen(input.trim())
        .map_err(|error| format!("failed to parse FEN: {error}"))?;
    Ok(position.to_fen())
}

pub fn split_for_normalized_fen(normalized_fen: &str) -> DatasetSplit {
    if fnv1a64(normalized_fen.as_bytes()).is_multiple_of(VALIDATION_MODULUS) {
        DatasetSplit::Validation
    } else {
        DatasetSplit::Train
    }
}

pub fn export_examples(input_path: &Path, output_path: &Path) -> Result<ExportSummary, String> {
    export_examples_with_config(input_path, output_path, LabelGenerationConfig::default())
}

pub fn export_examples_with_config(
    input_path: &Path,
    output_path: &Path,
    config: LabelGenerationConfig,
) -> Result<ExportSummary, String> {
    let input = File::open(input_path).map_err(|error| {
        format!(
            "failed to open FEN corpus '{}': {error}",
            input_path.display()
        )
    })?;

    let mut summary = ExportSummary::default();
    let mut work_items = Vec::new();
    for (line_number, line) in BufReader::new(input).lines().enumerate() {
        summary.input_lines += 1;
        let line = line
            .map_err(|error| format!("failed to read input line {}: {error}", line_number + 1))?;
        let fen = line.trim();
        if fen.is_empty() {
            summary.skipped_blank_lines += 1;
            continue;
        }
        work_items.push((line_number + 1, fen.to_owned()));
    }

    let results = process_export_work_items(&work_items, config)?;
    let mut output = BufWriter::new(File::create(temporary_output_path(output_path)).map_err(
        |error| {
            format!(
                "failed to create examples output '{}': {error}",
                output_path.display()
            )
        },
    )?);
    let manifest = ExampleExportManifest::from_label_config(config);
    write_examples_prologue(&mut output, &manifest)?;
    for (line_number, result) in results {
        match result.map_err(|error| format!("input line {line_number}: {error}"))? {
            ExportDecision::Example(example) => {
                write_example_record(&mut output, &example)?;
                summary.emitted_examples += 1;
            }
            ExportDecision::SkipTablebaseScope => {
                summary.skipped_tablebase_scope_positions += 1;
            }
            ExportDecision::SkipPositionFilter => {
                summary.skipped_position_filter_positions += 1;
            }
            ExportDecision::SkipIncompleteSearch => {
                summary.skipped_incomplete_search_positions += 1;
            }
        }
    }

    output.flush().map_err(|error| {
        format!(
            "failed to flush examples output '{}': {error}",
            output_path.display()
        )
    })?;
    fs::rename(temporary_output_path(output_path), output_path).map_err(|error| {
        format!(
            "failed to finalize examples output '{}': {error}",
            output_path.display()
        )
    })?;
    Ok(summary)
}

fn process_export_work_items(
    work_items: &[(usize, String)],
    config: LabelGenerationConfig,
) -> Result<Vec<(usize, Result<ExportDecision, String>)>, String> {
    if work_items.is_empty() {
        return Ok(Vec::new());
    }
    if config.workers <= 1 {
        let mut label_service = build_label_service(config)?;
        let mut results = Vec::with_capacity(work_items.len());
        for (line_number, fen) in work_items {
            results.push((
                *line_number,
                export_example_from_fen(fen, config, &mut label_service),
            ));
        }
        return Ok(results);
    }

    let worker_count = config.workers.min(work_items.len());
    let queue = Arc::new(Mutex::new(VecDeque::from(
        work_items
            .iter()
            .enumerate()
            .map(|(index, (line_number, fen))| (index, *line_number, fen.clone()))
            .collect::<Vec<_>>(),
    )));
    let (sender, receiver) = mpsc::channel::<(usize, usize, Result<ExportDecision, String>)>();
    let mut handles = Vec::with_capacity(worker_count);

    for _ in 0..worker_count {
        let queue = Arc::clone(&queue);
        let sender = sender.clone();
        handles.push(thread::spawn(move || -> Result<(), String> {
            let mut label_service = build_label_service(config)?;
            loop {
                let job = {
                    let mut queue = queue
                        .lock()
                        .map_err(|_| "search-label export queue mutex was poisoned".to_owned())?;
                    queue.pop_front()
                };
                let Some((index, line_number, fen)) = job else {
                    return Ok(());
                };
                let result = export_example_from_fen(&fen, config, &mut label_service);
                sender.send((index, line_number, result)).map_err(|error| {
                    format!("failed to send export result from worker: {error}")
                })?;
            }
        }));
    }
    drop(sender);

    let mut ordered = vec![None; work_items.len()];
    for _ in 0..work_items.len() {
        let (index, line_number, result) = receiver
            .recv()
            .map_err(|error| format!("failed to receive export result from worker: {error}"))?;
        ordered[index] = Some((line_number, result));
    }
    for handle in handles {
        handle
            .join()
            .map_err(|_| "search-label export worker thread panicked".to_owned())??;
    }

    ordered
        .into_iter()
        .map(|entry| {
            entry.ok_or_else(|| {
                "search-label export worker results did not cover every work item".to_owned()
            })
        })
        .collect()
}

pub fn expand_fen_corpus(
    input_path: &Path,
    output_path: &Path,
    config: CorpusExpansionConfig,
) -> Result<CorpusExpansionSummary, String> {
    if config.branching == 0 {
        return Err("corpus expansion branching must be at least 1".to_owned());
    }
    if config.max_positions == 0 {
        return Err("corpus expansion max_positions must be at least 1".to_owned());
    }

    let input = File::open(input_path).map_err(|error| {
        format!(
            "failed to open seed FEN corpus '{}': {error}",
            input_path.display()
        )
    })?;
    let mut output_positions = Vec::new();
    let mut queue = VecDeque::new();
    let mut seen = HashSet::new();
    let mut summary = CorpusExpansionSummary::default();

    for (line_number, line) in BufReader::new(input).lines().enumerate() {
        summary.input_lines += 1;
        let line = line
            .map_err(|error| format!("failed to read input line {}: {error}", line_number + 1))?;
        let fen = line.trim();
        if fen.is_empty() {
            summary.skipped_blank_lines += 1;
            continue;
        }

        let normalized = match normalize_fen(fen) {
            Ok(normalized) => normalized,
            Err(_) => {
                summary.skipped_invalid_positions += 1;
                continue;
            }
        };
        let mut position = Position::from_fen(&normalized).map_err(|error| {
            format!(
                "failed to parse normalized FEN on line {} '{}': {error}",
                line_number + 1,
                normalized
            )
        })?;
        if position.status() != PositionStatus::Ongoing {
            summary.skipped_terminal_positions += 1;
            continue;
        }
        if seen.insert(normalized.clone()) {
            output_positions.push(normalized.clone());
            queue.push_back((normalized, 0u8));
            if output_positions.len() >= config.max_positions {
                break;
            }
        }
    }

    while let Some((fen, ply)) = queue.pop_front() {
        if ply >= config.max_plies || output_positions.len() >= config.max_positions {
            continue;
        }

        let mut position = Position::from_fen(&fen)
            .map_err(|error| format!("failed to parse FEN '{fen}': {error}"))?;
        let mut moves = MoveList::new();
        position.generate_legal_moves(&mut moves);
        let children = evenly_spaced_children(&mut position, &moves, config.branching)?;
        for child in children {
            if seen.insert(child.clone()) {
                output_positions.push(child.clone());
                queue.push_back((child, ply + 1));
                if output_positions.len() >= config.max_positions {
                    break;
                }
            }
        }
    }

    let mut output = BufWriter::new(File::create(output_path).map_err(|error| {
        format!(
            "failed to create expanded corpus '{}': {error}",
            output_path.display()
        )
    })?);
    for fen in &output_positions {
        writeln!(output, "{fen}").map_err(|error| {
            format!(
                "failed to write expanded corpus '{}': {error}",
                output_path.display()
            )
        })?;
    }
    output.flush().map_err(|error| {
        format!(
            "failed to flush expanded corpus '{}': {error}",
            output_path.display()
        )
    })?;

    summary.emitted_positions = output_positions.len();
    Ok(summary)
}

pub fn generate_selfplay_corpus(
    input_path: &Path,
    output_path: &Path,
    config: SelfPlayCorpusConfig,
) -> Result<SelfPlayCorpusSummary, String> {
    if config.max_positions == 0 {
        return Err("self-play corpus max_positions must be at least 1".to_owned());
    }

    let input = File::open(input_path).map_err(|error| {
        format!(
            "failed to open self-play seed FEN corpus '{}': {error}",
            input_path.display()
        )
    })?;
    let mut service = build_match_service("", config.hash_mb)?;
    let mut seen = HashSet::new();
    let mut output_positions = Vec::new();
    let mut summary = SelfPlayCorpusSummary::default();

    for (line_number, line) in BufReader::new(input).lines().enumerate() {
        if output_positions.len() >= config.max_positions {
            break;
        }

        summary.input_lines += 1;
        let line = line
            .map_err(|error| format!("failed to read input line {}: {error}", line_number + 1))?;
        let fen = line.trim();
        if fen.is_empty() {
            summary.skipped_blank_lines += 1;
            continue;
        }

        let normalized = match normalize_fen(fen) {
            Ok(normalized) => normalized,
            Err(_) => {
                summary.skipped_invalid_positions += 1;
                continue;
            }
        };
        let mut position = Position::from_fen(&normalized).map_err(|error| {
            format!(
                "failed to parse normalized self-play FEN on line {} '{}': {error}",
                line_number + 1,
                normalized
            )
        })?;
        if position.status() != PositionStatus::Ongoing {
            summary.skipped_terminal_positions += 1;
            continue;
        }

        service.clear_hash();
        if seen.insert(normalized.clone()) {
            output_positions.push(normalized);
        }

        let mut plies_played = 0usize;
        while position.status() == PositionStatus::Ongoing
            && plies_played < config.max_plies
            && output_positions.len() < config.max_positions
        {
            let mut search_position = position.clone();
            let result = service.search(
                &mut search_position,
                build_match_request(MatchConfig {
                    hash_mb: config.hash_mb,
                    max_plies: config.max_plies,
                    mode: config.mode,
                }),
            );
            let best_move = result.best_move.ok_or_else(|| {
                format!(
                    "self-play search produced no best move for ongoing position '{}'",
                    position.to_fen()
                )
            })?;
            position
                .make_move(best_move)
                .map_err(|error| format!("self-play move '{best_move}' was not legal: {error}"))?;
            plies_played += 1;

            let child = position.to_fen();
            if seen.insert(child.clone()) {
                output_positions.push(child);
            }
        }

        summary.completed_games += 1;
    }

    let mut output = BufWriter::new(File::create(output_path).map_err(|error| {
        format!(
            "failed to create self-play corpus '{}': {error}",
            output_path.display()
        )
    })?);
    for fen in &output_positions {
        writeln!(output, "{fen}").map_err(|error| {
            format!(
                "failed to write self-play corpus '{}': {error}",
                output_path.display()
            )
        })?;
    }
    output.flush().map_err(|error| {
        format!(
            "failed to flush self-play corpus '{}': {error}",
            output_path.display()
        )
    })?;

    summary.emitted_positions = output_positions.len();
    Ok(summary)
}

pub fn compare_candidate_vs_fallback(
    openings_path: &Path,
    candidate_evalfile: &Path,
    config: MatchConfig,
) -> Result<MatchSummary, String> {
    let candidate_evalfile = candidate_evalfile.to_str().ok_or_else(|| {
        format!(
            "candidate eval file '{}' must be valid UTF-8",
            candidate_evalfile.display()
        )
    })?;
    let input = File::open(openings_path).map_err(|error| {
        format!(
            "failed to open openings corpus '{}': {error}",
            openings_path.display()
        )
    })?;

    let mut openings = Vec::new();
    for (line_number, line) in BufReader::new(input).lines().enumerate() {
        let line = line.map_err(|error| {
            format!("failed to read openings line {}: {error}", line_number + 1)
        })?;
        let fen = line.trim();
        if fen.is_empty() {
            continue;
        }
        openings.push(normalize_fen(fen).map_err(|error| {
            format!(
                "failed to normalize opening FEN on line {}: {error}",
                line_number + 1
            )
        })?);
    }

    let mut summary = MatchSummary {
        openings: openings.len(),
        ..MatchSummary::default()
    };
    for opening_fen in openings {
        for candidate_color in [Color::White, Color::Black] {
            let game = play_match_game(&opening_fen, candidate_evalfile, candidate_color, config)?;
            summary.games += 1;
            match game.outcome {
                MatchOutcome::CandidateWin => summary.candidate_wins += 1,
                MatchOutcome::FallbackWin => summary.fallback_wins += 1,
                MatchOutcome::Draw => summary.draws += 1,
            }
            summary.game_summaries.push(game);
        }
    }
    Ok(summary)
}

pub fn read_examples(
    path: &Path,
) -> Result<(ExampleExportManifest, Vec<ExampleRecord>, ExportSummary), String> {
    let file = File::open(path)
        .map_err(|error| format!("failed to open examples file '{}': {error}", path.display()))?;
    let mut lines = BufReader::new(file).lines();

    let magic_line = next_line(&mut lines, "examples magic header")?;
    let (magic, version) = parse_magic_header(&magic_line)?;
    if magic != EXAMPLES_MAGIC || version != EXAMPLES_VERSION {
        return Err(format!(
            "unsupported examples header '{magic}' version {version}"
        ));
    }

    let manifest_line = next_line(&mut lines, "examples manifest")?;
    let manifest_line = manifest_line
        .strip_prefix("# ")
        .ok_or_else(|| "examples manifest line must start with '# '".to_owned())?;
    let manifest: ExampleExportManifest = serde_json::from_str(manifest_line)
        .map_err(|error| format!("failed to parse examples manifest: {error}"))?;

    let columns = next_line(&mut lines, "examples column header")?;
    if columns != EXAMPLES_COLUMNS {
        return Err("examples column header did not match retained format".to_owned());
    }

    let mut records = Vec::new();
    let mut summary = ExportSummary::default();
    for (index, line) in lines.enumerate() {
        summary.input_lines += 1;
        let line =
            line.map_err(|error| format!("failed to read examples row {}: {error}", index + 1))?;
        if line.trim().is_empty() {
            summary.skipped_blank_lines += 1;
            continue;
        }
        records.push(
            parse_example_record(&line)
                .map_err(|error| format!("failed to parse examples row {}: {error}", index + 1))?,
        );
        summary.emitted_examples += 1;
    }

    Ok((manifest, records, summary))
}

pub fn read_checkpoint_manifest(checkpoint_dir: &Path) -> Result<CheckpointManifest, String> {
    let path = checkpoint_dir.join(CHECKPOINT_MANIFEST_FILE);
    let text = fs::read_to_string(&path).map_err(|error| {
        format!(
            "failed to read checkpoint manifest '{}': {error}",
            path.display()
        )
    })?;
    let manifest: CheckpointManifest = serde_json::from_str(&text).map_err(|error| {
        format!(
            "failed to parse checkpoint manifest '{}': {error}",
            path.display()
        )
    })?;
    validate_checkpoint_manifest(&manifest)?;
    Ok(manifest)
}

pub fn pack_checkpoint_to_volknnue(
    checkpoint_dir: &Path,
    output_path: &Path,
) -> Result<PackSummary, String> {
    let manifest = read_checkpoint_manifest(checkpoint_dir)?;
    let topology = supported_halfkp_topology(
        manifest.topology_id,
        manifest.hidden_size,
        manifest.output_inputs,
    )
    .map_err(|error| format!("checkpoint topology was not packable: {error}"))?;
    let hidden_biases = read_f32_slice(
        &checkpoint_dir.join(&manifest.hidden_biases_file),
        topology.hidden_size,
        "hidden biases",
    )?;
    let input_weights = read_f32_slice(
        &checkpoint_dir.join(&manifest.input_weights_file),
        topology.input_weight_count(),
        "input weights",
    )?;
    let output_weights = read_f32_slice(
        &checkpoint_dir.join(&manifest.output_weights_file),
        topology.output_inputs(),
        "output weights",
    )?;
    let output_bias = read_f32_slice(
        &checkpoint_dir.join(&manifest.output_bias_file),
        1,
        "output bias",
    )?;

    let hidden_biases = quantize_i16_vec(&hidden_biases, "hidden biases")?;
    let output_weights = quantize_i16_vec(&output_weights, "output weights")?;
    let input_weights = quantize_i16_vec(&input_weights, "input weights")?;
    let output_bias = quantize_i32_scalar(output_bias[0], "output bias")?;
    let bytes = encode_volknnue(
        topology,
        &hidden_biases,
        &input_weights,
        output_bias,
        &output_weights,
        manifest.output_scale,
    )?;

    let temp_output = temporary_output_path(output_path);
    fs::write(&temp_output, bytes).map_err(|error| {
        format!(
            "failed to write packed VOLKNNUE '{}': {error}",
            temp_output.display()
        )
    })?;
    fs::rename(&temp_output, output_path).map_err(|error| {
        format!(
            "failed to finalize packed VOLKNNUE '{}': {error}",
            output_path.display()
        )
    })?;

    let output_path_utf8 = output_path.to_str().ok_or_else(|| {
        format!(
            "packed VOLKNNUE path '{}' must be UTF-8",
            output_path.display()
        )
    })?;
    crate::search::nnue::NnueService::open_eval_file(output_path_utf8)?;

    let packed_manifest = PackedNetManifest {
        magic: PACK_MANIFEST_MAGIC.to_owned(),
        version: PACK_MANIFEST_VERSION,
        packer_version: PACKER_VERSION.to_owned(),
        source_engine_commit: SOURCE_COMMIT.to_owned(),
        output_net: output_path.display().to_string(),
        topology_name: topology.name.to_owned(),
        topology_id: topology.id,
        feature_count: NNUE_FEATURE_COUNT,
        hidden_size: topology.hidden_size,
        output_inputs: topology.output_inputs(),
        output_scale: manifest.output_scale,
        checkpoint_manifest: manifest,
    };
    let manifest_path = packed_manifest_path(output_path);
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&packed_manifest)
            .map_err(|error| format!("failed to encode pack manifest JSON: {error}"))?,
    )
    .map_err(|error| {
        format!(
            "failed to write pack manifest '{}': {error}",
            manifest_path.display()
        )
    })?;

    Ok(PackSummary {
        output_path: output_path.to_path_buf(),
        manifest_path,
    })
}

pub fn validate_volknnue(evalfile: &Path) -> Result<ValidationSummary, String> {
    let evalfile_utf8 = evalfile
        .to_str()
        .ok_or_else(|| format!("VOLKNNUE path '{}' must be UTF-8", evalfile.display()))?;
    crate::search::nnue::NnueService::open_eval_file(evalfile_utf8)?;
    let manifest_path = packed_manifest_path(evalfile);
    Ok(ValidationSummary {
        net_path: evalfile.to_path_buf(),
        manifest_path: manifest_path.exists().then_some(manifest_path),
    })
}

pub fn write_checkpoint_artifacts(
    checkpoint_dir: &Path,
    manifest: &CheckpointManifest,
    input_weights: &[f32],
    hidden_biases: &[f32],
    output_weights: &[f32],
    output_bias: &[f32],
) -> Result<(), String> {
    validate_checkpoint_manifest(manifest)?;
    if output_bias.len() != 1 {
        return Err(format!(
            "output bias length {} did not match retained length 1",
            output_bias.len()
        ));
    }

    fs::create_dir_all(checkpoint_dir).map_err(|error| {
        format!(
            "failed to create checkpoint dir '{}': {error}",
            checkpoint_dir.display()
        )
    })?;

    fs::write(
        checkpoint_dir.join(CHECKPOINT_MANIFEST_FILE),
        serde_json::to_string_pretty(manifest)
            .map_err(|error| format!("failed to encode checkpoint manifest JSON: {error}"))?,
    )
    .map_err(|error| {
        format!(
            "failed to write checkpoint manifest '{}': {error}",
            checkpoint_dir.join(CHECKPOINT_MANIFEST_FILE).display()
        )
    })?;

    write_f32_slice(
        &checkpoint_dir.join(CHECKPOINT_INPUT_WEIGHTS_FILE),
        input_weights,
        "input weights",
    )?;
    write_f32_slice(
        &checkpoint_dir.join(CHECKPOINT_HIDDEN_BIASES_FILE),
        hidden_biases,
        "hidden biases",
    )?;
    write_f32_slice(
        &checkpoint_dir.join(CHECKPOINT_OUTPUT_WEIGHTS_FILE),
        output_weights,
        "output weights",
    )?;
    write_f32_slice(
        &checkpoint_dir.join(CHECKPOINT_OUTPUT_BIAS_FILE),
        output_bias,
        "output bias",
    )?;

    Ok(())
}

fn export_example_from_fen(
    fen: &str,
    config: LabelGenerationConfig,
    label_service: &mut UciSearchService,
) -> Result<ExportDecision, String> {
    let position =
        Position::from_fen(fen).map_err(|error| format!("failed to parse FEN '{fen}': {error}"))?;
    if position_is_within_retained_scope(&position) {
        return Ok(ExportDecision::SkipTablebaseScope);
    }
    if !position_matches_filter(&position, config.position_filter) {
        return Ok(ExportDecision::SkipPositionFilter);
    }

    let normalized_fen = position.to_fen();
    let SparseFeaturePair { active, passive } = sparse_features_for_side_to_move(&position);
    let Some(raw_score_cp) = generate_label(&position, config, label_service)? else {
        return Ok(ExportDecision::SkipIncompleteSearch);
    };
    let target_cp = clip_target(raw_score_cp);

    Ok(ExportDecision::Example(ExampleRecord {
        fen: fen.to_owned(),
        normalized_fen,
        side_to_move: position.side_to_move(),
        raw_score_cp,
        target_cp,
        active_features: active,
        passive_features: passive,
    }))
}

fn generate_label(
    position: &Position,
    config: LabelGenerationConfig,
    service: &mut UciSearchService,
) -> Result<Option<i32>, String> {
    match config.mode {
        LabelMode::Search => generate_search_label(position, config, service),
        LabelMode::StaticEval => Ok(Some(generate_static_eval_label(position))),
    }
}

fn generate_search_label(
    position: &Position,
    config: LabelGenerationConfig,
    service: &mut UciSearchService,
) -> Result<Option<i32>, String> {
    service.clear_hash();
    let before = position.to_fen();
    let mut label_position = position.clone();
    let deadline = config
        .label_timeout_ms
        .map(|timeout_ms| Instant::now() + Duration::from_millis(timeout_ms));
    let result = service.search(
        &mut label_position,
        SearchRequest {
            limits: SearchLimits::new(config.depth)
                .with_hash_mb(config.hash_mb)
                .with_tt(config.tt_enabled),
            soft_deadline: deadline,
            hard_deadline: deadline,
            stop_flag: None,
        },
    );
    debug_assert_eq!(label_position.to_fen(), before);
    if result.depth < config.depth {
        return Ok(None);
    }
    Ok(Some(result.score.0))
}

fn generate_static_eval_label(position: &Position) -> i32 {
    let mut probe = position.clone();
    match probe.status() {
        PositionStatus::Checkmate => -TARGET_CLIP_CP,
        PositionStatus::Stalemate
        | PositionStatus::DrawByRepetition
        | PositionStatus::DrawByFiftyMove
        | PositionStatus::DrawByInsufficientMaterial => 0,
        PositionStatus::Ongoing => eval::evaluate(position).0,
    }
}

fn build_label_service(config: LabelGenerationConfig) -> Result<UciSearchService, String> {
    let mut service = UciSearchService::new();
    service.set_threads(1);
    service.set_syzygy_path("")?;
    service.set_eval_file("")?;
    service.resize_hash(config.hash_mb);
    Ok(service)
}

fn clip_target(raw_score_cp: i32) -> i32 {
    raw_score_cp.clamp(-TARGET_CLIP_CP, TARGET_CLIP_CP)
}

pub fn position_matches_filter(position: &Position, filter: PositionFilter) -> bool {
    match filter {
        PositionFilter::Any => true,
        PositionFilter::Quiet => position_is_quiet_for_supervision(position),
    }
}

pub fn position_is_quiet_for_supervision(position: &Position) -> bool {
    let mut probe = position.clone();
    if probe.status() != PositionStatus::Ongoing {
        return false;
    }
    if probe.is_in_check(probe.side_to_move()) {
        return false;
    }

    let mut tactical_moves = MoveList::new();
    probe.generate_pseudo_legal(MoveGenStage::Captures, &mut tactical_moves);
    let info = probe.check_info();
    if tactical_moves
        .as_slice()
        .iter()
        .copied()
        .any(|mv| probe.is_legal_fast(mv, &info))
    {
        return false;
    }

    let mut legal_moves = MoveList::new();
    probe.generate_legal_moves(&mut legal_moves);
    !legal_moves.as_slice().iter().any(|mv| mv.is_promotion())
}

fn write_examples_prologue(
    output: &mut dyn Write,
    manifest: &ExampleExportManifest,
) -> Result<(), String> {
    writeln!(output, "{EXAMPLES_MAGIC}\t{EXAMPLES_VERSION}")
        .map_err(|error| format!("failed to write examples header: {error}"))?;
    writeln!(
        output,
        "# {}",
        serde_json::to_string(manifest)
            .map_err(|error| format!("failed to encode examples manifest: {error}"))?
    )
    .map_err(|error| format!("failed to write examples manifest: {error}"))?;
    writeln!(output, "{EXAMPLES_COLUMNS}")
        .map_err(|error| format!("failed to write examples columns: {error}"))?;
    Ok(())
}

fn write_example_record(output: &mut dyn Write, record: &ExampleRecord) -> Result<(), String> {
    writeln!(
        output,
        "{}\t{}\t{}\t{}\t{}\t{}\t{}",
        record.fen,
        record.normalized_fen,
        color_code(record.side_to_move),
        record.raw_score_cp,
        record.target_cp,
        join_features(&record.active_features),
        join_features(&record.passive_features)
    )
    .map_err(|error| format!("failed to write example record: {error}"))
}

fn parse_example_record(line: &str) -> Result<ExampleRecord, String> {
    let columns: Vec<&str> = line.split('\t').collect();
    if columns.len() != 7 {
        return Err(format!(
            "expected 7 tab-separated columns, found {}",
            columns.len()
        ));
    }

    let side_to_move = match columns[2] {
        "w" => Color::White,
        "b" => Color::Black,
        other => return Err(format!("invalid side-to-move column '{other}'")),
    };
    let raw_score_cp = columns[3]
        .parse::<i32>()
        .map_err(|error| format!("invalid raw score '{}': {error}", columns[3]))?;
    let target_cp = columns[4]
        .parse::<i32>()
        .map_err(|error| format!("invalid target score '{}': {error}", columns[4]))?;
    let active_features = parse_features(columns[5])?;
    let passive_features = parse_features(columns[6])?;

    Ok(ExampleRecord {
        fen: columns[0].to_owned(),
        normalized_fen: columns[1].to_owned(),
        side_to_move,
        raw_score_cp,
        target_cp,
        active_features,
        passive_features,
    })
}

fn validate_checkpoint_manifest(manifest: &CheckpointManifest) -> Result<(), String> {
    if manifest.magic != CHECKPOINT_MAGIC && manifest.magic != LEGACY_CHECKPOINT_MAGIC {
        return Err(format!(
            "checkpoint magic '{}' did not match supported magics '{}' or '{}'",
            manifest.magic, CHECKPOINT_MAGIC, LEGACY_CHECKPOINT_MAGIC
        ));
    }
    if manifest.version != CHECKPOINT_VERSION {
        return Err(format!(
            "checkpoint version {} did not match retained version {}",
            manifest.version, CHECKPOINT_VERSION
        ));
    }
    let topology = supported_halfkp_topology(
        manifest.topology_id,
        manifest.hidden_size,
        manifest.output_inputs,
    )
    .map_err(|error| format!("checkpoint topology was not supported: {error}"))?;
    if manifest.topology_name != topology.name {
        return Err(format!(
            "checkpoint topology name '{}' did not match topology id {} / hidden size {} name '{}'",
            manifest.topology_name, manifest.topology_id, manifest.hidden_size, topology.name
        ));
    }
    if manifest.feature_count != NNUE_FEATURE_COUNT {
        return Err(format!(
            "checkpoint feature count {} did not match retained feature count {}",
            manifest.feature_count, NNUE_FEATURE_COUNT
        ));
    }
    if manifest.output_scale != OUTPUT_SCALE {
        return Err(format!(
            "checkpoint output scale {} did not match retained output scale {}",
            manifest.output_scale, OUTPUT_SCALE
        ));
    }
    Ok(())
}

fn read_f32_slice(path: &Path, expected_len: usize, label: &str) -> Result<Vec<f32>, String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("failed to read {label} '{}': {error}", path.display()))?;
    let expected_bytes = expected_len * std::mem::size_of::<f32>();
    if bytes.len() != expected_bytes {
        return Err(format!(
            "{label} '{}' contained {} bytes, expected {}",
            path.display(),
            bytes.len(),
            expected_bytes
        ));
    }

    let mut values = Vec::with_capacity(expected_len);
    for chunk in bytes.chunks_exact(4) {
        values.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(values)
}

fn write_f32_slice(path: &Path, values: &[f32], label: &str) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        if !value.is_finite() {
            return Err(format!("{label} contained a non-finite value"));
        }
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes)
        .map_err(|error| format!("failed to write {label} '{}': {error}", path.display()))
}

fn quantize_i16_vec(values: &[f32], label: &str) -> Result<Vec<i16>, String> {
    values
        .iter()
        .copied()
        .map(|value| quantize_i16_scalar(value, label))
        .collect()
}

fn quantize_i16_scalar(value: f32, label: &str) -> Result<i16, String> {
    if !value.is_finite() {
        return Err(format!("{label} contained a non-finite value"));
    }
    let rounded = value.round();
    if rounded < i16::MIN as f32 || rounded > i16::MAX as f32 {
        return Err(format!("{label} value {rounded} exceeded i16 range"));
    }
    Ok(rounded as i16)
}

fn quantize_i32_scalar(value: f32, label: &str) -> Result<i32, String> {
    if !value.is_finite() {
        return Err(format!("{label} contained a non-finite value"));
    }
    let rounded = value.round();
    if rounded < i32::MIN as f32 || rounded > i32::MAX as f32 {
        return Err(format!("{label} value {rounded} exceeded i32 range"));
    }
    Ok(rounded as i32)
}

fn next_line<I>(lines: &mut I, label: &str) -> Result<String, String>
where
    I: Iterator<Item = Result<String, std::io::Error>>,
{
    lines
        .next()
        .ok_or_else(|| format!("missing {label}"))?
        .map_err(|error| format!("failed to read {label}: {error}"))
}

fn parse_magic_header(line: &str) -> Result<(&str, u32), String> {
    let mut parts = line.split('\t');
    let magic = parts
        .next()
        .ok_or_else(|| "examples magic header was empty".to_owned())?;
    let version = parts
        .next()
        .ok_or_else(|| "examples magic header was missing version".to_owned())?
        .parse::<u32>()
        .map_err(|error| format!("invalid examples version: {error}"))?;
    Ok((magic, version))
}

fn join_features(features: &[u16]) -> String {
    features
        .iter()
        .map(|feature| feature.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn parse_features(text: &str) -> Result<Vec<u16>, String> {
    if text.is_empty() {
        return Ok(Vec::new());
    }

    text.split(',')
        .map(|part| {
            part.parse::<u16>()
                .map_err(|error| format!("invalid feature index '{part}': {error}"))
        })
        .collect()
}

fn color_code(color: Color) -> &'static str {
    match color {
        Color::White => "w",
        Color::Black => "b",
    }
}

fn temporary_output_path(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}{}", path.display(), EXAMPLES_TEMP_SUFFIX))
}

fn packed_manifest_path(output_path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.manifest.json", output_path.display()))
}

fn evenly_spaced_children(
    position: &mut Position,
    moves: &MoveList,
    limit: usize,
) -> Result<Vec<String>, String> {
    let mut candidates = Vec::with_capacity(moves.len());
    for mv in moves.as_slice().iter().copied() {
        let undo = position
            .make_move(mv)
            .map_err(|error| format!("failed to play expansion move '{mv}': {error}"))?;
        let child_fen = position.to_fen();
        let child_status = position.status();
        position.unmake_move(mv, undo);
        let Ok(normalized_child_fen) = normalize_fen(&child_fen) else {
            continue;
        };
        if child_status == PositionStatus::Ongoing {
            candidates.push((mv.to_string(), normalized_child_fen));
        }
    }
    candidates.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(select_evenly_spaced(&candidates, limit)
        .into_iter()
        .map(|(_, fen)| fen)
        .collect::<Vec<_>>())
}

fn select_evenly_spaced<T: Clone>(items: &[T], limit: usize) -> Vec<T> {
    if items.len() <= limit {
        return items.to_vec();
    }
    if limit == 1 {
        return vec![items[items.len() / 2].clone()];
    }

    let mut selected = Vec::with_capacity(limit);
    let max_index = items.len() - 1;
    for slot in 0..limit {
        let index = slot * max_index / (limit - 1);
        selected.push(items[index].clone());
    }
    selected
}

fn play_match_game(
    opening_fen: &str,
    candidate_evalfile: &str,
    candidate_color: Color,
    config: MatchConfig,
) -> Result<MatchGameSummary, String> {
    let mut position = Position::from_fen(opening_fen)
        .map_err(|error| format!("failed to parse opening FEN '{opening_fen}': {error}"))?;
    let mut fallback = build_match_service("", config.hash_mb)?;
    let mut candidate = build_match_service(candidate_evalfile, config.hash_mb)?;

    let mut plies_played = 0usize;
    let mut first_candidate_score_cp = None;
    let mut first_candidate_info_line = None;
    let mut first_fallback_score_cp = None;
    let mut first_fallback_info_line = None;

    loop {
        let status = position.status();
        if status != PositionStatus::Ongoing {
            return Ok(MatchGameSummary {
                opening_fen: opening_fen.to_owned(),
                candidate_color,
                outcome: match_outcome_from_status(
                    status,
                    position.side_to_move(),
                    candidate_color,
                ),
                terminal_status: status,
                plies_played,
                first_candidate_score_cp,
                first_candidate_info_line,
                first_fallback_score_cp,
                first_fallback_info_line,
            });
        }
        if plies_played >= config.max_plies {
            return Ok(MatchGameSummary {
                opening_fen: opening_fen.to_owned(),
                candidate_color,
                outcome: MatchOutcome::Draw,
                terminal_status: PositionStatus::Ongoing,
                plies_played,
                first_candidate_score_cp,
                first_candidate_info_line,
                first_fallback_score_cp,
                first_fallback_info_line,
            });
        }

        let side_to_move = position.side_to_move();
        let mut search_position = position.clone();
        let result = if side_to_move == candidate_color {
            candidate.search(&mut search_position, build_match_request(config))
        } else {
            fallback.search(&mut search_position, build_match_request(config))
        };

        if side_to_move == candidate_color && first_candidate_score_cp.is_none() {
            first_candidate_score_cp = Some(result.score.0);
            first_candidate_info_line = result.info_lines.last().cloned();
        } else if side_to_move != candidate_color && first_fallback_score_cp.is_none() {
            first_fallback_score_cp = Some(result.score.0);
            first_fallback_info_line = result.info_lines.last().cloned();
        }

        let best_move = result.best_move.ok_or_else(|| {
            format!(
                "match search produced no best move for ongoing position '{}'",
                position.to_fen()
            )
        })?;
        position
            .make_move(best_move)
            .map_err(|error| format!("match move '{best_move}' was not legal: {error}"))?;
        plies_played += 1;
    }
}

fn build_match_service(eval_file: &str, hash_mb: usize) -> Result<UciSearchService, String> {
    let mut service = UciSearchService::new();
    service.set_threads(1);
    service.set_syzygy_path("")?;
    service.set_eval_file(eval_file)?;
    service.resize_hash(hash_mb);
    Ok(service)
}

fn build_match_request(config: MatchConfig) -> SearchRequest {
    match config.mode {
        MatchMode::FixedDepth(depth) => SearchRequest {
            limits: SearchLimits::new(depth).with_hash_mb(config.hash_mb),
            soft_deadline: None,
            hard_deadline: None,
            stop_flag: None,
        },
        MatchMode::MoveTimeMs(movetime_ms) => {
            let deadline = Instant::now() + Duration::from_millis(movetime_ms);
            SearchRequest {
                limits: SearchLimits::new(127).with_hash_mb(config.hash_mb),
                soft_deadline: Some(deadline),
                hard_deadline: Some(deadline),
                stop_flag: None,
            }
        }
    }
}

fn match_outcome_from_status(
    status: PositionStatus,
    side_to_move: Color,
    candidate_color: Color,
) -> MatchOutcome {
    match status {
        PositionStatus::Checkmate => {
            if side_to_move.opposite() == candidate_color {
                MatchOutcome::CandidateWin
            } else {
                MatchOutcome::FallbackWin
            }
        }
        PositionStatus::Stalemate
        | PositionStatus::DrawByRepetition
        | PositionStatus::DrawByFiftyMove
        | PositionStatus::DrawByInsufficientMaterial
        | PositionStatus::Ongoing => MatchOutcome::Draw,
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}
