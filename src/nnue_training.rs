use std::{
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::{
    SOURCE_COMMIT,
    core::{Color, Position},
    search::{
        SearchLimits,
        nnue::{
            NNUE_FEATURE_COUNT, NNUE_HIDDEN_SIZE, NNUE_OUTPUT_INPUTS, NNUE_TOPOLOGY_HALFKP_128X2,
            SparseFeaturePair, encode_volknnue, sparse_features_for_side_to_move,
        },
        service::{SearchRequest, UciSearchService},
        tablebase::position_is_within_retained_scope,
    },
};

pub const EXAMPLES_MAGIC: &str = "VOLKRIX_EXAMPLES";
pub const EXAMPLES_VERSION: u32 = 1;
pub const CHECKPOINT_MAGIC: &str = "VOLKRIX_HALFKP128X2_CHECKPOINT";
pub const CHECKPOINT_VERSION: u32 = 1;
pub const PACK_MANIFEST_MAGIC: &str = "VOLKRIX_VOLKNNUE_PACK";
pub const PACK_MANIFEST_VERSION: u32 = 1;
pub const EXPORTER_VERSION: &str = "phase13-v1";
pub const TRAINER_VERSION: &str = "phase13-v1";
pub const PACKER_VERSION: &str = "phase13-v1";
pub const TOPOLOGY_NAME: &str = "HalfKP128x2";
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DatasetSplit {
    Train,
    Validation,
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
    pub normalized_fen_rule: String,
}

impl ExampleExportManifest {
    fn current() -> Self {
        Self {
            magic: EXAMPLES_MAGIC.to_owned(),
            version: EXAMPLES_VERSION,
            exporter_version: EXPORTER_VERSION.to_owned(),
            source_engine_commit: SOURCE_COMMIT.to_owned(),
            topology_name: TOPOLOGY_NAME.to_owned(),
            topology_id: NNUE_TOPOLOGY_HALFKP_128X2,
            feature_count: NNUE_FEATURE_COUNT,
            hidden_size: NNUE_HIDDEN_SIZE,
            output_inputs: NNUE_OUTPUT_INPUTS,
            label_depth: LABEL_DEPTH,
            target_clip_cp: TARGET_CLIP_CP,
            threads: 1,
            syzygy_path: String::new(),
            eval_file: String::new(),
            tt_enabled: false,
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
    let input = File::open(input_path).map_err(|error| {
        format!(
            "failed to open FEN corpus '{}': {error}",
            input_path.display()
        )
    })?;
    let mut output = BufWriter::new(File::create(temporary_output_path(output_path)).map_err(
        |error| {
            format!(
                "failed to create examples output '{}': {error}",
                output_path.display()
            )
        },
    )?);
    let manifest = ExampleExportManifest::current();
    write_examples_prologue(&mut output, &manifest)?;

    let mut summary = ExportSummary::default();
    for (line_number, line) in BufReader::new(input).lines().enumerate() {
        summary.input_lines += 1;
        let line = line
            .map_err(|error| format!("failed to read input line {}: {error}", line_number + 1))?;
        let fen = line.trim();
        if fen.is_empty() {
            summary.skipped_blank_lines += 1;
            continue;
        }

        let example = export_example_from_fen(fen)
            .map_err(|error| format!("input line {}: {error}", line_number + 1))?;
        let Some(example) = example else {
            summary.skipped_tablebase_scope_positions += 1;
            continue;
        };

        write_example_record(&mut output, &example)?;
        summary.emitted_examples += 1;
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
    let hidden_biases = read_f32_slice(
        &checkpoint_dir.join(&manifest.hidden_biases_file),
        NNUE_HIDDEN_SIZE,
        "hidden biases",
    )?;
    let input_weights = read_f32_slice(
        &checkpoint_dir.join(&manifest.input_weights_file),
        NNUE_FEATURE_COUNT * NNUE_HIDDEN_SIZE,
        "input weights",
    )?;
    let output_weights = read_f32_slice(
        &checkpoint_dir.join(&manifest.output_weights_file),
        NNUE_OUTPUT_INPUTS,
        "output weights",
    )?;
    let output_bias = read_f32_slice(
        &checkpoint_dir.join(&manifest.output_bias_file),
        1,
        "output bias",
    )?;

    let hidden_biases = quantize_i16_array::<NNUE_HIDDEN_SIZE>(&hidden_biases, "hidden biases")?;
    let output_weights =
        quantize_i16_array::<NNUE_OUTPUT_INPUTS>(&output_weights, "output weights")?;
    let input_weights = quantize_i16_vec(&input_weights, "input weights")?;
    let output_bias = quantize_i32_scalar(output_bias[0], "output bias")?;
    let bytes = encode_volknnue(
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
        topology_name: TOPOLOGY_NAME.to_owned(),
        topology_id: NNUE_TOPOLOGY_HALFKP_128X2,
        feature_count: NNUE_FEATURE_COUNT,
        hidden_size: NNUE_HIDDEN_SIZE,
        output_inputs: NNUE_OUTPUT_INPUTS,
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

fn export_example_from_fen(fen: &str) -> Result<Option<ExampleRecord>, String> {
    let position =
        Position::from_fen(fen).map_err(|error| format!("failed to parse FEN '{fen}': {error}"))?;
    if position_is_within_retained_scope(&position) {
        return Ok(None);
    }

    let normalized_fen = position.to_fen();
    let SparseFeaturePair { active, passive } = sparse_features_for_side_to_move(&position);
    let raw_score_cp = generate_search_label(&position)?;
    let target_cp = clip_target(raw_score_cp);

    Ok(Some(ExampleRecord {
        fen: fen.to_owned(),
        normalized_fen,
        side_to_move: position.side_to_move(),
        raw_score_cp,
        target_cp,
        active_features: active,
        passive_features: passive,
    }))
}

fn generate_search_label(position: &Position) -> Result<i32, String> {
    let mut service = UciSearchService::new();
    service.set_threads(1);
    service.set_syzygy_path("")?;
    service.set_eval_file("")?;

    let before = position.to_fen();
    let mut label_position = position.clone();
    let result = service.search(
        &mut label_position,
        SearchRequest {
            limits: SearchLimits::new(LABEL_DEPTH).without_tt(),
            soft_deadline: None,
            hard_deadline: None,
            stop_flag: None,
        },
    );
    debug_assert_eq!(label_position.to_fen(), before);
    Ok(result.score.0)
}

fn clip_target(raw_score_cp: i32) -> i32 {
    raw_score_cp.clamp(-TARGET_CLIP_CP, TARGET_CLIP_CP)
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
    if manifest.magic != CHECKPOINT_MAGIC {
        return Err(format!(
            "checkpoint magic '{}' did not match retained magic '{}'",
            manifest.magic, CHECKPOINT_MAGIC
        ));
    }
    if manifest.version != CHECKPOINT_VERSION {
        return Err(format!(
            "checkpoint version {} did not match retained version {}",
            manifest.version, CHECKPOINT_VERSION
        ));
    }
    if manifest.topology_id != NNUE_TOPOLOGY_HALFKP_128X2 || manifest.topology_name != TOPOLOGY_NAME
    {
        return Err("checkpoint topology did not match retained HalfKP128x2".to_owned());
    }
    if manifest.feature_count != NNUE_FEATURE_COUNT {
        return Err(format!(
            "checkpoint feature count {} did not match retained feature count {}",
            manifest.feature_count, NNUE_FEATURE_COUNT
        ));
    }
    if manifest.hidden_size != NNUE_HIDDEN_SIZE {
        return Err(format!(
            "checkpoint hidden size {} did not match retained hidden size {}",
            manifest.hidden_size, NNUE_HIDDEN_SIZE
        ));
    }
    if manifest.output_inputs != NNUE_OUTPUT_INPUTS {
        return Err(format!(
            "checkpoint output input count {} did not match retained output input count {}",
            manifest.output_inputs, NNUE_OUTPUT_INPUTS
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

fn quantize_i16_vec(values: &[f32], label: &str) -> Result<Vec<i16>, String> {
    values
        .iter()
        .copied()
        .map(|value| quantize_i16_scalar(value, label))
        .collect()
}

fn quantize_i16_array<const N: usize>(values: &[f32], label: &str) -> Result<[i16; N], String> {
    if values.len() != N {
        return Err(format!(
            "{label} length {} did not match retained length {}",
            values.len(),
            N
        ));
    }

    let mut output = [0i16; N];
    for (index, value) in values.iter().copied().enumerate() {
        output[index] = quantize_i16_scalar(value, label)?;
    }
    Ok(output)
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

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}
