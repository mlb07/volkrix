use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::{Arc, Mutex, mpsc},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use volkrix::{
    core::{Color, Position, PositionStatus},
    nnue_training::{MatchGameSummary, MatchOutcome, MatchSummary, normalize_fen},
};

const MANIFEST_VERSION: u32 = 2;
const CLASSICAL_EVAL: &str = "classical";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EngineOptions {
    pub path: PathBuf,
    /// An absolute network path, or the literal `classical`.
    pub eval_file: String,
    /// Optional absolute small network path. It is inert while policy is `off`.
    pub small_eval_file: Option<String>,
    pub dual_eval_policy: String,
    pub dual_eval_threshold: i32,
    pub threads: usize,
    pub hash_mb: usize,
    pub move_overhead_ms: u64,
    /// An absolute tablebase path, or the literal `none`.
    pub syzygy_path: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExternalTimeControl {
    FixedDepth { depth: u8 },
    MoveTime { milliseconds: u64 },
    Clock { initial_ms: u64, increment_ms: u64 },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExternalMatchConfig {
    pub baseline: EngineOptions,
    pub candidate: EngineOptions,
    pub time_control: ExternalTimeControl,
    pub max_plies: usize,
    pub max_openings: Option<usize>,
    pub protocol_timeout_ms: u64,
    pub stop_grace_ms: u64,
    pub artifacts_dir: PathBuf,
    pub resume: bool,
}

#[derive(Clone, Debug)]
pub struct EngineMatchReport {
    pub summary: MatchSummary,
    pub manifest_path: PathBuf,
    pub games_path: PathBuf,
    pub pgn_path: PathBuf,
    pub protocol_log_path: PathBuf,
    pub checkpoint_path: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct EngineArtifact {
    path: String,
    binary_sha256: String,
    eval_file: String,
    eval_sha256: Option<String>,
    small_eval_file: Option<String>,
    small_eval_sha256: Option<String>,
    dual_eval_policy: String,
    dual_eval_threshold: i32,
    threads: usize,
    hash_mb: usize,
    move_overhead_ms: u64,
    syzygy_path: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct ExperimentManifest {
    manifest_version: u32,
    created_unix_ms: u128,
    source_commit: String,
    opening_path: String,
    opening_sha256: String,
    opening_count: usize,
    baseline: EngineArtifact,
    candidate: EngineArtifact,
    time_control: ExternalTimeControl,
    max_plies: usize,
    protocol_timeout_ms: u64,
    stop_grace_ms: u64,
    environment: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct GameArtifact {
    game_number: usize,
    opening_index: usize,
    opening_fen: String,
    candidate_color: String,
    result: String,
    termination: String,
    plies: usize,
    moves_uci: Vec<String>,
    candidate_first_score_cp: Option<i32>,
    baseline_first_score_cp: Option<i32>,
    candidate_first_info: Option<String>,
    baseline_first_info: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct MatchCheckpoint {
    manifest_sha256: String,
    completed_openings: usize,
    completed_games: usize,
    candidate_wins: usize,
    draws: usize,
    baseline_wins: usize,
}

pub fn compare_external_engines(
    openings_path: &Path,
    config: ExternalMatchConfig,
) -> Result<EngineMatchReport, String> {
    validate_config(&config)?;
    let openings = load_openings(openings_path, config.max_openings)?;
    let paths = ArtifactPaths::new(&config.artifacts_dir);
    fs::create_dir_all(&config.artifacts_dir).map_err(|error| {
        format!(
            "failed to create artifact directory '{}': {error}",
            config.artifacts_dir.display()
        )
    })?;

    let manifest = build_manifest(openings_path, openings.len(), &config)?;
    let manifest_json = serde_json::to_string_pretty(&manifest)
        .map_err(|error| format!("failed to encode experiment manifest: {error}"))?;
    let manifest_sha256 = sha256_hex(manifest_json.as_bytes());
    let (mut summary, completed_openings, effective_manifest_sha256) = prepare_artifacts(
        &paths,
        &manifest,
        &manifest_json,
        &manifest_sha256,
        config.resume,
    )?;

    let protocol_log = Arc::new(Mutex::new(BufWriter::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&paths.protocol_log)
            .map_err(|error| format!("failed to open protocol log: {error}"))?,
    )));
    let mut baseline = ExternalEngine::spawn(
        "baseline",
        config.baseline.clone(),
        config.protocol_timeout_ms,
        config.stop_grace_ms,
        Arc::clone(&protocol_log),
    )?;
    let mut candidate = ExternalEngine::spawn(
        "candidate",
        config.candidate.clone(),
        config.protocol_timeout_ms,
        config.stop_grace_ms,
        Arc::clone(&protocol_log),
    )?;

    for (opening_index, opening_fen) in openings.iter().enumerate().skip(completed_openings) {
        for candidate_color in [Color::White, Color::Black] {
            let game_number = summary.games + 1;
            let game = play_match_game(
                game_number,
                opening_index,
                opening_fen,
                candidate_color,
                &config,
                &mut baseline,
                &mut candidate,
            )?;
            append_game_artifacts(&paths, &game)?;
            add_game_to_summary(&mut summary, game.to_summary()?);
        }
        write_checkpoint(&paths.checkpoint, &effective_manifest_sha256, &summary)?;
    }
    protocol_log
        .lock()
        .map_err(|_| "protocol log lock was poisoned".to_owned())?
        .flush()
        .map_err(|error| format!("failed to flush protocol log: {error}"))?;

    Ok(EngineMatchReport {
        summary,
        manifest_path: paths.manifest,
        games_path: paths.games,
        pgn_path: paths.pgn,
        protocol_log_path: paths.protocol_log,
        checkpoint_path: paths.checkpoint,
    })
}

fn validate_config(config: &ExternalMatchConfig) -> Result<(), String> {
    for (label, engine) in [
        ("baseline", &config.baseline),
        ("candidate", &config.candidate),
    ] {
        if !engine.path.is_file() {
            return Err(format!(
                "{label} engine '{}' is not a file",
                engine.path.display()
            ));
        }
        if engine.threads == 0 || engine.hash_mb == 0 {
            return Err(format!("{label} Threads and Hash must be positive"));
        }
        validate_eval_choice(label, &engine.eval_file)?;
        if let Some(small) = engine.small_eval_file.as_deref() {
            validate_eval_choice(label, small)?;
            if small == CLASSICAL_EVAL {
                return Err(format!("{label} SmallEvalFile cannot be classical"));
            }
        }
        if !matches!(engine.dual_eval_policy.as_str(), "off" | "small-fallback") {
            return Err(format!(
                "{label} DualEvalPolicy must be 'off' or 'small-fallback'"
            ));
        }
        if engine.dual_eval_policy == "small-fallback"
            && (engine.eval_file == CLASSICAL_EVAL || engine.small_eval_file.is_none())
        {
            return Err(format!(
                "{label} small-fallback requires network EvalFile and SmallEvalFile"
            ));
        }
        if !(0..=2_000).contains(&engine.dual_eval_threshold) {
            return Err(format!(
                "{label} DualEvalThreshold must be between 0 and 2000"
            ));
        }
        if engine.syzygy_path != "none" && !Path::new(&engine.syzygy_path).is_dir() {
            return Err(format!(
                "{label} SyzygyPath '{}' is not a directory (use 'none' explicitly to disable it)",
                engine.syzygy_path
            ));
        }
    }
    if config.protocol_timeout_ms == 0 || config.stop_grace_ms == 0 {
        return Err("protocol timeout and stop grace must be positive".to_owned());
    }
    if config.max_plies == 0 {
        return Err("max plies must be positive".to_owned());
    }
    Ok(())
}

fn validate_eval_choice(label: &str, value: &str) -> Result<(), String> {
    if value == CLASSICAL_EVAL {
        return Ok(());
    }
    let path = Path::new(value);
    if !path.is_absolute() {
        return Err(format!(
            "{label} EvalFile must be an absolute path or the literal 'classical'"
        ));
    }
    if !path.is_file() {
        return Err(format!(
            "{label} EvalFile '{}' is not a file",
            path.display()
        ));
    }
    Ok(())
}

fn build_manifest(
    openings_path: &Path,
    opening_count: usize,
    config: &ExternalMatchConfig,
) -> Result<ExperimentManifest, String> {
    let opening_path = canonical_text(openings_path)?;
    let mut environment = BTreeMap::new();
    environment.insert("os".to_owned(), std::env::consts::OS.to_owned());
    environment.insert("arch".to_owned(), std::env::consts::ARCH.to_owned());
    environment.insert("family".to_owned(), std::env::consts::FAMILY.to_owned());
    environment.insert(
        "current_dir".to_owned(),
        std::env::current_dir()
            .map_err(|error| format!("failed to read current directory: {error}"))?
            .display()
            .to_string(),
    );
    for name in ["RUSTFLAGS", "VOLKRIX_EVAL_FILE"] {
        if let Ok(value) = std::env::var(name) {
            environment.insert(name.to_owned(), value);
        }
    }

    Ok(ExperimentManifest {
        manifest_version: MANIFEST_VERSION,
        created_unix_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| format!("system clock precedes Unix epoch: {error}"))?
            .as_millis(),
        source_commit: volkrix::SOURCE_COMMIT.to_owned(),
        opening_path,
        opening_sha256: sha256_file(openings_path)?,
        opening_count,
        baseline: build_engine_artifact(&config.baseline)?,
        candidate: build_engine_artifact(&config.candidate)?,
        time_control: config.time_control,
        max_plies: config.max_plies,
        protocol_timeout_ms: config.protocol_timeout_ms,
        stop_grace_ms: config.stop_grace_ms,
        environment,
    })
}

fn build_engine_artifact(options: &EngineOptions) -> Result<EngineArtifact, String> {
    Ok(EngineArtifact {
        path: canonical_text(&options.path)?,
        binary_sha256: sha256_file(&options.path)?,
        eval_file: options.eval_file.clone(),
        eval_sha256: (options.eval_file != CLASSICAL_EVAL)
            .then(|| sha256_file(Path::new(&options.eval_file)))
            .transpose()?,
        small_eval_file: options.small_eval_file.clone(),
        small_eval_sha256: options
            .small_eval_file
            .as_deref()
            .map(|path| sha256_file(Path::new(path)))
            .transpose()?,
        dual_eval_policy: options.dual_eval_policy.clone(),
        dual_eval_threshold: options.dual_eval_threshold,
        threads: options.threads,
        hash_mb: options.hash_mb,
        move_overhead_ms: options.move_overhead_ms,
        syzygy_path: options.syzygy_path.clone(),
    })
}

fn canonical_text(path: &Path) -> Result<String, String> {
    path.canonicalize()
        .map(|path| path.display().to_string())
        .map_err(|error| format!("failed to canonicalize '{}': {error}", path.display()))
}

fn load_openings(openings_path: &Path, max_openings: Option<usize>) -> Result<Vec<String>, String> {
    let input = File::open(openings_path).map_err(|error| {
        format!(
            "failed to open openings corpus '{}': {error}",
            openings_path.display()
        )
    })?;

    let mut openings = Vec::new();
    for (line_number, line) in BufReader::new(input).lines().enumerate() {
        if max_openings.is_some_and(|limit| openings.len() >= limit) {
            break;
        }
        let line = line.map_err(|error| {
            format!("failed to read openings line {}: {error}", line_number + 1)
        })?;
        let fen = line.trim();
        if !fen.is_empty() {
            openings.push(normalize_fen(fen).map_err(|error| {
                format!(
                    "failed to normalize opening FEN on line {}: {error}",
                    line_number + 1
                )
            })?);
        }
    }
    if openings.is_empty() {
        return Err(format!(
            "openings corpus '{}' did not contain any usable FENs",
            openings_path.display()
        ));
    }
    Ok(openings)
}

struct ArtifactPaths {
    manifest: PathBuf,
    games: PathBuf,
    pgn: PathBuf,
    protocol_log: PathBuf,
    checkpoint: PathBuf,
}

impl ArtifactPaths {
    fn new(root: &Path) -> Self {
        Self {
            manifest: root.join("manifest.json"),
            games: root.join("games.jsonl"),
            pgn: root.join("games.pgn"),
            protocol_log: root.join("protocol.log"),
            checkpoint: root.join("checkpoint.json"),
        }
    }
}

fn prepare_artifacts(
    paths: &ArtifactPaths,
    manifest: &ExperimentManifest,
    manifest_json: &str,
    manifest_sha256: &str,
    resume: bool,
) -> Result<(MatchSummary, usize, String), String> {
    if resume {
        let previous_text = fs::read_to_string(&paths.manifest)
            .map_err(|error| format!("resume requires an existing manifest: {error}"))?;
        let mut previous: ExperimentManifest = serde_json::from_str(&previous_text)
            .map_err(|error| format!("failed to parse existing manifest: {error}"))?;
        // Creation time is provenance, not experiment identity.
        previous.created_unix_ms = manifest.created_unix_ms;
        if &previous != manifest {
            return Err("resume manifest does not match the requested experiment".to_owned());
        }
        let checkpoint: MatchCheckpoint = serde_json::from_str(
            &fs::read_to_string(&paths.checkpoint)
                .map_err(|error| format!("resume requires an existing checkpoint: {error}"))?,
        )
        .map_err(|error| format!("failed to parse checkpoint: {error}"))?;
        let previous_manifest_hash = sha256_hex(previous_text.as_bytes());
        if checkpoint.manifest_sha256 != previous_manifest_hash {
            return Err("checkpoint manifest checksum mismatch".to_owned());
        }
        if !checkpoint.completed_games.is_multiple_of(2)
            || checkpoint.completed_openings.checked_mul(2) != Some(checkpoint.completed_games)
            || checkpoint.completed_openings > manifest.opening_count
        {
            return Err("checkpoint is not at a valid opening-pair boundary".to_owned());
        }
        let mut summary = recover_committed_artifacts(paths, &checkpoint)?;
        summary.openings = manifest.opening_count;
        return Ok((
            summary,
            checkpoint.completed_openings,
            previous_manifest_hash,
        ));
    }

    for path in [
        &paths.manifest,
        &paths.games,
        &paths.pgn,
        &paths.protocol_log,
        &paths.checkpoint,
    ] {
        if path.exists() {
            return Err(format!(
                "artifact '{}' already exists; choose a new directory or pass --resume on",
                path.display()
            ));
        }
    }
    fs::write(&paths.manifest, manifest_json)
        .map_err(|error| format!("failed to write experiment manifest: {error}"))?;
    fs::write(&paths.games, "")
        .map_err(|error| format!("failed to initialize game log: {error}"))?;
    fs::write(&paths.pgn, "").map_err(|error| format!("failed to initialize PGN log: {error}"))?;
    fs::write(&paths.protocol_log, "")
        .map_err(|error| format!("failed to initialize protocol log: {error}"))?;
    let summary = MatchSummary {
        openings: manifest.opening_count,
        ..MatchSummary::default()
    };
    write_checkpoint(&paths.checkpoint, manifest_sha256, &summary)?;
    Ok((summary, 0, manifest_sha256.to_owned()))
}

fn play_match_game(
    game_number: usize,
    opening_index: usize,
    opening_fen: &str,
    candidate_color: Color,
    config: &ExternalMatchConfig,
    baseline: &mut ExternalEngine,
    candidate: &mut ExternalEngine,
) -> Result<GameArtifact, String> {
    let mut position = Position::from_fen(opening_fen)
        .map_err(|error| format!("failed to parse opening FEN '{opening_fen}': {error}"))?;
    baseline.new_game()?;
    candidate.new_game()?;

    let mut moves = Vec::new();
    let mut clocks = match config.time_control {
        ExternalTimeControl::Clock { initial_ms, .. } => [initial_ms; 2],
        _ => [0; 2],
    };
    let mut first_candidate_score_cp = None;
    let mut first_candidate_info_line = None;
    let mut first_baseline_score_cp = None;
    let mut first_baseline_info_line = None;

    loop {
        let status = position.status();
        if status != PositionStatus::Ongoing {
            let outcome =
                match_outcome_from_status(status, position.side_to_move(), candidate_color);
            return Ok(game_artifact(
                game_number,
                opening_index,
                opening_fen,
                candidate_color,
                outcome,
                board_termination(status),
                moves,
                first_candidate_score_cp,
                first_candidate_info_line,
                first_baseline_score_cp,
                first_baseline_info_line,
            ));
        }
        if moves.len() >= config.max_plies {
            return Ok(game_artifact(
                game_number,
                opening_index,
                opening_fen,
                candidate_color,
                MatchOutcome::Draw,
                "max_plies".to_owned(),
                moves,
                first_candidate_score_cp,
                first_candidate_info_line,
                first_baseline_score_cp,
                first_baseline_info_line,
            ));
        }

        let side = position.side_to_move();
        let moving_candidate = side == candidate_color;
        let command = search_command(config.time_control, clocks);
        let result = if moving_candidate {
            candidate.search(&position, &command)
        } else {
            baseline.search(&position, &command)
        };
        let result = match result {
            Ok(result) => result,
            Err(failure) => {
                let outcome = if moving_candidate {
                    MatchOutcome::FallbackWin
                } else {
                    MatchOutcome::CandidateWin
                };
                return Ok(game_artifact(
                    game_number,
                    opening_index,
                    opening_fen,
                    candidate_color,
                    outcome,
                    format!(
                        "{}_{}",
                        if moving_candidate {
                            "candidate"
                        } else {
                            "baseline"
                        },
                        failure.label()
                    ),
                    moves,
                    first_candidate_score_cp,
                    first_candidate_info_line,
                    first_baseline_score_cp,
                    first_baseline_info_line,
                ));
            }
        };

        if let ExternalTimeControl::Clock { increment_ms, .. } = config.time_control {
            let clock = &mut clocks[side.index()];
            let elapsed_ms = result.elapsed.as_millis().min(u64::MAX as u128) as u64;
            if elapsed_ms > *clock {
                let outcome = if moving_candidate {
                    MatchOutcome::FallbackWin
                } else {
                    MatchOutcome::CandidateWin
                };
                return Ok(game_artifact(
                    game_number,
                    opening_index,
                    opening_fen,
                    candidate_color,
                    outcome,
                    format!(
                        "{}_time_forfeit",
                        if moving_candidate {
                            "candidate"
                        } else {
                            "baseline"
                        }
                    ),
                    moves,
                    first_candidate_score_cp,
                    first_candidate_info_line,
                    first_baseline_score_cp,
                    first_baseline_info_line,
                ));
            }
            *clock = clock
                .saturating_sub(elapsed_ms)
                .saturating_add(increment_ms);
        }

        if moving_candidate && first_candidate_score_cp.is_none() {
            first_candidate_score_cp = result.score_cp;
            first_candidate_info_line = result.last_info_line.clone();
        } else if !moving_candidate && first_baseline_score_cp.is_none() {
            first_baseline_score_cp = result.score_cp;
            first_baseline_info_line = result.last_info_line.clone();
        }

        if let Err(error) = position.apply_uci_move(&result.best_move) {
            let outcome = if moving_candidate {
                MatchOutcome::FallbackWin
            } else {
                MatchOutcome::CandidateWin
            };
            return Ok(game_artifact(
                game_number,
                opening_index,
                opening_fen,
                candidate_color,
                outcome,
                format!(
                    "{}_illegal_move_{}_{}",
                    if moving_candidate {
                        "candidate"
                    } else {
                        "baseline"
                    },
                    sanitize_token(&result.best_move),
                    sanitize_token(&error.to_string())
                ),
                moves,
                first_candidate_score_cp,
                first_candidate_info_line,
                first_baseline_score_cp,
                first_baseline_info_line,
            ));
        }
        moves.push(result.best_move);
    }
}

fn search_command(time_control: ExternalTimeControl, clocks: [u64; 2]) -> String {
    match time_control {
        ExternalTimeControl::FixedDepth { depth } => format!("go depth {depth}"),
        ExternalTimeControl::MoveTime { milliseconds } => {
            format!("go movetime {milliseconds}")
        }
        ExternalTimeControl::Clock { increment_ms, .. } => format!(
            "go wtime {} btime {} winc {increment_ms} binc {increment_ms}",
            clocks[Color::White.index()],
            clocks[Color::Black.index()]
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn game_artifact(
    game_number: usize,
    opening_index: usize,
    opening_fen: &str,
    candidate_color: Color,
    outcome: MatchOutcome,
    termination: String,
    moves: Vec<String>,
    candidate_first_score_cp: Option<i32>,
    candidate_first_info: Option<String>,
    baseline_first_score_cp: Option<i32>,
    baseline_first_info: Option<String>,
) -> GameArtifact {
    GameArtifact {
        game_number,
        opening_index,
        opening_fen: opening_fen.to_owned(),
        candidate_color: color_name(candidate_color).to_owned(),
        result: outcome_name(outcome).to_owned(),
        termination,
        plies: moves.len(),
        moves_uci: moves,
        candidate_first_score_cp,
        baseline_first_score_cp,
        candidate_first_info,
        baseline_first_info,
    }
}

impl GameArtifact {
    fn to_summary(&self) -> Result<MatchGameSummary, String> {
        Ok(MatchGameSummary {
            opening_fen: self.opening_fen.clone(),
            candidate_color: parse_color(&self.candidate_color)?,
            outcome: parse_outcome(&self.result)?,
            terminal_status: parse_terminal_status(&self.termination),
            plies_played: self.plies,
            first_candidate_score_cp: self.candidate_first_score_cp,
            first_candidate_info_line: self.candidate_first_info.clone(),
            first_fallback_score_cp: self.baseline_first_score_cp,
            first_fallback_info_line: self.baseline_first_info.clone(),
        })
    }
}

fn add_game_to_summary(summary: &mut MatchSummary, game: MatchGameSummary) {
    summary.games += 1;
    match game.outcome {
        MatchOutcome::CandidateWin => summary.candidate_wins += 1,
        MatchOutcome::FallbackWin => summary.fallback_wins += 1,
        MatchOutcome::Draw => summary.draws += 1,
    }
    summary.game_summaries.push(game);
}

fn append_game_artifacts(paths: &ArtifactPaths, game: &GameArtifact) -> Result<(), String> {
    let mut games = OpenOptions::new()
        .append(true)
        .open(&paths.games)
        .map_err(|error| format!("failed to append game log: {error}"))?;
    serde_json::to_writer(&mut games, game)
        .map_err(|error| format!("failed to encode game record: {error}"))?;
    writeln!(games).map_err(|error| format!("failed to finish game record: {error}"))?;
    games
        .sync_all()
        .map_err(|error| format!("failed to flush game log: {error}"))?;

    let mut pgn = OpenOptions::new()
        .append(true)
        .open(&paths.pgn)
        .map_err(|error| format!("failed to append PGN log: {error}"))?;
    write_pgn_record(&mut pgn, game)
        .map_err(|error| format!("failed to write PGN record: {error}"))?;
    pgn.sync_all()
        .map_err(|error| format!("failed to flush PGN log: {error}"))
}

fn write_pgn_record(writer: &mut impl Write, game: &GameArtifact) -> std::io::Result<()> {
    let pgn_result = match game.result.as_str() {
        "candidate_win" if game.candidate_color == "white" => "1-0",
        "candidate_win" => "0-1",
        "baseline_win" if game.candidate_color == "white" => "0-1",
        "baseline_win" => "1-0",
        _ => "1/2-1/2",
    };
    writeln!(writer, "[Event \"Volkrix paired experiment\"]")
        .and_then(|_| writeln!(writer, "[Round \"{}\"]", game.game_number))
        .and_then(|_| writeln!(writer, "[SetUp \"1\"]"))
        .and_then(|_| writeln!(writer, "[FEN \"{}\"]", game.opening_fen))
        .and_then(|_| writeln!(writer, "[Result \"{pgn_result}\"]"))
        .and_then(|_| writeln!(writer, "[Termination \"{}\"]", game.termination))
        .and_then(|_| writeln!(writer))
        .and_then(|_| {
            writeln!(
                writer,
                "{{ UCI moves: {} }} {pgn_result}\n",
                game.moves_uci.join(" ")
            )
        })
}

fn write_checkpoint(
    path: &Path,
    manifest_sha256: &str,
    summary: &MatchSummary,
) -> Result<(), String> {
    let checkpoint = MatchCheckpoint {
        manifest_sha256: manifest_sha256.to_owned(),
        completed_openings: summary.games / 2,
        completed_games: summary.games,
        candidate_wins: summary.candidate_wins,
        draws: summary.draws,
        baseline_wins: summary.fallback_wins,
    };
    let mut bytes = serde_json::to_vec_pretty(&checkpoint)
        .map_err(|error| format!("failed to encode checkpoint: {error}"))?;
    bytes.push(b'\n');
    replace_file(path, &bytes, "checkpoint")
}

fn recover_committed_artifacts(
    paths: &ArtifactPaths,
    checkpoint: &MatchCheckpoint,
) -> Result<MatchSummary, String> {
    let input =
        File::open(&paths.games).map_err(|error| format!("failed to open game log: {error}"))?;
    let mut reader = BufReader::new(input);
    let mut games = Vec::with_capacity(checkpoint.completed_games);
    let mut line_number = 0usize;
    while games.len() < checkpoint.completed_games {
        let mut line = Vec::new();
        let bytes_read = reader
            .read_until(b'\n', &mut line)
            .map_err(|error| format!("failed to read game log: {error}"))?;
        if bytes_read == 0 {
            return Err(format!(
                "game log has {} committed records but checkpoint requires {}",
                games.len(),
                checkpoint.completed_games
            ));
        }
        line_number += 1;
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let artifact: GameArtifact = serde_json::from_slice(&line).map_err(|error| {
            format!("failed to parse committed game log line {line_number}: {error}")
        })?;
        validate_committed_game(&games, &artifact)?;
        games.push(artifact);
    }

    let mut summary = MatchSummary::default();
    for artifact in &games {
        add_game_to_summary(&mut summary, artifact.to_summary()?);
    }
    summary.openings = summary.games / 2;
    if summary.games != checkpoint.completed_games
        || summary.candidate_wins != checkpoint.candidate_wins
        || summary.draws != checkpoint.draws
        || summary.fallback_wins != checkpoint.baseline_wins
    {
        return Err("checkpoint result totals do not match committed game records".to_owned());
    }

    let mut jsonl = Vec::new();
    let mut pgn = Vec::new();
    for game in &games {
        serde_json::to_writer(&mut jsonl, game)
            .map_err(|error| format!("failed to encode recovered game record: {error}"))?;
        jsonl.push(b'\n');
        write_pgn_record(&mut pgn, game)
            .map_err(|error| format!("failed to encode recovered PGN record: {error}"))?;
    }
    // The checkpoint is the commit marker. Anything after its pair boundary is
    // an interrupted write and is discarded; PGN is derived from the committed
    // JSON records so the two views cannot remain out of sync after recovery.
    replace_file(&paths.games, &jsonl, "recovered game log")?;
    replace_file(&paths.pgn, &pgn, "recovered PGN")?;
    Ok(summary)
}

fn validate_committed_game(previous: &[GameArtifact], game: &GameArtifact) -> Result<(), String> {
    let index = previous.len();
    let expected_number = index + 1;
    let expected_opening = index / 2;
    let expected_color = if index.is_multiple_of(2) {
        "white"
    } else {
        "black"
    };
    if game.game_number != expected_number
        || game.opening_index != expected_opening
        || game.candidate_color != expected_color
    {
        return Err(format!(
            "committed game {} does not match opening-pair sequence",
            game.game_number
        ));
    }
    if !index.is_multiple_of(2)
        && previous
            .last()
            .is_some_and(|first| first.opening_fen != game.opening_fen)
    {
        return Err(format!(
            "committed opening pair {} uses different FENs",
            expected_opening
        ));
    }
    Ok(())
}

fn replace_file(path: &Path, bytes: &[u8], label: &str) -> Result<(), String> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .map_or_else(|| "tmp".to_owned(), |value| format!("{value}.tmp"));
    let temporary = path.with_extension(extension);
    let mut output = File::create(&temporary)
        .map_err(|error| format!("failed to create {label} temporary file: {error}"))?;
    output
        .write_all(bytes)
        .and_then(|_| output.sync_all())
        .map_err(|error| format!("failed to write {label}: {error}"))?;
    fs::rename(&temporary, path).map_err(|error| format!("failed to publish {label}: {error}"))
}

fn match_outcome_from_status(
    status: PositionStatus,
    side_to_move: Color,
    candidate_color: Color,
) -> MatchOutcome {
    match status {
        PositionStatus::Checkmate if side_to_move.opposite() == candidate_color => {
            MatchOutcome::CandidateWin
        }
        PositionStatus::Checkmate => MatchOutcome::FallbackWin,
        PositionStatus::Stalemate
        | PositionStatus::DrawByRepetition
        | PositionStatus::DrawByFiftyMove
        | PositionStatus::DrawByInsufficientMaterial
        | PositionStatus::Ongoing => MatchOutcome::Draw,
    }
}

fn board_termination(status: PositionStatus) -> String {
    match status {
        PositionStatus::Ongoing => "ongoing",
        PositionStatus::Checkmate => "checkmate",
        PositionStatus::Stalemate => "stalemate",
        PositionStatus::DrawByRepetition => "repetition",
        PositionStatus::DrawByFiftyMove => "fifty_move",
        PositionStatus::DrawByInsufficientMaterial => "insufficient_material",
    }
    .to_owned()
}

fn parse_terminal_status(termination: &str) -> PositionStatus {
    match termination {
        "checkmate" => PositionStatus::Checkmate,
        "stalemate" => PositionStatus::Stalemate,
        "repetition" => PositionStatus::DrawByRepetition,
        "fifty_move" => PositionStatus::DrawByFiftyMove,
        "insufficient_material" => PositionStatus::DrawByInsufficientMaterial,
        _ => PositionStatus::Ongoing,
    }
}

fn color_name(color: Color) -> &'static str {
    match color {
        Color::White => "white",
        Color::Black => "black",
    }
}

fn parse_color(value: &str) -> Result<Color, String> {
    match value {
        "white" => Ok(Color::White),
        "black" => Ok(Color::Black),
        _ => Err(format!("invalid recorded color '{value}'")),
    }
}

fn outcome_name(outcome: MatchOutcome) -> &'static str {
    match outcome {
        MatchOutcome::CandidateWin => "candidate_win",
        MatchOutcome::FallbackWin => "baseline_win",
        MatchOutcome::Draw => "draw",
    }
}

fn parse_outcome(value: &str) -> Result<MatchOutcome, String> {
    match value {
        "candidate_win" => Ok(MatchOutcome::CandidateWin),
        "baseline_win" => Ok(MatchOutcome::FallbackWin),
        "draw" => Ok(MatchOutcome::Draw),
        _ => Err(format!("invalid recorded result '{value}'")),
    }
}

fn sanitize_token(text: &str) -> String {
    text.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect()
}

#[derive(Debug)]
struct EngineSearchResult {
    best_move: String,
    score_cp: Option<i32>,
    last_info_line: Option<String>,
    elapsed: Duration,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EngineFailure {
    Crash,
    Hang,
    TimeForfeit,
    Protocol,
}

impl EngineFailure {
    const fn label(self) -> &'static str {
        match self {
            Self::Crash => "crash",
            Self::Hang => "hang",
            Self::TimeForfeit => "time_forfeit",
            Self::Protocol => "protocol_error",
        }
    }
}

enum ReaderEvent {
    Line(String),
    Eof,
    Error(String),
}

struct ExternalEngine {
    label: &'static str,
    options: EngineOptions,
    protocol_timeout: Duration,
    stop_grace: Duration,
    child: Option<Child>,
    stdin: Option<BufWriter<ChildStdin>>,
    stdout: Option<mpsc::Receiver<ReaderEvent>>,
    protocol_log: Arc<Mutex<BufWriter<File>>>,
}

impl ExternalEngine {
    fn spawn(
        label: &'static str,
        options: EngineOptions,
        protocol_timeout_ms: u64,
        stop_grace_ms: u64,
        protocol_log: Arc<Mutex<BufWriter<File>>>,
    ) -> Result<Self, String> {
        let mut engine = Self {
            label,
            options,
            protocol_timeout: Duration::from_millis(protocol_timeout_ms),
            stop_grace: Duration::from_millis(stop_grace_ms),
            child: None,
            stdin: None,
            stdout: None,
            protocol_log,
        };
        engine.start()?;
        Ok(engine)
    }

    fn start(&mut self) -> Result<(), String> {
        let mut child = Command::new(&self.options.path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| {
                format!(
                    "failed to launch {} engine '{}': {error}",
                    self.label,
                    self.options.path.display()
                )
            })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| format!("failed to acquire stdin for {} engine", self.label))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| format!("failed to acquire stdout for {} engine", self.label))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| format!("failed to acquire stderr for {} engine", self.label))?;
        let (sender, receiver) = mpsc::channel();
        let label = self.label;
        let stdout_log = Arc::clone(&self.protocol_log);
        thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            loop {
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) => {
                        let _ = sender.send(ReaderEvent::Eof);
                        break;
                    }
                    Ok(_) => {
                        log_protocol(&stdout_log, label, "out", line.trim_end());
                        if sender.send(ReaderEvent::Line(line)).is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = sender.send(ReaderEvent::Error(error.to_string()));
                        break;
                    }
                }
            }
        });
        let stderr_log = Arc::clone(&self.protocol_log);
        thread::spawn(move || {
            for line in BufReader::new(stderr).lines() {
                match line {
                    Ok(line) => log_protocol(&stderr_log, label, "err", &line),
                    Err(error) => {
                        log_protocol(&stderr_log, label, "err-read", &error.to_string());
                        break;
                    }
                }
            }
        });
        self.child = Some(child);
        self.stdin = Some(BufWriter::new(stdin));
        self.stdout = Some(receiver);
        self.initialize()
    }

    fn initialize(&mut self) -> Result<(), String> {
        self.send_command("uci")?;
        let handshake = self.read_until("uciok", self.protocol_timeout)?;
        for required in ["Hash", "Threads", "Move Overhead", "SyzygyPath", "EvalFile"] {
            if !handshake
                .iter()
                .any(|line| line.starts_with(&format!("option name {required} ")))
            {
                return Err(format!(
                    "{} engine did not advertise required UCI option '{required}'",
                    self.label
                ));
            }
        }
        self.send_command(&format!(
            "setoption name Threads value {}",
            self.options.threads
        ))?;
        self.send_command(&format!(
            "setoption name Hash value {}",
            self.options.hash_mb
        ))?;
        self.send_command(&format!(
            "setoption name Move Overhead value {}",
            self.options.move_overhead_ms
        ))?;
        if self.options.syzygy_path == "none" {
            self.send_command("setoption name SyzygyPath value")?;
        } else {
            self.send_command(&format!(
                "setoption name SyzygyPath value {}",
                self.options.syzygy_path
            ))?;
        }
        if self.options.eval_file == CLASSICAL_EVAL {
            self.send_command("setoption name EvalFile value")?;
        } else {
            self.send_command(&format!(
                "setoption name EvalFile value {}",
                self.options.eval_file
            ))?;
        }
        if let Some(path) = self.options.small_eval_file.as_deref() {
            for required in ["SmallEvalFile", "DualEvalPolicy", "DualEvalThreshold"] {
                if !handshake
                    .iter()
                    .any(|line| line.starts_with(&format!("option name {required} ")))
                {
                    return Err(format!(
                        "{} engine did not advertise required dual-eval UCI option '{required}'",
                        self.label
                    ));
                }
            }
            self.send_command(&format!("setoption name SmallEvalFile value {path}"))?;
            self.send_command(&format!(
                "setoption name DualEvalThreshold value {}",
                self.options.dual_eval_threshold
            ))?;
            self.send_command(&format!(
                "setoption name DualEvalPolicy value {}",
                self.options.dual_eval_policy
            ))?;
        }
        self.wait_ready()
    }

    fn new_game(&mut self) -> Result<(), String> {
        if self.child.is_none() {
            self.start()?;
        }
        self.send_command("ucinewgame")?;
        self.wait_ready()
    }

    fn search(
        &mut self,
        position: &Position,
        command: &str,
    ) -> Result<EngineSearchResult, EngineFailure> {
        self.send_command(&format!("position fen {}", position.to_fen()))
            .map_err(|_| EngineFailure::Crash)?;
        self.send_command(command)
            .map_err(|_| EngineFailure::Crash)?;
        let started = Instant::now();
        let hard_budget = search_timeout(
            command,
            position.side_to_move(),
            self.protocol_timeout,
            self.stop_grace,
        );
        let deadline = started + hard_budget;
        let mut last_info_line = None;
        let mut score_cp = None;
        loop {
            let event = self.read_event_until(deadline)?;
            match event {
                ReaderEvent::Line(line) => {
                    let trimmed = line.trim();
                    if is_uci_error(trimmed) {
                        self.terminate();
                        return Err(EngineFailure::Protocol);
                    }
                    if trimmed.starts_with("info ") {
                        last_info_line = Some(trimmed.to_owned());
                        score_cp = parse_score_cp(trimmed).or(score_cp);
                    } else if let Some(best_move) = parse_bestmove(trimmed) {
                        if best_move == "0000" {
                            return Err(EngineFailure::Protocol);
                        }
                        let elapsed = started.elapsed();
                        if command.starts_with("go movetime ") && elapsed > hard_budget {
                            self.terminate();
                            return Err(EngineFailure::TimeForfeit);
                        }
                        return Ok(EngineSearchResult {
                            best_move: best_move.to_owned(),
                            score_cp,
                            last_info_line,
                            elapsed,
                        });
                    }
                }
                ReaderEvent::Eof | ReaderEvent::Error(_) => {
                    self.terminate();
                    return Err(EngineFailure::Crash);
                }
            }
        }
    }

    fn wait_ready(&mut self) -> Result<(), String> {
        self.send_command("isready")?;
        self.read_until("readyok", self.protocol_timeout)
            .map(|_| ())
    }

    fn send_command(&mut self, command: &str) -> Result<(), String> {
        log_protocol(&self.protocol_log, self.label, "in", command);
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| format!("{} engine is not running", self.label))?;
        writeln!(stdin, "{command}")
            .and_then(|_| stdin.flush())
            .map_err(|error| format!("failed to write command to {} engine: {error}", self.label))
    }

    fn read_until(&mut self, target: &str, timeout: Duration) -> Result<Vec<String>, String> {
        let deadline = Instant::now() + timeout;
        let mut lines = Vec::new();
        loop {
            match self.read_event_until(deadline).map_err(|failure| {
                format!(
                    "{} engine failed while waiting for {target}: {}",
                    self.label,
                    failure.label()
                )
            })? {
                ReaderEvent::Line(line) => {
                    let trimmed = line.trim().to_owned();
                    if is_uci_error(&trimmed) {
                        return Err(format!(
                            "{} engine rejected its configuration: {trimmed}",
                            self.label
                        ));
                    }
                    if trimmed == target {
                        return Ok(lines);
                    }
                    lines.push(trimmed);
                }
                ReaderEvent::Eof => {
                    return Err(format!("{} engine closed stdout unexpectedly", self.label));
                }
                ReaderEvent::Error(error) => {
                    return Err(format!("{} engine stdout failed: {error}", self.label));
                }
            }
        }
    }

    fn read_event_until(&mut self, deadline: Instant) -> Result<ReaderEvent, EngineFailure> {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let event = self
            .stdout
            .as_ref()
            .ok_or(EngineFailure::Crash)?
            .recv_timeout(remaining);
        match event {
            Ok(event) => Ok(event),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(EngineFailure::Crash),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let _ = self.send_command("stop");
                let grace_deadline = Instant::now() + self.stop_grace;
                while let Some(receiver) = self.stdout.as_ref() {
                    let Ok(event) = receiver
                        .recv_timeout(grace_deadline.saturating_duration_since(Instant::now()))
                    else {
                        break;
                    };
                    if matches!(&event, ReaderEvent::Line(line) if parse_bestmove(line.trim()).is_some())
                    {
                        self.terminate();
                        return Err(EngineFailure::TimeForfeit);
                    }
                    if matches!(event, ReaderEvent::Eof | ReaderEvent::Error(_)) {
                        self.terminate();
                        return Err(EngineFailure::Crash);
                    }
                    if Instant::now() >= grace_deadline {
                        break;
                    }
                }
                self.terminate();
                Err(EngineFailure::Hang)
            }
        }
    }

    fn terminate(&mut self) {
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.child = None;
        self.stdin = None;
        self.stdout = None;
    }
}

impl Drop for ExternalEngine {
    fn drop(&mut self) {
        if self.child.is_some() {
            let _ = self.send_command("quit");
        }
        self.terminate();
    }
}

fn search_timeout(
    command: &str,
    side_to_move: Color,
    protocol_timeout: Duration,
    stop_grace: Duration,
) -> Duration {
    let mut tokens = command.split_whitespace();
    if tokens.next() == Some("go")
        && tokens.next() == Some("movetime")
        && let Some(milliseconds) = tokens.next().and_then(|value| value.parse::<u64>().ok())
    {
        return Duration::from_millis(milliseconds).saturating_add(stop_grace);
    }
    if command.starts_with("go wtime ") {
        let fields = command.split_whitespace().collect::<Vec<_>>();
        let white = fields.get(2).and_then(|value| value.parse::<u64>().ok());
        let black = fields.get(4).and_then(|value| value.parse::<u64>().ok());
        let remaining = match side_to_move {
            Color::White => white,
            Color::Black => black,
        };
        if let Some(remaining) = remaining {
            return Duration::from_millis(remaining).saturating_add(stop_grace);
        }
    }
    protocol_timeout
}

fn log_protocol(log: &Arc<Mutex<BufWriter<File>>>, engine: &str, direction: &str, line: &str) {
    if let Ok(mut log) = log.lock() {
        let _ = writeln!(log, "[{engine} {direction}] {line}");
        let _ = log.flush();
    }
}

fn is_uci_error(line: &str) -> bool {
    line.starts_with("info string error")
}

fn parse_bestmove(line: &str) -> Option<&str> {
    let mut tokens = line.split_whitespace();
    (tokens.next()? == "bestmove")
        .then(|| tokens.next())
        .flatten()
}

fn parse_score_cp(line: &str) -> Option<i32> {
    let mut tokens = line.split_whitespace();
    while let Some(token) = tokens.next() {
        if token == "score" {
            return (tokens.next()? == "cp")
                .then(|| tokens.next()?.parse::<i32>().ok())
                .flatten();
        }
    }
    None
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut input = File::open(path)
        .map_err(|error| format!("failed to hash '{}': {error}", path.display()))?;
    let mut state = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = input
            .read(&mut buffer)
            .map_err(|error| format!("failed to hash '{}': {error}", path.display()))?;
        if count == 0 {
            break;
        }
        state.update(&buffer[..count]);
    }
    Ok(state.finish_hex())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut state = Sha256::new();
    state.update(bytes);
    state.finish_hex()
}

struct Sha256 {
    state: [u32; 8],
    pending: Vec<u8>,
    length_bytes: u64,
}

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            pending: Vec::with_capacity(64),
            length_bytes: 0,
        }
    }

    fn update(&mut self, bytes: &[u8]) {
        self.length_bytes = self.length_bytes.saturating_add(bytes.len() as u64);
        self.pending.extend_from_slice(bytes);
        while self.pending.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&self.pending[..64]);
            self.compress(&block);
            self.pending.drain(..64);
        }
    }

    fn finish_hex(mut self) -> String {
        let bit_length = self.length_bytes.wrapping_mul(8);
        self.pending.push(0x80);
        while self.pending.len() % 64 != 56 {
            self.pending.push(0);
        }
        self.pending.extend_from_slice(&bit_length.to_be_bytes());
        while !self.pending.is_empty() {
            let mut block = [0u8; 64];
            block.copy_from_slice(&self.pending[..64]);
            self.compress(&block);
            self.pending.drain(..64);
        }
        self.state
            .iter()
            .map(|word| format!("{word:08x}"))
            .collect()
    }

    fn compress(&mut self, block: &[u8; 64]) {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        let mut words = [0u32; 64];
        for (index, chunk) in block.chunks_exact(4).enumerate() {
            words[index] = u32::from_be_bytes(chunk.try_into().expect("four-byte chunk"));
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(choice)
                .wrapping_add(K[index])
                .wrapping_add(words[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        for (target, value) in self.state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *target = target.wrapping_add(value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_standard_vectors() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn parses_bestmove_and_centipawn_score() {
        assert_eq!(parse_bestmove("bestmove e2e4 ponder e7e5"), Some("e2e4"));
        assert_eq!(
            parse_score_cp("info depth 8 score cp -31 nodes 20"),
            Some(-31)
        );
        assert_eq!(parse_score_cp("info depth 8 score mate 2"), None);
    }

    #[test]
    fn explicit_evaluator_validation_rejects_implicit_empty_choice() {
        assert!(validate_eval_choice("candidate", "").is_err());
        assert!(validate_eval_choice("candidate", CLASSICAL_EVAL).is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn supervised_engine_rejects_uci_configuration_errors_and_kills_hangs() {
        use std::os::unix::fs::PermissionsExt;

        let root = std::env::temp_dir().join(format!(
            "volkrix-engine-fixture-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let log_path = root.join("protocol.log");
        let log = Arc::new(Mutex::new(BufWriter::new(File::create(&log_path).unwrap())));

        let error_script = root.join("error.sh");
        fs::write(&error_script, fixture_script("error")).unwrap();
        fs::set_permissions(&error_script, fs::Permissions::from_mode(0o755)).unwrap();
        let error = ExternalEngine::spawn(
            "candidate",
            fixture_options(error_script),
            10_000,
            25,
            Arc::clone(&log),
        )
        .err()
        .expect("configuration error must reject engine");
        assert!(
            error.contains("rejected its configuration"),
            "unexpected initialization error: {error}"
        );

        let hang_script = root.join("hang.sh");
        fs::write(&hang_script, fixture_script("hang")).unwrap();
        fs::set_permissions(&hang_script, fs::Permissions::from_mode(0o755)).unwrap();
        let mut hanging =
            ExternalEngine::spawn("candidate", fixture_options(hang_script), 10_000, 20, log)
                .expect("hanging engine must initialize");
        hanging.protocol_timeout = Duration::from_millis(50);
        let failure = hanging
            .search(&Position::startpos(), "go depth 1")
            .expect_err("hung search must fail");
        assert_eq!(failure, EngineFailure::Hang);
        assert!(hanging.child.is_none());

        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn paired_match_recovers_interrupted_artifact_tails_at_pair_boundary() {
        use std::os::unix::fs::PermissionsExt;

        let root = std::env::temp_dir().join(format!(
            "volkrix-paired-fixture-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let engine_path = root.join("engine.sh");
        fs::write(&engine_path, fixture_script("ok")).unwrap();
        fs::set_permissions(&engine_path, fs::Permissions::from_mode(0o755)).unwrap();
        let openings_path = root.join("openings.fens");
        fs::write(&openings_path, format!("{}\n", volkrix::core::STARTPOS_FEN)).unwrap();
        let config = ExternalMatchConfig {
            baseline: fixture_options(engine_path.clone()),
            candidate: fixture_options(engine_path),
            time_control: ExternalTimeControl::FixedDepth { depth: 1 },
            max_plies: 1,
            max_openings: None,
            protocol_timeout_ms: 10_000,
            stop_grace_ms: 25,
            artifacts_dir: root.join("artifacts"),
            resume: false,
        };

        let report = compare_external_engines(&openings_path, config.clone()).unwrap();
        assert_eq!(report.summary.games, 2);
        assert_eq!(report.summary.draws, 2);
        for path in [
            &report.manifest_path,
            &report.games_path,
            &report.pgn_path,
            &report.protocol_log_path,
            &report.checkpoint_path,
        ] {
            assert!(path.is_file(), "missing artifact {}", path.display());
        }
        assert_eq!(
            fs::read_to_string(&report.games_path)
                .unwrap()
                .lines()
                .count(),
            2
        );
        let manifest = fs::read_to_string(&report.manifest_path).unwrap();
        assert!(manifest.contains("\"binary_sha256\""));
        assert!(manifest.contains("\"opening_sha256\""));
        let checkpoint: MatchCheckpoint =
            serde_json::from_str(&fs::read_to_string(&report.checkpoint_path).unwrap()).unwrap();
        assert_eq!(checkpoint.completed_openings, 1);
        assert_eq!(checkpoint.completed_games, 2);

        let committed_games = fs::read_to_string(&report.games_path).unwrap();
        let committed_pgn = fs::read_to_string(&report.pgn_path).unwrap();
        let mut orphan: GameArtifact =
            serde_json::from_str(committed_games.lines().next().unwrap()).unwrap();
        orphan.game_number = 3;
        orphan.opening_index = 1;
        let mut interrupted_games = committed_games.clone();
        interrupted_games.push_str(&serde_json::to_string(&orphan).unwrap());
        interrupted_games.push('\n');
        interrupted_games.push_str("{\"game_number\":4,\"interrupted\"");
        fs::write(&report.games_path, interrupted_games).unwrap();
        fs::write(
            &report.pgn_path,
            format!("{committed_pgn}[Event \"interrupted PGN"),
        )
        .unwrap();

        let resumed = compare_external_engines(
            &openings_path,
            ExternalMatchConfig {
                resume: true,
                ..config
            },
        )
        .unwrap();
        assert_eq!(resumed.summary.games, 2);
        assert_eq!(
            fs::read_to_string(&resumed.games_path).unwrap(),
            committed_games
        );
        assert_eq!(
            fs::read_to_string(&resumed.pgn_path).unwrap(),
            committed_pgn
        );

        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    fn fixture_options(path: PathBuf) -> EngineOptions {
        EngineOptions {
            path,
            eval_file: CLASSICAL_EVAL.to_owned(),
            small_eval_file: None,
            dual_eval_policy: "off".to_owned(),
            dual_eval_threshold: 200,
            threads: 1,
            hash_mb: 1,
            move_overhead_ms: 0,
            syzygy_path: "none".to_owned(),
        }
    }

    #[cfg(unix)]
    fn fixture_script(mode: &str) -> String {
        format!(
            r#"#!/bin/sh
mode='{mode}'
while IFS= read -r line; do
    case "$line" in
        uci)
            printf '%s\n' \
                'option name Hash type spin default 1 min 1 max 512' \
                'option name Threads type spin default 1 min 1 max 64' \
                'option name Move Overhead type spin default 0 min 0 max 5000' \
                'option name SyzygyPath type string default' \
                'option name EvalFile type string default' \
                'uciok'
            ;;
        setoption*)
            if [ "$mode" = error ]; then
                printf '%s\n' 'info string error: injected'
                mode=ok
            fi
            ;;
        isready)
            printf '%s\n' readyok
            ;;
        go*)
            if [ "$mode" != hang ]; then
                printf '%s\n' 'bestmove e2e4'
            fi
            ;;
        quit)
            exit 0
            ;;
    esac
done
"#
        )
    }
}
