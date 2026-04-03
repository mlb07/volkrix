use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
};

use volkrix::{
    core::{Color, Position, PositionStatus},
    nnue_training::{MatchConfig, MatchGameSummary, MatchMode, MatchOutcome, MatchSummary, normalize_fen},
};

pub fn compare_external_engines(
    openings_path: &Path,
    baseline_engine_path: &Path,
    candidate_engine_path: &Path,
    config: MatchConfig,
    max_openings: Option<usize>,
) -> Result<MatchSummary, String> {
    let openings = load_openings(openings_path, max_openings)?;
    let mut baseline = ExternalEngine::spawn(baseline_engine_path, config.hash_mb)?;
    let mut candidate = ExternalEngine::spawn(candidate_engine_path, config.hash_mb)?;

    let mut summary = MatchSummary {
        openings: openings.len(),
        ..MatchSummary::default()
    };
    for opening_fen in openings {
        for candidate_color in [Color::White, Color::Black] {
            let game = play_match_game(
                &opening_fen,
                candidate_color,
                config,
                &mut baseline,
                &mut candidate,
            )?;
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

        let line = line
            .map_err(|error| format!("failed to read openings line {}: {error}", line_number + 1))?;
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

    if openings.is_empty() {
        return Err(format!(
            "openings corpus '{}' did not contain any usable FENs",
            openings_path.display()
        ));
    }

    Ok(openings)
}

fn play_match_game(
    opening_fen: &str,
    candidate_color: Color,
    config: MatchConfig,
    baseline: &mut ExternalEngine,
    candidate: &mut ExternalEngine,
) -> Result<MatchGameSummary, String> {
    let mut position = Position::from_fen(opening_fen)
        .map_err(|error| format!("failed to parse opening FEN '{opening_fen}': {error}"))?;
    baseline.new_game()?;
    candidate.new_game()?;

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
                outcome: match_outcome_from_status(status, position.side_to_move(), candidate_color),
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
        let result = if side_to_move == candidate_color {
            candidate.search(&position, config.mode)?
        } else {
            baseline.search(&position, config.mode)?
        };

        if side_to_move == candidate_color && first_candidate_score_cp.is_none() {
            first_candidate_score_cp = result.score_cp;
            first_candidate_info_line = result.last_info_line;
        } else if side_to_move != candidate_color && first_fallback_score_cp.is_none() {
            first_fallback_score_cp = result.score_cp;
            first_fallback_info_line = result.last_info_line;
        }

        position.apply_uci_move(&result.best_move).map_err(|error| {
            format!(
                "engine move '{}' was not legal from '{}': {error}",
                result.best_move,
                position.to_fen()
            )
        })?;
        plies_played += 1;
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
        | PositionStatus::DrawByInsufficientMaterial => MatchOutcome::Draw,
        PositionStatus::Ongoing => MatchOutcome::Draw,
    }
}

#[derive(Debug)]
struct EngineSearchResult {
    best_move: String,
    score_cp: Option<i32>,
    last_info_line: Option<String>,
}

struct ExternalEngine {
    path: String,
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
}

impl ExternalEngine {
    fn spawn(path: &Path, hash_mb: usize) -> Result<Self, String> {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| format!("failed to launch engine '{}': {error}", path.display()))?;
        let stdin = child.stdin.take().ok_or_else(|| {
            format!(
                "failed to acquire stdin for engine '{}'",
                path.display()
            )
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            format!(
                "failed to acquire stdout for engine '{}'",
                path.display()
            )
        })?;

        let mut engine = Self {
            path: path.display().to_string(),
            child,
            stdin: BufWriter::new(stdin),
            stdout: BufReader::new(stdout),
        };
        engine.initialize(hash_mb)?;
        Ok(engine)
    }

    fn initialize(&mut self, hash_mb: usize) -> Result<(), String> {
        self.send_command("uci")?;
        self.read_until("uciok")?;
        self.send_command("setoption name Threads value 1")?;
        self.send_command(&format!("setoption name Hash value {hash_mb}"))?;
        self.send_command("setoption name SyzygyPath value")?;
        self.send_command("setoption name EvalFile value")?;
        self.wait_ready()
    }

    fn new_game(&mut self) -> Result<(), String> {
        self.send_command("ucinewgame")?;
        self.wait_ready()
    }

    fn search(
        &mut self,
        position: &Position,
        mode: MatchMode,
    ) -> Result<EngineSearchResult, String> {
        self.send_command(&format!("position fen {}", position.to_fen()))?;
        match mode {
            MatchMode::FixedDepth(depth) => self.send_command(&format!("go depth {depth}"))?,
            MatchMode::MoveTimeMs(movetime_ms) => {
                self.send_command(&format!("go movetime {movetime_ms}"))?
            }
        }

        let mut last_info_line = None;
        let mut score_cp = None;
        loop {
            let line = self.read_line()?;
            let trimmed = line.trim();
            if trimmed.starts_with("info ") {
                last_info_line = Some(trimmed.to_owned());
                score_cp = parse_score_cp(trimmed).or(score_cp);
                continue;
            }
            if let Some(best_move) = parse_bestmove(trimmed) {
                if best_move == "0000" {
                    return Err(format!(
                        "engine '{}' returned bestmove 0000 for ongoing position '{}'",
                        self.path,
                        position.to_fen()
                    ));
                }
                return Ok(EngineSearchResult {
                    best_move: best_move.to_owned(),
                    score_cp,
                    last_info_line,
                });
            }
        }
    }

    fn wait_ready(&mut self) -> Result<(), String> {
        self.send_command("isready")?;
        self.read_until("readyok")
    }

    fn send_command(&mut self, command: &str) -> Result<(), String> {
        writeln!(self.stdin, "{command}")
            .map_err(|error| format!("failed to write command to '{}': {error}", self.path))?;
        self.stdin
            .flush()
            .map_err(|error| format!("failed to flush command to '{}': {error}", self.path))
    }

    fn read_until(&mut self, target: &str) -> Result<(), String> {
        loop {
            if self.read_line()?.trim() == target {
                return Ok(());
            }
        }
    }

    fn read_line(&mut self) -> Result<String, String> {
        let mut line = String::new();
        let read = self
            .stdout
            .read_line(&mut line)
            .map_err(|error| format!("failed to read response from '{}': {error}", self.path))?;
        if read == 0 {
            return Err(format!(
                "engine '{}' closed stdout unexpectedly",
                self.path
            ));
        }
        Ok(line)
    }
}

impl Drop for ExternalEngine {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "quit");
        let _ = self.stdin.flush();
        let _ = self.child.wait();
    }
}

fn parse_bestmove(line: &str) -> Option<&str> {
    let mut tokens = line.split_whitespace();
    if tokens.next()? != "bestmove" {
        return None;
    }
    tokens.next()
}

fn parse_score_cp(line: &str) -> Option<i32> {
    let mut tokens = line.split_whitespace();
    while let Some(token) = tokens.next() {
        if token != "score" {
            continue;
        }
        let kind = tokens.next()?;
        let value = tokens.next()?;
        if kind == "cp" {
            return value.parse::<i32>().ok();
        }
        return None;
    }
    None
}
