use std::{
    io::{self, BufRead, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender},
    },
    thread,
    time::{Duration, Instant},
};

use crate::{
    ENGINE_AUTHOR, ENGINE_NAME,
    core::{Color, Position},
    search::{
        SearchLimits,
        service::{DEFAULT_THREADS, SearchRequest, UciSearchService},
    },
};

const DEFAULT_GO_DEPTH: u8 = 1;
const MAX_GO_DEPTH: u8 = 127;
const MIN_HASH_MB: usize = 1;
const MAX_HASH_MB: usize = 512;
const MIN_THREADS: usize = 1;
const MAX_THREADS: usize = 64;

pub struct UciResponse {
    pub lines: Vec<String>,
    pub should_quit: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct GoOptions {
    depth: Option<u8>,
    movetime_ms: Option<u64>,
    wtime_ms: Option<u64>,
    btime_ms: Option<u64>,
    winc_ms: u64,
    binc_ms: u64,
    movestogo: Option<u32>,
    infinite: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SetOptionCommand {
    Hash(usize),
    ClearHash,
    Threads(usize),
}

enum RuntimeInput {
    Command(String),
    QuitRequested,
}

pub struct UciEngine {
    position: Position,
    search_service: UciSearchService,
}

impl UciEngine {
    pub fn new() -> Self {
        Self {
            position: Position::startpos(),
            search_service: UciSearchService::new(),
        }
    }

    pub fn position(&self) -> &Position {
        &self.position
    }

    #[cfg(any(test, debug_assertions))]
    #[doc(hidden)]
    pub fn debug_hash_mb(&self) -> usize {
        self.search_service.hash_mb()
    }

    #[cfg(any(test, debug_assertions))]
    #[doc(hidden)]
    pub fn debug_threads(&self) -> usize {
        self.search_service.threads()
    }

    #[cfg(any(test, debug_assertions))]
    #[doc(hidden)]
    pub fn debug_tt_entry_count(&self) -> usize {
        self.search_service.debug_tt_entry_count()
    }

    #[cfg(any(test, debug_assertions))]
    #[doc(hidden)]
    pub fn debug_worker_count(&self) -> usize {
        self.search_service.debug_worker_count()
    }

    #[cfg(any(test, debug_assertions))]
    #[doc(hidden)]
    pub fn debug_active_helper_count(&self) -> usize {
        self.search_service.debug_active_helper_count()
    }

    pub fn handle_line(&mut self, line: &str) -> UciResponse {
        self.handle_line_with_stop(line, None)
    }

    fn handle_line_with_stop(
        &mut self,
        line: &str,
        stop_flag: Option<Arc<AtomicBool>>,
    ) -> UciResponse {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return UciResponse {
                lines: Vec::new(),
                should_quit: false,
            };
        }

        let tokens: Vec<&str> = trimmed.split_whitespace().collect();
        match tokens[0] {
            "uci" => UciResponse {
                lines: vec![
                    format!("id name {ENGINE_NAME}"),
                    format!("id author {ENGINE_AUTHOR}"),
                    format!(
                        "option name Hash type spin default {} min {} max {}",
                        self.search_service.hash_mb(),
                        MIN_HASH_MB,
                        MAX_HASH_MB
                    ),
                    format!(
                        "option name Threads type spin default {} min {} max {}",
                        DEFAULT_THREADS, MIN_THREADS, MAX_THREADS
                    ),
                    "option name Clear Hash type button".to_owned(),
                    "uciok".to_owned(),
                ],
                should_quit: false,
            },
            "isready" => UciResponse {
                lines: vec!["readyok".to_owned()],
                should_quit: false,
            },
            "ucinewgame" => {
                self.position = Position::startpos();
                self.search_service.clear_hash();
                UciResponse {
                    lines: Vec::new(),
                    should_quit: false,
                }
            }
            "position" => UciResponse {
                lines: self.handle_position(&tokens),
                should_quit: false,
            },
            "setoption" => UciResponse {
                lines: self.handle_setoption(&tokens),
                should_quit: false,
            },
            "go" => UciResponse {
                lines: self.handle_go(&tokens, stop_flag),
                should_quit: false,
            },
            "stop" => {
                if let Some(stop_flag) = stop_flag.as_ref() {
                    stop_flag.store(true, Ordering::Relaxed);
                }
                UciResponse {
                    lines: Vec::new(),
                    should_quit: false,
                }
            }
            "quit" => {
                if let Some(stop_flag) = stop_flag.as_ref() {
                    stop_flag.store(true, Ordering::Relaxed);
                }
                UciResponse {
                    lines: Vec::new(),
                    should_quit: true,
                }
            }
            _ => UciResponse {
                lines: vec![format!(
                    "info string error: unsupported command '{trimmed}'"
                )],
                should_quit: false,
            },
        }
    }

    fn handle_position(&mut self, tokens: &[&str]) -> Vec<String> {
        if tokens.len() < 2 {
            return vec!["info string error: position requires startpos or fen".to_owned()];
        }

        let mut cursor = 1usize;
        let mut next_position = match tokens[cursor] {
            "startpos" => {
                cursor += 1;
                Position::startpos()
            }
            "fen" => {
                cursor += 1;
                if tokens.len() < cursor + 6 {
                    return vec![
                        "info string error: position fen requires 6 FEN fields".to_owned(),
                    ];
                }
                let fen = tokens[cursor..cursor + 6].join(" ");
                cursor += 6;
                match Position::from_fen(&fen) {
                    Ok(position) => position,
                    Err(error) => {
                        return vec![format!("info string error: {error}")];
                    }
                }
            }
            other => {
                return vec![format!(
                    "info string error: unsupported position source '{other}'"
                )];
            }
        };

        if cursor < tokens.len() {
            if tokens[cursor] != "moves" {
                return vec![format!(
                    "info string error: expected 'moves' after position source, found '{}'",
                    tokens[cursor]
                )];
            }
            cursor += 1;
            for move_text in &tokens[cursor..] {
                if let Err(error) = next_position.apply_uci_move(move_text) {
                    return vec![format!("info string error: {error}")];
                }
            }
        }

        self.position = next_position;
        Vec::new()
    }

    fn handle_setoption(&mut self, tokens: &[&str]) -> Vec<String> {
        match parse_setoption(tokens) {
            Ok(SetOptionCommand::Hash(hash_mb)) => {
                self.search_service.resize_hash(hash_mb);
                Vec::new()
            }
            Ok(SetOptionCommand::ClearHash) => {
                self.search_service.clear_hash();
                Vec::new()
            }
            Ok(SetOptionCommand::Threads(threads)) => {
                self.search_service.set_threads(threads);
                Vec::new()
            }
            Err(error) => vec![format!("info string error: {error}")],
        }
    }

    fn handle_go(&mut self, tokens: &[&str], stop_flag: Option<Arc<AtomicBool>>) -> Vec<String> {
        let options = match parse_go(tokens) {
            Ok(options) => options,
            Err(error) => return vec![format!("info string error: {error}")],
        };

        if let Some(stop_flag) = stop_flag.as_ref() {
            stop_flag.store(false, Ordering::Relaxed);
        }

        let request = match self.build_search_request(options, stop_flag) {
            Ok(request) => request,
            Err(error) => return vec![format!("info string error: {error}")],
        };

        let result = self.search_service.search(&mut self.position, request);
        let mut lines = result.info_lines;
        let bestmove = result
            .best_move
            .map_or_else(|| "0000".to_owned(), |mv| mv.to_string());
        lines.push(format!("bestmove {bestmove}"));
        lines
    }

    fn build_search_request(
        &self,
        options: GoOptions,
        stop_flag: Option<Arc<AtomicBool>>,
    ) -> Result<SearchRequest, String> {
        let now = Instant::now();
        if options.infinite {
            if stop_flag.is_none() {
                return Err("go infinite requires the stdio runtime".to_owned());
            }

            return Ok(SearchRequest {
                limits: SearchLimits::new(MAX_GO_DEPTH),
                soft_deadline: None,
                hard_deadline: None,
                stop_flag,
            });
        }

        if let Some(movetime_ms) = options.movetime_ms {
            let deadline = now + Duration::from_millis(movetime_ms);
            return Ok(SearchRequest {
                limits: SearchLimits::new(MAX_GO_DEPTH),
                soft_deadline: Some(deadline),
                hard_deadline: Some(deadline),
                stop_flag,
            });
        }

        if options.wtime_ms.is_some() || options.btime_ms.is_some() {
            let (soft_ms, hard_ms) = self.clock_budget_ms(options)?;
            return Ok(SearchRequest {
                limits: SearchLimits::new(MAX_GO_DEPTH),
                soft_deadline: Some(now + Duration::from_millis(soft_ms)),
                hard_deadline: Some(now + Duration::from_millis(hard_ms)),
                stop_flag,
            });
        }

        Ok(SearchRequest {
            limits: SearchLimits::new(options.depth.unwrap_or(DEFAULT_GO_DEPTH)),
            soft_deadline: None,
            hard_deadline: None,
            stop_flag,
        })
    }

    fn clock_budget_ms(&self, options: GoOptions) -> Result<(u64, u64), String> {
        let side = self.position.side_to_move();
        let (remaining, increment) = match side {
            Color::White => (
                options
                    .wtime_ms
                    .ok_or_else(|| "go clock mode requires both wtime and btime".to_owned())?,
                options.winc_ms,
            ),
            Color::Black => (
                options
                    .btime_ms
                    .ok_or_else(|| "go clock mode requires both wtime and btime".to_owned())?,
                options.binc_ms,
            ),
        };

        if options.wtime_ms.is_none() || options.btime_ms.is_none() {
            return Err("go clock mode requires both wtime and btime".to_owned());
        }

        let safety = 25u64.max(remaining / 50);
        let available = remaining.saturating_sub(safety);
        let moves_to_go = options.movestogo.unwrap_or(25).clamp(1, 40) as u64;
        let soft = available.min(available / moves_to_go + (3 * increment / 4));
        let hard = available.min(soft + 25u64.max(soft / 2));
        Ok((soft, hard))
    }
}

impl Default for UciEngine {
    fn default() -> Self {
        Self::new()
    }
}

fn parse_go(tokens: &[&str]) -> Result<GoOptions, String> {
    let mut options = GoOptions::default();
    let mut index = 1usize;
    while index < tokens.len() {
        match tokens[index] {
            "depth" => {
                let Some(value) = tokens.get(index + 1) else {
                    return Err("go depth requires a value".to_owned());
                };
                options.depth = Some(parse_depth(value)?);
                index += 2;
            }
            "movetime" => {
                let Some(value) = tokens.get(index + 1) else {
                    return Err("go movetime requires a value".to_owned());
                };
                options.movetime_ms = Some(parse_u64_arg(value, "go movetime")?);
                index += 2;
            }
            "wtime" => {
                let Some(value) = tokens.get(index + 1) else {
                    return Err("go wtime requires a value".to_owned());
                };
                options.wtime_ms = Some(parse_u64_arg(value, "go wtime")?);
                index += 2;
            }
            "btime" => {
                let Some(value) = tokens.get(index + 1) else {
                    return Err("go btime requires a value".to_owned());
                };
                options.btime_ms = Some(parse_u64_arg(value, "go btime")?);
                index += 2;
            }
            "winc" => {
                let Some(value) = tokens.get(index + 1) else {
                    return Err("go winc requires a value".to_owned());
                };
                options.winc_ms = parse_u64_arg(value, "go winc")?;
                index += 2;
            }
            "binc" => {
                let Some(value) = tokens.get(index + 1) else {
                    return Err("go binc requires a value".to_owned());
                };
                options.binc_ms = parse_u64_arg(value, "go binc")?;
                index += 2;
            }
            "movestogo" => {
                let Some(value) = tokens.get(index + 1) else {
                    return Err("go movestogo requires a value".to_owned());
                };
                options.movestogo = Some(parse_u32_arg(value, "go movestogo")?);
                index += 2;
            }
            "infinite" => {
                options.infinite = true;
                index += 1;
            }
            "ponder" | "ponderhit" | "searchmoves" | "nodes" | "mate" => {
                return Err(format!("unsupported go argument '{}'", tokens[index]));
            }
            other => {
                return Err(format!("unsupported go argument '{other}'"));
            }
        }
    }

    let explicit_depth = options.depth.is_some();
    let has_time_mode =
        options.movetime_ms.is_some() || options.wtime_ms.is_some() || options.btime_ms.is_some();
    if options.infinite && (explicit_depth || has_time_mode) {
        return Err("go infinite cannot be combined with depth or time controls".to_owned());
    }
    if options.movetime_ms.is_some()
        && (explicit_depth || options.wtime_ms.is_some() || options.btime_ms.is_some())
    {
        return Err("go movetime cannot be combined with depth or clock controls".to_owned());
    }
    if (options.wtime_ms.is_some() || options.btime_ms.is_some()) && explicit_depth {
        return Err("go clock mode cannot be combined with depth".to_owned());
    }

    Ok(options)
}

fn parse_setoption(tokens: &[&str]) -> Result<SetOptionCommand, String> {
    if tokens.len() < 3 || tokens[1] != "name" {
        return Err("setoption requires 'name'".to_owned());
    }

    let value_index = tokens.iter().position(|token| *token == "value");
    let name_tokens = match value_index {
        Some(index) => &tokens[2..index],
        None => &tokens[2..],
    };
    if name_tokens.is_empty() {
        return Err("setoption requires an option name".to_owned());
    }

    let name = name_tokens.join(" ");
    match name.as_str() {
        "Hash" => {
            let Some(value_index) = value_index else {
                return Err("setoption name Hash requires 'value <mb>'".to_owned());
            };
            if value_index + 2 != tokens.len() {
                return Err("setoption name Hash requires exactly one value".to_owned());
            }
            let hash_mb = parse_usize_arg(tokens[value_index + 1], "setoption name Hash value")?;
            if !(MIN_HASH_MB..=MAX_HASH_MB).contains(&hash_mb) {
                return Err(format!(
                    "Hash value must be between {MIN_HASH_MB} and {MAX_HASH_MB}"
                ));
            }
            Ok(SetOptionCommand::Hash(hash_mb))
        }
        "Clear Hash" => {
            if value_index.is_some() {
                return Err("setoption name Clear Hash does not take a value".to_owned());
            }
            Ok(SetOptionCommand::ClearHash)
        }
        "Threads" => {
            let Some(value_index) = value_index else {
                return Err("setoption name Threads requires 'value <n>'".to_owned());
            };
            if value_index + 2 != tokens.len() {
                return Err("setoption name Threads requires exactly one value".to_owned());
            }
            let threads = parse_usize_arg(tokens[value_index + 1], "setoption name Threads value")?;
            if !(MIN_THREADS..=MAX_THREADS).contains(&threads) {
                return Err(format!(
                    "Threads value must be between {MIN_THREADS} and {MAX_THREADS}"
                ));
            }
            Ok(SetOptionCommand::Threads(threads))
        }
        _ => Err(format!("unsupported option '{name}'")),
    }
}

fn parse_depth(value: &str) -> Result<u8, String> {
    parse_u32_arg(value, "go depth").map(|depth| depth.clamp(1, MAX_GO_DEPTH as u32) as u8)
}

fn parse_u32_arg(value: &str, label: &str) -> Result<u32, String> {
    value
        .parse::<u32>()
        .map_err(|_| format!("invalid {label} value '{value}'"))
}

fn parse_u64_arg(value: &str, label: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|_| format!("invalid {label} value '{value}'"))
}

fn parse_usize_arg(value: &str, label: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("invalid {label} value '{value}'"))
}

fn run_runtime_session<W: Write>(
    engine: &mut UciEngine,
    receiver: &Receiver<RuntimeInput>,
    output: &mut W,
    stop_flag: &Arc<AtomicBool>,
    quit_flag: &AtomicBool,
) -> io::Result<()> {
    while let Ok(message) = receiver.recv() {
        match message {
            RuntimeInput::Command(line) => {
                let response = engine.handle_line_with_stop(&line, Some(Arc::clone(stop_flag)));
                let suppress_output =
                    quit_flag.load(Ordering::Relaxed) && line.trim_start().starts_with("go");
                if !suppress_output {
                    for output_line in response.lines {
                        writeln!(output, "{output_line}")?;
                    }
                    output.flush()?;
                }

                if response.should_quit || quit_flag.load(Ordering::Relaxed) {
                    break;
                }
            }
            RuntimeInput::QuitRequested => break,
        }
    }

    Ok(())
}

fn handle_input_line(
    line: String,
    sender: &Sender<RuntimeInput>,
    stop_flag: &AtomicBool,
    quit_flag: &AtomicBool,
) -> io::Result<bool> {
    match line.trim() {
        "stop" => {
            stop_flag.store(true, Ordering::Relaxed);
            Ok(true)
        }
        "quit" => {
            stop_flag.store(true, Ordering::Relaxed);
            quit_flag.store(true, Ordering::Relaxed);
            sender.send(RuntimeInput::QuitRequested).map_err(|error| {
                io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    format!("uci runtime send failed: {error}"),
                )
            })?;
            Ok(false)
        }
        _ => {
            sender.send(RuntimeInput::Command(line)).map_err(|error| {
                io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    format!("uci runtime send failed: {error}"),
                )
            })?;
            Ok(true)
        }
    }
}

pub fn run_stdio() -> io::Result<()> {
    let (sender, receiver) = mpsc::channel();
    let stop_flag = Arc::new(AtomicBool::new(false));
    let quit_flag = Arc::new(AtomicBool::new(false));
    let helper_sender = sender.clone();
    let helper_stop = Arc::clone(&stop_flag);
    let helper_quit = Arc::clone(&quit_flag);

    let helper = thread::spawn(move || -> io::Result<()> {
        let stdin = io::stdin();
        for line_result in stdin.lock().lines() {
            let line = line_result?;
            if !handle_input_line(line, &helper_sender, &helper_stop, &helper_quit)? {
                break;
            }
        }
        Ok(())
    });

    drop(sender);

    let stdout = io::stdout();
    let mut output = io::BufWriter::new(stdout.lock());
    let mut engine = UciEngine::new();
    let runtime_result = run_runtime_session(
        &mut engine,
        &receiver,
        &mut output,
        &stop_flag,
        quit_flag.as_ref(),
    );

    let helper_result = helper
        .join()
        .map_err(|_| io::Error::other("uci input helper thread panicked"))?;

    runtime_result?;
    helper_result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::mpsc, thread, time::Duration};

    fn interrupted_position_fen() -> &'static str {
        "r2q1rk1/ppp2ppp/2npbn2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8"
    }

    fn make_runtime_engine() -> UciEngine {
        let mut engine = UciEngine::new();
        let response = engine.handle_line(&format!("position fen {}", interrupted_position_fen()));
        assert!(response.lines.is_empty());
        engine
    }

    fn make_runtime_engine_with_threads(threads: usize) -> UciEngine {
        let mut engine = make_runtime_engine();
        let response = engine.handle_line(&format!("setoption name Threads value {threads}"));
        assert!(response.lines.is_empty());
        engine
    }

    fn run_with_external_stop(engine: &mut UciEngine, command: &str, delay_ms: u64) -> UciResponse {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stopper = {
            let stop_flag = Arc::clone(&stop_flag);
            thread::spawn(move || {
                thread::sleep(Duration::from_millis(delay_ms));
                stop_flag.store(true, Ordering::Relaxed);
            })
        };

        let response = engine.handle_line_with_stop(command, Some(Arc::clone(&stop_flag)));
        stopper.join().expect("stop helper must join");
        response
    }

    #[test]
    fn go_infinite_requires_runtime_stop_path() {
        let mut engine = UciEngine::new();
        let response = engine.handle_line("go infinite");
        assert!(
            response
                .lines
                .iter()
                .any(|line| line.contains("go infinite requires the stdio runtime"))
        );
    }

    #[test]
    fn hard_deadline_stop_leaves_root_position_unchanged() {
        let mut engine = make_runtime_engine();
        let before = engine.position().to_fen();
        let before_search_key = engine.position().debug_search_key();
        let before_history = engine.position().debug_repetition_history_snapshot();

        let response = engine.handle_line("go movetime 0");
        assert!(
            response
                .lines
                .iter()
                .any(|line| line.starts_with("bestmove "))
        );

        assert_eq!(engine.position().to_fen(), before);
        assert_eq!(engine.position().debug_search_key(), before_search_key);
        assert_eq!(
            engine.position().debug_repetition_history_snapshot(),
            before_history
        );
        engine
            .position()
            .validate()
            .expect("position must remain valid");
    }

    #[test]
    fn external_stop_leaves_root_position_unchanged() {
        let mut engine = make_runtime_engine();
        let before = engine.position().to_fen();
        let before_search_key = engine.position().debug_search_key();
        let before_history = engine.position().debug_repetition_history_snapshot();

        let response = run_with_external_stop(&mut engine, "go infinite", 10);
        assert!(
            response
                .lines
                .iter()
                .any(|line| line.starts_with("bestmove "))
        );

        assert_eq!(engine.position().to_fen(), before);
        assert_eq!(engine.position().debug_search_key(), before_search_key);
        assert_eq!(
            engine.position().debug_repetition_history_snapshot(),
            before_history
        );
        engine
            .position()
            .validate()
            .expect("position must remain valid");
    }

    #[test]
    fn threaded_external_stop_leaves_root_position_unchanged() {
        let mut engine = make_runtime_engine_with_threads(2);
        let before = engine.position().to_fen();
        let before_search_key = engine.position().debug_search_key();
        let before_history = engine.position().debug_repetition_history_snapshot();

        let response = run_with_external_stop(&mut engine, "go infinite", 10);
        assert!(
            response
                .lines
                .iter()
                .any(|line| line.starts_with("bestmove "))
        );

        assert_eq!(engine.position().to_fen(), before);
        assert_eq!(engine.position().debug_search_key(), before_search_key);
        assert_eq!(
            engine.position().debug_repetition_history_snapshot(),
            before_history
        );
        assert_eq!(engine.debug_active_helper_count(), 0);
        engine
            .position()
            .validate()
            .expect("threaded position must remain valid");
    }

    #[test]
    fn interrupted_search_leaves_tt_service_valid_for_next_command() {
        let mut engine = make_runtime_engine();
        let interrupted = run_with_external_stop(&mut engine, "go infinite", 10);
        assert!(
            interrupted
                .lines
                .iter()
                .any(|line| line.starts_with("bestmove "))
        );

        let follow_up = engine.handle_line("go depth 2");
        assert!(
            follow_up
                .lines
                .iter()
                .any(|line| line.starts_with("bestmove "))
        );
        assert!(engine.debug_tt_entry_count() > 0);
    }

    #[test]
    fn helper_stop_and_quit_are_immediate_commands() {
        let (sender, receiver) = mpsc::channel();
        let stop_flag = AtomicBool::new(false);
        let quit_flag = AtomicBool::new(false);

        assert!(
            handle_input_line("stop".to_owned(), &sender, &stop_flag, &quit_flag)
                .expect("stop handling must succeed")
        );
        assert!(stop_flag.load(Ordering::Relaxed));
        assert!(receiver.try_recv().is_err());

        assert!(
            !handle_input_line("quit".to_owned(), &sender, &stop_flag, &quit_flag)
                .expect("quit handling must succeed")
        );
        assert!(quit_flag.load(Ordering::Relaxed));
        assert!(matches!(
            receiver.try_recv().expect("quit wakeup must be queued"),
            RuntimeInput::QuitRequested
        ));
    }

    #[test]
    fn position_received_during_search_does_not_mutate_live_search_state_mid_search() {
        let (sender, receiver) = mpsc::channel();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let quit_flag = Arc::new(AtomicBool::new(false));

        let helper = {
            let sender = sender.clone();
            let stop_flag = Arc::clone(&stop_flag);
            let quit_flag = Arc::clone(&quit_flag);
            thread::spawn(move || -> io::Result<()> {
                handle_input_line(
                    format!("position fen {}", interrupted_position_fen()),
                    &sender,
                    stop_flag.as_ref(),
                    quit_flag.as_ref(),
                )?;
                handle_input_line(
                    "go infinite".to_owned(),
                    &sender,
                    stop_flag.as_ref(),
                    quit_flag.as_ref(),
                )?;
                thread::sleep(Duration::from_millis(15));
                handle_input_line(
                    "position startpos moves e2e4".to_owned(),
                    &sender,
                    stop_flag.as_ref(),
                    quit_flag.as_ref(),
                )?;
                thread::sleep(Duration::from_millis(10));
                handle_input_line(
                    "stop".to_owned(),
                    &sender,
                    stop_flag.as_ref(),
                    quit_flag.as_ref(),
                )?;
                Ok(())
            })
        };

        drop(sender);

        let mut output = Vec::new();
        let mut engine = UciEngine::new();
        run_runtime_session(
            &mut engine,
            &receiver,
            &mut output,
            &stop_flag,
            quit_flag.as_ref(),
        )
        .expect("runtime must complete");
        helper
            .join()
            .expect("helper thread must join")
            .expect("helper must succeed");

        let output_text = String::from_utf8(output).expect("runtime output must be utf8");
        assert_eq!(output_text.matches("bestmove ").count(), 1);
        assert_eq!(
            engine.position().to_fen(),
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        );
    }

    #[test]
    fn setoption_hash_received_during_search_takes_effect_only_after_stop_unwind() {
        let (sender, receiver) = mpsc::channel();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let quit_flag = Arc::new(AtomicBool::new(false));

        let helper = {
            let sender = sender.clone();
            let stop_flag = Arc::clone(&stop_flag);
            let quit_flag = Arc::clone(&quit_flag);
            thread::spawn(move || -> io::Result<()> {
                handle_input_line(
                    format!("position fen {}", interrupted_position_fen()),
                    &sender,
                    stop_flag.as_ref(),
                    quit_flag.as_ref(),
                )?;
                handle_input_line(
                    "go infinite".to_owned(),
                    &sender,
                    stop_flag.as_ref(),
                    quit_flag.as_ref(),
                )?;
                thread::sleep(Duration::from_millis(15));
                handle_input_line(
                    "setoption name Hash value 32".to_owned(),
                    &sender,
                    stop_flag.as_ref(),
                    quit_flag.as_ref(),
                )?;
                thread::sleep(Duration::from_millis(10));
                handle_input_line(
                    "stop".to_owned(),
                    &sender,
                    stop_flag.as_ref(),
                    quit_flag.as_ref(),
                )?;
                Ok(())
            })
        };

        drop(sender);

        let mut output = Vec::new();
        let mut engine = UciEngine::new();
        let original_hash = engine.debug_hash_mb();
        run_runtime_session(
            &mut engine,
            &receiver,
            &mut output,
            &stop_flag,
            quit_flag.as_ref(),
        )
        .expect("runtime must complete");
        helper
            .join()
            .expect("helper thread must join")
            .expect("helper must succeed");

        assert_eq!(original_hash, MIN_HASH_MB.max(16));
        assert_eq!(engine.debug_hash_mb(), 32);
        assert_eq!(
            String::from_utf8(output)
                .expect("utf8")
                .matches("bestmove ")
                .count(),
            1
        );
    }

    #[test]
    fn setoption_threads_received_during_search_takes_effect_only_after_stop_unwind() {
        let (sender, receiver) = mpsc::channel();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let quit_flag = Arc::new(AtomicBool::new(false));

        let helper = {
            let sender = sender.clone();
            let stop_flag = Arc::clone(&stop_flag);
            let quit_flag = Arc::clone(&quit_flag);
            thread::spawn(move || -> io::Result<()> {
                handle_input_line(
                    format!("position fen {}", interrupted_position_fen()),
                    &sender,
                    stop_flag.as_ref(),
                    quit_flag.as_ref(),
                )?;
                handle_input_line(
                    "go infinite".to_owned(),
                    &sender,
                    stop_flag.as_ref(),
                    quit_flag.as_ref(),
                )?;
                thread::sleep(Duration::from_millis(15));
                handle_input_line(
                    "setoption name Threads value 2".to_owned(),
                    &sender,
                    stop_flag.as_ref(),
                    quit_flag.as_ref(),
                )?;
                thread::sleep(Duration::from_millis(10));
                handle_input_line(
                    "stop".to_owned(),
                    &sender,
                    stop_flag.as_ref(),
                    quit_flag.as_ref(),
                )?;
                Ok(())
            })
        };

        drop(sender);

        let mut output = Vec::new();
        let mut engine = UciEngine::new();
        assert_eq!(engine.debug_threads(), 1);
        run_runtime_session(
            &mut engine,
            &receiver,
            &mut output,
            &stop_flag,
            quit_flag.as_ref(),
        )
        .expect("runtime must complete");
        helper
            .join()
            .expect("helper thread must join")
            .expect("helper must succeed");

        assert_eq!(engine.debug_threads(), 2);
        assert_eq!(
            String::from_utf8(output)
                .expect("utf8")
                .matches("bestmove ")
                .count(),
            1
        );
    }

    #[test]
    fn setoption_clear_hash_received_during_search_takes_effect_only_after_stop_unwind() {
        let (sender, receiver) = mpsc::channel();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let quit_flag = Arc::new(AtomicBool::new(false));

        let helper = {
            let sender = sender.clone();
            let stop_flag = Arc::clone(&stop_flag);
            let quit_flag = Arc::clone(&quit_flag);
            thread::spawn(move || -> io::Result<()> {
                handle_input_line(
                    format!("position fen {}", interrupted_position_fen()),
                    &sender,
                    stop_flag.as_ref(),
                    quit_flag.as_ref(),
                )?;
                handle_input_line(
                    "go depth 1".to_owned(),
                    &sender,
                    stop_flag.as_ref(),
                    quit_flag.as_ref(),
                )?;
                handle_input_line(
                    "go infinite".to_owned(),
                    &sender,
                    stop_flag.as_ref(),
                    quit_flag.as_ref(),
                )?;
                thread::sleep(Duration::from_millis(15));
                handle_input_line(
                    "setoption name Clear Hash".to_owned(),
                    &sender,
                    stop_flag.as_ref(),
                    quit_flag.as_ref(),
                )?;
                thread::sleep(Duration::from_millis(10));
                handle_input_line(
                    "stop".to_owned(),
                    &sender,
                    stop_flag.as_ref(),
                    quit_flag.as_ref(),
                )?;
                Ok(())
            })
        };

        drop(sender);

        let mut output = Vec::new();
        let mut engine = UciEngine::new();
        run_runtime_session(
            &mut engine,
            &receiver,
            &mut output,
            &stop_flag,
            quit_flag.as_ref(),
        )
        .expect("runtime must complete");
        helper
            .join()
            .expect("helper thread must join")
            .expect("helper must succeed");

        assert_eq!(engine.debug_tt_entry_count(), 0);
        assert_eq!(
            String::from_utf8(output)
                .expect("utf8")
                .matches("bestmove ")
                .count(),
            2
        );
    }

    #[test]
    fn ucinewgame_received_during_search_is_deferred_until_after_search_termination() {
        let (sender, receiver) = mpsc::channel();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let quit_flag = Arc::new(AtomicBool::new(false));

        let helper = {
            let sender = sender.clone();
            let stop_flag = Arc::clone(&stop_flag);
            let quit_flag = Arc::clone(&quit_flag);
            thread::spawn(move || -> io::Result<()> {
                handle_input_line(
                    format!("position fen {}", interrupted_position_fen()),
                    &sender,
                    stop_flag.as_ref(),
                    quit_flag.as_ref(),
                )?;
                handle_input_line(
                    "go infinite".to_owned(),
                    &sender,
                    stop_flag.as_ref(),
                    quit_flag.as_ref(),
                )?;
                thread::sleep(Duration::from_millis(15));
                handle_input_line(
                    "ucinewgame".to_owned(),
                    &sender,
                    stop_flag.as_ref(),
                    quit_flag.as_ref(),
                )?;
                thread::sleep(Duration::from_millis(10));
                handle_input_line(
                    "stop".to_owned(),
                    &sender,
                    stop_flag.as_ref(),
                    quit_flag.as_ref(),
                )?;
                Ok(())
            })
        };

        drop(sender);

        let mut output = Vec::new();
        let mut engine = UciEngine::new();
        run_runtime_session(
            &mut engine,
            &receiver,
            &mut output,
            &stop_flag,
            quit_flag.as_ref(),
        )
        .expect("runtime must complete");
        helper
            .join()
            .expect("helper thread must join")
            .expect("helper must succeed");

        assert_eq!(engine.position().to_fen(), Position::startpos().to_fen());
        assert_eq!(engine.debug_tt_entry_count(), 0);
        assert_eq!(
            String::from_utf8(output)
                .expect("utf8")
                .matches("bestmove ")
                .count(),
            1
        );
    }

    #[test]
    fn quit_during_search_exits_cleanly_without_bestmove_output() {
        let (sender, receiver) = mpsc::channel();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let quit_flag = Arc::new(AtomicBool::new(false));

        let helper = {
            let sender = sender.clone();
            let stop_flag = Arc::clone(&stop_flag);
            let quit_flag = Arc::clone(&quit_flag);
            thread::spawn(move || -> io::Result<()> {
                handle_input_line(
                    format!("position fen {}", interrupted_position_fen()),
                    &sender,
                    stop_flag.as_ref(),
                    quit_flag.as_ref(),
                )?;
                handle_input_line(
                    "go infinite".to_owned(),
                    &sender,
                    stop_flag.as_ref(),
                    quit_flag.as_ref(),
                )?;
                thread::sleep(Duration::from_millis(15));
                handle_input_line(
                    "quit".to_owned(),
                    &sender,
                    stop_flag.as_ref(),
                    quit_flag.as_ref(),
                )?;
                Ok(())
            })
        };

        drop(sender);

        let mut output = Vec::new();
        let mut engine = UciEngine::new();
        run_runtime_session(
            &mut engine,
            &receiver,
            &mut output,
            &stop_flag,
            quit_flag.as_ref(),
        )
        .expect("runtime must complete");
        helper
            .join()
            .expect("helper thread must join")
            .expect("helper must succeed");

        assert!(
            !String::from_utf8(output)
                .expect("utf8")
                .contains("bestmove ")
        );
        assert!(quit_flag.load(Ordering::Relaxed));
    }

    #[test]
    fn threaded_quit_during_search_exits_cleanly_without_bestmove_output() {
        let (sender, receiver) = mpsc::channel();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let quit_flag = Arc::new(AtomicBool::new(false));

        let helper = {
            let sender = sender.clone();
            let stop_flag = Arc::clone(&stop_flag);
            let quit_flag = Arc::clone(&quit_flag);
            thread::spawn(move || -> io::Result<()> {
                handle_input_line(
                    format!("position fen {}", interrupted_position_fen()),
                    &sender,
                    stop_flag.as_ref(),
                    quit_flag.as_ref(),
                )?;
                handle_input_line(
                    "setoption name Threads value 2".to_owned(),
                    &sender,
                    stop_flag.as_ref(),
                    quit_flag.as_ref(),
                )?;
                handle_input_line(
                    "go infinite".to_owned(),
                    &sender,
                    stop_flag.as_ref(),
                    quit_flag.as_ref(),
                )?;
                thread::sleep(Duration::from_millis(15));
                handle_input_line(
                    "quit".to_owned(),
                    &sender,
                    stop_flag.as_ref(),
                    quit_flag.as_ref(),
                )?;
                Ok(())
            })
        };

        drop(sender);

        let mut output = Vec::new();
        let mut engine = UciEngine::new();
        run_runtime_session(
            &mut engine,
            &receiver,
            &mut output,
            &stop_flag,
            quit_flag.as_ref(),
        )
        .expect("runtime must complete");
        helper
            .join()
            .expect("helper thread must join")
            .expect("helper must succeed");

        assert!(
            !String::from_utf8(output)
                .expect("utf8")
                .contains("bestmove ")
        );
        assert_eq!(engine.debug_active_helper_count(), 0);
        assert!(quit_flag.load(Ordering::Relaxed));
    }

    #[test]
    fn movetime_uses_equal_soft_and_hard_deadlines() {
        let engine = UciEngine::new();
        let request = engine
            .build_search_request(
                GoOptions {
                    movetime_ms: Some(25),
                    ..GoOptions::default()
                },
                None,
            )
            .expect("movetime request must build");

        assert_eq!(request.soft_deadline, request.hard_deadline);
    }

    #[test]
    fn clock_budget_uses_sudden_death_defaults() {
        let engine = UciEngine::new();
        let (soft, hard) = engine
            .clock_budget_ms(GoOptions {
                wtime_ms: Some(1_000),
                btime_ms: Some(1_000),
                ..GoOptions::default()
            })
            .expect("clock budget must build");

        assert_eq!(soft, 39);
        assert_eq!(hard, 64);
    }

    #[test]
    fn clock_budget_honors_movestogo_and_increment() {
        let engine = UciEngine::new();
        let (soft, hard) = engine
            .clock_budget_ms(GoOptions {
                wtime_ms: Some(5_000),
                btime_ms: Some(5_000),
                winc_ms: 1_000,
                movestogo: Some(10),
                ..GoOptions::default()
            })
            .expect("clock budget must build");

        assert_eq!(soft, 1_240);
        assert_eq!(hard, 1_860);
    }

    #[test]
    fn clock_budget_keeps_low_time_safety_floor() {
        let engine = UciEngine::new();
        let (soft, hard) = engine
            .clock_budget_ms(GoOptions {
                wtime_ms: Some(20),
                btime_ms: Some(20),
                ..GoOptions::default()
            })
            .expect("clock budget must build");

        assert_eq!(soft, 0);
        assert_eq!(hard, 0);
    }
}
