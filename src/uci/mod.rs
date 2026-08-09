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

#[cfg(any(test, debug_assertions, feature = "internal-testing"))]
use std::path::Path;

use crate::{
    ENGINE_AUTHOR, ENGINE_NAME, VERSION,
    core::{Color, Move, MoveList, ParsedMove, Position},
    search::{
        PonderState, SearchLimits,
        service::{
            DEFAULT_THREADS, DualEvalPolicy, MAX_DUAL_EVAL_THRESHOLD, SearchRequest,
            UciSearchService,
        },
        tablebase::MAX_SYZYGY_PIECES,
    },
};

#[cfg(feature = "spsa-tuning")]
use crate::search::parameters::{
    PARAMETER_SPECS, SearchParameters, manifest_lines, parameter_spec,
};

const DEFAULT_GO_DEPTH: u8 = 1;
const MAX_GO_DEPTH: u8 = 127;
const MIN_HASH_MB: usize = 1;
const MAX_HASH_MB: usize = 512;
const MIN_THREADS: usize = 1;
const MAX_THREADS: usize = 64;
const DEFAULT_MOVE_OVERHEAD_MS: u64 = 10;
const MAX_MOVE_OVERHEAD_MS: u64 = 5_000;

pub struct UciResponse {
    pub lines: Vec<String>,
    pub should_quit: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct GoOptions {
    depth: Option<u8>,
    nodes: Option<u64>,
    movetime_ms: Option<u64>,
    wtime_ms: Option<u64>,
    btime_ms: Option<u64>,
    winc_ms: u64,
    binc_ms: u64,
    movestogo: Option<u32>,
    infinite: bool,
    ponder: bool,
    searchmoves: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SetOptionCommand {
    Hash(usize),
    ClearHash,
    Threads(usize),
    MoveOverhead(u64),
    Ponder(bool),
    SyzygyPath,
    SyzygyProbeLimit(u8),
    Syzygy50MoveRule(bool),
    EvalFile,
    SmallEvalFile,
    DualEvalPolicy(DualEvalPolicy),
    DualEvalThreshold(i32),
    #[cfg(feature = "spsa-tuning")]
    TuneParameter(&'static str, i32),
    #[cfg(feature = "spsa-tuning")]
    TuneManifest,
}

enum RuntimeInput {
    Command(String, Option<Arc<PonderState>>),
    PonderHitRequested,
    StopRequested,
    QuitRequested,
}

pub struct UciEngine {
    position: Position,
    search_service: UciSearchService,
    move_overhead_ms: u64,
    ponder_enabled: bool,
    #[cfg(feature = "spsa-tuning")]
    search_parameters: SearchParameters,
}

impl UciEngine {
    pub fn new() -> Self {
        Self {
            position: Position::startpos(),
            search_service: UciSearchService::new(),
            move_overhead_ms: DEFAULT_MOVE_OVERHEAD_MS,
            ponder_enabled: false,
            #[cfg(feature = "spsa-tuning")]
            search_parameters: SearchParameters::DEFAULT,
        }
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    #[doc(hidden)]
    pub fn debug_new_with_eval_discovery(
        environment_path: Option<&str>,
        executable_path: Option<&Path>,
    ) -> Self {
        Self {
            position: Position::startpos(),
            search_service: UciSearchService::new_with_eval_discovery(
                environment_path,
                executable_path,
            ),
            move_overhead_ms: DEFAULT_MOVE_OVERHEAD_MS,
            ponder_enabled: false,
            #[cfg(feature = "spsa-tuning")]
            search_parameters: SearchParameters::DEFAULT,
        }
    }

    pub fn position(&self) -> &Position {
        &self.position
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    #[doc(hidden)]
    pub fn debug_hash_mb(&self) -> usize {
        self.search_service.hash_mb()
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    #[doc(hidden)]
    pub fn debug_threads(&self) -> usize {
        self.search_service.threads()
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    #[doc(hidden)]
    pub fn debug_syzygy_path(&self) -> &str {
        self.search_service.syzygy_path()
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    #[doc(hidden)]
    pub fn debug_syzygy_probe_limit(&self) -> u8 {
        self.search_service.syzygy_probe_limit()
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    #[doc(hidden)]
    pub fn debug_syzygy_50_move_rule(&self) -> bool {
        self.search_service.syzygy_50_move_rule()
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    #[doc(hidden)]
    pub fn debug_eval_file(&self) -> &str {
        self.search_service.eval_file()
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    #[doc(hidden)]
    pub fn debug_small_eval_file(&self) -> &str {
        self.search_service.small_eval_file()
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    #[doc(hidden)]
    pub fn debug_dual_eval_config(&self) -> (&'static str, i32) {
        (
            self.search_service.dual_eval_policy().as_str(),
            self.search_service.dual_eval_threshold(),
        )
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    #[doc(hidden)]
    pub fn debug_dual_eval_counters(&self) -> (u64, u64) {
        let counters = self.search_service.dual_eval_counters();
        (counters.small_selected, counters.big_fallbacks)
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    #[doc(hidden)]
    pub fn debug_tt_entry_count(&self) -> usize {
        self.search_service.debug_tt_entry_count()
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    #[doc(hidden)]
    pub fn debug_worker_count(&self) -> usize {
        self.search_service.debug_worker_count()
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    #[doc(hidden)]
    pub fn debug_active_helper_count(&self) -> usize {
        self.search_service.debug_active_helper_count()
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    #[doc(hidden)]
    pub fn debug_move_overhead_ms(&self) -> u64 {
        self.move_overhead_ms
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    #[doc(hidden)]
    pub fn debug_ponder_enabled(&self) -> bool {
        self.ponder_enabled
    }

    pub fn handle_line(&mut self, line: &str) -> UciResponse {
        self.handle_line_with_stop_and_info(line, None, None)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn handle_line_with_stop(
        &mut self,
        line: &str,
        stop_flag: Option<Arc<AtomicBool>>,
    ) -> UciResponse {
        self.handle_line_with_stop_and_info(line, stop_flag, None)
    }

    fn handle_line_with_stop_and_info(
        &mut self,
        line: &str,
        stop_flag: Option<Arc<AtomicBool>>,
        info_reporter: Option<&mut dyn FnMut(&str)>,
    ) -> UciResponse {
        self.handle_line_with_runtime_ponder(line, stop_flag, info_reporter, None)
    }

    fn handle_line_with_runtime_ponder(
        &mut self,
        line: &str,
        stop_flag: Option<Arc<AtomicBool>>,
        info_reporter: Option<&mut dyn FnMut(&str)>,
        ponder_state: Option<Arc<PonderState>>,
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
                lines: {
                    let mut lines = vec![
                        format!("id name {ENGINE_NAME} {VERSION}"),
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
                        format!(
                            "option name Move Overhead type spin default {DEFAULT_MOVE_OVERHEAD_MS} min 0 max {MAX_MOVE_OVERHEAD_MS}"
                        ),
                        "option name Ponder type check default false".to_owned(),
                        "option name SyzygyPath type string default".to_owned(),
                        format!(
                            "option name SyzygyProbeLimit type spin default {} min 0 max {MAX_SYZYGY_PIECES}",
                            self.search_service.syzygy_probe_limit()
                        ),
                        format!(
                            "option name Syzygy50MoveRule type check default {}",
                            self.search_service.syzygy_50_move_rule()
                        ),
                        if self.search_service.eval_file().is_empty() {
                            "option name EvalFile type string default".to_owned()
                        } else {
                            format!(
                                "option name EvalFile type string default {}",
                                self.search_service.eval_file()
                            )
                        },
                        if self.search_service.small_eval_file().is_empty() {
                            "option name SmallEvalFile type string default".to_owned()
                        } else {
                            format!(
                                "option name SmallEvalFile type string default {}",
                                self.search_service.small_eval_file()
                            )
                        },
                        format!(
                            "option name DualEvalPolicy type combo default {} var off var small-fallback",
                            self.search_service.dual_eval_policy().as_str()
                        ),
                        format!(
                            "option name DualEvalThreshold type spin default {} min 0 max {}",
                            self.search_service.dual_eval_threshold(),
                            MAX_DUAL_EVAL_THRESHOLD
                        ),
                        "option name Clear Hash type button".to_owned(),
                    ];
                    #[cfg(feature = "spsa-tuning")]
                    {
                        lines.extend(PARAMETER_SPECS.iter().map(|spec| {
                            format!(
                                "option name {} type spin default {} min {} max {}",
                                spec.name, spec.default, spec.min, spec.max
                            )
                        }));
                        lines.push("option name TuneManifest type button".to_owned());
                    }
                    if let Some(diagnostic) = self.search_service.eval_discovery_diagnostic() {
                        lines.push(format!(
                            "info string warning: {}",
                            diagnostic.replace(['\r', '\n'], " ")
                        ));
                    }
                    lines.push("uciok".to_owned());
                    lines
                },
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
                lines: self.handle_setoption(trimmed, &tokens),
                should_quit: false,
            },
            "go" => UciResponse {
                lines: self.handle_go(&tokens, stop_flag, info_reporter, ponder_state),
                should_quit: false,
            },
            "debug" => UciResponse {
                lines: handle_debug_command(&tokens),
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

    fn handle_setoption(&mut self, line: &str, tokens: &[&str]) -> Vec<String> {
        match parse_setoption(line, tokens) {
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
            Ok(SetOptionCommand::MoveOverhead(overhead_ms)) => {
                self.move_overhead_ms = overhead_ms;
                Vec::new()
            }
            Ok(SetOptionCommand::Ponder(enabled)) => {
                self.ponder_enabled = enabled;
                Vec::new()
            }
            Ok(SetOptionCommand::SyzygyPath) => {
                let path = parse_syzygy_path_value(line);
                match self.search_service.set_syzygy_path(&path) {
                    Ok(()) if path.is_empty() => {
                        vec!["info string syzygy disabled".to_owned()]
                    }
                    Ok(()) => {
                        let loaded = self
                            .search_service
                            .syzygy_loaded_cardinality()
                            .map_or_else(|| "unknown".to_owned(), |pieces| pieces.to_string());
                        vec![format!(
                            "info string syzygy loaded max_pieces {loaded} path {}",
                            path.replace(['\r', '\n'], " ")
                        )]
                    }
                    Err(error) => vec![format!("info string error: {error}")],
                }
            }
            Ok(SetOptionCommand::SyzygyProbeLimit(limit)) => {
                self.search_service.set_syzygy_probe_limit(limit);
                Vec::new()
            }
            Ok(SetOptionCommand::Syzygy50MoveRule(enabled)) => {
                self.search_service.set_syzygy_50_move_rule(enabled);
                Vec::new()
            }
            Ok(SetOptionCommand::EvalFile) => {
                let path = parse_eval_file_value(line);
                match self.search_service.set_eval_file(&path) {
                    Ok(()) => Vec::new(),
                    Err(error) => vec![format!("info string error: {error}")],
                }
            }
            Ok(SetOptionCommand::SmallEvalFile) => {
                let path = parse_eval_file_value(line);
                match self.search_service.set_small_eval_file(&path) {
                    Ok(()) => Vec::new(),
                    Err(error) => vec![format!("info string error: {error}")],
                }
            }
            Ok(SetOptionCommand::DualEvalPolicy(policy)) => {
                match self.search_service.set_dual_eval_policy(policy) {
                    Ok(()) => Vec::new(),
                    Err(error) => vec![format!("info string error: {error}")],
                }
            }
            Ok(SetOptionCommand::DualEvalThreshold(threshold)) => {
                match self.search_service.set_dual_eval_threshold(threshold) {
                    Ok(()) => Vec::new(),
                    Err(error) => vec![format!("info string error: {error}")],
                }
            }
            #[cfg(feature = "spsa-tuning")]
            Ok(SetOptionCommand::TuneParameter(name, value)) => {
                match self.search_parameters.set(name, value) {
                    Ok(()) => {
                        // Bounds and static-evaluation entries from a different
                        // parameter vector must never leak into the next run.
                        self.search_service.clear_hash();
                        Vec::new()
                    }
                    Err(error) => vec![format!("info string error: {error}")],
                }
            }
            #[cfg(feature = "spsa-tuning")]
            Ok(SetOptionCommand::TuneManifest) => manifest_lines(self.search_parameters),
            Err(error) => vec![format!("info string error: {error}")],
        }
    }

    fn handle_go(
        &mut self,
        tokens: &[&str],
        stop_flag: Option<Arc<AtomicBool>>,
        info_reporter: Option<&mut dyn FnMut(&str)>,
        ponder_state: Option<Arc<PonderState>>,
    ) -> Vec<String> {
        let options = match parse_go(tokens) {
            Ok(options) => options,
            Err(error) => return vec![format!("info string error: {error}")],
        };
        let is_ponder = options.ponder;
        if is_ponder {
            let Some(state) = ponder_state.as_ref() else {
                return vec!["info string error: go ponder requires the stdio runtime".to_owned()];
            };
            state.arm(Instant::now());
        }
        let ponder_waiter = ponder_state.clone();
        let stop_waiter = stop_flag.clone();

        let request = match self.build_search_request(options, stop_flag) {
            Ok(request) => request,
            Err(error) => return vec![format!("info string error: {error}")],
        };

        let live_info = info_reporter.is_some();
        let mut last_live_info = None;
        let syzygy_before = self.search_service.syzygy_probe_stats();
        let result = if let Some(reporter) = info_reporter {
            self.search_service.search_with_info_and_ponder(
                &mut self.position,
                request,
                Some(Box::new(|line| {
                    last_live_info = Some(line.to_owned());
                    reporter(line);
                })),
                ponder_state,
            )
        } else {
            self.search_service.search_with_info_and_ponder(
                &mut self.position,
                request,
                None,
                ponder_state,
            )
        };
        if is_ponder && let Some(state) = ponder_waiter {
            // Terminal positions, tablebase hits, and a completed maximum-depth
            // search can finish before `ponderhit`. UCI still forbids emitting
            // bestmove until the GUI releases the ponder search.
            state.wait_until_released(stop_waiter.as_deref());
        }
        let mut lines = if live_info {
            Vec::new()
        } else {
            result.info_lines.clone()
        };
        if live_info
            && let Some(final_info) = result.info_lines.last()
            && last_live_info.as_deref() != Some(final_info.as_str())
        {
            // Helper statistics are known only after every worker has stopped. Emit one corrected
            // final line when aggregate nodes/TT hits/seldepth differ from the last live main-thread
            // iteration instead of leaving GUIs with incomplete SMP accounting.
            lines.push(final_info.clone());
        }
        let syzygy = self
            .search_service
            .syzygy_probe_stats()
            .delta_since(syzygy_before);
        if syzygy.attempts() != 0 {
            lines.push(format!(
                "info string syzygy probes {} root {} wdl {} hits {} misses {} errors {}",
                syzygy.attempts(),
                syzygy.root_attempts,
                syzygy.wdl_attempts,
                syzygy.hits,
                syzygy.misses,
                syzygy.errors
            ));
            if syzygy.errors != 0
                && let Some(error) = self.search_service.syzygy_last_probe_error()
            {
                lines.push(format!(
                    "info string syzygy last_error {}",
                    error.replace(['\r', '\n'], " ")
                ));
            }
        }
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
        let root_moves = self.resolve_searchmoves(&options.searchmoves)?;
        let depth_limit = options.depth.unwrap_or(if options.nodes.is_some() {
            MAX_GO_DEPTH
        } else {
            DEFAULT_GO_DEPTH
        });
        let limits = self
            .search_limits(depth_limit)
            .with_node_limit(options.nodes);
        if options.infinite {
            if stop_flag.is_none() {
                return Err("go infinite requires the stdio runtime".to_owned());
            }

            return Ok(SearchRequest {
                limits: self.search_limits(MAX_GO_DEPTH),
                soft_deadline: None,
                hard_deadline: None,
                stop_flag,
                root_moves,
            });
        }

        if options.ponder && stop_flag.is_none() {
            return Err("go ponder requires the stdio runtime".to_owned());
        }

        if let Some(movetime_ms) = options.movetime_ms {
            let search_time_ms = movetime_ms.saturating_sub(self.move_overhead_ms);
            let deadline = checked_deadline(now, search_time_ms, "go movetime")?;
            return Ok(SearchRequest {
                limits: self
                    .search_limits(MAX_GO_DEPTH)
                    .with_node_limit(options.nodes),
                soft_deadline: Some(deadline),
                hard_deadline: Some(deadline),
                stop_flag,
                root_moves,
            });
        }

        if options.wtime_ms.is_some() || options.btime_ms.is_some() {
            let (soft_ms, hard_ms) = self.clock_budget_ms(&options)?;
            return Ok(SearchRequest {
                limits: self
                    .search_limits(MAX_GO_DEPTH)
                    .with_node_limit(options.nodes),
                soft_deadline: Some(checked_deadline(now, soft_ms, "go clock soft limit")?),
                hard_deadline: Some(checked_deadline(now, hard_ms, "go clock hard limit")?),
                stop_flag,
                root_moves,
            });
        }

        Ok(SearchRequest {
            limits,
            soft_deadline: None,
            hard_deadline: None,
            stop_flag,
            root_moves,
        })
    }

    fn search_limits(&self, depth: u8) -> SearchLimits {
        let limits = SearchLimits::new(depth);
        #[cfg(feature = "spsa-tuning")]
        let limits = limits.with_parameters(self.search_parameters);
        limits
    }

    fn resolve_searchmoves(&self, searchmoves: &[String]) -> Result<Option<Vec<Move>>, String> {
        if searchmoves.is_empty() {
            return Ok(None);
        }

        let mut position = self.position.clone();
        let mut legal_moves = MoveList::new();
        position.generate_legal_moves(&mut legal_moves);

        let mut root_moves = Vec::with_capacity(searchmoves.len());
        for move_text in searchmoves {
            let parsed = ParsedMove::parse(move_text)
                .map_err(|_| format!("invalid go searchmoves move '{move_text}'"))?;
            let Some(mv) = legal_moves
                .as_slice()
                .iter()
                .copied()
                .find(|mv| mv.matches_parsed(parsed))
            else {
                return Err(format!("illegal go searchmoves move '{move_text}'"));
            };
            if !root_moves.contains(&mv) {
                root_moves.push(mv);
            }
        }

        Ok(Some(root_moves))
    }

    fn clock_budget_ms(&self, options: &GoOptions) -> Result<(u64, u64), String> {
        let side = self.position.side_to_move();
        let (remaining, increment) = match side {
            Color::White => (
                options
                    .wtime_ms
                    .or(options.btime_ms)
                    .ok_or_else(|| "go clock mode requires wtime or btime".to_owned())?,
                options.winc_ms,
            ),
            Color::Black => (
                options
                    .btime_ms
                    .or(options.wtime_ms)
                    .ok_or_else(|| "go clock mode requires wtime or btime".to_owned())?,
                options.binc_ms,
            ),
        };

        let reserve = self.move_overhead_ms.saturating_add(remaining / 100);
        let available = remaining.saturating_sub(reserve);
        // Honor the GUI's complete control horizon. Artificially clamping large
        // `movestogo` values spends too aggressively in tournament controls.
        let moves_to_go = u64::from(options.movestogo.unwrap_or(25).max(1));
        let base = available / moves_to_go;
        #[cfg(not(feature = "spsa-tuning"))]
        let soft = available.min(base.saturating_add(increment.saturating_mul(3) / 4));
        #[cfg(feature = "spsa-tuning")]
        let soft = available.min(base.saturating_add(
            increment.saturating_mul(self.search_parameters.time_increment_pct as u64) / 100,
        ));
        #[cfg(not(feature = "spsa-tuning"))]
        let hard = available.min(
            soft.saturating_mul(3)
                .saturating_div(2)
                .max(soft.saturating_add(10)),
        );
        #[cfg(feature = "spsa-tuning")]
        let hard = available.min(
            soft.saturating_mul(self.search_parameters.time_hard_pct as u64)
                .saturating_div(100)
                .max(soft.saturating_add(10)),
        );
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
    let mut seen_depth = false;
    let mut seen_nodes = false;
    let mut seen_movetime = false;
    let mut seen_wtime = false;
    let mut seen_btime = false;
    let mut seen_winc = false;
    let mut seen_binc = false;
    let mut seen_movestogo = false;
    let mut seen_infinite = false;
    let mut seen_ponder = false;
    let mut seen_searchmoves = false;
    let mut index = 1usize;
    while index < tokens.len() {
        match tokens[index] {
            "depth" => {
                mark_go_option_seen(&mut seen_depth, "depth")?;
                let Some(value) = tokens.get(index + 1) else {
                    return Err("go depth requires a value".to_owned());
                };
                options.depth = Some(parse_depth(value)?);
                index += 2;
            }
            "nodes" => {
                mark_go_option_seen(&mut seen_nodes, "nodes")?;
                let Some(value) = tokens.get(index + 1) else {
                    return Err("go nodes requires a value".to_owned());
                };
                let nodes = parse_u64_arg(value, "go nodes")?;
                if nodes == 0 {
                    return Err("go nodes must be at least 1".to_owned());
                }
                options.nodes = Some(nodes);
                index += 2;
            }
            "movetime" => {
                mark_go_option_seen(&mut seen_movetime, "movetime")?;
                let Some(value) = tokens.get(index + 1) else {
                    return Err("go movetime requires a value".to_owned());
                };
                options.movetime_ms = Some(parse_u64_arg(value, "go movetime")?);
                index += 2;
            }
            "wtime" => {
                mark_go_option_seen(&mut seen_wtime, "wtime")?;
                let Some(value) = tokens.get(index + 1) else {
                    return Err("go wtime requires a value".to_owned());
                };
                options.wtime_ms = Some(parse_u64_arg(value, "go wtime")?);
                index += 2;
            }
            "btime" => {
                mark_go_option_seen(&mut seen_btime, "btime")?;
                let Some(value) = tokens.get(index + 1) else {
                    return Err("go btime requires a value".to_owned());
                };
                options.btime_ms = Some(parse_u64_arg(value, "go btime")?);
                index += 2;
            }
            "winc" => {
                mark_go_option_seen(&mut seen_winc, "winc")?;
                let Some(value) = tokens.get(index + 1) else {
                    return Err("go winc requires a value".to_owned());
                };
                options.winc_ms = parse_u64_arg(value, "go winc")?;
                index += 2;
            }
            "binc" => {
                mark_go_option_seen(&mut seen_binc, "binc")?;
                let Some(value) = tokens.get(index + 1) else {
                    return Err("go binc requires a value".to_owned());
                };
                options.binc_ms = parse_u64_arg(value, "go binc")?;
                index += 2;
            }
            "movestogo" => {
                mark_go_option_seen(&mut seen_movestogo, "movestogo")?;
                let Some(value) = tokens.get(index + 1) else {
                    return Err("go movestogo requires a value".to_owned());
                };
                let moves_to_go = parse_u32_arg(value, "go movestogo")?;
                if moves_to_go == 0 {
                    return Err("go movestogo must be at least 1".to_owned());
                }
                options.movestogo = Some(moves_to_go);
                index += 2;
            }
            "infinite" => {
                mark_go_option_seen(&mut seen_infinite, "infinite")?;
                options.infinite = true;
                index += 1;
            }
            "ponder" => {
                mark_go_option_seen(&mut seen_ponder, "ponder")?;
                options.ponder = true;
                index += 1;
            }
            "searchmoves" => {
                mark_go_option_seen(&mut seen_searchmoves, "searchmoves")?;
                index += 1;
                while index < tokens.len() && !is_go_option_token(tokens[index]) {
                    options.searchmoves.push(tokens[index].to_owned());
                    index += 1;
                }
                if options.searchmoves.is_empty() {
                    return Err("go searchmoves requires at least one move".to_owned());
                }
            }
            "ponderhit" | "mate" => {
                return Err(format!("unsupported go argument '{}'", tokens[index]));
            }
            other => {
                return Err(format!("unsupported go argument '{other}'"));
            }
        }
    }

    let explicit_depth = options.depth.is_some();
    let has_clock_arguments = seen_wtime || seen_btime || seen_winc || seen_binc || seen_movestogo;
    if options.infinite && (explicit_depth || seen_movetime || has_clock_arguments || seen_nodes) {
        return Err(
            "go infinite cannot be combined with depth, nodes, movetime, or clock controls"
                .to_owned(),
        );
    }
    if options.ponder
        && (options.infinite
            || explicit_depth
            || seen_movetime
            || seen_nodes
            || !has_clock_arguments)
    {
        return Err(
            "go ponder requires clock controls and cannot be combined with depth, nodes, movetime, or infinite"
                .to_owned(),
        );
    }
    if options.movetime_ms.is_some() && (explicit_depth || has_clock_arguments) {
        return Err("go movetime cannot be combined with depth or clock controls".to_owned());
    }
    if has_clock_arguments && explicit_depth {
        return Err("go clock mode cannot be combined with depth".to_owned());
    }
    if has_clock_arguments && !seen_wtime && !seen_btime {
        return Err("go clock mode requires wtime or btime".to_owned());
    }

    Ok(options)
}

fn mark_go_option_seen(seen: &mut bool, option: &str) -> Result<(), String> {
    if *seen {
        return Err(format!("duplicate go {option} argument"));
    }
    *seen = true;
    Ok(())
}

fn handle_debug_command(tokens: &[&str]) -> Vec<String> {
    match tokens {
        [_, "on"] | [_, "off"] => Vec::new(),
        [_] => vec!["info string error: debug requires on or off".to_owned()],
        _ => vec![format!(
            "info string error: unsupported debug argument '{}'",
            tokens[1]
        )],
    }
}

fn is_go_option_token(token: &str) -> bool {
    matches!(
        token,
        "depth"
            | "movetime"
            | "wtime"
            | "btime"
            | "winc"
            | "binc"
            | "movestogo"
            | "infinite"
            | "searchmoves"
            | "ponder"
            | "ponderhit"
            | "nodes"
            | "mate"
    )
}

fn parse_setoption(line: &str, tokens: &[&str]) -> Result<SetOptionCommand, String> {
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
        "Move Overhead" => {
            let Some(value_index) = value_index else {
                return Err("setoption name Move Overhead requires 'value <ms>'".to_owned());
            };
            if value_index + 2 != tokens.len() {
                return Err("setoption name Move Overhead requires exactly one value".to_owned());
            }
            let overhead_ms = parse_u64_arg(
                tokens[value_index + 1],
                "setoption name Move Overhead value",
            )?;
            if overhead_ms > MAX_MOVE_OVERHEAD_MS {
                return Err(format!(
                    "Move Overhead value must be between 0 and {MAX_MOVE_OVERHEAD_MS}"
                ));
            }
            Ok(SetOptionCommand::MoveOverhead(overhead_ms))
        }
        "Ponder" => {
            let Some(value_index) = value_index else {
                return Err("setoption name Ponder requires 'value <true|false>'".to_owned());
            };
            if value_index + 2 != tokens.len() {
                return Err("setoption name Ponder requires exactly one value".to_owned());
            }
            match tokens[value_index + 1] {
                "true" => Ok(SetOptionCommand::Ponder(true)),
                "false" => Ok(SetOptionCommand::Ponder(false)),
                value => Err(format!(
                    "invalid setoption name Ponder value '{value}'; expected true or false"
                )),
            }
        }
        "SyzygyPath" => {
            if !line.contains(" value") {
                return Err("setoption name SyzygyPath requires 'value <path>'".to_owned());
            }
            Ok(SetOptionCommand::SyzygyPath)
        }
        "SyzygyProbeLimit" => {
            let Some(value_index) = value_index else {
                return Err("setoption name SyzygyProbeLimit requires 'value <pieces>'".to_owned());
            };
            if value_index + 2 != tokens.len() {
                return Err("setoption name SyzygyProbeLimit requires exactly one value".to_owned());
            }
            let limit = parse_u32_arg(
                tokens[value_index + 1],
                "setoption name SyzygyProbeLimit value",
            )?;
            if limit > u32::from(MAX_SYZYGY_PIECES) {
                return Err(format!(
                    "SyzygyProbeLimit value must be between 0 and {MAX_SYZYGY_PIECES}"
                ));
            }
            Ok(SetOptionCommand::SyzygyProbeLimit(limit as u8))
        }
        "Syzygy50MoveRule" => {
            let Some(value_index) = value_index else {
                return Err(
                    "setoption name Syzygy50MoveRule requires 'value <true|false>'".to_owned(),
                );
            };
            if value_index + 2 != tokens.len() {
                return Err("setoption name Syzygy50MoveRule requires exactly one value".to_owned());
            }
            match tokens[value_index + 1] {
                "true" => Ok(SetOptionCommand::Syzygy50MoveRule(true)),
                "false" => Ok(SetOptionCommand::Syzygy50MoveRule(false)),
                value => Err(format!(
                    "invalid setoption name Syzygy50MoveRule value '{value}'; expected true or false"
                )),
            }
        }
        "EvalFile" => {
            if !line.contains(" value") {
                return Err("setoption name EvalFile requires 'value <path>'".to_owned());
            }
            Ok(SetOptionCommand::EvalFile)
        }
        "SmallEvalFile" => {
            if !line.contains(" value") {
                return Err("setoption name SmallEvalFile requires 'value <path>'".to_owned());
            }
            Ok(SetOptionCommand::SmallEvalFile)
        }
        "DualEvalPolicy" => {
            let Some(value_index) = value_index else {
                return Err(
                    "setoption name DualEvalPolicy requires 'value <off|small-fallback>'"
                        .to_owned(),
                );
            };
            if value_index + 2 != tokens.len() {
                return Err("setoption name DualEvalPolicy requires exactly one value".to_owned());
            }
            match tokens[value_index + 1] {
                "off" => Ok(SetOptionCommand::DualEvalPolicy(DualEvalPolicy::Off)),
                "small-fallback" => Ok(SetOptionCommand::DualEvalPolicy(
                    DualEvalPolicy::SmallFallback,
                )),
                value => Err(format!(
                    "invalid setoption name DualEvalPolicy value '{value}'; expected off or small-fallback"
                )),
            }
        }
        "DualEvalThreshold" => {
            let Some(value_index) = value_index else {
                return Err(
                    "setoption name DualEvalThreshold requires 'value <centipawns>'".to_owned(),
                );
            };
            if value_index + 2 != tokens.len() {
                return Err(
                    "setoption name DualEvalThreshold requires exactly one value".to_owned(),
                );
            }
            let threshold = tokens[value_index + 1]
                .parse::<i32>()
                .map_err(|_| "invalid setoption name DualEvalThreshold value".to_owned())?;
            if !(0..=MAX_DUAL_EVAL_THRESHOLD).contains(&threshold) {
                return Err(format!(
                    "DualEvalThreshold value must be between 0 and {MAX_DUAL_EVAL_THRESHOLD}"
                ));
            }
            Ok(SetOptionCommand::DualEvalThreshold(threshold))
        }
        #[cfg(feature = "spsa-tuning")]
        "TuneManifest" => {
            if value_index.is_some() {
                return Err("setoption name TuneManifest does not take a value".to_owned());
            }
            Ok(SetOptionCommand::TuneManifest)
        }
        #[cfg(feature = "spsa-tuning")]
        name if parameter_spec(name).is_some() => {
            let Some(value_index) = value_index else {
                return Err(format!("setoption name {name} requires 'value <integer>'"));
            };
            if value_index + 2 != tokens.len() {
                return Err(format!("setoption name {name} requires exactly one value"));
            }
            let value = tokens[value_index + 1]
                .parse::<i32>()
                .map_err(|_| format!("invalid setoption name {name} value"))?;
            let spec = parameter_spec(name).expect("guarded tuning parameter must exist");
            if !(spec.min..=spec.max).contains(&value) {
                return Err(format!(
                    "{name} value must be between {} and {}",
                    spec.min, spec.max
                ));
            }
            Ok(SetOptionCommand::TuneParameter(spec.name, value))
        }
        _ => Err(format!("unsupported option '{name}'")),
    }
}

fn parse_syzygy_path_value(line: &str) -> String {
    let trimmed = line.trim();
    if let Some((_, value)) = trimmed.split_once(" value ") {
        return value.trim().to_owned();
    }
    if trimmed.ends_with(" value") {
        return String::new();
    }
    String::new()
}

fn parse_eval_file_value(line: &str) -> String {
    let trimmed = line.trim();
    if let Some((_, value)) = trimmed.split_once(" value ") {
        return value.trim().to_owned();
    }
    if trimmed.ends_with(" value") {
        return String::new();
    }
    String::new()
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

fn checked_deadline(now: Instant, milliseconds: u64, label: &str) -> Result<Instant, String> {
    now.checked_add(Duration::from_millis(milliseconds))
        .ok_or_else(|| format!("{label} is too large for this platform"))
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
            RuntimeInput::Command(line, ponder_state) => {
                let (response, live_info_error) = {
                    let mut live_info_error = None;
                    let mut live_info = |info_line: &str| {
                        if live_info_error.is_some() {
                            return;
                        }
                        if let Err(error) =
                            writeln!(output, "{info_line}").and_then(|_| output.flush())
                        {
                            live_info_error = Some(error);
                        }
                    };
                    let response = engine.handle_line_with_runtime_ponder(
                        &line,
                        Some(Arc::clone(stop_flag)),
                        Some(&mut live_info),
                        ponder_state,
                    );
                    (response, live_info_error)
                };
                if let Some(error) = live_info_error {
                    return Err(error);
                }
                let suppress_output =
                    quit_flag.load(Ordering::Relaxed) && line.trim_start().starts_with("go");
                if !suppress_output {
                    for output_line in response.lines {
                        writeln!(output, "{output_line}")?;
                    }
                    output.flush()?;
                }

                if response.should_quit {
                    break;
                }
            }
            RuntimeInput::PonderHitRequested => {}
            RuntimeInput::StopRequested => {
                stop_flag.store(false, Ordering::Relaxed);
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
            sender.send(RuntimeInput::StopRequested).map_err(|error| {
                io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    format!("uci runtime send failed: {error}"),
                )
            })?;
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
            sender
                .send(RuntimeInput::Command(line, None))
                .map_err(|error| {
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
        let mut active_ponder = None::<Arc<PonderState>>;
        for line_result in stdin.lock().lines() {
            let line = line_result?;
            if line.trim() == "ponderhit" {
                if let Some(state) = active_ponder.take() {
                    state.hit(Instant::now());
                    helper_sender
                        .send(RuntimeInput::PonderHitRequested)
                        .map_err(|error| {
                            io::Error::new(
                                io::ErrorKind::BrokenPipe,
                                format!("uci runtime send failed: {error}"),
                            )
                        })?;
                } else {
                    helper_sender
                        .send(RuntimeInput::Command(line, None))
                        .map_err(|error| {
                            io::Error::new(
                                io::ErrorKind::BrokenPipe,
                                format!("uci runtime send failed: {error}"),
                            )
                        })?;
                }
                continue;
            }
            let tokens = line.split_whitespace().collect::<Vec<_>>();
            if tokens.first() == Some(&"go")
                && parse_go(&tokens).is_ok_and(|options| options.ponder)
            {
                if let Some(state) = active_ponder.take() {
                    state.cancel();
                }
                let state = Arc::new(PonderState::new(Instant::now()));
                active_ponder = Some(Arc::clone(&state));
                helper_sender
                    .send(RuntimeInput::Command(line, Some(state)))
                    .map_err(|error| {
                        io::Error::new(
                            io::ErrorKind::BrokenPipe,
                            format!("uci runtime send failed: {error}"),
                        )
                    })?;
                continue;
            }
            if matches!(line.trim(), "stop" | "quit")
                && let Some(state) = active_ponder.take()
            {
                state.cancel();
            }
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
    use crate::search::nnue::tiny_test_evalfile_path;
    use crate::search::tablebase::{MockTablebaseBackend, TablebaseService, WdlOutcome};
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

    #[cfg(not(feature = "spsa-tuning"))]
    #[test]
    fn production_build_does_not_expose_experimental_tuning_controls() {
        let mut engine = UciEngine::new();
        assert!(
            engine
                .handle_line("uci")
                .lines
                .iter()
                .all(|line| !line.starts_with("option name Tune"))
        );
        assert_eq!(
            engine
                .handle_line("setoption name TuneFutilityBase value 100")
                .lines,
            ["info string error: unsupported option 'TuneFutilityBase'"]
        );
    }

    #[cfg(feature = "spsa-tuning")]
    #[test]
    fn tuning_build_advertises_exact_bounded_schema() {
        let mut engine = UciEngine::new();
        let response = engine.handle_line("uci");
        for spec in PARAMETER_SPECS {
            let expected = format!(
                "option name {} type spin default {} min {} max {}",
                spec.name, spec.default, spec.min, spec.max
            );
            assert!(response.lines.contains(&expected), "missing {expected}");
        }
        assert!(
            response
                .lines
                .contains(&"option name TuneManifest type button".to_owned())
        );
    }

    #[cfg(feature = "spsa-tuning")]
    #[test]
    fn tuning_options_validate_mutate_and_report_deterministically() {
        let mut engine = UciEngine::new();
        let defaults = engine.handle_line("setoption name TuneManifest").lines;
        assert!(defaults[0].starts_with("info string tuning manifest version 1 checksum "));
        assert_eq!(
            defaults,
            engine.handle_line("setoption name TuneManifest").lines
        );

        let searched = engine.handle_line("go depth 2");
        assert!(
            searched
                .lines
                .iter()
                .any(|line| line.starts_with("bestmove "))
        );
        assert!(engine.debug_tt_entry_count() > 0);
        assert!(
            engine
                .handle_line("setoption name TuneFutilityBase value 100")
                .lines
                .is_empty()
        );
        assert_eq!(engine.debug_tt_entry_count(), 0);
        let changed = engine.handle_line("setoption name TuneManifest").lines;
        assert_ne!(defaults[0], changed[0]);
        assert!(changed.iter().any(|line| {
            line == "info string tuning parameter TuneFutilityBase value 100 default 90 min 0 max 240 step 10"
        }));

        let rejected = engine.handle_line("setoption name TuneFutilityBase value 241");
        assert_eq!(
            rejected.lines,
            ["info string error: TuneFutilityBase value must be between 0 and 240"]
        );
        assert_eq!(
            changed,
            engine.handle_line("setoption name TuneManifest").lines,
            "a rejected update must not change the live vector"
        );
    }

    #[test]
    fn root_tablebase_probe_is_reported_before_bestmove() {
        let fen = "8/8/8/8/8/3Q4/2K5/k7 w - - 0 1";
        let tablebases = TablebaseService::from_backend_for_tests(
            "/mock/syzygy",
            Arc::new(MockTablebaseBackend::new().with_root_probe(
                fen,
                "d3d7",
                WdlOutcome::Win,
                Some(1),
            )),
        );
        let mut engine = UciEngine::new();
        engine
            .search_service
            .debug_install_tablebases("/mock/syzygy", tablebases);
        assert!(
            engine
                .handle_line(&format!("position fen {fen}"))
                .lines
                .is_empty()
        );

        let response = engine.handle_line("go depth 5");
        assert_eq!(
            response.lines.last().map(String::as_str),
            Some("bestmove d3d7")
        );
        assert!(response.lines.iter().any(|line| {
            line == "info string syzygy probes 1 root 1 wdl 0 hits 1 misses 0 errors 0"
        }));
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
        assert!(matches!(
            receiver.try_recv().expect("stop wakeup must be queued"),
            RuntimeInput::StopRequested
        ));

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
    fn queued_quit_does_not_suppress_startup_identification() {
        let (sender, receiver) = mpsc::channel();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let quit_flag = Arc::new(AtomicBool::new(false));

        handle_input_line(
            "uci".to_owned(),
            &sender,
            stop_flag.as_ref(),
            quit_flag.as_ref(),
        )
        .expect("uci command must queue");
        handle_input_line(
            "isready".to_owned(),
            &sender,
            stop_flag.as_ref(),
            quit_flag.as_ref(),
        )
        .expect("isready command must queue");
        assert!(
            !handle_input_line(
                "quit".to_owned(),
                &sender,
                stop_flag.as_ref(),
                quit_flag.as_ref(),
            )
            .expect("quit command must queue")
        );
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
        let output = String::from_utf8(output).expect("runtime output must be utf8");

        assert!(
            output
                .lines()
                .any(|line| line.starts_with("id name Volkrix"))
        );
        assert!(output.contains("id author Monty Bognar\n"));
        assert!(output.contains("uciok\n"));
        assert!(output.contains("readyok\n"));
    }

    #[test]
    fn runtime_streams_search_info_before_bestmove_without_repeating_it() {
        let (sender, receiver) = mpsc::channel();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let quit_flag = Arc::new(AtomicBool::new(false));

        handle_input_line(
            "go depth 2".to_owned(),
            &sender,
            stop_flag.as_ref(),
            quit_flag.as_ref(),
        )
        .expect("go command must queue");
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
        let output = String::from_utf8(output).expect("runtime output must be utf8");
        let info_count = output
            .lines()
            .filter(|line| line.starts_with("info depth "))
            .count();
        let first_info = output
            .find("info depth ")
            .expect("runtime must stream an info line");
        let bestmove = output
            .find("bestmove ")
            .expect("runtime must emit bestmove");

        assert_eq!(info_count, 2);
        assert!(first_info < bestmove);
        assert!(output.contains(" nps "));
        assert_eq!(output.matches("bestmove ").count(), 1);
    }

    #[test]
    fn runtime_ponder_waits_for_hit_and_then_uses_its_clock_budget() {
        let (sender, receiver) = mpsc::channel();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let quit_flag = Arc::new(AtomicBool::new(false));
        let ponder_started = Instant::now();
        let ponder = Arc::new(PonderState::new(ponder_started));
        sender
            .send(RuntimeInput::Command(
                "go ponder wtime 20 btime 20".to_owned(),
                Some(Arc::clone(&ponder)),
            ))
            .expect("ponder command must queue");

        let hitter = thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));
            ponder.hit(Instant::now());
            sender
                .send(RuntimeInput::PonderHitRequested)
                .expect("ponderhit notification must queue");
        });

        let mut output = Vec::new();
        let mut engine = UciEngine::new();
        let position_response = engine.handle_line("position fen 7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
        assert!(position_response.lines.is_empty());
        run_runtime_session(
            &mut engine,
            &receiver,
            &mut output,
            &stop_flag,
            quit_flag.as_ref(),
        )
        .expect("runtime must complete");
        hitter.join().expect("ponderhit helper must join");

        let elapsed = ponder_started.elapsed();
        let output = String::from_utf8(output).expect("runtime output must be utf8");
        assert!(elapsed >= Duration::from_millis(20));
        assert_eq!(output.matches("bestmove ").count(), 1);
        assert!(output.contains("bestmove 0000"));
        assert!(!output.contains("unsupported go argument 'ponder'"));
    }

    #[test]
    fn runtime_emits_corrected_final_smp_statistics() {
        let (sender, receiver) = mpsc::channel();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let quit_flag = Arc::new(AtomicBool::new(false));

        for command in ["setoption name Threads value 4", "go nodes 4001"] {
            handle_input_line(
                command.to_owned(),
                &sender,
                stop_flag.as_ref(),
                quit_flag.as_ref(),
            )
            .expect("command must queue");
        }
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
        let output = String::from_utf8(output).expect("runtime output must be utf8");
        let final_info = output
            .lines()
            .rfind(|line| line.starts_with("info depth "))
            .expect("runtime must emit final aggregate info");

        assert!(final_info.contains(" nodes 4001 "), "{final_info}");
        assert_eq!(output.matches("bestmove ").count(), 1);
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
    fn setoption_syzygypath_received_during_search_is_deferred_until_after_stop_unwind() {
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
                    "setoption name SyzygyPath value /tmp/syzygy".to_owned(),
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
        assert_eq!(engine.debug_syzygy_path(), "");
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

        let output = String::from_utf8(output).expect("utf8");
        assert_eq!(engine.debug_syzygy_path(), "");
        assert!(output.contains("did not load any supported Syzygy tablebase files"));
        assert_eq!(output.matches("bestmove ").count(), 1);
    }

    #[test]
    fn setoption_evalfile_received_during_search_is_deferred_until_after_stop_unwind() {
        let (sender, receiver) = mpsc::channel();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let quit_flag = Arc::new(AtomicBool::new(false));
        let eval_file = tiny_test_evalfile_path()
            .to_str()
            .expect("tiny test eval file path must be UTF-8")
            .to_owned();

        let helper = {
            let sender = sender.clone();
            let stop_flag = Arc::clone(&stop_flag);
            let quit_flag = Arc::clone(&quit_flag);
            let eval_file = eval_file.clone();
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
                    format!("setoption name EvalFile value {eval_file}"),
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
        assert_eq!(engine.debug_eval_file(), "");
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

        assert_eq!(engine.debug_eval_file(), eval_file);
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
    fn move_overhead_is_configurable_and_reserved_from_movetime() {
        let mut engine = UciEngine::new();
        let response = engine.handle_line("setoption name Move Overhead value 25");
        assert!(response.lines.is_empty());
        assert_eq!(engine.move_overhead_ms, 25);

        let now = Instant::now();
        let request = engine
            .build_search_request(
                GoOptions {
                    movetime_ms: Some(100),
                    ..GoOptions::default()
                },
                None,
            )
            .expect("movetime request must build");
        let deadline = request.hard_deadline.expect("hard deadline must exist");
        let allocated = deadline.saturating_duration_since(now);
        assert!(allocated >= Duration::from_millis(75));
        assert!(allocated <= Duration::from_millis(80));
    }

    #[test]
    fn clock_budget_uses_sudden_death_defaults() {
        let engine = UciEngine::new();
        let (soft, hard) = engine
            .clock_budget_ms(&GoOptions {
                wtime_ms: Some(1_000),
                btime_ms: Some(1_000),
                ..GoOptions::default()
            })
            .expect("clock budget must build");

        assert_eq!(soft, 39);
        assert_eq!(hard, 58);
    }

    #[test]
    fn clock_budget_honors_movestogo_and_increment() {
        let engine = UciEngine::new();
        let (soft, hard) = engine
            .clock_budget_ms(&GoOptions {
                wtime_ms: Some(5_000),
                btime_ms: Some(5_000),
                winc_ms: 1_000,
                movestogo: Some(10),
                ..GoOptions::default()
            })
            .expect("clock budget must build");

        assert_eq!(soft, 1_244);
        assert_eq!(hard, 1_866);
    }

    #[test]
    fn clock_budget_honors_large_movestogo_without_overspending() {
        let engine = UciEngine::new();
        let (soft, hard) = engine
            .clock_budget_ms(&GoOptions {
                wtime_ms: Some(5_000),
                btime_ms: Some(5_000),
                movestogo: Some(100),
                ..GoOptions::default()
            })
            .expect("clock budget must build");

        assert_eq!((soft, hard), (49, 73));
    }

    #[test]
    fn ponder_requires_runtime_and_unambiguous_clock_controls() {
        let options = parse_go(&["go", "ponder", "wtime", "1_000", "btime", "1_000"])
            .expect_err("underscores are not valid UCI integers");
        assert!(options.contains("invalid go wtime value"));

        let options = parse_go(&["go", "ponder", "wtime", "1000", "btime", "1000"])
            .expect("clocked ponder must parse");
        assert!(options.ponder);
        for command in [
            ["go", "ponder", "depth", "5"].as_slice(),
            ["go", "ponder", "nodes", "100"].as_slice(),
            ["go", "ponder", "movetime", "100"].as_slice(),
            ["go", "ponder"].as_slice(),
        ] {
            assert!(parse_go(command).is_err(), "{command:?} must be rejected");
        }

        let response = UciEngine::new().handle_line("go ponder wtime 1000 btime 1000");
        assert!(
            response
                .lines
                .iter()
                .any(|line| line.contains("go ponder requires the stdio runtime"))
        );
        assert!(
            response
                .lines
                .iter()
                .all(|line| !line.starts_with("bestmove "))
        );
    }

    #[test]
    fn clock_budget_never_spends_the_low_time_reserve() {
        let engine = UciEngine::new();
        let (soft, hard) = engine
            .clock_budget_ms(&GoOptions {
                wtime_ms: Some(20),
                btime_ms: Some(20),
                ..GoOptions::default()
            })
            .expect("clock budget must build");

        assert_eq!(soft, 0);
        assert_eq!(hard, 10);

        let (soft, hard) = engine
            .clock_budget_ms(&GoOptions {
                wtime_ms: Some(DEFAULT_MOVE_OVERHEAD_MS),
                btime_ms: Some(DEFAULT_MOVE_OVERHEAD_MS),
                ..GoOptions::default()
            })
            .expect("clock budget must build");
        assert_eq!((soft, hard), (0, 0));
    }
}
