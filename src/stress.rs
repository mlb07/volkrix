//! Deterministic, dependency-free state-machine stress harness.
//!
//! This module is available only to tests and the `internal-testing` feature. It
//! intentionally uses an independent slow legal-move oracle and reproducible
//! pseudo-random walks instead of adding a runtime fuzzing dependency.

use std::fmt;

use crate::{
    core::{Move, MoveList, ParsedMove, Position, STARTPOS_FEN, UndoState},
    uci::UciEngine,
};

const MAX_STRESS_PLIES: u32 = 3_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StressConfig {
    pub seed: u64,
    pub walks: u32,
    pub plies: u32,
    pub parser_cases: u32,
}

impl Default for StressConfig {
    fn default() -> Self {
        Self {
            seed: 0x6a09_e667_f3bc_c909,
            walks: 16,
            plies: 96,
            parser_cases: 2_048,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StressReport {
    pub seed: u64,
    pub corpus_roots: u64,
    pub walks: u64,
    pub visited_positions: u64,
    pub played_moves: u64,
    pub fen_cases: u64,
    pub uci_cases: u64,
    pub trace_digest: u64,
}

impl fmt::Display for StressReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "stress seed=0x{:016x} roots={} walks={} positions={} moves={} fen_cases={} uci_cases={} digest=0x{:016x}",
            self.seed,
            self.corpus_roots,
            self.walks,
            self.visited_positions,
            self.played_moves,
            self.fen_cases,
            self.uci_cases,
            self.trace_digest
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PositionSnapshot {
    fen: String,
    zobrist: u64,
    search_key: u64,
    repetition_history: Vec<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct BoardSnapshot {
    fen: String,
    zobrist: u64,
    search_key: u64,
}

#[derive(Clone, Copy)]
struct MoveExpectation {
    uci: &'static str,
    legal: bool,
}

#[derive(Clone, Copy)]
struct CorpusSeed {
    name: &'static str,
    fen: &'static str,
    prefix: &'static [&'static str],
    expectations: &'static [MoveExpectation],
    expect_repetition: bool,
}

const NO_MOVES: &[&str] = &[];
const NO_EXPECTATIONS: &[MoveExpectation] = &[];
const CASTLING_EXPECTATIONS: &[MoveExpectation] = &[
    MoveExpectation {
        uci: "e1g1",
        legal: true,
    },
    MoveExpectation {
        uci: "e1c1",
        legal: true,
    },
];
const LEGAL_EP: &[MoveExpectation] = &[MoveExpectation {
    uci: "e5d6",
    legal: true,
}];
const PINNED_EP: &[MoveExpectation] = &[MoveExpectation {
    uci: "e5d6",
    legal: false,
}];
const WHITE_PROMOTION: &[MoveExpectation] = &[MoveExpectation {
    uci: "a7a8q",
    legal: true,
}];
const CAPTURE_PROMOTION: &[MoveExpectation] = &[MoveExpectation {
    uci: "g7h8q",
    legal: true,
}];
const BLACK_PROMOTION: &[MoveExpectation] = &[MoveExpectation {
    uci: "b2b1q",
    legal: true,
}];
const REPETITION_PREFIX: &[&str] = &[
    "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8",
];

const CORPUS: &[CorpusSeed] = &[
    CorpusSeed {
        name: "startpos",
        fen: STARTPOS_FEN,
        prefix: NO_MOVES,
        expectations: NO_EXPECTATIONS,
        expect_repetition: false,
    },
    CorpusSeed {
        name: "castling-both-sides",
        fen: "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        prefix: NO_MOVES,
        expectations: CASTLING_EXPECTATIONS,
        expect_repetition: false,
    },
    CorpusSeed {
        name: "legal-en-passant",
        fen: "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        prefix: NO_MOVES,
        expectations: LEGAL_EP,
        expect_repetition: false,
    },
    CorpusSeed {
        name: "pinned-en-passant",
        fen: "k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        prefix: NO_MOVES,
        expectations: PINNED_EP,
        expect_repetition: false,
    },
    CorpusSeed {
        name: "quiet-white-promotion",
        fen: "7k/P7/8/8/8/8/8/K7 w - - 0 1",
        prefix: NO_MOVES,
        expectations: WHITE_PROMOTION,
        expect_repetition: false,
    },
    CorpusSeed {
        name: "capture-promotion",
        fen: "4k2r/6P1/8/8/8/8/8/4K3 w - - 0 1",
        prefix: NO_MOVES,
        expectations: CAPTURE_PROMOTION,
        expect_repetition: false,
    },
    CorpusSeed {
        name: "quiet-black-promotion",
        fen: "7k/8/8/8/8/8/1p6/7K b - - 0 1",
        prefix: NO_MOVES,
        expectations: BLACK_PROMOTION,
        expect_repetition: false,
    },
    CorpusSeed {
        name: "threefold-repetition",
        fen: STARTPOS_FEN,
        prefix: REPETITION_PREFIX,
        expectations: NO_EXPECTATIONS,
        expect_repetition: true,
    },
    CorpusSeed {
        name: "kiwipete",
        fen: "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        prefix: NO_MOVES,
        expectations: NO_EXPECTATIONS,
        expect_repetition: false,
    },
    CorpusSeed {
        name: "perft-position-three",
        fen: "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        prefix: NO_MOVES,
        expectations: NO_EXPECTATIONS,
        expect_repetition: false,
    },
];

const INVALID_FENS: &[&str] = &[
    "",
    "8/8/8/8/8/8/8/8 w - - 0",
    "8/8/8/8/8/8/8/8/8 w - - 0 1",
    "9/8/8/8/8/8/8/K6k w - - 0 1",
    "7x/8/8/8/8/8/8/K6k w - - 0 1",
    "8/8/8/8/8/8/8/K6k x - - 0 1",
    "8/8/8/8/8/8/8/K6k w KK - 0 1",
    "8/8/8/8/8/8/8/K6k w - e4 0 1",
    "8/8/8/8/8/8/8/K6k w - - -1 1",
    "8/8/8/8/8/8/8/K6k w - - 0 0",
    "8/8/8/8/8/8/8/Kk6 w - - 0 1",
    "♞7/8/8/8/8/8/8/K6k w - - 0 1",
];

pub fn run(config: StressConfig) -> Result<StressReport, String> {
    if config.walks == 0 {
        return Err("stress walks must be at least 1".to_owned());
    }
    if config.plies == 0 || config.plies > MAX_STRESS_PLIES {
        return Err(format!(
            "stress plies must be between 1 and {MAX_STRESS_PLIES}"
        ));
    }

    let mut rng = DeterministicRng::new(config.seed);
    let mut digest = TraceDigest::new(config.seed);
    let mut report = StressReport {
        seed: config.seed,
        ..StressReport::default()
    };

    for corpus in CORPUS {
        let mut position = seeded_position(corpus)?;
        let moves = audit_generation(&mut position, corpus.name, 0)?;
        assert_expectations(corpus, &moves)?;
        audit_all_moves(&position, &moves, corpus.name)?;
        digest.mix(corpus.name.as_bytes());
        digest.mix(position.to_fen().as_bytes());
        report.corpus_roots += 1;
    }

    for walk in 0..config.walks {
        let corpus = &CORPUS[walk as usize % CORPUS.len()];
        run_walk(
            corpus,
            walk,
            config.plies,
            &mut rng,
            &mut digest,
            &mut report,
        )?;
        report.walks += 1;
    }

    stress_parsers(config.parser_cases, &mut rng, &mut digest, &mut report)?;
    report.trace_digest = digest.finish();
    Ok(report)
}

fn run_walk(
    corpus: &CorpusSeed,
    walk: u32,
    plies: u32,
    rng: &mut DeterministicRng,
    digest: &mut TraceDigest,
    report: &mut StressReport,
) -> Result<(), String> {
    let mut position = seeded_position(corpus)?;
    let root = snapshot(&position);
    let mut played: Vec<(Move, UndoState, PositionSnapshot)> = Vec::new();

    for ply in 0..plies {
        let label = format!(
            "seed=0x{:016x} walk={walk} {} ply={ply}",
            report.seed, corpus.name
        );
        let before = snapshot(&position);
        let moves = audit_generation(&mut position, &label, ply)?;
        report.visited_positions += 1;
        digest.mix(before.fen.as_bytes());
        if moves.is_empty() {
            break;
        }

        let mv = moves.get(rng.index(moves.len()));
        let move_text = mv.to_string();
        let mut uci_applied = position.clone();
        let undo = position
            .make_move(mv)
            .map_err(|error| format!("{label}: generated move {move_text} failed: {error}"))?;
        validate_position(&position, &format!("{label} after {move_text}"))?;

        uci_applied
            .apply_uci_move(&move_text)
            .map_err(|error| format!("{label}: UCI replay of {move_text} failed: {error}"))?;
        if snapshot(&uci_applied) != snapshot(&position) {
            return Err(format!(
                "{label}: checked move and UCI application disagree for {move_text}"
            ));
        }

        assert_fen_round_trip(&position, &format!("{label} after {move_text}"))?;
        digest.mix(move_text.as_bytes());
        report.played_moves += 1;
        played.push((mv, undo, before));
    }

    while let Some((mv, undo, before)) = played.pop() {
        position.unmake_move(mv, undo);
        validate_position(&position, "stress unwind")?;
        if snapshot(&position) != before {
            return Err(format!(
                "seed=0x{:016x} walk={walk} {}: unmake mismatch after {mv}",
                report.seed, corpus.name
            ));
        }
    }
    if snapshot(&position) != root {
        return Err(format!(
            "seed=0x{:016x} walk={walk} {}: walk did not restore its root",
            report.seed, corpus.name
        ));
    }
    Ok(())
}

fn audit_generation(position: &mut Position, label: &str, ply: u32) -> Result<MoveList, String> {
    validate_position(
        position,
        &format!("{label}: before generation at ply {ply}"),
    )?;
    let before = snapshot(position);
    let mut fast = MoveList::new();
    position.generate_legal_moves(&mut fast);
    if snapshot(position) != before {
        return Err(format!(
            "{label}: fast legal generation mutated position at ply {ply}"
        ));
    }

    let mut slow = MoveList::new();
    position.debug_generate_legal_moves_slow(&mut slow);
    if snapshot(position) != before {
        return Err(format!(
            "{label}: slow legal generation mutated position at ply {ply}"
        ));
    }

    let fast_moves = sorted_move_text(&fast)?;
    let slow_moves = sorted_move_text(&slow)?;
    if fast_moves != slow_moves {
        return Err(format!(
            "{label}: fast/slow legal disagreement at ply {ply}\nfast={fast_moves:?}\nslow={slow_moves:?}"
        ));
    }
    Ok(fast)
}

fn audit_all_moves(position: &Position, moves: &MoveList, label: &str) -> Result<(), String> {
    let root = snapshot(position);
    for mv in moves.as_slice().iter().copied() {
        let move_text = mv.to_string();
        let mut checked = position.clone();
        let undo = checked
            .make_move(mv)
            .map_err(|error| format!("{label}: generated root move {move_text} failed: {error}"))?;
        validate_position(&checked, &format!("{label}: after root move {move_text}"))?;

        let mut uci_applied = position.clone();
        uci_applied
            .apply_uci_move(&move_text)
            .map_err(|error| format!("{label}: root UCI move {move_text} failed: {error}"))?;
        if snapshot(&checked) != snapshot(&uci_applied) {
            return Err(format!(
                "{label}: root checked/UCI state mismatch for {move_text}"
            ));
        }

        checked.unmake_move(mv, undo);
        if snapshot(&checked) != root {
            return Err(format!(
                "{label}: root move {move_text} failed exact unmake"
            ));
        }
    }
    Ok(())
}

fn stress_parsers(
    cases: u32,
    rng: &mut DeterministicRng,
    digest: &mut TraceDigest,
    report: &mut StressReport,
) -> Result<(), String> {
    for invalid in INVALID_FENS {
        if Position::from_fen(invalid).is_ok() {
            return Err(format!(
                "invalid FEN corpus entry was accepted: {invalid:?}"
            ));
        }
        digest.mix(invalid.as_bytes());
        report.fen_cases += 1;
    }

    let mut engine = UciEngine::new();
    for case in 0..cases {
        let length = 1 + rng.index(96);
        let text = random_parser_text(rng, length);
        let _ = Position::from_fen(&text);
        digest.mix(text.as_bytes());
        report.fen_cases += 1;

        let command = match case % 5 {
            0 => format!("fuzz_{text}"),
            1 => format!("position fen {text}"),
            2 => format!("go depth bad{text}"),
            3 => format!("setoption name Hash value bad{text}"),
            _ => format!("setoption name SyzygyProbeLimit value bad{text}"),
        };
        let before = snapshot(engine.position());
        let response = engine.handle_line(&command);
        if !response
            .lines
            .iter()
            .any(|line| line.starts_with("info string error:"))
        {
            return Err(format!(
                "malformed UCI stress command was not rejected: {command:?}"
            ));
        }
        if snapshot(engine.position()) != before {
            return Err(format!(
                "malformed UCI stress command mutated position: {command:?}"
            ));
        }
        validate_position(engine.position(), "after malformed UCI command")?;
        digest.mix(command.as_bytes());
        report.uci_cases += 1;
    }
    Ok(())
}

fn seeded_position(corpus: &CorpusSeed) -> Result<Position, String> {
    let mut position = Position::from_fen(corpus.fen)
        .map_err(|error| format!("{} seed FEN failed: {error}", corpus.name))?;
    for move_text in corpus.prefix {
        position.apply_uci_move(move_text).map_err(|error| {
            format!(
                "{} seed prefix move {move_text} failed: {error}",
                corpus.name
            )
        })?;
    }
    validate_position(&position, corpus.name)?;
    if position.is_draw_by_repetition() != corpus.expect_repetition {
        return Err(format!(
            "{} repetition expectation was {}, observed {}",
            corpus.name,
            corpus.expect_repetition,
            position.is_draw_by_repetition()
        ));
    }
    Ok(position)
}

fn assert_expectations(corpus: &CorpusSeed, moves: &MoveList) -> Result<(), String> {
    let move_text: Vec<String> = moves.as_slice().iter().map(ToString::to_string).collect();
    for expectation in corpus.expectations {
        let present = move_text.iter().any(|mv| mv == expectation.uci);
        if present != expectation.legal {
            return Err(format!(
                "{} expected move {} legal={}, observed legal={present}",
                corpus.name, expectation.uci, expectation.legal
            ));
        }
    }
    Ok(())
}

fn sorted_move_text(moves: &MoveList) -> Result<Vec<String>, String> {
    let mut result = Vec::with_capacity(moves.len());
    for mv in moves.as_slice() {
        let text = mv.to_string();
        let parsed = ParsedMove::parse(&text)
            .map_err(|_| format!("generated move does not parse as UCI: {text}"))?;
        if !mv.matches_parsed(parsed) {
            return Err(format!(
                "generated move does not round-trip through UCI: {text}"
            ));
        }
        result.push(text);
    }
    result.sort_unstable();
    if let Some(duplicate) = result.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(format!(
            "legal generator returned duplicate move {}",
            duplicate[0]
        ));
    }
    Ok(result)
}

fn assert_fen_round_trip(position: &Position, label: &str) -> Result<(), String> {
    let reparsed = Position::from_fen(&position.to_fen())
        .map_err(|error| format!("{label}: emitted FEN failed to parse: {error}"))?;
    validate_position(&reparsed, &format!("{label}: reparsed FEN"))?;
    if board_snapshot(&reparsed) != board_snapshot(position) {
        return Err(format!("{label}: FEN round-trip changed board state"));
    }
    Ok(())
}

fn snapshot(position: &Position) -> PositionSnapshot {
    PositionSnapshot {
        fen: position.to_fen(),
        zobrist: position.zobrist_key(),
        search_key: position.debug_search_key(),
        repetition_history: position.debug_repetition_history_snapshot(),
    }
}

fn board_snapshot(position: &Position) -> BoardSnapshot {
    BoardSnapshot {
        fen: position.to_fen(),
        zobrist: position.zobrist_key(),
        search_key: position.debug_search_key(),
    }
}

fn validate_position(position: &Position, label: &str) -> Result<(), String> {
    position
        .validate()
        .map_err(|error| format!("{label}: position invariant failed: {error}"))
}

fn random_parser_text(rng: &mut DeterministicRng, len: usize) -> String {
    const ALPHABET: &[u8] = b"prnbqkPRNBQK1234567890/ -_+xKQkqabcdefghABCDEFGH\t!?@#$%^&*()[]{}";
    let mut result = String::with_capacity(len);
    for _ in 0..len {
        result.push(ALPHABET[rng.index(ALPHABET.len())] as char);
    }
    result
}

struct DeterministicRng(u64);

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0 = self.0.wrapping_mul(0x2545_f491_4f6c_dd1d);
        self.0
    }

    fn index(&mut self, len: usize) -> usize {
        debug_assert!(len != 0);
        (self.next_u64() as usize) % len
    }
}

struct TraceDigest(u64);

impl TraceDigest {
    fn new(seed: u64) -> Self {
        Self(0xcbf2_9ce4_8422_2325 ^ seed)
    }

    fn mix(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
        self.0 ^= 0xff;
        self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
    }

    fn finish(self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_stress_quick_profile_is_reproducible() {
        let config = StressConfig {
            walks: 10,
            plies: 32,
            parser_cases: 256,
            ..StressConfig::default()
        };
        let first = run(config).expect("deterministic stress run must pass");
        let second = run(config).expect("repeated deterministic stress run must pass");
        assert_eq!(first, second);
        assert_eq!(
            first,
            StressReport {
                seed: config.seed,
                corpus_roots: CORPUS.len() as u64,
                walks: u64::from(config.walks),
                visited_positions: 320,
                played_moves: 320,
                fen_cases: INVALID_FENS.len() as u64 + u64::from(config.parser_cases),
                uci_cases: u64::from(config.parser_cases),
                trace_digest: 0x6f91_b131_3475_fed9,
            }
        );
    }

    #[test]
    #[ignore = "long deterministic state/parser stress; use the volkrix-stress binary or scheduled CI"]
    fn deterministic_stress_long_profile() {
        let report = run(StressConfig {
            walks: 128,
            plies: 512,
            parser_cases: 50_000,
            ..StressConfig::default()
        })
        .expect("long deterministic stress run must pass");
        println!("{report}");
    }
}
