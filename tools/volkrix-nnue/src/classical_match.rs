use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use volkrix::{
    core::{Color, Position, PositionStatus},
    nnue_training::{MatchGameSummary, MatchOutcome, MatchSummary, normalize_fen},
    search::{
        SearchLimits,
        service::{SearchRequest, UciSearchService},
    },
};

use crate::texel_tuning::read_classical_weights;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClassicalMatchConfig {
    pub depth: u8,
    pub max_plies: usize,
    pub max_openings: Option<usize>,
}

pub fn compare_classical_weights(
    openings_path: &Path,
    baseline_weights_path: Option<&Path>,
    candidate_weights_path: &Path,
    config: ClassicalMatchConfig,
) -> Result<MatchSummary, String> {
    let baseline_weights = baseline_weights_path
        .map(read_classical_weights)
        .transpose()?;
    let candidate_weights = read_classical_weights(candidate_weights_path)?;
    let openings = load_openings(openings_path, config.max_openings)?;
    let mut baseline = UciSearchService::new();
    baseline.resize_hash(64);
    baseline.set_threads(1);
    baseline.set_classical_weights(baseline_weights);
    let mut candidate = UciSearchService::new();
    candidate.resize_hash(64);
    candidate.set_threads(1);
    candidate.set_classical_weights(Some(candidate_weights));

    let mut summary = MatchSummary {
        openings: openings.len(),
        ..MatchSummary::default()
    };
    for opening_fen in openings {
        for candidate_color in [Color::White, Color::Black] {
            let game = play_match_game(
                &opening_fen,
                &mut baseline,
                &mut candidate,
                candidate_color,
                config,
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
    baseline: &mut UciSearchService,
    candidate: &mut UciSearchService,
    candidate_color: Color,
    config: ClassicalMatchConfig,
) -> Result<MatchGameSummary, String> {
    let mut position = Position::from_fen(opening_fen)
        .map_err(|error| format!("failed to parse opening FEN '{opening_fen}': {error}"))?;
    baseline.clear_hash();
    candidate.clear_hash();
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

        let limits = SearchLimits::new(config.depth);
        let side_to_move = position.side_to_move();
        let result = if side_to_move == candidate_color {
            candidate.search(
                &mut position,
                SearchRequest {
                    limits,
                    soft_deadline: None,
                    hard_deadline: None,
                    stop_flag: None,
                },
            )
        } else {
            baseline.search(
                &mut position,
                SearchRequest {
                    limits,
                    soft_deadline: None,
                    hard_deadline: None,
                    stop_flag: None,
                },
            )
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
                "in-process classical match produced no best move for ongoing position '{}'",
                position.to_fen()
            )
        })?;
        position
            .apply_uci_move(&best_move.to_string())
            .map_err(|error| {
                format!(
                    "engine move '{}' was not legal from '{}': {error}",
                    best_move,
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
