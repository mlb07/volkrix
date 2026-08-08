use volkrix::{
    core::PositionStatus,
    nnue_training::{MatchOutcome, MatchSummary},
};

const Z_95: f64 = 1.959_963_984_540_054;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TerminationCounts {
    pub checkmates: usize,
    pub stalemates: usize,
    pub repetitions: usize,
    pub fifty_move_draws: usize,
    pub insufficient_material_draws: usize,
    pub max_ply_draws: usize,
    pub adjudicated_decisive_games: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MatchStatistics {
    /// Pair buckets in candidate half-points: 0, 1, 2, 3, and 4.
    pub pentanomial: [usize; 5],
    pub pairs: usize,
    pub score: f64,
    pub score_ci95: Option<(f64, f64)>,
    pub elo: Option<f64>,
    pub elo_ci95: Option<(f64, f64)>,
    pub terminations: TerminationCounts,
}

impl MatchStatistics {
    pub fn from_paired_summary(summary: &MatchSummary) -> Result<Self, String> {
        validate_summary_totals(summary)?;

        if summary.games != summary.openings.saturating_mul(2) {
            return Err(format!(
                "paired match expected exactly two games per opening, found {} games over {} openings",
                summary.games, summary.openings
            ));
        }

        let mut pentanomial = [0usize; 5];
        let mut pair_scores = Vec::with_capacity(summary.openings);
        for (pair_index, pair) in summary.game_summaries.chunks_exact(2).enumerate() {
            let first = &pair[0];
            let second = &pair[1];
            if first.opening_fen != second.opening_fen {
                return Err(format!(
                    "pair {} uses different openings: '{}' and '{}'",
                    pair_index + 1,
                    first.opening_fen,
                    second.opening_fen
                ));
            }
            if first.candidate_color == second.candidate_color {
                return Err(format!(
                    "pair {} did not reverse candidate colors for opening '{}'",
                    pair_index + 1,
                    first.opening_fen
                ));
            }

            let half_points =
                outcome_half_points(first.outcome) + outcome_half_points(second.outcome);
            pentanomial[half_points] += 1;
            pair_scores.push(half_points as f64 / 2.0);
        }

        let score = if summary.games == 0 {
            0.0
        } else {
            (summary.candidate_wins as f64 + 0.5 * summary.draws as f64) / summary.games as f64
        };
        let score_ci95 = paired_score_interval(&pair_scores);
        let elo = elo_from_score(score);
        let elo_ci95 =
            score_ci95.and_then(|(low, high)| Some((elo_from_score(low)?, elo_from_score(high)?)));

        Ok(Self {
            pentanomial,
            pairs: pair_scores.len(),
            score,
            score_ci95,
            elo,
            elo_ci95,
            terminations: termination_counts(summary),
        })
    }
}

pub fn print_match_statistics(summary: &MatchSummary) -> Result<(), String> {
    let statistics = MatchStatistics::from_paired_summary(summary)?;
    println!(
        "paired result: {} pairs; pentanomial [{}, {}, {}, {}, {}]",
        statistics.pairs,
        statistics.pentanomial[0],
        statistics.pentanomial[1],
        statistics.pentanomial[2],
        statistics.pentanomial[3],
        statistics.pentanomial[4]
    );

    match statistics.score_ci95 {
        Some((low, high)) => println!(
            "candidate score: {:.2}% (paired Wilson-style 95% CI {:.2}%..{:.2}%)",
            statistics.score * 100.0,
            low * 100.0,
            high * 100.0
        ),
        None => println!(
            "candidate score: {:.2}% (no opening pairs available for an interval)",
            statistics.score * 100.0
        ),
    }
    if let Some(elo) = statistics.elo {
        if let Some((low, high)) = statistics.elo_ci95 {
            println!("estimated Elo: {elo:+.1} (95% CI {low:+.1}..{high:+.1})");
        } else {
            println!("estimated Elo: {elo:+.1} (finite confidence bounds unavailable)");
        }
    } else {
        println!("estimated Elo: unbounded at a 0% or 100% score");
    }

    let terminations = statistics.terminations;
    println!(
        "terminations: checkmate {} stalemate {} repetition {} fifty-move {} insufficient-material {} max-ply-adjudicated-draw {} decisive-adjudication {}",
        terminations.checkmates,
        terminations.stalemates,
        terminations.repetitions,
        terminations.fifty_move_draws,
        terminations.insufficient_material_draws,
        terminations.max_ply_draws,
        terminations.adjudicated_decisive_games
    );

    if statistics.pairs < 100 {
        println!(
            "warning: fewer than 100 opening pairs; treat this result as exploratory, not a strength promotion"
        );
    }
    if terminations.max_ply_draws > 0 {
        println!(
            "warning: max-ply draws are adjudications, not board-rule draws; review the cutoff before interpreting Elo"
        );
    }

    Ok(())
}

fn validate_summary_totals(summary: &MatchSummary) -> Result<(), String> {
    if summary.game_summaries.len() != summary.games {
        return Err(format!(
            "match summary contains {} game records but reports {} games",
            summary.game_summaries.len(),
            summary.games
        ));
    }
    if summary.candidate_wins + summary.fallback_wins + summary.draws != summary.games {
        return Err("match W/D/L totals do not add up to the reported game count".to_owned());
    }
    Ok(())
}

fn outcome_half_points(outcome: MatchOutcome) -> usize {
    match outcome {
        MatchOutcome::CandidateWin => 2,
        MatchOutcome::Draw => 1,
        MatchOutcome::FallbackWin => 0,
    }
}

fn paired_score_interval(pair_scores: &[f64]) -> Option<(f64, f64)> {
    if pair_scores.is_empty() {
        return None;
    }

    let pair_count = pair_scores.len() as f64;
    let score = pair_scores.iter().sum::<f64>() / (2.0 * pair_count);
    // Treat each color-reversed opening pair as the independent sampling unit.
    // The Wilson variance is deliberately conservative for bounded fractional
    // pair scores and, unlike a raw sample-variance interval, remains non-zero
    // when a small match happens to contain only draws.
    let z_squared = Z_95 * Z_95;
    let denominator = 1.0 + z_squared / pair_count;
    let center = (score + z_squared / (2.0 * pair_count)) / denominator;
    let radius = Z_95
        * (score * (1.0 - score) / pair_count + z_squared / (4.0 * pair_count * pair_count)).sqrt()
        / denominator;
    Some(((center - radius).max(0.0), (center + radius).min(1.0)))
}

fn elo_from_score(score: f64) -> Option<f64> {
    if !(0.0..1.0).contains(&score) {
        return None;
    }
    let elo = -400.0 * ((1.0 / score) - 1.0).log10();
    Some(if elo.abs() < 0.05 { 0.0 } else { elo })
}

fn termination_counts(summary: &MatchSummary) -> TerminationCounts {
    let mut counts = TerminationCounts::default();
    for game in &summary.game_summaries {
        match game.terminal_status {
            PositionStatus::Checkmate => counts.checkmates += 1,
            PositionStatus::Stalemate => counts.stalemates += 1,
            PositionStatus::DrawByRepetition => counts.repetitions += 1,
            PositionStatus::DrawByFiftyMove => counts.fifty_move_draws += 1,
            PositionStatus::DrawByInsufficientMaterial => {
                counts.insufficient_material_draws += 1;
            }
            PositionStatus::Ongoing => match game.outcome {
                MatchOutcome::Draw => counts.max_ply_draws += 1,
                MatchOutcome::CandidateWin | MatchOutcome::FallbackWin => {
                    counts.adjudicated_decisive_games += 1;
                }
            },
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use volkrix::{
        core::{Color, PositionStatus},
        nnue_training::{MatchGameSummary, MatchOutcome, MatchSummary},
    };

    use super::MatchStatistics;

    #[test]
    fn paired_statistics_preserve_color_reversed_openings() {
        let summary = summary_with_outcomes(&[
            (MatchOutcome::CandidateWin, MatchOutcome::Draw),
            (MatchOutcome::FallbackWin, MatchOutcome::CandidateWin),
        ]);
        let statistics = MatchStatistics::from_paired_summary(&summary).expect("valid pairs");

        assert_eq!(statistics.pentanomial, [0, 0, 1, 1, 0]);
        assert_eq!(statistics.pairs, 2);
        assert!((statistics.score - 0.625).abs() < f64::EPSILON);
        assert!(statistics.score_ci95.is_some());
        assert_eq!(statistics.terminations.max_ply_draws, 1);
        assert_eq!(statistics.terminations.checkmates, 3);
    }

    #[test]
    fn rejects_a_pair_without_reversed_colors() {
        let mut summary =
            summary_with_outcomes(&[(MatchOutcome::CandidateWin, MatchOutcome::FallbackWin)]);
        summary.game_summaries[1].candidate_color = Color::White;

        let error = MatchStatistics::from_paired_summary(&summary).expect_err("invalid pair");
        assert!(error.contains("did not reverse candidate colors"));
    }

    #[test]
    fn all_draw_sample_keeps_nonzero_uncertainty() {
        let summary = summary_with_outcomes(&[
            (MatchOutcome::Draw, MatchOutcome::Draw),
            (MatchOutcome::Draw, MatchOutcome::Draw),
        ]);
        let statistics = MatchStatistics::from_paired_summary(&summary).expect("valid pairs");
        let (low, high) = statistics.score_ci95.expect("interval");

        assert!(low < 0.5);
        assert!(high > 0.5);
        assert_eq!(statistics.elo, Some(0.0));
    }

    fn summary_with_outcomes(pairs: &[(MatchOutcome, MatchOutcome)]) -> MatchSummary {
        let mut summary = MatchSummary {
            openings: pairs.len(),
            ..MatchSummary::default()
        };
        for (index, (first, second)) in pairs.iter().copied().enumerate() {
            for (candidate_color, outcome) in [(Color::White, first), (Color::Black, second)] {
                summary.games += 1;
                match outcome {
                    MatchOutcome::CandidateWin => summary.candidate_wins += 1,
                    MatchOutcome::FallbackWin => summary.fallback_wins += 1,
                    MatchOutcome::Draw => summary.draws += 1,
                }
                summary.game_summaries.push(MatchGameSummary {
                    opening_fen: format!("opening-{index}"),
                    candidate_color,
                    outcome,
                    terminal_status: if outcome == MatchOutcome::Draw {
                        PositionStatus::Ongoing
                    } else {
                        PositionStatus::Checkmate
                    },
                    plies_played: 100,
                    first_candidate_score_cp: None,
                    first_candidate_info_line: None,
                    first_fallback_score_cp: None,
                    first_fallback_info_line: None,
                });
            }
        }
        summary
    }
}
