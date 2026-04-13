use std::{
    fs,
    path::{Path, PathBuf},
};

use serde_json::Value;
use serde::{Deserialize, Serialize};
use volkrix::{
    SOURCE_COMMIT,
    core::Position,
    nnue_training::{read_examples, split_for_normalized_fen},
    search::eval::{ClassicalEvalWeights, PhaseScore, evaluate_with_weights},
};

const TEXEL_WEIGHTS_MAGIC: &str = "VOLKRIX_TEXEL_WEIGHTS";
const TEXEL_WEIGHTS_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TexelTuningConfig {
    pub iterations: u32,
    pub initial_step: i32,
    pub sigmoid_scale: f64,
    pub regularization: f64,
    pub max_examples: Option<usize>,
}

impl Default for TexelTuningConfig {
    fn default() -> Self {
        Self {
            iterations: 6,
            initial_step: 8,
            sigmoid_scale: 400.0,
            regularization: 1e-6,
            max_examples: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TexelWeightsFile {
    pub magic: String,
    pub version: u32,
    pub source_engine_commit: String,
    pub examples_path: String,
    pub iterations: u32,
    pub initial_step: i32,
    pub sigmoid_scale: f64,
    pub regularization: f64,
    pub train_examples: usize,
    pub validation_examples: usize,
    pub initial_train_loss: f64,
    pub final_train_loss: f64,
    pub initial_validation_loss: Option<f64>,
    pub final_validation_loss: Option<f64>,
    pub accepted_updates: usize,
    pub tuned_weights: ClassicalEvalWeights,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TexelTuningSummary {
    pub output_path: PathBuf,
    pub train_examples: usize,
    pub validation_examples: usize,
    pub initial_train_loss: f64,
    pub final_train_loss: f64,
    pub initial_validation_loss: Option<f64>,
    pub final_validation_loss: Option<f64>,
    pub accepted_updates: usize,
    pub iterations_run: u32,
}

#[derive(Clone, Debug)]
struct TuningSample {
    position: Position,
    target_probability: f64,
}

struct ParameterSpec {
    #[allow(dead_code)]
    name: &'static str,
    get: Box<dyn Fn(&ClassicalEvalWeights) -> i32>,
    set: Box<dyn Fn(&mut ClassicalEvalWeights, i32)>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct TuneRunResult {
    weights: ClassicalEvalWeights,
    initial_train_loss: f64,
    final_train_loss: f64,
    initial_validation_loss: Option<f64>,
    final_validation_loss: Option<f64>,
    accepted_updates: usize,
    iterations_run: u32,
}

pub fn tune_from_examples(
    examples_path: &Path,
    output_path: &Path,
    config: TexelTuningConfig,
) -> Result<TexelTuningSummary, String> {
    if config.iterations == 0 {
        return Err("texel tuning iterations must be at least 1".to_owned());
    }
    if config.initial_step <= 0 {
        return Err("texel tuning initial_step must be positive".to_owned());
    }
    if !(config.sigmoid_scale.is_finite() && config.sigmoid_scale > 0.0) {
        return Err("texel tuning sigmoid_scale must be a finite positive number".to_owned());
    }
    if !config.regularization.is_finite() || config.regularization < 0.0 {
        return Err("texel tuning regularization must be finite and non-negative".to_owned());
    }

    let (_, records, _) = read_examples(examples_path)?;
    let limited_records = if let Some(limit) = config.max_examples {
        records.into_iter().take(limit).collect::<Vec<_>>()
    } else {
        records
    };

    let mut train = Vec::new();
    let mut validation = Vec::new();
    for record in limited_records {
        let sample = TuningSample {
            position: Position::from_fen(&record.normalized_fen).map_err(|error| {
                format!(
                    "failed to parse normalized FEN '{}' from examples file: {error}",
                    record.normalized_fen
                )
            })?,
            target_probability: score_to_probability(record.target_cp as f64, config.sigmoid_scale),
        };
        match split_for_normalized_fen(&record.normalized_fen) {
            volkrix::nnue_training::DatasetSplit::Train => train.push(sample),
            volkrix::nnue_training::DatasetSplit::Validation => validation.push(sample),
        }
    }

    if train.is_empty() {
        return Err("texel tuning requires at least one training example".to_owned());
    }

    let run = run_coordinate_descent(
        &train,
        &validation,
        ClassicalEvalWeights::default(),
        config,
    );
    let weights_file = TexelWeightsFile {
        magic: TEXEL_WEIGHTS_MAGIC.to_owned(),
        version: TEXEL_WEIGHTS_VERSION,
        source_engine_commit: SOURCE_COMMIT.to_owned(),
        examples_path: examples_path.display().to_string(),
        iterations: run.iterations_run,
        initial_step: config.initial_step,
        sigmoid_scale: config.sigmoid_scale,
        regularization: config.regularization,
        train_examples: train.len(),
        validation_examples: validation.len(),
        initial_train_loss: run.initial_train_loss,
        final_train_loss: run.final_train_loss,
        initial_validation_loss: run.initial_validation_loss,
        final_validation_loss: run.final_validation_loss,
        accepted_updates: run.accepted_updates,
        tuned_weights: run.weights,
    };

    let temp_output = temporary_output_path(output_path);
    fs::write(
        &temp_output,
        serde_json::to_string_pretty(&weights_file)
            .map_err(|error| format!("failed to encode texel weights JSON: {error}"))?,
    )
    .map_err(|error| {
        format!(
            "failed to write texel weights output '{}': {error}",
            temp_output.display()
        )
    })?;
    fs::rename(&temp_output, output_path).map_err(|error| {
        format!(
            "failed to finalize texel weights output '{}': {error}",
            output_path.display()
        )
    })?;

    Ok(TexelTuningSummary {
        output_path: output_path.to_path_buf(),
        train_examples: train.len(),
        validation_examples: validation.len(),
        initial_train_loss: run.initial_train_loss,
        final_train_loss: run.final_train_loss,
        initial_validation_loss: run.initial_validation_loss,
        final_validation_loss: run.final_validation_loss,
        accepted_updates: run.accepted_updates,
        iterations_run: run.iterations_run,
    })
}

pub fn read_classical_weights(path: &Path) -> Result<ClassicalEvalWeights, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("failed to read classical weights '{}': {error}", path.display()))?;
    let value: Value = serde_json::from_str(&text)
        .map_err(|error| format!("failed to parse classical weights JSON '{}': {error}", path.display()))?;

    if let Some(tuned_weights) = value.get("tuned_weights") {
        serde_json::from_value::<ClassicalEvalWeights>(tuned_weights.clone()).map_err(|error| {
            format!(
                "failed to parse tuned_weights from '{}': {error}",
                path.display()
            )
        })
    } else {
        serde_json::from_value::<ClassicalEvalWeights>(value).map_err(|error| {
            format!(
                "failed to parse classical weight object from '{}': {error}",
                path.display()
            )
        })
    }
}

fn run_coordinate_descent(
    train: &[TuningSample],
    validation: &[TuningSample],
    initial: ClassicalEvalWeights,
    config: TexelTuningConfig,
) -> TuneRunResult {
    let baseline = initial;
    let mut weights = initial;
    let params = parameter_specs();
    let mut step = config.initial_step.max(1);
    let initial_train_loss = objective_loss(train, &weights, &baseline, config);
    let initial_validation_loss = average_log_loss(validation, &weights, config.sigmoid_scale);
    let mut current_train_loss = initial_train_loss;
    let mut accepted_updates = 0usize;
    let mut iterations_run = 0u32;

    // Coordinate descent keeps the implementation deterministic and simple enough
    // for the existing offline toolchain while still giving us a practical first-pass tuner.
    for _ in 0..config.iterations {
        iterations_run += 1;
        let mut improved_any = false;

        for param in &params {
            let current_value = (param.get)(&weights);
            let mut best_value = current_value;
            let mut best_loss = current_train_loss;

            for candidate in [current_value + step, current_value - step] {
                let mut candidate_weights = weights;
                (param.set)(&mut candidate_weights, candidate);
                let candidate_loss = objective_loss(train, &candidate_weights, &baseline, config);
                if candidate_loss + 1e-12 < best_loss {
                    best_loss = candidate_loss;
                    best_value = candidate;
                }
            }

            if best_value != current_value {
                (param.set)(&mut weights, best_value);
                current_train_loss = best_loss;
                accepted_updates += 1;
                improved_any = true;
            }
        }

        if !improved_any {
            if step == 1 {
                break;
            }
            step = (step / 2).max(1);
        }
    }

    TuneRunResult {
        weights,
        initial_train_loss,
        final_train_loss: average_log_loss(train, &weights, config.sigmoid_scale).unwrap_or(0.0)
            + regularization_penalty(&weights, &baseline, config.regularization),
        initial_validation_loss,
        final_validation_loss: average_log_loss(validation, &weights, config.sigmoid_scale),
        accepted_updates,
        iterations_run,
    }
}

fn objective_loss(
    samples: &[TuningSample],
    weights: &ClassicalEvalWeights,
    baseline: &ClassicalEvalWeights,
    config: TexelTuningConfig,
) -> f64 {
    average_log_loss(samples, weights, config.sigmoid_scale).unwrap_or(0.0)
        + regularization_penalty(weights, baseline, config.regularization)
}

fn average_log_loss(
    samples: &[TuningSample],
    weights: &ClassicalEvalWeights,
    sigmoid_scale: f64,
) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }

    let total = samples
        .iter()
        .map(|sample| sample_log_loss(sample, weights, sigmoid_scale))
        .sum::<f64>();
    Some(total / samples.len() as f64)
}

fn sample_log_loss(sample: &TuningSample, weights: &ClassicalEvalWeights, sigmoid_scale: f64) -> f64 {
    let predicted_score = evaluate_with_weights(&sample.position, weights).0 as f64;
    let predicted_probability = score_to_probability(predicted_score, sigmoid_scale).clamp(1e-12, 1.0 - 1e-12);
    let target_probability = sample.target_probability.clamp(1e-12, 1.0 - 1e-12);
    -(target_probability * predicted_probability.ln()
        + (1.0 - target_probability) * (1.0 - predicted_probability).ln())
}

fn score_to_probability(score_cp: f64, sigmoid_scale: f64) -> f64 {
    1.0 / (1.0 + (-score_cp / sigmoid_scale).exp())
}

fn regularization_penalty(
    weights: &ClassicalEvalWeights,
    baseline: &ClassicalEvalWeights,
    regularization: f64,
) -> f64 {
    if regularization == 0.0 {
        return 0.0;
    }

    let squared = parameter_specs()
        .iter()
        .map(|param| {
            let delta = (param.get)(weights) - (param.get)(baseline);
            (delta * delta) as f64
        })
        .sum::<f64>();
    regularization * squared
}

fn temporary_output_path(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.tmp", path.display()))
}

fn parameter_specs() -> Vec<ParameterSpec> {
    vec![
        ParameterSpec { name: "mg_pawn", get: Box::new(|w| w.mg_values[0]), set: Box::new(|w, v| w.mg_values[0] = v) },
        ParameterSpec { name: "mg_knight", get: Box::new(|w| w.mg_values[1]), set: Box::new(|w, v| w.mg_values[1] = v) },
        ParameterSpec { name: "mg_bishop", get: Box::new(|w| w.mg_values[2]), set: Box::new(|w, v| w.mg_values[2] = v) },
        ParameterSpec { name: "mg_rook", get: Box::new(|w| w.mg_values[3]), set: Box::new(|w, v| w.mg_values[3] = v) },
        ParameterSpec { name: "mg_queen", get: Box::new(|w| w.mg_values[4]), set: Box::new(|w, v| w.mg_values[4] = v) },
        ParameterSpec { name: "eg_pawn", get: Box::new(|w| w.eg_values[0]), set: Box::new(|w, v| w.eg_values[0] = v) },
        ParameterSpec { name: "eg_knight", get: Box::new(|w| w.eg_values[1]), set: Box::new(|w, v| w.eg_values[1] = v) },
        ParameterSpec { name: "eg_bishop", get: Box::new(|w| w.eg_values[2]), set: Box::new(|w, v| w.eg_values[2] = v) },
        ParameterSpec { name: "eg_rook", get: Box::new(|w| w.eg_values[3]), set: Box::new(|w, v| w.eg_values[3] = v) },
        ParameterSpec { name: "eg_queen", get: Box::new(|w| w.eg_values[4]), set: Box::new(|w, v| w.eg_values[4] = v) },
        phase_score_param("knight_mobility_mg", |w| &w.knight_mobility, |w| &mut w.knight_mobility, true),
        phase_score_param("knight_mobility_eg", |w| &w.knight_mobility, |w| &mut w.knight_mobility, false),
        phase_score_param("bishop_mobility_mg", |w| &w.bishop_mobility, |w| &mut w.bishop_mobility, true),
        phase_score_param("bishop_mobility_eg", |w| &w.bishop_mobility, |w| &mut w.bishop_mobility, false),
        phase_score_param("rook_mobility_mg", |w| &w.rook_mobility, |w| &mut w.rook_mobility, true),
        phase_score_param("rook_mobility_eg", |w| &w.rook_mobility, |w| &mut w.rook_mobility, false),
        phase_score_param("queen_mobility_mg", |w| &w.queen_mobility, |w| &mut w.queen_mobility, true),
        phase_score_param("queen_mobility_eg", |w| &w.queen_mobility, |w| &mut w.queen_mobility, false),
        phase_score_param("knight_outpost_bonus_mg", |w| &w.knight_outpost_bonus, |w| &mut w.knight_outpost_bonus, true),
        phase_score_param("knight_outpost_bonus_eg", |w| &w.knight_outpost_bonus, |w| &mut w.knight_outpost_bonus, false),
        phase_score_param("bishop_pair_bonus_mg", |w| &w.bishop_pair_bonus, |w| &mut w.bishop_pair_bonus, true),
        phase_score_param("bishop_pair_bonus_eg", |w| &w.bishop_pair_bonus, |w| &mut w.bishop_pair_bonus, false),
        phase_score_param("doubled_pawn_penalty_mg", |w| &w.doubled_pawn_penalty, |w| &mut w.doubled_pawn_penalty, true),
        phase_score_param("doubled_pawn_penalty_eg", |w| &w.doubled_pawn_penalty, |w| &mut w.doubled_pawn_penalty, false),
        phase_score_param("isolated_pawn_penalty_mg", |w| &w.isolated_pawn_penalty, |w| &mut w.isolated_pawn_penalty, true),
        phase_score_param("isolated_pawn_penalty_eg", |w| &w.isolated_pawn_penalty, |w| &mut w.isolated_pawn_penalty, false),
        phase_score_param("pawn_island_penalty_mg", |w| &w.pawn_island_penalty, |w| &mut w.pawn_island_penalty, true),
        phase_score_param("pawn_island_penalty_eg", |w| &w.pawn_island_penalty, |w| &mut w.pawn_island_penalty, false),
        phase_score_param("phalanx_pawn_bonus_mg", |w| &w.phalanx_pawn_bonus, |w| &mut w.phalanx_pawn_bonus, true),
        phase_score_param("phalanx_pawn_bonus_eg", |w| &w.phalanx_pawn_bonus, |w| &mut w.phalanx_pawn_bonus, false),
        phase_score_param("open_file_rook_bonus_mg", |w| &w.open_file_rook_bonus, |w| &mut w.open_file_rook_bonus, true),
        phase_score_param("open_file_rook_bonus_eg", |w| &w.open_file_rook_bonus, |w| &mut w.open_file_rook_bonus, false),
        phase_score_param("semi_open_file_rook_bonus_mg", |w| &w.semi_open_file_rook_bonus, |w| &mut w.semi_open_file_rook_bonus, true),
        phase_score_param("semi_open_file_rook_bonus_eg", |w| &w.semi_open_file_rook_bonus, |w| &mut w.semi_open_file_rook_bonus, false),
        phase_score_param("rook_on_seventh_bonus_mg", |w| &w.rook_on_seventh_bonus, |w| &mut w.rook_on_seventh_bonus, true),
        phase_score_param("rook_on_seventh_bonus_eg", |w| &w.rook_on_seventh_bonus, |w| &mut w.rook_on_seventh_bonus, false),
        phase_array_param("passed_pawn_bonus_r2_mg", |w| &w.passed_pawn_bonus, |w| &mut w.passed_pawn_bonus, 1, true),
        phase_array_param("passed_pawn_bonus_r2_eg", |w| &w.passed_pawn_bonus, |w| &mut w.passed_pawn_bonus, 1, false),
        phase_array_param("passed_pawn_bonus_r3_mg", |w| &w.passed_pawn_bonus, |w| &mut w.passed_pawn_bonus, 2, true),
        phase_array_param("passed_pawn_bonus_r3_eg", |w| &w.passed_pawn_bonus, |w| &mut w.passed_pawn_bonus, 2, false),
        phase_array_param("passed_pawn_bonus_r4_mg", |w| &w.passed_pawn_bonus, |w| &mut w.passed_pawn_bonus, 3, true),
        phase_array_param("passed_pawn_bonus_r4_eg", |w| &w.passed_pawn_bonus, |w| &mut w.passed_pawn_bonus, 3, false),
        phase_array_param("passed_pawn_bonus_r5_mg", |w| &w.passed_pawn_bonus, |w| &mut w.passed_pawn_bonus, 4, true),
        phase_array_param("passed_pawn_bonus_r5_eg", |w| &w.passed_pawn_bonus, |w| &mut w.passed_pawn_bonus, 4, false),
        phase_array_param("passed_pawn_bonus_r6_mg", |w| &w.passed_pawn_bonus, |w| &mut w.passed_pawn_bonus, 5, true),
        phase_array_param("passed_pawn_bonus_r6_eg", |w| &w.passed_pawn_bonus, |w| &mut w.passed_pawn_bonus, 5, false),
        phase_array_param("protected_passed_pawn_bonus_r3_mg", |w| &w.protected_passed_pawn_bonus, |w| &mut w.protected_passed_pawn_bonus, 2, true),
        phase_array_param("protected_passed_pawn_bonus_r3_eg", |w| &w.protected_passed_pawn_bonus, |w| &mut w.protected_passed_pawn_bonus, 2, false),
        phase_array_param("protected_passed_pawn_bonus_r4_mg", |w| &w.protected_passed_pawn_bonus, |w| &mut w.protected_passed_pawn_bonus, 3, true),
        phase_array_param("protected_passed_pawn_bonus_r4_eg", |w| &w.protected_passed_pawn_bonus, |w| &mut w.protected_passed_pawn_bonus, 3, false),
        phase_array_param("protected_passed_pawn_bonus_r5_mg", |w| &w.protected_passed_pawn_bonus, |w| &mut w.protected_passed_pawn_bonus, 4, true),
        phase_array_param("protected_passed_pawn_bonus_r5_eg", |w| &w.protected_passed_pawn_bonus, |w| &mut w.protected_passed_pawn_bonus, 4, false),
        phase_array_param("protected_passed_pawn_bonus_r6_mg", |w| &w.protected_passed_pawn_bonus, |w| &mut w.protected_passed_pawn_bonus, 5, true),
        phase_array_param("protected_passed_pawn_bonus_r6_eg", |w| &w.protected_passed_pawn_bonus, |w| &mut w.protected_passed_pawn_bonus, 5, false),
        phase_score_param("pawn_threat_minor_mg", |w| &w.pawn_threat_minor, |w| &mut w.pawn_threat_minor, true),
        phase_score_param("pawn_threat_minor_eg", |w| &w.pawn_threat_minor, |w| &mut w.pawn_threat_minor, false),
        phase_score_param("pawn_threat_rook_mg", |w| &w.pawn_threat_rook, |w| &mut w.pawn_threat_rook, true),
        phase_score_param("pawn_threat_rook_eg", |w| &w.pawn_threat_rook, |w| &mut w.pawn_threat_rook, false),
        phase_score_param("pawn_threat_queen_mg", |w| &w.pawn_threat_queen, |w| &mut w.pawn_threat_queen, true),
        phase_score_param("pawn_threat_queen_eg", |w| &w.pawn_threat_queen, |w| &mut w.pawn_threat_queen, false),
        phase_score_param("minor_threat_rook_mg", |w| &w.minor_threat_rook, |w| &mut w.minor_threat_rook, true),
        phase_score_param("minor_threat_rook_eg", |w| &w.minor_threat_rook, |w| &mut w.minor_threat_rook, false),
        phase_score_param("minor_threat_queen_mg", |w| &w.minor_threat_queen, |w| &mut w.minor_threat_queen, true),
        phase_score_param("minor_threat_queen_eg", |w| &w.minor_threat_queen, |w| &mut w.minor_threat_queen, false),
    ]
}

fn phase_score_param(
    name: &'static str,
    get_field: fn(&ClassicalEvalWeights) -> &PhaseScore,
    get_field_mut: fn(&mut ClassicalEvalWeights) -> &mut PhaseScore,
    middlegame: bool,
) -> ParameterSpec {
    ParameterSpec {
        name,
        get: Box::new(move |weights| {
            let score = get_field(weights);
            if middlegame { score.mg } else { score.eg }
        }),
        set: Box::new(move |weights, value| {
            let score = get_field_mut(weights);
            if middlegame {
                score.mg = value;
            } else {
                score.eg = value;
            }
        }),
    }
}

fn phase_array_param(
    name: &'static str,
    get_field: fn(&ClassicalEvalWeights) -> &[PhaseScore; 8],
    get_field_mut: fn(&mut ClassicalEvalWeights) -> &mut [PhaseScore; 8],
    index: usize,
    middlegame: bool,
) -> ParameterSpec {
    ParameterSpec {
        name,
        get: Box::new(move |weights| {
            let score = &get_field(weights)[index];
            if middlegame { score.mg } else { score.eg }
        }),
        set: Box::new(move |weights, value| {
            let score = &mut get_field_mut(weights)[index];
            if middlegame {
                score.mg = value;
            } else {
                score.eg = value;
            }
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use volkrix::search::eval::ClassicalEvalWeights;

    fn sample_from_target_weights(
        fen: &str,
        target_weights: &ClassicalEvalWeights,
        sigmoid_scale: f64,
    ) -> TuningSample {
        let position = Position::from_fen(fen).expect("FEN parse must succeed");
        let target_score = evaluate_with_weights(&position, target_weights).0 as f64;
        TuningSample {
            position,
            target_probability: score_to_probability(target_score, sigmoid_scale),
        }
    }

    #[test]
    fn coordinate_descent_reduces_training_loss_on_controlled_samples() {
        let mut target = ClassicalEvalWeights::default();
        target.doubled_pawn_penalty.mg += 24;
        target.doubled_pawn_penalty.eg += 16;
        target.isolated_pawn_penalty.mg += 12;
        target.isolated_pawn_penalty.eg += 8;

        let config = TexelTuningConfig {
            iterations: 8,
            initial_step: 8,
            sigmoid_scale: 400.0,
            regularization: 0.0,
            max_examples: None,
        };
        let train = vec![
            sample_from_target_weights("4k3/8/8/8/8/3P4/2P1P3/4K3 w - - 0 1", &target, config.sigmoid_scale),
            sample_from_target_weights("4k3/8/8/8/8/2P5/P1P5/4K3 w - - 0 1", &target, config.sigmoid_scale),
            sample_from_target_weights("4k3/8/8/8/8/8/3PP3/4K3 w - - 0 1", &target, config.sigmoid_scale),
            sample_from_target_weights("4k3/8/8/8/8/8/2P1P3/4K3 w - - 0 1", &target, config.sigmoid_scale),
        ];
        let result = run_coordinate_descent(&train, &[], ClassicalEvalWeights::default(), config);

        assert!(result.final_train_loss < result.initial_train_loss);
        assert!(result.weights.doubled_pawn_penalty.mg >= ClassicalEvalWeights::default().doubled_pawn_penalty.mg);
        assert!(result.accepted_updates > 0);
    }

    #[test]
    fn parameter_specs_are_unique() {
        let params = parameter_specs();
        let mut names = params.iter().map(|param| param.name).collect::<Vec<_>>();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), params.len());
    }
}
