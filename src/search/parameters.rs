//! Bounded search parameters for controlled SPSA/OpenBench experiments.
//!
//! This module is compiled only with `spsa-tuning`. Normal production builds
//! continue to use the literal constants in the search and time-management
//! code, keeping their code generation and UCI surface isolated from tuning.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ParameterSpec {
    pub(crate) name: &'static str,
    pub(crate) default: i32,
    pub(crate) min: i32,
    pub(crate) max: i32,
    pub(crate) step: i32,
}

macro_rules! tuning_parameters {
    ($(($field:ident, $name:literal, $default:literal, $min:literal, $max:literal, $step:literal)),+ $(,)?) => {
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub(crate) struct SearchParameters {
            $(pub(crate) $field: i32,)+
        }

        pub(crate) const PARAMETER_SPECS: &[ParameterSpec] = &[
            $(ParameterSpec {
                name: $name,
                default: $default,
                min: $min,
                max: $max,
                step: $step,
            },)+
        ];

        impl SearchParameters {
            pub(crate) const DEFAULT: Self = Self {
                $($field: $default,)+
            };

            pub(crate) fn get(self, name: &str) -> Option<i32> {
                match name {
                    $($name => Some(self.$field),)+
                    _ => None,
                }
            }

            pub(crate) fn set(&mut self, name: &str, value: i32) -> Result<(), String> {
                let spec = parameter_spec(name)
                    .ok_or_else(|| format!("unknown tuning parameter '{name}'"))?;
                if !(spec.min..=spec.max).contains(&value) {
                    return Err(format!(
                        "{} value must be between {} and {}",
                        spec.name, spec.min, spec.max
                    ));
                }
                match name {
                    $($name => self.$field = value,)+
                    _ => unreachable!("validated parameter name must have a storage field"),
                }
                Ok(())
            }
        }
    };
}

// Bounds are intentionally conservative. They exclude arithmetic hazards and
// nonsensical search shapes while leaving enough range for an initial broad
// SPSA pass. `step` is a recommended OpenBench starting step, not a validation
// constraint: SPSA must be free to converge to every integer in the interval.
tuning_parameters!(
    (aspiration_delta, "TuneAspirationDelta", 36, 8, 128, 4),
    (lmr_divisor_pct, "TuneLmrDivisorPct", 150, 80, 300, 10),
    (null_base_reduction, "TuneNullBaseReduction", 2, 1, 5, 1),
    (null_depth_divisor, "TuneNullDepthDivisor", 6, 3, 12, 1),
    (null_eval_divisor, "TuneNullEvalDivisor", 256, 96, 512, 16),
    (null_static_margin, "TuneNullStaticMargin", 32, 0, 160, 8),
    (null_verify_depth, "TuneNullVerifyDepth", 10, 6, 16, 1),
    (
        reverse_futility_slope,
        "TuneReverseFutilitySlope",
        140,
        60,
        240,
        10
    ),
    (futility_base, "TuneFutilityBase", 90, 0, 240, 10),
    (futility_slope, "TuneFutilitySlope", 120, 40, 240, 10),
    (late_move_base, "TuneLateMoveBase", 3, 1, 8, 1),
    (late_move_slope, "TuneLateMoveSlope", 3, 1, 8, 1),
    (see_margin, "TuneSeeMargin", 70, 20, 160, 10),
    (
        history_prune_threshold,
        "TuneHistoryPruneThreshold",
        2000,
        500,
        6000,
        250
    ),
    (history_bonus_scale, "TuneHistoryBonusScale", 32, 8, 96, 4),
    (probcut_base, "TuneProbCutBase", 180, 80, 300, 10),
    (probcut_slope, "TuneProbCutSlope", 5, 0, 15, 1),
    (
        probcut_static_offset,
        "TuneProbCutStaticOffset",
        80,
        0,
        200,
        10
    ),
    (time_increment_pct, "TuneTimeIncrementPct", 75, 25, 125, 5),
    (time_hard_pct, "TuneTimeHardPct", 150, 110, 250, 10),
    (time_stable1_pct, "TuneTimeStable1Pct", 95, 70, 110, 5),
    (time_stable2_pct, "TuneTimeStable2Pct", 82, 55, 100, 5),
    (time_stable3_pct, "TuneTimeStable3Pct", 70, 40, 90, 5),
    (time_unstable_pct, "TuneTimeUnstablePct", 125, 100, 180, 5),
    (
        time_score_swing_pct,
        "TuneTimeScoreSwingPct",
        145,
        110,
        220,
        5
    ),
    (time_score_swing_cp, "TuneTimeScoreSwingCp", 80, 20, 200, 10),
);

impl Default for SearchParameters {
    fn default() -> Self {
        Self::DEFAULT
    }
}

pub(crate) fn parameter_spec(name: &str) -> Option<&'static ParameterSpec> {
    PARAMETER_SPECS.iter().find(|spec| spec.name == name)
}

/// Stable, dependency-free FNV-1a identity of the full schema and live values.
pub(crate) fn manifest_checksum(parameters: SearchParameters) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in b"volkrix-tuning-manifest-v1\n" {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    for spec in PARAMETER_SPECS {
        let record = format!(
            "{}|{}|{}|{}|{}|{}\n",
            spec.name,
            parameters
                .get(spec.name)
                .expect("every spec must have a value"),
            spec.default,
            spec.min,
            spec.max,
            spec.step
        );
        for byte in record.bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

pub(crate) fn manifest_lines(parameters: SearchParameters) -> Vec<String> {
    let mut lines = Vec::with_capacity(PARAMETER_SPECS.len() + 2);
    lines.push(format!(
        "info string tuning manifest version 1 checksum {:016x}",
        manifest_checksum(parameters)
    ));
    lines.extend(PARAMETER_SPECS.iter().map(|spec| {
        format!(
            "info string tuning parameter {} value {} default {} min {} max {} step {}",
            spec.name,
            parameters
                .get(spec.name)
                .expect("every spec must have a value"),
            spec.default,
            spec.min,
            spec.max,
            spec.step
        )
    }));
    lines.push("info string tuning manifest end".to_owned());
    lines
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{PARAMETER_SPECS, SearchParameters, manifest_checksum, manifest_lines};

    #[test]
    fn schema_names_are_unique_and_defaults_are_valid() {
        let mut names = HashSet::new();
        for spec in PARAMETER_SPECS {
            assert!(names.insert(spec.name));
            assert!(spec.min <= spec.default && spec.default <= spec.max);
            assert!(spec.step > 0 && spec.step <= spec.max - spec.min);
            assert_eq!(SearchParameters::DEFAULT.get(spec.name), Some(spec.default));
        }
    }

    #[test]
    fn mutation_is_transactional_and_bounded() {
        let mut parameters = SearchParameters::DEFAULT;
        parameters.set("TuneAspirationDelta", 48).unwrap();
        assert_eq!(parameters.aspiration_delta, 48);
        let before = parameters;
        assert!(parameters.set("TuneAspirationDelta", 129).is_err());
        assert!(parameters.set("NotAParameter", 1).is_err());
        assert_eq!(parameters, before);
    }

    #[test]
    fn manifest_is_deterministic_and_value_sensitive() {
        let defaults = SearchParameters::DEFAULT;
        let first = manifest_lines(defaults);
        assert_eq!(first, manifest_lines(defaults));
        assert_eq!(first.len(), PARAMETER_SPECS.len() + 2);
        let mut changed = defaults;
        changed.set("TuneFutilityBase", 100).unwrap();
        assert_ne!(manifest_checksum(defaults), manifest_checksum(changed));
    }
}
