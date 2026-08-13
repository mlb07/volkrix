//! Pure aspiration and time-management policy helpers.
//!
//! The production clock allocation remains available as an exact, frozen
//! function.  Richer behavior is deliberately opt-in until paired game testing
//! demonstrates a gain.  Keeping the arithmetic here free of wall-clock reads
//! makes overflow, time-loss, and search-instability behavior deterministic.

#[cfg(any(test, debug_assertions, feature = "internal-testing"))]
const DEFAULT_MOVES_TO_GO: u64 = 25;
#[cfg(any(test, debug_assertions, feature = "internal-testing"))]
const MAX_WINDOW: i32 = 31_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ClockBudgetInput {
    pub(crate) remaining_ms: u64,
    pub(crate) increment_ms: u64,
    pub(crate) moves_to_go: Option<u32>,
    pub(crate) overhead_ms: u64,
    pub(crate) opponent_remaining_ms: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ClockBudget {
    pub(crate) soft_ms: u64,
    pub(crate) hard_ms: u64,
    pub(crate) available_ms: u64,
}

impl ClockBudget {
    fn new(soft_ms: u64, hard_ms: u64, available_ms: u64) -> Self {
        debug_assert!(soft_ms <= hard_ms);
        debug_assert!(hard_ms <= available_ms);
        Self {
            soft_ms,
            hard_ms,
            available_ms,
        }
    }
}

/// The exact production allocation, centralized here to make equivalence and
/// safety properties independently testable.
pub(crate) fn production_clock_budget(input: ClockBudgetInput) -> ClockBudget {
    let reserve = input.overhead_ms.saturating_add(input.remaining_ms / 100);
    let available = input.remaining_ms.saturating_sub(reserve);
    let moves_to_go = u64::from(input.moves_to_go.unwrap_or(25).max(1));
    let base = available / moves_to_go;
    let soft = available.min(base.saturating_add(input.increment_ms.saturating_mul(3) / 4));
    let hard = available.min(
        soft.saturating_mul(3)
            .saturating_div(2)
            .max(soft.saturating_add(10)),
    );
    ClockBudget::new(soft, hard, available)
}

/// Candidate clock allocation with explicit low-clock and opponent-clock
/// handling. It never spends the reserve and cannot construct a deadline past
/// the usable clock, including for adversarial `u64::MAX` UCI inputs.
#[cfg(any(test, debug_assertions, feature = "internal-testing"))]
pub(crate) fn candidate_clock_budget(input: ClockBudgetInput) -> ClockBudget {
    let percentage_reserve = input.remaining_ms / 100;
    let emergency_reserve = input.increment_ms.min(input.remaining_ms / 50);
    let reserve = input
        .overhead_ms
        .saturating_add(percentage_reserve)
        .saturating_add(emergency_reserve);
    let available = input.remaining_ms.saturating_sub(reserve);
    if available == 0 {
        return ClockBudget::new(0, 0, 0);
    }

    let horizon = u64::from(
        input
            .moves_to_go
            .unwrap_or(if input.increment_ms == 0 {
                DEFAULT_MOVES_TO_GO as u32
            } else {
                28
            })
            .max(1),
    );
    let base = available / horizon;
    let low_clock =
        input.increment_ms > 0 && input.remaining_ms <= input.increment_ms.saturating_mul(20);
    let increment_pct = if low_clock { 80 } else { 75 };
    let mut soft = base.saturating_add(
        input
            .increment_ms
            .saturating_mul(increment_pct)
            .saturating_div(100),
    );

    // When substantially ahead on the clock, preserve that practical edge;
    // when behind, spend a little more to avoid compounding a weak position.
    if let Some(opponent) = input.opponent_remaining_ms.filter(|&ms| ms > 0) {
        if input.remaining_ms >= opponent.saturating_mul(2) {
            soft = soft.saturating_mul(95).saturating_div(100);
        } else if opponent >= input.remaining_ms.saturating_mul(2) {
            soft = soft.saturating_mul(110).saturating_div(100);
        }
    }

    soft = soft.min(available);
    let hard = available.min(
        soft.saturating_mul(3)
            .saturating_div(2)
            .max(soft.saturating_add(10)),
    );
    ClockBudget::new(soft, hard, available)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg(any(test, debug_assertions, feature = "internal-testing"))]
pub(crate) struct AspirationWindow {
    pub(crate) alpha: i32,
    pub(crate) beta: i32,
    pub(crate) delta: i32,
}

#[cfg(any(test, debug_assertions, feature = "internal-testing"))]
impl AspirationWindow {
    pub(crate) fn initial(guess: i32, base_delta: i32, depth: u8, volatility_cp: i32) -> Self {
        let base = base_delta.clamp(1, MAX_WINDOW / 2);
        let center = guess.clamp(-MAX_WINDOW, MAX_WINDOW);
        let depth_extra = i32::from(depth.saturating_sub(4)).saturating_mul(2);
        let volatility_extra = volatility_cp.max(0) / 4;
        let delta = base
            .saturating_add(depth_extra)
            .saturating_add(volatility_extra)
            .min(base.saturating_mul(2))
            .min(MAX_WINDOW);
        Self {
            alpha: center.saturating_sub(delta).max(-MAX_WINDOW),
            beta: center.saturating_add(delta).min(MAX_WINDOW),
            delta,
        }
    }

    /// Fail-low widening is deliberately stronger than fail-high widening:
    /// collapsing scores more often indicate a tactical refutation and need a
    /// faster escape from repeated narrow re-searches.
    pub(crate) fn widen_low(self) -> Self {
        let delta = self.delta.saturating_mul(2).min(MAX_WINDOW);
        Self {
            alpha: self.alpha.saturating_sub(delta).max(-MAX_WINDOW),
            beta: self.beta,
            delta,
        }
    }

    pub(crate) fn widen_high(self) -> Self {
        let delta = self
            .delta
            .saturating_mul(3)
            .saturating_div(2)
            .max(self.delta.saturating_add(1))
            .min(MAX_WINDOW);
        Self {
            alpha: self.alpha,
            beta: self.beta.saturating_add(delta).min(MAX_WINDOW),
            delta,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[cfg(any(test, debug_assertions, feature = "internal-testing"))]
pub(crate) struct IterationSignals {
    pub(crate) best_move: Option<u32>,
    pub(crate) pv_signature: u64,
    pub(crate) score_cp: i32,
    pub(crate) second_best_margin_cp: Option<i32>,
    pub(crate) aspiration_researches: u8,
}

/// Compact completed-iteration history used only for deciding whether another
/// iteration is worth its clock cost. Partial iterations are never recorded.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[cfg(any(test, debug_assertions, feature = "internal-testing"))]
pub(crate) struct IterationInstability {
    previous: Option<IterationSignals>,
    comparisons: u8,
    stable_iterations: u8,
    move_churn: u8,
    pv_churn: u8,
    score_volatility_cp: i32,
    latest_margin_cp: Option<i32>,
    latest_researches: u8,
}

#[cfg(any(test, debug_assertions, feature = "internal-testing"))]
impl IterationInstability {
    pub(crate) fn record(&mut self, signals: IterationSignals) {
        if let Some(previous) = self.previous {
            self.comparisons = self.comparisons.saturating_add(1);
            let move_changed = signals.best_move != previous.best_move;
            let pv_changed = signals.pv_signature != previous.pv_signature;
            let score_delta = signals
                .score_cp
                .saturating_sub(previous.score_cp)
                .saturating_abs();
            self.score_volatility_cp = (self.score_volatility_cp / 2).max(score_delta);
            self.move_churn = self
                .move_churn
                .saturating_sub(1)
                .saturating_add(u8::from(move_changed).saturating_mul(2));
            self.pv_churn = self
                .pv_churn
                .saturating_sub(1)
                .saturating_add(u8::from(pv_changed));
            if !move_changed && !pv_changed && score_delta <= 20 {
                self.stable_iterations = self.stable_iterations.saturating_add(1);
            } else {
                self.stable_iterations = 0;
            }
        }
        self.latest_margin_cp = signals.second_best_margin_cp;
        self.latest_researches = signals.aspiration_researches;
        self.previous = Some(signals);
    }

    pub(crate) const fn score_volatility_cp(self) -> i32 {
        self.score_volatility_cp
    }

    pub(crate) fn soft_budget_factor(self) -> f64 {
        f64::from(self.soft_budget_percent()) / 100.0
    }

    pub(crate) fn soft_budget_percent(self) -> u16 {
        if self.comparisons == 0 {
            return 100;
        }
        let stable_base: i32 = match self.stable_iterations {
            3.. => 70,
            2 => 82,
            1 => 95,
            _ => 110,
        };
        let churn = i32::from(self.move_churn).saturating_mul(8)
            + i32::from(self.pv_churn).saturating_mul(4);
        let volatility = (self.score_volatility_cp / 10).clamp(0, 30);
        let ambiguity = match self.latest_margin_cp {
            Some(margin) if margin <= 20 => 15,
            Some(margin) if margin >= 120 && self.stable_iterations > 0 => -5,
            _ => 0,
        };
        let research_cost = i32::from(self.latest_researches).min(3) * 5;
        (stable_base + churn + volatility + ambiguity + research_cost).clamp(60, 175) as u16
    }

    /// Predicts the next iteration from the last two completed costs. Search
    /// growth is clamped so one pathological re-search cannot consume the
    /// entire remaining hard window.
    pub(crate) fn predicted_next_iteration_ms(
        self,
        previous_ms: Option<u64>,
        latest_ms: u64,
    ) -> u64 {
        let growth_pct = previous_ms
            .filter(|&previous| previous > 0)
            .map_or(200, |previous| {
                latest_ms
                    .saturating_mul(100)
                    .saturating_div(previous)
                    .clamp(125, 300)
            });
        let research_pct = u64::from(self.latest_researches).min(3) * 15;
        latest_ms
            .saturating_mul(growth_pct.saturating_add(research_pct))
            .saturating_div(100)
    }

    pub(crate) fn should_start_next_iteration(
        self,
        previous_ms: Option<u64>,
        latest_ms: u64,
        remaining_ms: u64,
    ) -> bool {
        self.predicted_next_iteration_ms(previous_ms, latest_ms) < remaining_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signals(mv: u32, pv: u64, score: i32, margin: i32) -> IterationSignals {
        IterationSignals {
            best_move: Some(mv),
            pv_signature: pv,
            score_cp: score,
            second_best_margin_cp: Some(margin),
            aspiration_researches: 0,
        }
    }

    #[test]
    fn production_budget_matches_frozen_clock_modes() {
        let sudden_death = production_clock_budget(ClockBudgetInput {
            remaining_ms: 1_000,
            increment_ms: 0,
            moves_to_go: None,
            overhead_ms: 10,
            opponent_remaining_ms: Some(1_000),
        });
        assert_eq!((sudden_death.soft_ms, sudden_death.hard_ms), (39, 58));

        let increment = production_clock_budget(ClockBudgetInput {
            remaining_ms: 5_000,
            increment_ms: 1_000,
            moves_to_go: Some(10),
            overhead_ms: 10,
            opponent_remaining_ms: Some(5_000),
        });
        assert_eq!((increment.soft_ms, increment.hard_ms), (1_244, 1_866));
    }

    #[test]
    fn candidate_budget_uses_increment_and_opponent_clock() {
        let common = ClockBudgetInput {
            remaining_ms: 10_000,
            increment_ms: 100,
            moves_to_go: None,
            overhead_ms: 10,
            opponent_remaining_ms: Some(10_000),
        };
        let even = candidate_clock_budget(common);
        let ahead = candidate_clock_budget(ClockBudgetInput {
            opponent_remaining_ms: Some(4_000),
            ..common
        });
        let behind = candidate_clock_budget(ClockBudgetInput {
            opponent_remaining_ms: Some(25_000),
            ..common
        });
        assert!(ahead.soft_ms < even.soft_ms);
        assert!(behind.soft_ms > even.soft_ms);
        assert!(
            even.soft_ms
                > candidate_clock_budget(ClockBudgetInput {
                    increment_ms: 0,
                    ..common
                })
                .soft_ms
        );
    }

    #[test]
    fn candidate_budget_is_time_loss_safe_at_extremes() {
        for input in [
            ClockBudgetInput {
                remaining_ms: 5,
                increment_ms: u64::MAX,
                moves_to_go: Some(1),
                overhead_ms: 10,
                opponent_remaining_ms: Some(u64::MAX),
            },
            ClockBudgetInput {
                remaining_ms: u64::MAX,
                increment_ms: u64::MAX,
                moves_to_go: Some(u32::MAX),
                overhead_ms: u64::MAX,
                opponent_remaining_ms: Some(1),
            },
        ] {
            let budget = candidate_clock_budget(input);
            assert!(budget.soft_ms <= budget.hard_ms);
            assert!(budget.hard_ms <= budget.available_ms);
            assert!(budget.available_ms <= input.remaining_ms);
        }
    }

    #[test]
    fn aspiration_widening_is_asymmetric_and_bounded() {
        let initial = AspirationWindow::initial(30, 36, 10, 40);
        assert!(initial.delta > 36);
        let low = initial.widen_low();
        let high = initial.widen_high();
        assert!(initial.alpha - low.alpha > high.beta - initial.beta);
        let edge = AspirationWindow::initial(i32::MAX, i32::MAX, u8::MAX, i32::MAX);
        assert_eq!(edge.beta, MAX_WINDOW);
        assert!(edge.alpha >= -MAX_WINDOW);
    }

    #[test]
    fn bestmove_and_pv_churn_extend_the_budget() {
        let mut stable = IterationInstability::default();
        let mut churn = IterationInstability::default();
        for depth in 0..4 {
            stable.record(signals(1, 11, 20 + depth, 80));
            churn.record(signals(1 + depth as u32 % 2, 11 + depth as u64, 20, 80));
        }
        assert!(stable.soft_budget_factor() < 1.0);
        assert!(churn.soft_budget_factor() > stable.soft_budget_factor());
    }

    #[test]
    fn score_volatility_and_second_best_margin_are_independent_signals() {
        let mut quiet = IterationInstability::default();
        quiet.record(signals(1, 1, 10, 150));
        quiet.record(signals(1, 1, 12, 150));

        let mut volatile = IterationInstability::default();
        volatile.record(signals(1, 1, 10, 150));
        volatile.record(signals(1, 1, 130, 150));

        let mut ambiguous = IterationInstability::default();
        ambiguous.record(signals(1, 1, 10, 10));
        ambiguous.record(signals(1, 1, 12, 10));

        assert!(volatile.score_volatility_cp() > quiet.score_volatility_cp());
        assert!(volatile.soft_budget_percent() > quiet.soft_budget_percent());
        assert!(ambiguous.soft_budget_percent() > quiet.soft_budget_percent());
    }

    #[test]
    fn next_iteration_cost_uses_growth_and_researches() {
        let mut state = IterationInstability::default();
        let mut sample = signals(1, 1, 10, 50);
        sample.aspiration_researches = 2;
        state.record(sample);
        assert_eq!(state.predicted_next_iteration_ms(Some(50), 100), 230);
        assert!(!state.should_start_next_iteration(Some(50), 100, 230));
        assert!(state.should_start_next_iteration(Some(50), 100, 231));
    }
}
