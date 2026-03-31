#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SearchHeuristics {
    pub(crate) pv_move_ordering: bool,
    pub(crate) capture_buckets: bool,
    pub(crate) killer_moves: bool,
    pub(crate) counter_moves: bool,
    pub(crate) quiet_history: bool,
    pub(crate) aspiration_windows: bool,
    pub(crate) late_move_reductions: bool,
    pub(crate) null_move_pruning: bool,
    pub(crate) reverse_futility_pruning: bool,
    pub(crate) futility_pruning: bool,
    pub(crate) late_move_pruning: bool,
}

impl SearchHeuristics {
    pub(crate) const fn phase8_baseline() -> Self {
        Self {
            pv_move_ordering: true,
            capture_buckets: true,
            killer_moves: true,
            counter_moves: true,
            quiet_history: true,
            aspiration_windows: true,
            late_move_reductions: false,
            null_move_pruning: false,
            reverse_futility_pruning: false,
            futility_pruning: false,
            late_move_pruning: false,
        }
    }

    pub(crate) const fn phase9_default() -> Self {
        Self {
            pv_move_ordering: true,
            capture_buckets: true,
            killer_moves: true,
            counter_moves: true,
            quiet_history: true,
            aspiration_windows: true,
            late_move_reductions: true,
            null_move_pruning: true,
            reverse_futility_pruning: true,
            futility_pruning: true,
            late_move_pruning: true,
        }
    }

    pub(crate) const fn with_aspiration_windows(mut self, enabled: bool) -> Self {
        self.aspiration_windows = enabled;
        self
    }

    pub(crate) const fn with_late_move_reductions(mut self, enabled: bool) -> Self {
        self.late_move_reductions = enabled;
        self
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) const fn with_null_move_pruning(mut self, enabled: bool) -> Self {
        self.null_move_pruning = enabled;
        self
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) const fn with_reverse_futility_pruning(mut self, enabled: bool) -> Self {
        self.reverse_futility_pruning = enabled;
        self
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) const fn with_futility_pruning(mut self, enabled: bool) -> Self {
        self.futility_pruning = enabled;
        self
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) const fn with_late_move_pruning(mut self, enabled: bool) -> Self {
        self.late_move_pruning = enabled;
        self
    }
}

impl Default for SearchHeuristics {
    fn default() -> Self {
        Self::phase9_default()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SearchLimits {
    pub depth: u8,
    pub tt_enabled: bool,
    pub hash_mb: usize,
    pub(crate) heuristics: SearchHeuristics,
}

impl SearchLimits {
    pub const fn new(depth: u8) -> Self {
        Self {
            depth,
            tt_enabled: true,
            hash_mb: super::tt::DEFAULT_HASH_MB,
            heuristics: SearchHeuristics::phase9_default(),
        }
    }

    pub const fn with_tt(mut self, enabled: bool) -> Self {
        self.tt_enabled = enabled;
        self
    }

    pub const fn without_tt(mut self) -> Self {
        self.tt_enabled = false;
        self
    }

    pub const fn with_hash_mb(mut self, hash_mb: usize) -> Self {
        self.hash_mb = hash_mb;
        self
    }

    pub(crate) const fn with_heuristics(mut self, heuristics: SearchHeuristics) -> Self {
        self.heuristics = heuristics;
        self
    }

    pub(crate) const fn with_phase8_baseline(mut self) -> Self {
        self.heuristics = SearchHeuristics::phase8_baseline();
        self
    }
}

impl Default for SearchLimits {
    fn default() -> Self {
        Self::new(1)
    }
}
