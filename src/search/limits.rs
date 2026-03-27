#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SearchHeuristics {
    pub(crate) pv_move_ordering: bool,
    pub(crate) capture_buckets: bool,
    pub(crate) killer_moves: bool,
    pub(crate) quiet_history: bool,
    pub(crate) aspiration_windows: bool,
}

impl SearchHeuristics {
    pub(crate) const fn phase5_baseline() -> Self {
        Self {
            pv_move_ordering: false,
            capture_buckets: false,
            killer_moves: false,
            quiet_history: false,
            aspiration_windows: false,
        }
    }

    pub(crate) const fn phase6_default() -> Self {
        Self {
            pv_move_ordering: true,
            capture_buckets: true,
            killer_moves: true,
            quiet_history: true,
            aspiration_windows: true,
        }
    }

    pub(crate) const fn with_pv_move_ordering(mut self, enabled: bool) -> Self {
        self.pv_move_ordering = enabled;
        self
    }

    pub(crate) const fn with_capture_buckets(mut self, enabled: bool) -> Self {
        self.capture_buckets = enabled;
        self
    }

    pub(crate) const fn with_killer_moves(mut self, enabled: bool) -> Self {
        self.killer_moves = enabled;
        self
    }

    pub(crate) const fn with_quiet_history(mut self, enabled: bool) -> Self {
        self.quiet_history = enabled;
        self
    }

    pub(crate) const fn with_aspiration_windows(mut self, enabled: bool) -> Self {
        self.aspiration_windows = enabled;
        self
    }
}

impl Default for SearchHeuristics {
    fn default() -> Self {
        Self::phase6_default()
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
            heuristics: SearchHeuristics::phase6_default(),
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

    pub(crate) const fn with_phase5_baseline(mut self) -> Self {
        self.heuristics = SearchHeuristics::phase5_baseline();
        self
    }
}

impl Default for SearchLimits {
    fn default() -> Self {
        Self::new(1)
    }
}
