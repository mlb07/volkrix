#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SearchHeuristics {
    pub(crate) pv_move_ordering: bool,
    pub(crate) capture_buckets: bool,
    pub(crate) killer_moves: bool,
    pub(crate) quiet_history: bool,
    pub(crate) continuation_history: bool,
    pub(crate) aspiration_windows: bool,
    pub(crate) late_move_reductions: bool,
    pub(crate) null_move_pruning: bool,
    pub(crate) reverse_futility_pruning: bool,
    pub(crate) futility_pruning: bool,
    pub(crate) late_move_pruning: bool,
    pub(crate) capture_history: bool,
    pub(crate) internal_iterative_reduction: bool,
    pub(crate) see_pruning: bool,
    pub(crate) history_pruning: bool,
    pub(crate) probcut: bool,
    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    pub(crate) razoring: bool,
    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    pub(crate) multi_cut: bool,
    pub(crate) qsearch_tt: bool,
    pub(crate) tt_static_eval: bool,
    pub(crate) history_maluses: bool,
    pub(crate) contextual_lmr: bool,
    pub(crate) correction_history: bool,
    pub(crate) singular_extensions: bool,
}

impl SearchHeuristics {
    #[cfg_attr(
        not(any(test, debug_assertions, feature = "internal-testing")),
        allow(dead_code)
    )]
    pub(crate) const fn phase8_baseline() -> Self {
        Self {
            pv_move_ordering: true,
            capture_buckets: true,
            killer_moves: true,
            quiet_history: true,
            continuation_history: false,
            aspiration_windows: true,
            late_move_reductions: false,
            null_move_pruning: false,
            reverse_futility_pruning: false,
            futility_pruning: false,
            late_move_pruning: false,
            capture_history: false,
            internal_iterative_reduction: false,
            see_pruning: false,
            history_pruning: false,
            probcut: false,
            #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
            razoring: false,
            #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
            multi_cut: false,
            qsearch_tt: false,
            tt_static_eval: false,
            history_maluses: false,
            contextual_lmr: false,
            correction_history: false,
            singular_extensions: false,
        }
    }

    pub(crate) const fn phase9_default() -> Self {
        Self {
            pv_move_ordering: true,
            capture_buckets: true,
            killer_moves: true,
            quiet_history: true,
            continuation_history: true,
            aspiration_windows: true,
            late_move_reductions: true,
            null_move_pruning: true,
            reverse_futility_pruning: true,
            futility_pruning: true,
            late_move_pruning: true,
            // Capture-history updates are implemented, but the current classical evaluator's
            // depth-7 profile regresses badly when they influence ordering. Keep the feature
            // available for controlled NNUE/SPRT tuning instead of paying that cost by default.
            capture_history: false,
            internal_iterative_reduction: true,
            see_pruning: true,
            history_pruning: true,
            probcut: true,
            #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
            // Razoring is a speculative fail-low shortcut. A held-out 1,000-game paired test
            // rejected it at 47.00%, so keep the experiment compiled out of production.
            razoring: false,
            #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
            // Multi-Cut is implemented as a deliberately conservative experimental seam. Its
            // probabilistic cutoff must earn promotion through paired SPRT before it may affect
            // the evidence-backed default search.
            multi_cut: false,
            qsearch_tt: true,
            tt_static_eval: true,
            history_maluses: true,
            // The contextual adjustment remains available for match testing, but the isolated
            // depth-7 profile expanded the tree slightly. The logarithmic reduction table is the
            // proven default until this extra adjustment passes SPRT.
            contextual_lmr: false,
            // The mechanism remains available behind an isolated seam, but a 300-game match
            // rejected it for the default profile (46.17%, about -26.7 Elo). Raw evaluator
            // values remain authoritative in the TT regardless of this toggle.
            correction_history: false,
            // Singular verification is implemented behind an isolated match-test seam. Keep it
            // off until fixed-depth behavior and game outcomes justify promotion.
            singular_extensions: false,
        }
    }

    #[cfg_attr(
        not(any(test, debug_assertions, feature = "internal-testing")),
        allow(dead_code)
    )]
    pub(crate) const fn with_aspiration_windows(mut self, enabled: bool) -> Self {
        self.aspiration_windows = enabled;
        self
    }

    #[cfg_attr(
        not(any(test, debug_assertions, feature = "internal-testing")),
        allow(dead_code)
    )]
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

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) const fn with_correction_history(mut self, enabled: bool) -> Self {
        self.correction_history = enabled;
        self
    }

    #[cfg_attr(
        not(any(test, debug_assertions, feature = "internal-testing")),
        allow(dead_code)
    )]
    pub(crate) const fn with_singular_extensions(mut self, enabled: bool) -> Self {
        self.singular_extensions = enabled;
        self
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    pub(crate) const fn with_razoring(mut self, enabled: bool) -> Self {
        self.razoring = enabled;
        self
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    pub(crate) const fn with_multi_cut(mut self, enabled: bool) -> Self {
        self.multi_cut = enabled;
        self
    }

    #[allow(dead_code)]
    pub(crate) const fn with_modern_search(mut self, enabled: bool) -> Self {
        self.capture_history = enabled;
        self.internal_iterative_reduction = enabled;
        self.see_pruning = enabled;
        self.history_pruning = enabled;
        self.probcut = enabled;
        self.qsearch_tt = enabled;
        self.tt_static_eval = enabled;
        self.history_maluses = enabled;
        self.contextual_lmr = enabled;
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
    pub(crate) node_limit: Option<u64>,
    pub(crate) heuristics: SearchHeuristics,
    #[cfg(feature = "spsa-tuning")]
    pub(crate) parameters: super::parameters::SearchParameters,
}

impl SearchLimits {
    pub const fn new(depth: u8) -> Self {
        Self {
            depth,
            tt_enabled: true,
            hash_mb: super::tt::DEFAULT_HASH_MB,
            node_limit: None,
            heuristics: SearchHeuristics::phase9_default(),
            #[cfg(feature = "spsa-tuning")]
            parameters: super::parameters::SearchParameters::DEFAULT,
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

    pub(crate) const fn with_node_limit(mut self, node_limit: Option<u64>) -> Self {
        self.node_limit = node_limit;
        self
    }

    pub(crate) const fn with_heuristics(mut self, heuristics: SearchHeuristics) -> Self {
        self.heuristics = heuristics;
        self
    }

    #[cfg(feature = "spsa-tuning")]
    pub(crate) const fn with_parameters(
        mut self,
        parameters: super::parameters::SearchParameters,
    ) -> Self {
        self.parameters = parameters;
        self
    }

    #[cfg_attr(
        not(any(test, debug_assertions, feature = "internal-testing")),
        allow(dead_code)
    )]
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
