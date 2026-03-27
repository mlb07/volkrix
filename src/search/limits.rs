#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SearchLimits {
    pub depth: u8,
    pub tt_enabled: bool,
    pub hash_mb: usize,
}

impl SearchLimits {
    pub const fn new(depth: u8) -> Self {
        Self {
            depth,
            tt_enabled: true,
            hash_mb: super::tt::DEFAULT_HASH_MB,
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
}

impl Default for SearchLimits {
    fn default() -> Self {
        Self::new(1)
    }
}
