#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SearchLimits {
    pub depth: u8,
}

impl SearchLimits {
    pub const fn new(depth: u8) -> Self {
        Self { depth }
    }
}

impl Default for SearchLimits {
    fn default() -> Self {
        Self { depth: 1 }
    }
}
