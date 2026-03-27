use std::{sync::atomic::AtomicBool, time::Instant};

use crate::core::Position;

use super::{
    SearchLimits, SearchResult,
    root::{self, SearchControl},
    tt::{DEFAULT_HASH_MB, TranspositionTable},
};

pub(crate) struct SearchRequest<'a> {
    pub(crate) limits: SearchLimits,
    pub(crate) soft_deadline: Option<Instant>,
    pub(crate) hard_deadline: Option<Instant>,
    pub(crate) stop_flag: Option<&'a AtomicBool>,
}

#[derive(Clone, Debug)]
pub(crate) struct UciSearchService {
    hash_mb: usize,
    tt: TranspositionTable,
}

impl UciSearchService {
    pub(crate) fn new() -> Self {
        Self {
            hash_mb: DEFAULT_HASH_MB,
            tt: TranspositionTable::new_mb(DEFAULT_HASH_MB),
        }
    }

    pub(crate) fn hash_mb(&self) -> usize {
        self.hash_mb
    }

    pub(crate) fn resize_hash(&mut self, hash_mb: usize) {
        let hash_mb = hash_mb.max(1);
        self.hash_mb = hash_mb;
        self.tt = TranspositionTable::new_mb(hash_mb);
    }

    pub(crate) fn clear_hash(&mut self) {
        self.tt.clear();
    }

    pub(crate) fn search(
        &mut self,
        position: &mut Position,
        request: SearchRequest<'_>,
    ) -> SearchResult {
        let limits = request.limits.with_hash_mb(self.hash_mb).with_tt(true);
        root::search_with_control(
            position,
            limits,
            Some(&mut self.tt),
            SearchControl {
                stop_flag: request.stop_flag,
                soft_deadline: request.soft_deadline,
                hard_deadline: request.hard_deadline,
            },
        )
    }

    #[cfg(any(test, debug_assertions))]
    pub(crate) fn debug_tt_entry_count(&self) -> usize {
        self.tt.debug_entry_count()
    }
}

impl Default for UciSearchService {
    fn default() -> Self {
        Self::new()
    }
}
