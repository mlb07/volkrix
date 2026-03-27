pub(crate) const MAX_REPETITION_HISTORY: usize = 4096;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct HistoryOverflow;

#[derive(Clone, Debug)]
pub(crate) struct RepetitionHistory {
    keys: [u64; MAX_REPETITION_HISTORY],
    len: usize,
}

impl RepetitionHistory {
    pub(crate) const fn empty() -> Self {
        Self {
            keys: [0; MAX_REPETITION_HISTORY],
            len: 0,
        }
    }

    pub(crate) fn clear_and_seed(&mut self, current_key: u64) {
        self.keys[0] = current_key;
        self.len = 1;
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    #[cfg(test)]
    pub(crate) fn as_slice(&self) -> &[u64] {
        &self.keys[..self.len]
    }

    pub(crate) fn current(&self) -> Option<u64> {
        if self.len == 0 {
            None
        } else {
            Some(self.keys[self.len - 1])
        }
    }

    pub(crate) fn push(&mut self, key: u64) -> Result<(), HistoryOverflow> {
        if self.len >= MAX_REPETITION_HISTORY {
            return Err(HistoryOverflow);
        }
        self.keys[self.len] = key;
        self.len += 1;
        Ok(())
    }

    pub(crate) fn pop(&mut self) {
        debug_assert!(
            self.len > 1,
            "persistent history must retain the current root key"
        );
        if self.len > 1 {
            self.len -= 1;
        }
    }

    pub(crate) fn is_threefold_repetition(&self, halfmove_clock: u16) -> bool {
        if self.len < 5 {
            return false;
        }

        let current = self.keys[self.len - 1];
        let mut occurrences = 1usize;
        let mut plies_back = 2usize;
        let mut index = self.len as isize - 3;
        let max_plies_back = halfmove_clock as usize;

        while index >= 0 && plies_back <= max_plies_back {
            if self.keys[index as usize] == current {
                occurrences += 1;
                if occurrences >= 3 {
                    return true;
                }
            }
            index -= 2;
            plies_back += 2;
        }

        false
    }
}
