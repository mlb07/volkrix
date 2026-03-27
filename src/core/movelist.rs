use super::chess_move::Move;

pub const MAX_MOVES: usize = 256;

#[derive(Clone)]
pub struct MoveList {
    moves: [Move; MAX_MOVES],
    len: usize,
}

impl MoveList {
    pub fn new() -> Self {
        Self {
            moves: [Move::NONE; MAX_MOVES],
            len: 0,
        }
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn push(&mut self, mv: Move) {
        assert!(self.len < MAX_MOVES, "move list overflow");
        self.moves[self.len] = mv;
        self.len += 1;
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[Move] {
        &self.moves[..self.len]
    }

    pub fn get(&self, index: usize) -> Move {
        self.as_slice()[index]
    }

    pub fn swap(&mut self, a: usize, b: usize) {
        self.moves.swap(a, b);
    }

    pub fn iter(&self) -> impl Iterator<Item = &Move> {
        self.as_slice().iter()
    }
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}
