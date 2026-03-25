use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Square(u8);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SquareParseError;

impl Square {
    pub const A1: Self = Self(0);
    pub const B1: Self = Self(1);
    pub const C1: Self = Self(2);
    pub const D1: Self = Self(3);
    pub const E1: Self = Self(4);
    pub const F1: Self = Self(5);
    pub const G1: Self = Self(6);
    pub const H1: Self = Self(7);
    pub const A8: Self = Self(56);
    pub const B8: Self = Self(57);
    pub const C8: Self = Self(58);
    pub const D8: Self = Self(59);
    pub const E8: Self = Self(60);
    pub const F8: Self = Self(61);
    pub const G8: Self = Self(62);
    pub const H8: Self = Self(63);

    pub const fn from_index_unchecked(index: u8) -> Self {
        Self(index)
    }

    pub fn try_from_index(index: u8) -> Option<Self> {
        (index < 64).then_some(Self(index))
    }

    pub fn from_coords(file: u8, rank: u8) -> Option<Self> {
        (file < 8 && rank < 8).then_some(Self(rank * 8 + file))
    }

    pub fn from_coord_text(text: &str) -> Result<Self, SquareParseError> {
        let bytes = text.as_bytes();
        if bytes.len() != 2 {
            return Err(SquareParseError);
        }

        let file = bytes[0];
        let rank = bytes[1];
        if !(b'a'..=b'h').contains(&file) || !(b'1'..=b'8').contains(&rank) {
            return Err(SquareParseError);
        }

        let file_index = file - b'a';
        let rank_index = rank - b'1';
        Self::from_coords(file_index, rank_index).ok_or(SquareParseError)
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }

    pub const fn file(self) -> u8 {
        self.0 % 8
    }

    pub const fn rank(self) -> u8 {
        self.0 / 8
    }

    pub const fn bit(self) -> u64 {
        1u64 << self.0
    }

    pub fn offset(self, file_delta: i8, rank_delta: i8) -> Option<Self> {
        let file = self.file() as i8 + file_delta;
        let rank = self.rank() as i8 + rank_delta;
        if !(0..=7).contains(&file) || !(0..=7).contains(&rank) {
            return None;
        }
        Self::from_coords(file as u8, rank as u8)
    }

    pub fn to_coord(self) -> String {
        let file = (b'a' + self.file()) as char;
        let rank = (b'1' + self.rank()) as char;
        let mut value = String::with_capacity(2);
        value.push(file);
        value.push(rank);
        value
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_coord())
    }
}
