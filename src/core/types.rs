#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Color {
    White,
    Black,
}

impl Color {
    pub const ALL: [Self; 2] = [Self::White, Self::Black];

    pub const fn index(self) -> usize {
        match self {
            Self::White => 0,
            Self::Black => 1,
        }
    }

    pub const fn opposite(self) -> Self {
        match self {
            Self::White => Self::Black,
            Self::Black => Self::White,
        }
    }

    pub const fn pawn_direction(self) -> i8 {
        match self {
            Self::White => 1,
            Self::Black => -1,
        }
    }

    pub const fn pawn_start_rank(self) -> u8 {
        match self {
            Self::White => 1,
            Self::Black => 6,
        }
    }

    pub const fn promotion_from_rank(self) -> u8 {
        match self {
            Self::White => 6,
            Self::Black => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub struct CastlingRights(u8);

impl CastlingRights {
    pub const NONE: Self = Self(0);
    pub const WHITE_KINGSIDE: Self = Self(1 << 0);
    pub const WHITE_QUEENSIDE: Self = Self(1 << 1);
    pub const BLACK_KINGSIDE: Self = Self(1 << 2);
    pub const BLACK_QUEENSIDE: Self = Self(1 << 3);

    pub const fn bits(self) -> u8 {
        self.0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub const fn contains(self, rights: Self) -> bool {
        self.0 & rights.0 == rights.0
    }

    pub fn insert(&mut self, rights: Self) {
        self.0 |= rights.0;
    }

    pub fn remove(&mut self, rights: Self) {
        self.0 &= !rights.0;
    }

    pub const fn has_kingside(self, color: Color) -> bool {
        match color {
            Color::White => self.contains(Self::WHITE_KINGSIDE),
            Color::Black => self.contains(Self::BLACK_KINGSIDE),
        }
    }

    pub const fn has_queenside(self, color: Color) -> bool {
        match color {
            Color::White => self.contains(Self::WHITE_QUEENSIDE),
            Color::Black => self.contains(Self::BLACK_QUEENSIDE),
        }
    }

    pub fn remove_color(&mut self, color: Color) {
        match color {
            Color::White => self.remove(Self::WHITE_KINGSIDE),
            Color::Black => self.remove(Self::BLACK_KINGSIDE),
        }
        match color {
            Color::White => self.remove(Self::WHITE_QUEENSIDE),
            Color::Black => self.remove(Self::BLACK_QUEENSIDE),
        }
    }

    pub fn to_fen(self) -> String {
        if self.is_empty() {
            return "-".to_owned();
        }

        let mut value = String::new();
        if self.contains(Self::WHITE_KINGSIDE) {
            value.push('K');
        }
        if self.contains(Self::WHITE_QUEENSIDE) {
            value.push('Q');
        }
        if self.contains(Self::BLACK_KINGSIDE) {
            value.push('k');
        }
        if self.contains(Self::BLACK_QUEENSIDE) {
            value.push('q');
        }
        value
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Default)]
pub struct Score(pub i32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Default)]
pub struct Depth(pub i16);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Default)]
pub struct Value(pub i16);
