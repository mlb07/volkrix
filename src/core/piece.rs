use super::types::Color;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum PieceType {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

impl PieceType {
    pub const ALL: [Self; 6] = [
        Self::Pawn,
        Self::Knight,
        Self::Bishop,
        Self::Rook,
        Self::Queen,
        Self::King,
    ];

    pub const fn index(self) -> usize {
        match self {
            Self::Pawn => 0,
            Self::Knight => 1,
            Self::Bishop => 2,
            Self::Rook => 3,
            Self::Queen => 4,
            Self::King => 5,
        }
    }

    pub const fn promotion_code(self) -> Option<u8> {
        match self {
            Self::Knight => Some(1),
            Self::Bishop => Some(2),
            Self::Rook => Some(3),
            Self::Queen => Some(4),
            Self::Pawn | Self::King => None,
        }
    }

    pub const fn from_promotion_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::Knight),
            2 => Some(Self::Bishop),
            3 => Some(Self::Rook),
            4 => Some(Self::Queen),
            _ => None,
        }
    }

    pub const fn promotion_char(self) -> Option<char> {
        match self {
            Self::Knight => Some('n'),
            Self::Bishop => Some('b'),
            Self::Rook => Some('r'),
            Self::Queen => Some('q'),
            Self::Pawn | Self::King => None,
        }
    }

    pub fn from_promotion_char(value: char) -> Option<Self> {
        match value.to_ascii_lowercase() {
            'n' => Some(Self::Knight),
            'b' => Some(Self::Bishop),
            'r' => Some(Self::Rook),
            'q' => Some(Self::Queen),
            _ => None,
        }
    }

    pub const fn promotion_pieces() -> [Self; 4] {
        [Self::Knight, Self::Bishop, Self::Rook, Self::Queen]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Piece {
    WhitePawn,
    WhiteKnight,
    WhiteBishop,
    WhiteRook,
    WhiteQueen,
    WhiteKing,
    BlackPawn,
    BlackKnight,
    BlackBishop,
    BlackRook,
    BlackQueen,
    BlackKing,
}

impl Piece {
    pub const fn color(self) -> Color {
        match self {
            Self::WhitePawn
            | Self::WhiteKnight
            | Self::WhiteBishop
            | Self::WhiteRook
            | Self::WhiteQueen
            | Self::WhiteKing => Color::White,
            Self::BlackPawn
            | Self::BlackKnight
            | Self::BlackBishop
            | Self::BlackRook
            | Self::BlackQueen
            | Self::BlackKing => Color::Black,
        }
    }

    pub const fn piece_type(self) -> PieceType {
        match self {
            Self::WhitePawn | Self::BlackPawn => PieceType::Pawn,
            Self::WhiteKnight | Self::BlackKnight => PieceType::Knight,
            Self::WhiteBishop | Self::BlackBishop => PieceType::Bishop,
            Self::WhiteRook | Self::BlackRook => PieceType::Rook,
            Self::WhiteQueen | Self::BlackQueen => PieceType::Queen,
            Self::WhiteKing | Self::BlackKing => PieceType::King,
        }
    }

    pub const fn from_parts(color: Color, piece_type: PieceType) -> Self {
        match (color, piece_type) {
            (Color::White, PieceType::Pawn) => Self::WhitePawn,
            (Color::White, PieceType::Knight) => Self::WhiteKnight,
            (Color::White, PieceType::Bishop) => Self::WhiteBishop,
            (Color::White, PieceType::Rook) => Self::WhiteRook,
            (Color::White, PieceType::Queen) => Self::WhiteQueen,
            (Color::White, PieceType::King) => Self::WhiteKing,
            (Color::Black, PieceType::Pawn) => Self::BlackPawn,
            (Color::Black, PieceType::Knight) => Self::BlackKnight,
            (Color::Black, PieceType::Bishop) => Self::BlackBishop,
            (Color::Black, PieceType::Rook) => Self::BlackRook,
            (Color::Black, PieceType::Queen) => Self::BlackQueen,
            (Color::Black, PieceType::King) => Self::BlackKing,
        }
    }

    pub const fn fen_char(self) -> char {
        match self {
            Self::WhitePawn => 'P',
            Self::WhiteKnight => 'N',
            Self::WhiteBishop => 'B',
            Self::WhiteRook => 'R',
            Self::WhiteQueen => 'Q',
            Self::WhiteKing => 'K',
            Self::BlackPawn => 'p',
            Self::BlackKnight => 'n',
            Self::BlackBishop => 'b',
            Self::BlackRook => 'r',
            Self::BlackQueen => 'q',
            Self::BlackKing => 'k',
        }
    }

    pub fn from_fen_char(value: char) -> Option<Self> {
        match value {
            'P' => Some(Self::WhitePawn),
            'N' => Some(Self::WhiteKnight),
            'B' => Some(Self::WhiteBishop),
            'R' => Some(Self::WhiteRook),
            'Q' => Some(Self::WhiteQueen),
            'K' => Some(Self::WhiteKing),
            'p' => Some(Self::BlackPawn),
            'n' => Some(Self::BlackKnight),
            'b' => Some(Self::BlackBishop),
            'r' => Some(Self::BlackRook),
            'q' => Some(Self::BlackQueen),
            'k' => Some(Self::BlackKing),
            _ => None,
        }
    }
}
