use std::fmt;

use super::{
    piece::PieceType,
    square::{Square, SquareParseError},
};

const FROM_MASK: u32 = 0x3f;
const TO_SHIFT: u32 = 6;
const TO_MASK: u32 = 0x3f << TO_SHIFT;
const PROMOTION_SHIFT: u32 = 12;
const PROMOTION_MASK: u32 = 0x7 << PROMOTION_SHIFT;

pub const FLAG_CAPTURE: u32 = 1 << 15;
pub const FLAG_DOUBLE_PAWN_PUSH: u32 = 1 << 16;
pub const FLAG_EN_PASSANT: u32 = 1 << 17;
pub const FLAG_CASTLE: u32 = 1 << 18;
pub const FLAG_PROMOTION: u32 = 1 << 19;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Move(u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParsedMove {
    from: Square,
    to: Square,
    promotion: Option<PieceType>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ParseUciMoveError {
    InvalidLength,
    InvalidFromSquare,
    InvalidToSquare,
    InvalidPromotion,
}

impl Move {
    pub const NONE: Self = Self(u32::MAX);

    pub const fn new(from: Square, to: Square) -> Self {
        Self((from.index() as u32) | ((to.index() as u32) << TO_SHIFT))
    }

    pub fn with_flags(self, flags: u32) -> Self {
        Self(self.0 | flags)
    }

    pub fn with_promotion(self, piece_type: PieceType) -> Self {
        let code = piece_type
            .promotion_code()
            .expect("promotion piece must be knight, bishop, rook, or queen");
        Self((self.0 & !PROMOTION_MASK) | FLAG_PROMOTION | ((code as u32) << PROMOTION_SHIFT))
    }

    pub fn from(self) -> Square {
        Square::from_index_unchecked((self.0 & FROM_MASK) as u8)
    }

    pub fn to(self) -> Square {
        Square::from_index_unchecked(((self.0 & TO_MASK) >> TO_SHIFT) as u8)
    }

    pub fn promotion(self) -> Option<PieceType> {
        let code = ((self.0 & PROMOTION_MASK) >> PROMOTION_SHIFT) as u8;
        PieceType::from_promotion_code(code)
    }

    pub const fn is_none(self) -> bool {
        self.0 == u32::MAX
    }

    pub const fn is_capture(self) -> bool {
        self.0 & FLAG_CAPTURE != 0
    }

    pub const fn is_double_pawn_push(self) -> bool {
        self.0 & FLAG_DOUBLE_PAWN_PUSH != 0
    }

    pub const fn is_en_passant(self) -> bool {
        self.0 & FLAG_EN_PASSANT != 0
    }

    pub const fn is_castle(self) -> bool {
        self.0 & FLAG_CASTLE != 0
    }

    pub const fn is_promotion(self) -> bool {
        self.0 & FLAG_PROMOTION != 0
    }

    pub fn matches_parsed(self, parsed: ParsedMove) -> bool {
        self.from() == parsed.from && self.to() == parsed.to && self.promotion() == parsed.promotion
    }

    pub fn to_uci(self) -> String {
        let mut value = String::with_capacity(if self.is_promotion() { 5 } else { 4 });
        value.push_str(&self.from().to_coord());
        value.push_str(&self.to().to_coord());
        if let Some(piece_type) = self.promotion() {
            value.push(
                piece_type
                    .promotion_char()
                    .expect("promotion moves must encode a promotion piece"),
            );
        }
        value
    }
}

impl ParsedMove {
    pub fn parse(value: &str) -> Result<Self, ParseUciMoveError> {
        let bytes = value.as_bytes();
        if bytes.len() != 4 && bytes.len() != 5 {
            return Err(ParseUciMoveError::InvalidLength);
        }

        let from = Square::from_coord_text(&value[0..2])
            .map_err(|SquareParseError| ParseUciMoveError::InvalidFromSquare)?;
        let to = Square::from_coord_text(&value[2..4])
            .map_err(|SquareParseError| ParseUciMoveError::InvalidToSquare)?;
        let promotion = if bytes.len() == 5 {
            let promotion_char = value
                .chars()
                .nth(4)
                .expect("length checked to be at least five characters");
            Some(
                PieceType::from_promotion_char(promotion_char)
                    .ok_or(ParseUciMoveError::InvalidPromotion)?,
            )
        } else {
            None
        };

        Ok(Self {
            from,
            to,
            promotion,
        })
    }

    pub const fn from(self) -> Square {
        self.from
    }

    pub const fn to(self) -> Square {
        self.to
    }

    pub const fn promotion(self) -> Option<PieceType> {
        self.promotion
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_uci())
    }
}
