use std::{error::Error, fmt};

use super::{
    piece::Piece,
    position::Position,
    square::Square,
    types::{CastlingRights, Color},
};

pub const STARTPOS_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FenError {
    WrongFieldCount,
    InvalidPiecePlacement(String),
    InvalidSideToMove(String),
    InvalidCastlingRights(String),
    InvalidEnPassant(String),
    InvalidHalfmoveClock(String),
    InvalidFullmoveNumber(String),
    InvalidPosition(String),
}

impl Position {
    pub fn startpos() -> Self {
        Self::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("hard-coded start position FEN must be valid")
    }

    pub fn from_fen(fen: &str) -> Result<Self, FenError> {
        let fields: Vec<&str> = fen.split_whitespace().collect();
        if fields.len() != 6 {
            return Err(FenError::WrongFieldCount);
        }

        let mut position = Position::empty();

        let ranks: Vec<&str> = fields[0].split('/').collect();
        if ranks.len() != 8 {
            return Err(FenError::InvalidPiecePlacement(
                "piece placement must contain 8 ranks".to_owned(),
            ));
        }

        for (rank_index, rank_text) in ranks.iter().enumerate() {
            let board_rank = 7 - rank_index as u8;
            let mut file = 0u8;

            for symbol in rank_text.chars() {
                if let Some(empty_count) = symbol.to_digit(10) {
                    if empty_count == 0 || empty_count > 8 {
                        return Err(FenError::InvalidPiecePlacement(format!(
                            "invalid empty-square count '{symbol}'"
                        )));
                    }
                    file = file.saturating_add(empty_count as u8);
                } else {
                    let piece = Piece::from_fen_char(symbol).ok_or_else(|| {
                        FenError::InvalidPiecePlacement(format!("invalid piece symbol '{symbol}'"))
                    })?;
                    let square = Square::from_coords(file, board_rank).ok_or_else(|| {
                        FenError::InvalidPiecePlacement(
                            "piece placement extends past the end of the rank".to_owned(),
                        )
                    })?;
                    position
                        .place_piece(square, piece)
                        .map_err(FenError::InvalidPiecePlacement)?;
                    file = file.saturating_add(1);
                }
            }

            if file != 8 {
                return Err(FenError::InvalidPiecePlacement(format!(
                    "rank '{}' does not contain exactly 8 files",
                    rank_text
                )));
            }
        }

        let side_to_move = match fields[1] {
            "w" => Color::White,
            "b" => Color::Black,
            other => {
                return Err(FenError::InvalidSideToMove(format!(
                    "invalid side-to-move field '{other}'"
                )));
            }
        };
        position.set_side_to_move(side_to_move);

        let castling_rights = parse_castling_rights(fields[2])?;
        position.set_castling_rights(castling_rights);

        let en_passant = if fields[3] == "-" {
            None
        } else {
            let square = Square::from_coord_text(fields[3]).map_err(|_| {
                FenError::InvalidEnPassant(format!("invalid en passant square '{}'", fields[3]))
            })?;
            let expected_rank = match side_to_move {
                Color::White => 5,
                Color::Black => 2,
            };
            if square.rank() != expected_rank {
                return Err(FenError::InvalidEnPassant(
                    "en passant square rank does not match side to move".to_owned(),
                ));
            }
            Some(square)
        };
        position.set_en_passant(en_passant);

        let halfmove_clock = fields[4].parse::<u16>().map_err(|_| {
            FenError::InvalidHalfmoveClock(format!("invalid halfmove clock '{}'", fields[4]))
        })?;
        position.set_halfmove_clock(halfmove_clock);

        let fullmove_number = fields[5].parse::<u16>().map_err(|_| {
            FenError::InvalidFullmoveNumber(format!("invalid fullmove number '{}'", fields[5]))
        })?;
        if fullmove_number == 0 {
            return Err(FenError::InvalidFullmoveNumber(
                "fullmove number must be at least 1".to_owned(),
            ));
        }
        position.set_fullmove_number(fullmove_number);

        position.validate().map_err(FenError::InvalidPosition)?;

        Ok(position)
    }

    pub fn to_fen(&self) -> String {
        let mut placement = String::new();
        for rank in (0..8).rev() {
            let mut empty_run = 0u8;
            for file in 0..8 {
                let square = Square::from_coords(file, rank).expect("board square must exist");
                match self.piece_at(square) {
                    Some(piece) => {
                        if empty_run > 0 {
                            placement.push(char::from(b'0' + empty_run));
                            empty_run = 0;
                        }
                        placement.push(piece.fen_char());
                    }
                    None => {
                        empty_run += 1;
                    }
                }
            }
            if empty_run > 0 {
                placement.push(char::from(b'0' + empty_run));
            }
            if rank != 0 {
                placement.push('/');
            }
        }

        let side_to_move = match self.side_to_move() {
            Color::White => "w",
            Color::Black => "b",
        };

        let en_passant = self
            .en_passant()
            .map_or_else(|| "-".to_owned(), |square| square.to_coord());

        format!(
            "{} {} {} {} {} {}",
            placement,
            side_to_move,
            self.castling_rights().to_fen(),
            en_passant,
            self.halfmove_clock(),
            self.fullmove_number()
        )
    }
}

impl fmt::Display for FenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongFieldCount => f.write_str("FEN must contain exactly 6 fields"),
            Self::InvalidPiecePlacement(value) => write!(f, "invalid piece placement: {value}"),
            Self::InvalidSideToMove(value) => write!(f, "invalid side-to-move field: {value}"),
            Self::InvalidCastlingRights(value) => write!(f, "invalid castling rights: {value}"),
            Self::InvalidEnPassant(value) => write!(f, "invalid en passant field: {value}"),
            Self::InvalidHalfmoveClock(value) => write!(f, "invalid halfmove clock: {value}"),
            Self::InvalidFullmoveNumber(value) => write!(f, "invalid fullmove number: {value}"),
            Self::InvalidPosition(value) => write!(f, "invalid position: {value}"),
        }
    }
}

impl Error for FenError {}

fn parse_castling_rights(value: &str) -> Result<CastlingRights, FenError> {
    if value == "-" {
        return Ok(CastlingRights::NONE);
    }

    let mut rights = CastlingRights::NONE;
    for symbol in value.chars() {
        match symbol {
            'K' => rights.insert(CastlingRights::WHITE_KINGSIDE),
            'Q' => rights.insert(CastlingRights::WHITE_QUEENSIDE),
            'k' => rights.insert(CastlingRights::BLACK_KINGSIDE),
            'q' => rights.insert(CastlingRights::BLACK_QUEENSIDE),
            _ => {
                return Err(FenError::InvalidCastlingRights(format!(
                    "invalid castling symbol '{symbol}'"
                )));
            }
        }
    }
    Ok(rights)
}
