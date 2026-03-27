use crate::core::{Piece, PieceType, Position, Score, Square, see, types::Color};

const FILE_CENTRALITY: [i32; 8] = [0, 4, 8, 12, 12, 8, 4, 0];
const RANK_CENTRALITY: [i32; 8] = [0, 4, 8, 12, 12, 8, 4, 0];

pub fn evaluate(position: &Position) -> Score {
    let mut white_score = 0i32;
    let mut black_score = 0i32;

    for index in 0..64 {
        let square = Square::from_index_unchecked(index as u8);
        let Some(piece) = position.piece_at(square) else {
            continue;
        };

        let value =
            see::piece_value(piece.piece_type()).0 as i32 + piece_square_bonus(piece, square);

        match piece.color() {
            Color::White => white_score += value,
            Color::Black => black_score += value,
        }
    }

    let score = white_score - black_score;
    match position.side_to_move() {
        Color::White => Score(score),
        Color::Black => Score(-score),
    }
}

fn piece_square_bonus(piece: Piece, square: Square) -> i32 {
    let rank = match piece.color() {
        Color::White => square.rank() as usize,
        Color::Black => 7usize - square.rank() as usize,
    };
    let file = square.file() as usize;
    let center = FILE_CENTRALITY[file] + RANK_CENTRALITY[rank];
    let advance = rank as i32;

    match piece.piece_type() {
        PieceType::Pawn => advance * 8 + FILE_CENTRALITY[file] / 2,
        PieceType::Knight => center,
        PieceType::Bishop => center / 2 + advance * 2,
        PieceType::Rook => advance * 3 + FILE_CENTRALITY[file] / 3,
        PieceType::Queen => center / 3,
        PieceType::King => 14 - center,
    }
}
