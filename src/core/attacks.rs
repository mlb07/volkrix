use std::sync::OnceLock;

use super::{magics, square::Square, types::Color};

const fn square_from_coords(file: i8, rank: i8) -> Option<Square> {
    if file < 0 || file > 7 || rank < 0 || rank > 7 {
        return None;
    }

    Some(Square::from_index_unchecked((rank as u8) * 8 + file as u8))
}

const fn leaper_attacks(square: Square, offsets: &[(i8, i8)]) -> u64 {
    let mut attacks = 0u64;
    let mut index = 0usize;
    while index < offsets.len() {
        let (file_delta, rank_delta) = offsets[index];
        let file = square.file() as i8 + file_delta;
        let rank = square.rank() as i8 + rank_delta;
        if let Some(target) = square_from_coords(file, rank) {
            attacks |= target.bit();
        }
        index += 1;
    }
    attacks
}

const fn pawn_attacks_from(square: Square, color: Color) -> u64 {
    match color {
        Color::White => leaper_attacks(square, &[(-1, 1), (1, 1)]),
        Color::Black => leaper_attacks(square, &[(-1, -1), (1, -1)]),
    }
}

const KNIGHT_OFFSETS: [(i8, i8); 8] = [
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
];
const KING_OFFSETS: [(i8, i8); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

const fn build_knight_attacks() -> [u64; 64] {
    let mut table = [0u64; 64];
    let mut index = 0usize;
    while index < 64 {
        let square = Square::from_index_unchecked(index as u8);
        table[index] = leaper_attacks(square, &KNIGHT_OFFSETS);
        index += 1;
    }
    table
}

const fn build_king_attacks() -> [u64; 64] {
    let mut table = [0u64; 64];
    let mut index = 0usize;
    while index < 64 {
        let square = Square::from_index_unchecked(index as u8);
        table[index] = leaper_attacks(square, &KING_OFFSETS);
        index += 1;
    }
    table
}

const fn build_pawn_attacks(color: Color) -> [u64; 64] {
    let mut table = [0u64; 64];
    let mut index = 0usize;
    while index < 64 {
        let square = Square::from_index_unchecked(index as u8);
        table[index] = pawn_attacks_from(square, color);
        index += 1;
    }
    table
}

const fn direction(square_a: Square, square_b: Square) -> Option<(i8, i8)> {
    let file_delta = square_b.file() as i8 - square_a.file() as i8;
    let rank_delta = square_b.rank() as i8 - square_a.rank() as i8;

    if file_delta == 0 && rank_delta != 0 {
        return Some((0, rank_delta.signum()));
    }
    if rank_delta == 0 && file_delta != 0 {
        return Some((file_delta.signum(), 0));
    }
    if file_delta.abs() == rank_delta.abs() && file_delta != 0 {
        return Some((file_delta.signum(), rank_delta.signum()));
    }

    None
}

fn build_between() -> [[u64; 64]; 64] {
    let mut table = [[0u64; 64]; 64];
    let mut from_index = 0usize;
    while from_index < 64 {
        let from = Square::from_index_unchecked(from_index as u8);
        let mut to_index = 0usize;
        while to_index < 64 {
            let to = Square::from_index_unchecked(to_index as u8);
            if let Some((file_delta, rank_delta)) = direction(from, to) {
                let mut current = from;
                let mut line = 0u64;
                loop {
                    let next = current.offset(file_delta, rank_delta);
                    if next.is_none() {
                        break;
                    }
                    let next_square = next.expect("checked above");
                    if next_square == to {
                        break;
                    }
                    line |= next_square.bit();
                    current = next_square;
                }
                table[from_index][to_index] = line;
            }
            to_index += 1;
        }
        from_index += 1;
    }
    table
}

fn build_line() -> [[u64; 64]; 64] {
    let mut table = [[0u64; 64]; 64];
    let mut from_index = 0usize;
    while from_index < 64 {
        let from = Square::from_index_unchecked(from_index as u8);
        let mut to_index = 0usize;
        while to_index < 64 {
            let to = Square::from_index_unchecked(to_index as u8);
            if let Some((file_delta, rank_delta)) = direction(from, to) {
                let mut line = from.bit() | to.bit();

                let mut current = from;
                loop {
                    let next = current.offset(-file_delta, -rank_delta);
                    if next.is_none() {
                        break;
                    }
                    let next_square = next.expect("checked above");
                    line |= next_square.bit();
                    current = next_square;
                }

                current = from;
                loop {
                    let next = current.offset(file_delta, rank_delta);
                    if next.is_none() {
                        break;
                    }
                    let next_square = next.expect("checked above");
                    line |= next_square.bit();
                    current = next_square;
                }

                table[from_index][to_index] = line;
            }
            to_index += 1;
        }
        from_index += 1;
    }
    table
}

const KNIGHT_ATTACKS: [u64; 64] = build_knight_attacks();
const KING_ATTACKS: [u64; 64] = build_king_attacks();
const WHITE_PAWN_ATTACKS: [u64; 64] = build_pawn_attacks(Color::White);
const BLACK_PAWN_ATTACKS: [u64; 64] = build_pawn_attacks(Color::Black);
static BETWEEN: OnceLock<[[u64; 64]; 64]> = OnceLock::new();
static LINE: OnceLock<[[u64; 64]; 64]> = OnceLock::new();

pub(crate) fn pawn_attacks(color: Color, square: Square) -> u64 {
    match color {
        Color::White => WHITE_PAWN_ATTACKS[square.index()],
        Color::Black => BLACK_PAWN_ATTACKS[square.index()],
    }
}

pub(crate) fn pawn_attackers_to(square: Square, by_color: Color) -> u64 {
    pawn_attacks(by_color.opposite(), square)
}

pub(crate) fn knight_attacks(square: Square) -> u64 {
    KNIGHT_ATTACKS[square.index()]
}

pub(crate) fn king_attacks(square: Square) -> u64 {
    KING_ATTACKS[square.index()]
}

pub(crate) fn bishop_attacks(square: Square, occupied: u64) -> u64 {
    magics::bishop_attacks(square, occupied)
}

pub(crate) fn rook_attacks(square: Square, occupied: u64) -> u64 {
    magics::rook_attacks(square, occupied)
}

pub(crate) fn queen_attacks(square: Square, occupied: u64) -> u64 {
    bishop_attacks(square, occupied) | rook_attacks(square, occupied)
}

pub(crate) fn between(from: Square, to: Square) -> u64 {
    BETWEEN.get_or_init(build_between)[from.index()][to.index()]
}

pub(crate) fn line(from: Square, to: Square) -> u64 {
    LINE.get_or_init(build_line)[from.index()][to.index()]
}
