use std::sync::OnceLock;

use super::{piece::Piece, square::Square, types::CastlingRights};

struct ZobristTables {
    piece_square: [[u64; 64]; 12],
    castling: [u64; 16],
    en_passant_file: [u64; 8],
    side_to_move: u64,
}

static TABLES: OnceLock<ZobristTables> = OnceLock::new();

pub(crate) fn piece_square(piece: Piece, square: Square) -> u64 {
    tables().piece_square[piece as usize][square.index()]
}

pub(crate) fn castling(rights: CastlingRights) -> u64 {
    tables().castling[rights.bits() as usize]
}

pub(crate) fn en_passant_file(file: u8) -> u64 {
    tables().en_passant_file[file as usize]
}

pub(crate) fn side_to_move() -> u64 {
    tables().side_to_move
}

fn tables() -> &'static ZobristTables {
    TABLES.get_or_init(build_tables)
}

fn build_tables() -> ZobristTables {
    let mut seed = 0x4d595df4d0f33173u64;
    let mut piece_square = [[0u64; 64]; 12];
    let mut castling = [0u64; 16];
    let mut en_passant_file = [0u64; 8];

    for table in &mut piece_square {
        for entry in table {
            *entry = next_u64(&mut seed);
        }
    }
    for entry in &mut castling {
        *entry = next_u64(&mut seed);
    }
    for entry in &mut en_passant_file {
        *entry = next_u64(&mut seed);
    }

    let side_to_move = next_u64(&mut seed);

    ZobristTables {
        piece_square,
        castling,
        en_passant_file,
        side_to_move,
    }
}

fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}
