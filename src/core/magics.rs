use std::sync::OnceLock;

use super::square::Square;

const BISHOP_DIRECTIONS: [(i8, i8); 4] = [(-1, -1), (-1, 1), (1, -1), (1, 1)];
const ROOK_DIRECTIONS: [(i8, i8); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

struct SlidingTables {
    bishop_masks: [u64; 64],
    bishop_offsets: [usize; 64],
    bishop_tables: Box<[u64]>,
    rook_masks: [u64; 64],
    rook_offsets: [usize; 64],
    rook_tables: Box<[u64]>,
}

static TABLES: OnceLock<SlidingTables> = OnceLock::new();

pub(crate) fn bishop_attacks(square: Square, occupied: u64) -> u64 {
    let tables = tables();
    let mask = tables.bishop_masks[square.index()];
    let index = subset_index(occupied & mask, mask);
    tables.bishop_tables[tables.bishop_offsets[square.index()] + index]
}

pub(crate) fn rook_attacks(square: Square, occupied: u64) -> u64 {
    let tables = tables();
    let mask = tables.rook_masks[square.index()];
    let index = subset_index(occupied & mask, mask);
    tables.rook_tables[tables.rook_offsets[square.index()] + index]
}

fn tables() -> &'static SlidingTables {
    TABLES.get_or_init(build_tables)
}

fn build_tables() -> SlidingTables {
    let (bishop_masks, bishop_offsets, bishop_tables) = build_piece_tables(&BISHOP_DIRECTIONS);
    let (rook_masks, rook_offsets, rook_tables) = build_piece_tables(&ROOK_DIRECTIONS);

    SlidingTables {
        bishop_masks,
        bishop_offsets,
        bishop_tables: bishop_tables.into_boxed_slice(),
        rook_masks,
        rook_offsets,
        rook_tables: rook_tables.into_boxed_slice(),
    }
}

fn build_piece_tables(directions: &[(i8, i8)]) -> ([u64; 64], [usize; 64], Vec<u64>) {
    let mut masks = [0u64; 64];
    let mut offsets = [0usize; 64];
    let mut total_entries = 0usize;

    for index in 0..64 {
        let square = Square::from_index_unchecked(index as u8);
        let mask = relevant_mask(square, directions);
        masks[index] = mask;
        offsets[index] = total_entries;
        total_entries += 1usize << mask.count_ones();
    }

    let mut table = vec![0u64; total_entries];
    for index in 0..64 {
        let square = Square::from_index_unchecked(index as u8);
        let mask = masks[index];
        let subset_count = 1usize << mask.count_ones();
        for subset_index_value in 0..subset_count {
            let occupied = occupancy_from_index(mask, subset_index_value);
            table[offsets[index] + subset_index_value] =
                slider_attacks_on_the_fly(square, occupied, directions);
        }
    }

    (masks, offsets, table)
}

fn relevant_mask(square: Square, directions: &[(i8, i8)]) -> u64 {
    let mut mask = 0u64;
    for &(file_delta, rank_delta) in directions {
        let mut current = square;
        while let Some(next) = current.offset(file_delta, rank_delta) {
            if next.offset(file_delta, rank_delta).is_none() {
                break;
            }
            mask |= next.bit();
            current = next;
        }
    }
    mask
}

fn occupancy_from_index(mask: u64, index: usize) -> u64 {
    let mut result = 0u64;
    let mut working_mask = mask;
    let mut bit_index = 0usize;

    while working_mask != 0 {
        let square_bit = working_mask & working_mask.wrapping_neg();
        if (index >> bit_index) & 1 == 1 {
            result |= square_bit;
        }
        working_mask &= working_mask - 1;
        bit_index += 1;
    }

    result
}

fn subset_index(occupied: u64, mask: u64) -> usize {
    let mut index = 0usize;
    let mut working_mask = mask;
    let mut bit_index = 0usize;

    while working_mask != 0 {
        let square_bit = working_mask & working_mask.wrapping_neg();
        if occupied & square_bit != 0 {
            index |= 1usize << bit_index;
        }
        working_mask &= working_mask - 1;
        bit_index += 1;
    }

    index
}

fn slider_attacks_on_the_fly(square: Square, occupied: u64, directions: &[(i8, i8)]) -> u64 {
    let mut attacks = 0u64;
    for &(file_delta, rank_delta) in directions {
        let mut current = square;
        while let Some(next) = current.offset(file_delta, rank_delta) {
            attacks |= next.bit();
            if occupied & next.bit() != 0 {
                break;
            }
            current = next;
        }
    }
    attacks
}
