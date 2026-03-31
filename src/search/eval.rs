use crate::core::{PieceType, Position, Score, Square, attacks, types::Color};

const TOTAL_PHASE: i32 = 24;
const FILE_CENTRALITY: [i32; 8] = [0, 4, 8, 12, 12, 8, 4, 0];
const RANK_CENTRALITY: [i32; 8] = [0, 4, 8, 12, 12, 8, 4, 0];
const PHASE_WEIGHTS: [i32; 6] = [0, 1, 1, 2, 4, 0];
const MG_VALUES: [i32; 6] = [100, 325, 335, 500, 900, 0];
const EG_VALUES: [i32; 6] = [100, 310, 345, 520, 900, 0];
const KNIGHT_MOBILITY: PhaseScore = PhaseScore::new(4, 3);
const BISHOP_MOBILITY: PhaseScore = PhaseScore::new(5, 5);
const ROOK_MOBILITY: PhaseScore = PhaseScore::new(2, 4);
const QUEEN_MOBILITY: PhaseScore = PhaseScore::new(1, 2);
const KNIGHT_OUTPOST_BONUS: PhaseScore = PhaseScore::new(18, 10);
const BISHOP_PAIR_BONUS: PhaseScore = PhaseScore::new(28, 42);
const DOUBLED_PAWN_PENALTY: PhaseScore = PhaseScore::new(10, 14);
const ISOLATED_PAWN_PENALTY: PhaseScore = PhaseScore::new(12, 10);
const PAWN_ISLAND_PENALTY: PhaseScore = PhaseScore::new(8, 10);
const PHALANX_PAWN_BONUS: PhaseScore = PhaseScore::new(8, 12);
const OPEN_FILE_ROOK_BONUS: PhaseScore = PhaseScore::new(18, 12);
const SEMI_OPEN_FILE_ROOK_BONUS: PhaseScore = PhaseScore::new(10, 6);
const ROOK_ON_SEVENTH_BONUS: PhaseScore = PhaseScore::new(10, 24);
const PASSED_PAWN_BONUS: [PhaseScore; 8] = [
    PhaseScore::new(0, 0),
    PhaseScore::new(8, 10),
    PhaseScore::new(14, 20),
    PhaseScore::new(24, 36),
    PhaseScore::new(40, 62),
    PhaseScore::new(68, 104),
    PhaseScore::new(0, 0),
    PhaseScore::new(0, 0),
];
const PROTECTED_PASSED_PAWN_BONUS: [PhaseScore; 8] = [
    PhaseScore::new(0, 0),
    PhaseScore::new(0, 0),
    PhaseScore::new(4, 8),
    PhaseScore::new(8, 14),
    PhaseScore::new(12, 20),
    PhaseScore::new(18, 28),
    PhaseScore::new(0, 0),
    PhaseScore::new(0, 0),
];
const PAWN_THREAT_MINOR: PhaseScore = PhaseScore::new(12, 8);
const PAWN_THREAT_ROOK: PhaseScore = PhaseScore::new(18, 12);
const PAWN_THREAT_QUEEN: PhaseScore = PhaseScore::new(26, 18);
const MINOR_THREAT_ROOK: PhaseScore = PhaseScore::new(10, 8);
const MINOR_THREAT_QUEEN: PhaseScore = PhaseScore::new(14, 10);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct PhaseScore {
    mg: i32,
    eg: i32,
}

impl PhaseScore {
    const fn new(mg: i32, eg: i32) -> Self {
        Self { mg, eg }
    }

    fn blend(self, phase: i32) -> i32 {
        (self.mg * phase + self.eg * (TOTAL_PHASE - phase)) / TOTAL_PHASE
    }
}

impl std::ops::AddAssign for PhaseScore {
    fn add_assign(&mut self, rhs: Self) {
        self.mg += rhs.mg;
        self.eg += rhs.eg;
    }
}

impl std::ops::Sub for PhaseScore {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.mg - rhs.mg, self.eg - rhs.eg)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EvalBreakdown {
    pub material_and_piece_square: i32,
    pub mobility: i32,
    pub king_safety: i32,
    pub pawn_structure: i32,
    pub passed_pawns: i32,
    pub bishop_pair: i32,
    pub rook_placement: i32,
    pub threats: i32,
    pub middlegame: i32,
    pub endgame: i32,
    pub phase: i32,
    pub blended: i32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct EvalTerms {
    material_and_piece_square: PhaseScore,
    mobility: PhaseScore,
    king_safety: PhaseScore,
    pawn_structure: PhaseScore,
    passed_pawns: PhaseScore,
    bishop_pair: PhaseScore,
    rook_placement: PhaseScore,
    threats: PhaseScore,
}

impl EvalTerms {
    fn total(self) -> PhaseScore {
        let mut total = self.material_and_piece_square;
        total += self.mobility;
        total += self.king_safety;
        total += self.pawn_structure;
        total += self.passed_pawns;
        total += self.bishop_pair;
        total += self.rook_placement;
        total += self.threats;
        total
    }
}

pub fn evaluate(position: &Position) -> Score {
    let breakdown = evaluate_breakdown(position);
    let score = match position.side_to_move() {
        Color::White => breakdown.blended,
        Color::Black => -breakdown.blended,
    };
    Score(score)
}

#[cfg(any(test, debug_assertions, feature = "internal-testing"))]
#[doc(hidden)]
pub fn debug_evaluate_breakdown(position: &Position) -> EvalBreakdown {
    evaluate_breakdown(position)
}

fn evaluate_breakdown(position: &Position) -> EvalBreakdown {
    let phase = game_phase(position);
    let white = evaluate_color(position, Color::White);
    let black = evaluate_color(position, Color::Black);
    let diff = white.total() - black.total();

    EvalBreakdown {
        material_and_piece_square: (white.material_and_piece_square
            - black.material_and_piece_square)
            .blend(phase),
        mobility: (white.mobility - black.mobility).blend(phase),
        king_safety: (white.king_safety - black.king_safety).blend(phase),
        pawn_structure: (white.pawn_structure - black.pawn_structure).blend(phase),
        passed_pawns: (white.passed_pawns - black.passed_pawns).blend(phase),
        bishop_pair: (white.bishop_pair - black.bishop_pair).blend(phase),
        rook_placement: (white.rook_placement - black.rook_placement).blend(phase),
        threats: (white.threats - black.threats).blend(phase),
        middlegame: diff.mg,
        endgame: diff.eg,
        phase,
        blended: diff.blend(phase),
    }
}

fn evaluate_color(position: &Position, color: Color) -> EvalTerms {
    EvalTerms {
        material_and_piece_square: material_and_piece_square(position, color),
        mobility: mobility(position, color),
        king_safety: king_safety(position, color),
        pawn_structure: pawn_structure(position, color),
        passed_pawns: passed_pawns(position, color),
        bishop_pair: bishop_pair(color, position),
        rook_placement: rook_placement(position, color),
        threats: threats(position, color),
    }
}

fn material_and_piece_square(position: &Position, color: Color) -> PhaseScore {
    let mut score = PhaseScore::default();
    let own_pawns = position.pieces(color, PieceType::Pawn);
    let enemy_pawns = position.pieces(color.opposite(), PieceType::Pawn);
    for piece_type in PieceType::ALL {
        let mut pieces = position.pieces(color, piece_type);
        while let Some(square) = pop_lsb(&mut pieces) {
            score += PhaseScore::new(MG_VALUES[piece_type.index()], EG_VALUES[piece_type.index()]);
            score += piece_square_term(piece_type, color, square);
            score += piece_positional_term(piece_type, color, square, own_pawns, enemy_pawns);
        }
    }
    score
}

fn piece_square_term(piece_type: PieceType, color: Color, square: Square) -> PhaseScore {
    let relative_rank = relative_rank(color, square);
    let file = square.file() as usize;
    let center = FILE_CENTRALITY[file] + RANK_CENTRALITY[relative_rank as usize];
    let advance = relative_rank as i32;

    match piece_type {
        PieceType::Pawn => PhaseScore::new(advance * 6 + FILE_CENTRALITY[file] / 2, advance * 12),
        PieceType::Knight => PhaseScore::new(center * 2, center),
        PieceType::Bishop => PhaseScore::new(center + advance * 2, center + advance * 3),
        PieceType::Rook => PhaseScore::new(advance * 2 + FILE_CENTRALITY[file] / 2, advance * 5),
        PieceType::Queen => PhaseScore::new(center / 2, center / 3),
        PieceType::King => PhaseScore::new(18 - center - advance * 3, center + advance * 2),
    }
}

fn piece_positional_term(
    piece_type: PieceType,
    color: Color,
    square: Square,
    own_pawns: u64,
    enemy_pawns: u64,
) -> PhaseScore {
    match piece_type {
        PieceType::Knight if knight_is_supported_outpost(color, square, own_pawns, enemy_pawns) => {
            KNIGHT_OUTPOST_BONUS
        }
        _ => PhaseScore::default(),
    }
}

fn mobility(position: &Position, color: Color) -> PhaseScore {
    let occupied = position.occupancy();
    let friendly = position.occupancy_by(color);
    let mut score = PhaseScore::default();

    let mut knights = position.pieces(color, PieceType::Knight);
    while let Some(square) = pop_lsb(&mut knights) {
        score += scale_by_count(
            KNIGHT_MOBILITY,
            (attacks::knight_attacks(square) & !friendly).count_ones() as i32,
        );
    }

    let mut bishops = position.pieces(color, PieceType::Bishop);
    while let Some(square) = pop_lsb(&mut bishops) {
        score += scale_by_count(
            BISHOP_MOBILITY,
            (attacks::bishop_attacks(square, occupied) & !friendly).count_ones() as i32,
        );
    }

    let mut rooks = position.pieces(color, PieceType::Rook);
    while let Some(square) = pop_lsb(&mut rooks) {
        score += scale_by_count(
            ROOK_MOBILITY,
            (attacks::rook_attacks(square, occupied) & !friendly).count_ones() as i32,
        );
    }

    let mut queens = position.pieces(color, PieceType::Queen);
    while let Some(square) = pop_lsb(&mut queens) {
        score += scale_by_count(
            QUEEN_MOBILITY,
            (attacks::queen_attacks(square, occupied) & !friendly).count_ones() as i32,
        );
    }

    score
}

fn king_safety(position: &Position, color: Color) -> PhaseScore {
    let king_square = position.king_square(color);
    let pawns = position.pieces(color, PieceType::Pawn);
    let enemy_pawns = position.pieces(color.opposite(), PieceType::Pawn);
    let mut score = pawn_shield(color, king_square, pawns);

    for file in neighboring_files(king_square.file()) {
        let own_file_pawns = pawns & file_mask(file);
        let enemy_file_pawns = enemy_pawns & file_mask(file);
        let file_weight = if file == king_square.file() { 12 } else { 7 };
        if own_file_pawns == 0 {
            score += PhaseScore::new(-file_weight, -4);
            if enemy_file_pawns == 0 {
                score += PhaseScore::new(-(file_weight / 2), -2);
            }
        }
    }

    score
}

fn pawn_shield(color: Color, king_square: Square, pawns: u64) -> PhaseScore {
    let mut score = PhaseScore::default();
    let direction = color.pawn_direction();

    for file_delta in -1..=1 {
        if let Some(front) = king_square.offset(file_delta, direction) {
            if pawns & front.bit() != 0 {
                score += PhaseScore::new(10, 2);
            } else {
                score += PhaseScore::new(-8, -1);
            }

            if let Some(front_two) = front.offset(0, direction)
                && pawns & front_two.bit() != 0
            {
                score += PhaseScore::new(4, 1);
            }
        }
    }

    score
}

fn pawn_structure(position: &Position, color: Color) -> PhaseScore {
    let pawns = position.pieces(color, PieceType::Pawn);
    let mut score = PhaseScore::default();
    let mut islands = 0i32;
    let mut previous_file_occupied = false;

    for file in 0..8u8 {
        let file_has_pawn = pawns & file_mask(file) != 0;
        let pawns_on_file = (pawns & file_mask(file)).count_ones() as i32;
        if file_has_pawn && !previous_file_occupied {
            islands += 1;
        }
        previous_file_occupied = file_has_pawn;
        if pawns_on_file > 1 {
            score += scale_by_count(DOUBLED_PAWN_PENALTY, -(pawns_on_file - 1));
        }
        if pawns_on_file == 0 {
            continue;
        }

        let left = file.checked_sub(1).map(file_mask).unwrap_or(0);
        let right = if file < 7 { file_mask(file + 1) } else { 0 };
        if pawns & (left | right) == 0 {
            score += scale_by_count(ISOLATED_PAWN_PENALTY, -pawns_on_file);
        }
    }

    let phalanx_pairs = count_phalanx_pairs(pawns);
    if phalanx_pairs > 0 {
        score += scale_by_count(PHALANX_PAWN_BONUS, phalanx_pairs);
    }
    if islands > 1 {
        score += scale_by_count(PAWN_ISLAND_PENALTY, -(islands - 1));
    }

    score
}

fn passed_pawns(position: &Position, color: Color) -> PhaseScore {
    let mut pawns = position.pieces(color, PieceType::Pawn);
    let own_pawns = pawns;
    let mut score = PhaseScore::default();

    while let Some(square) = pop_lsb(&mut pawns) {
        if is_passed_pawn(position, color, square) {
            let relative_rank = relative_rank(color, square) as usize;
            score += PASSED_PAWN_BONUS[relative_rank];
            if pawn_is_protected_by_friendly_pawn(color, square, own_pawns) {
                score += PROTECTED_PASSED_PAWN_BONUS[relative_rank];
            }
        }
    }

    score
}

fn bishop_pair(color: Color, position: &Position) -> PhaseScore {
    if position.pieces(color, PieceType::Bishop).count_ones() >= 2 {
        BISHOP_PAIR_BONUS
    } else {
        PhaseScore::default()
    }
}

fn rook_placement(position: &Position, color: Color) -> PhaseScore {
    let rooks = position.pieces(color, PieceType::Rook);
    let own_pawns = position.pieces(color, PieceType::Pawn);
    let enemy_pawns = position.pieces(color.opposite(), PieceType::Pawn);
    let mut score = PhaseScore::default();
    let mut rooks_remaining = rooks;

    while let Some(square) = pop_lsb(&mut rooks_remaining) {
        let file = square.file();
        let own_file_pawns = own_pawns & file_mask(file);
        let enemy_file_pawns = enemy_pawns & file_mask(file);
        if relative_rank(color, square) == 6 && enemy_pawns != 0 {
            score += ROOK_ON_SEVENTH_BONUS;
        }
        if own_file_pawns == 0 {
            score += if enemy_file_pawns == 0 {
                OPEN_FILE_ROOK_BONUS
            } else {
                SEMI_OPEN_FILE_ROOK_BONUS
            };
        }
    }

    score
}

fn threats(position: &Position, color: Color) -> PhaseScore {
    let enemy = color.opposite();
    let enemy_knights = position.pieces(enemy, PieceType::Knight);
    let enemy_bishops = position.pieces(enemy, PieceType::Bishop);
    let enemy_rooks = position.pieces(enemy, PieceType::Rook);
    let enemy_queens = position.pieces(enemy, PieceType::Queen);
    let occupied = position.occupancy();

    let mut pawn_attacks_all = 0u64;
    let mut pawns = position.pieces(color, PieceType::Pawn);
    while let Some(square) = pop_lsb(&mut pawns) {
        pawn_attacks_all |= attacks::pawn_attacks(color, square);
    }

    let mut minor_attacks_all = 0u64;
    let mut knights = position.pieces(color, PieceType::Knight);
    while let Some(square) = pop_lsb(&mut knights) {
        minor_attacks_all |= attacks::knight_attacks(square);
    }
    let mut bishops = position.pieces(color, PieceType::Bishop);
    while let Some(square) = pop_lsb(&mut bishops) {
        minor_attacks_all |= attacks::bishop_attacks(square, occupied);
    }

    let mut score = PhaseScore::default();
    score += scale_by_count(
        PAWN_THREAT_MINOR,
        (pawn_attacks_all & (enemy_knights | enemy_bishops)).count_ones() as i32,
    );
    score += scale_by_count(
        PAWN_THREAT_ROOK,
        (pawn_attacks_all & enemy_rooks).count_ones() as i32,
    );
    score += scale_by_count(
        PAWN_THREAT_QUEEN,
        (pawn_attacks_all & enemy_queens).count_ones() as i32,
    );
    score += scale_by_count(
        MINOR_THREAT_ROOK,
        (minor_attacks_all & enemy_rooks).count_ones() as i32,
    );
    score += scale_by_count(
        MINOR_THREAT_QUEEN,
        (minor_attacks_all & enemy_queens).count_ones() as i32,
    );
    score
}

fn game_phase(position: &Position) -> i32 {
    let mut phase = 0i32;
    for color in Color::ALL {
        for piece_type in PieceType::ALL {
            phase += PHASE_WEIGHTS[piece_type.index()]
                * position.pieces(color, piece_type).count_ones() as i32;
        }
    }
    phase.min(TOTAL_PHASE)
}

fn scale_by_count(term: PhaseScore, count: i32) -> PhaseScore {
    PhaseScore::new(term.mg * count, term.eg * count)
}

fn relative_rank(color: Color, square: Square) -> u8 {
    match color {
        Color::White => square.rank(),
        Color::Black => 7 - square.rank(),
    }
}

fn is_passed_pawn(position: &Position, color: Color, square: Square) -> bool {
    let enemy_pawns = position.pieces(color.opposite(), PieceType::Pawn);
    let file = square.file() as i8;
    let rank = square.rank() as i8;
    let direction = color.pawn_direction();

    for file_delta in -1..=1 {
        let target_file = file + file_delta;
        if !(0..=7).contains(&target_file) {
            continue;
        }

        let mut target_rank = rank + direction;
        while (0..=7).contains(&target_rank) {
            let target = Square::from_coords(target_file as u8, target_rank as u8)
                .expect("checked board bounds above");
            if enemy_pawns & target.bit() != 0 {
                return false;
            }
            target_rank += direction;
        }
    }

    true
}

fn knight_is_supported_outpost(
    color: Color,
    square: Square,
    own_pawns: u64,
    enemy_pawns: u64,
) -> bool {
    let relative_rank = relative_rank(color, square);
    if relative_rank < 3 {
        return false;
    }

    let supported_by_pawn = attacks::pawn_attackers_to(square, color) & own_pawns != 0;
    let attacked_by_enemy_pawn =
        attacks::pawn_attackers_to(square, color.opposite()) & enemy_pawns != 0;
    supported_by_pawn && !attacked_by_enemy_pawn
}

fn pawn_is_protected_by_friendly_pawn(color: Color, square: Square, own_pawns: u64) -> bool {
    let backward_rank_delta = -color.pawn_direction();
    [-1, 1].into_iter().any(|file_delta| {
        square
            .offset(file_delta, backward_rank_delta)
            .is_some_and(|support_square| own_pawns & support_square.bit() != 0)
    })
}

fn count_phalanx_pairs(pawns: u64) -> i32 {
    let mut remaining = pawns;
    let mut pairs = 0i32;
    while let Some(square) = pop_lsb(&mut remaining) {
        if square.file() == 7 {
            continue;
        }
        let right = Square::from_coords(square.file() + 1, square.rank())
            .expect("file+1 must remain on board");
        if pawns & right.bit() != 0 {
            pairs += 1;
        }
    }
    pairs
}

fn neighboring_files(center: u8) -> impl Iterator<Item = u8> {
    let start = center.saturating_sub(1);
    let end = (center + 1).min(7);
    start..=end
}

fn file_mask(file: u8) -> u64 {
    0x0101_0101_0101_0101u64 << file
}

fn pop_lsb(bits: &mut u64) -> Option<Square> {
    if *bits == 0 {
        return None;
    }

    let index = bits.trailing_zeros() as u8;
    *bits &= *bits - 1;
    Some(Square::from_index_unchecked(index))
}
