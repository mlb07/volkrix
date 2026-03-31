use std::{error::Error, fmt};

use super::{
    attacks,
    chess_move::{
        FLAG_CAPTURE, FLAG_CASTLE, FLAG_DOUBLE_PAWN_PUSH, FLAG_EN_PASSANT, Move, ParsedMove,
    },
    movelist::MoveList,
    piece::{Piece, PieceType},
    repetition::RepetitionHistory,
    see,
    square::Square,
    types::{CastlingRights, Color, Value},
    zobrist,
};

const PIECE_LIST_CAPACITY: usize = 16;
const OCCUPANCY_WHITE: usize = 0;
const OCCUPANCY_BLACK: usize = 1;
const OCCUPANCY_ALL: usize = 2;
const ALL_TARGETS: u64 = u64::MAX;

#[derive(Clone, Copy, Debug)]
struct SquareList {
    squares: [Square; PIECE_LIST_CAPACITY],
    len: u8,
}

impl SquareList {
    const fn new() -> Self {
        Self {
            squares: [Square::A1; PIECE_LIST_CAPACITY],
            len: 0,
        }
    }

    fn push(&mut self, square: Square) {
        let index = self.len as usize;
        assert!(index < self.squares.len(), "piece list overflow");
        self.squares[index] = square;
        self.len += 1;
    }

    fn remove(&mut self, square: Square) -> bool {
        let mut index = 0usize;
        while index < self.len as usize {
            if self.squares[index] == square {
                let last_index = self.len as usize - 1;
                self.squares[index] = self.squares[last_index];
                self.len -= 1;
                return true;
            }
            index += 1;
        }
        false
    }

    fn len(&self) -> usize {
        self.len as usize
    }

    fn iter(&self) -> impl Iterator<Item = Square> + '_ {
        self.squares[..self.len as usize].iter().copied()
    }
}

#[derive(Clone, Debug)]
pub struct Position {
    board: [Option<Piece>; 64],
    piece_bitboards: [[u64; 6]; 2],
    piece_lists: [[SquareList; 6]; 2],
    king_squares: [Square; 2],
    occupancies: [u64; 3],
    side_to_move: Color,
    castling_rights: CastlingRights,
    en_passant: Option<Square>,
    halfmove_clock: u16,
    fullmove_number: u16,
    zobrist_key: u64,
    repetition_history: RepetitionHistory,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UndoState {
    pub captured_piece: Option<Piece>,
    pub moved_piece: Piece,
    pub previous_castling_rights: CastlingRights,
    pub previous_en_passant: Option<Square>,
    pub previous_halfmove_clock: u16,
    pub previous_fullmove_number: u16,
    pub previous_zobrist_key: u64,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct NullMoveState {
    previous_en_passant: Option<Square>,
    previous_halfmove_clock: u16,
    previous_fullmove_number: u16,
    previous_zobrist_key: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MoveGenStage {
    All,
    Captures,
    Quiets,
    Evasions,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct CheckInfo {
    pub checkers: u64,
    pub pinned: u64,
    pub block_mask: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PositionStatus {
    Ongoing,
    Checkmate,
    Stalemate,
    DrawByRepetition,
    DrawByFiftyMove,
    DrawByInsufficientMaterial,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HistoryMode {
    Persistent,
    Transient,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MoveError {
    InvalidUciMove(String),
    NoPieceOnSource,
    WrongSideToMove,
    DestinationOccupiedByOwnPiece,
    IllegalEnPassant,
    IllegalCastle,
    InvalidPromotion,
    LeavesKingInCheck,
    HistoryOverflow,
    IllegalMove(String),
}

impl fmt::Display for MoveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidUciMove(value) => write!(f, "invalid UCI move: {value}"),
            Self::NoPieceOnSource => f.write_str("no piece on source square"),
            Self::WrongSideToMove => f.write_str("piece does not match the side to move"),
            Self::DestinationOccupiedByOwnPiece => {
                f.write_str("destination square is occupied by own piece")
            }
            Self::IllegalEnPassant => f.write_str("illegal en passant move"),
            Self::IllegalCastle => f.write_str("illegal castling move"),
            Self::InvalidPromotion => f.write_str("invalid promotion move"),
            Self::LeavesKingInCheck => f.write_str("move leaves king in check"),
            Self::HistoryOverflow => f.write_str("repetition history capacity exceeded"),
            Self::IllegalMove(value) => write!(f, "illegal move: {value}"),
        }
    }
}

impl Error for MoveError {}

impl Position {
    pub fn empty() -> Self {
        let mut position = Self {
            board: [None; 64],
            piece_bitboards: [[0; 6]; 2],
            piece_lists: [[SquareList::new(); 6]; 2],
            king_squares: [Square::E1, Square::E8],
            occupancies: [0; 3],
            side_to_move: Color::White,
            castling_rights: CastlingRights::NONE,
            en_passant: None,
            halfmove_clock: 0,
            fullmove_number: 1,
            zobrist_key: 0,
            repetition_history: RepetitionHistory::empty(),
        };
        position.refresh_derived_state_from_scratch();
        position
    }

    pub fn side_to_move(&self) -> Color {
        self.side_to_move
    }

    pub fn castling_rights(&self) -> CastlingRights {
        self.castling_rights
    }

    pub fn en_passant(&self) -> Option<Square> {
        self.en_passant
    }

    pub fn halfmove_clock(&self) -> u16 {
        self.halfmove_clock
    }

    pub fn fullmove_number(&self) -> u16 {
        self.fullmove_number
    }

    pub fn zobrist_key(&self) -> u64 {
        self.zobrist_key
    }

    pub(crate) fn search_key(&self) -> u64 {
        let history = self.repetition_history.as_slice();
        let relevant_len = usize::min(history.len(), self.halfmove_clock as usize + 1);
        let start = history.len().saturating_sub(relevant_len);

        let mut hash = mix_u64(self.zobrist_key ^ ((self.halfmove_clock as u64) << 48));
        hash ^= mix_u64(relevant_len as u64);
        for (offset, key) in history[start..].iter().enumerate() {
            let mixed =
                mix_u64(key.wrapping_add((offset as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)));
            hash ^= mixed;
            hash = hash.rotate_left(11);
        }
        hash
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    pub fn debug_repetition_history_snapshot(&self) -> Vec<u64> {
        self.repetition_history.as_slice().to_vec()
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    pub fn debug_search_key(&self) -> u64 {
        self.search_key()
    }

    pub fn is_draw_by_repetition(&self) -> bool {
        self.repetition_history
            .is_threefold_repetition(self.halfmove_clock)
    }

    pub fn is_draw_by_fifty_move(&self) -> bool {
        self.halfmove_clock >= 100
    }

    pub fn is_insufficient_material(&self) -> bool {
        if self.pieces(Color::White, PieceType::Pawn) != 0
            || self.pieces(Color::Black, PieceType::Pawn) != 0
            || self.pieces(Color::White, PieceType::Rook) != 0
            || self.pieces(Color::Black, PieceType::Rook) != 0
            || self.pieces(Color::White, PieceType::Queen) != 0
            || self.pieces(Color::Black, PieceType::Queen) != 0
        {
            return false;
        }

        let white_knights = self.pieces(Color::White, PieceType::Knight).count_ones();
        let black_knights = self.pieces(Color::Black, PieceType::Knight).count_ones();
        let white_bishops = self.pieces(Color::White, PieceType::Bishop).count_ones();
        let black_bishops = self.pieces(Color::Black, PieceType::Bishop).count_ones();
        let total_minors = white_knights + black_knights + white_bishops + black_bishops;

        matches!(
            (
                white_knights,
                black_knights,
                white_bishops,
                black_bishops,
                total_minors
            ),
            (0, 0, 0, 0, 0)
                | (1, 0, 0, 0, 1)
                | (0, 1, 0, 0, 1)
                | (0, 0, 1, 0, 1)
                | (0, 0, 0, 1, 1)
                | (0, 0, 1, 1, 2)
        )
    }

    pub fn status(&mut self) -> PositionStatus {
        let mut legal_moves = MoveList::new();
        self.generate_legal_moves(&mut legal_moves);
        if legal_moves.is_empty() {
            return if self.is_in_check(self.side_to_move) {
                PositionStatus::Checkmate
            } else {
                PositionStatus::Stalemate
            };
        }
        if self.is_draw_by_repetition() {
            return PositionStatus::DrawByRepetition;
        }
        if self.is_draw_by_fifty_move() {
            return PositionStatus::DrawByFiftyMove;
        }
        if self.is_insufficient_material() {
            return PositionStatus::DrawByInsufficientMaterial;
        }
        PositionStatus::Ongoing
    }

    pub fn see(&self, mv: Move) -> Value {
        self.see_internal(mv)
    }

    pub fn piece_at(&self, square: Square) -> Option<Piece> {
        self.board[square.index()]
    }

    pub(crate) fn set_side_to_move(&mut self, color: Color) {
        self.side_to_move = color;
    }

    pub(crate) fn set_castling_rights(&mut self, rights: CastlingRights) {
        self.castling_rights = rights;
    }

    pub(crate) fn set_en_passant(&mut self, square: Option<Square>) {
        self.en_passant = square;
    }

    pub(crate) fn set_halfmove_clock(&mut self, value: u16) {
        self.halfmove_clock = value;
    }

    pub(crate) fn set_fullmove_number(&mut self, value: u16) {
        self.fullmove_number = value;
    }

    pub(crate) fn refresh_derived_state_from_scratch(&mut self) {
        self.zobrist_key = self.recompute_zobrist();
        self.repetition_history.clear_and_seed(self.zobrist_key);
    }

    fn recompute_zobrist(&self) -> u64 {
        let mut key = 0u64;
        for index in 0..64 {
            let square = Square::from_index_unchecked(index as u8);
            if let Some(piece) = self.board[index] {
                key ^= zobrist::piece_square(piece, square);
            }
        }
        key ^= zobrist::castling(self.castling_rights);
        if let Some(file) = self.hashed_en_passant_file() {
            key ^= zobrist::en_passant_file(file);
        }
        if self.side_to_move == Color::Black {
            key ^= zobrist::side_to_move();
        }
        key
    }

    fn hashed_en_passant_file(&self) -> Option<u8> {
        let en_passant = self.en_passant?;
        let attackers = attacks::pawn_attackers_to(en_passant, self.side_to_move)
            & self.pieces(self.side_to_move, PieceType::Pawn);
        if attackers == 0 {
            None
        } else {
            Some(en_passant.file())
        }
    }

    fn xor_hashed_en_passant(&mut self) {
        if let Some(file) = self.hashed_en_passant_file() {
            self.zobrist_key ^= zobrist::en_passant_file(file);
        }
    }

    fn set_en_passant_hashed(&mut self, value: Option<Square>) {
        self.xor_hashed_en_passant();
        self.en_passant = value;
        self.xor_hashed_en_passant();
    }

    fn set_castling_rights_hashed(&mut self, value: CastlingRights) {
        self.zobrist_key ^= zobrist::castling(self.castling_rights);
        self.castling_rights = value;
        self.zobrist_key ^= zobrist::castling(self.castling_rights);
    }

    fn set_side_to_move_hashed(&mut self, value: Color) {
        if self.side_to_move == value {
            return;
        }
        self.xor_hashed_en_passant();
        self.side_to_move = value;
        self.zobrist_key ^= zobrist::side_to_move();
        self.xor_hashed_en_passant();
    }

    pub(crate) fn place_piece(&mut self, square: Square, piece: Piece) -> Result<(), String> {
        if self.piece_at(square).is_some() {
            return Err(format!("square {square} is already occupied"));
        }
        self.add_piece_unchecked(square, piece);
        Ok(())
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut expected_piece_bitboards = [[0u64; 6]; 2];
        let mut expected_occupancies = [0u64; 3];
        let mut expected_piece_lists = [[0usize; 6]; 2];
        let mut king_counts = [0usize; 2];

        for index in 0..64 {
            let square = Square::from_index_unchecked(index as u8);
            if let Some(piece) = self.board[index] {
                let color_index = piece.color().index();
                let piece_index = piece.piece_type().index();
                expected_piece_bitboards[color_index][piece_index] |= square.bit();
                expected_occupancies[color_index] |= square.bit();
                expected_piece_lists[color_index][piece_index] += 1;
                if piece.piece_type() == PieceType::King {
                    king_counts[color_index] += 1;
                    if self.king_squares[color_index] != square {
                        return Err(format!(
                            "king square cache mismatch for {:?}: expected {}, found {}",
                            piece.color(),
                            square,
                            self.king_squares[color_index]
                        ));
                    }
                }
            }
        }

        expected_occupancies[OCCUPANCY_ALL] =
            expected_occupancies[OCCUPANCY_WHITE] | expected_occupancies[OCCUPANCY_BLACK];

        if self.piece_bitboards != expected_piece_bitboards {
            return Err("piece bitboards do not match board state".to_owned());
        }

        if self.occupancies != expected_occupancies {
            return Err("occupancy bitboards do not match board state".to_owned());
        }

        if king_counts != [1, 1] {
            return Err("position must contain exactly one king of each color".to_owned());
        }

        if self.king_squares[Color::White.index()]
            .file()
            .abs_diff(self.king_squares[Color::Black.index()].file())
            <= 1
            && self.king_squares[Color::White.index()]
                .rank()
                .abs_diff(self.king_squares[Color::Black.index()].rank())
                <= 1
        {
            return Err("kings may not be adjacent".to_owned());
        }

        for color in Color::ALL {
            for piece_type in PieceType::ALL {
                let list = &self.piece_lists[color.index()][piece_type.index()];
                if list.len() != expected_piece_lists[color.index()][piece_type.index()] {
                    return Err(format!(
                        "piece list count mismatch for {:?} {:?}",
                        color, piece_type
                    ));
                }

                let mut seen = 0u64;
                for square in list.iter() {
                    if self.piece_at(square) != Some(Piece::from_parts(color, piece_type)) {
                        return Err(format!(
                            "piece list entry mismatch for {:?} {:?} at {}",
                            color, piece_type, square
                        ));
                    }
                    if seen & square.bit() != 0 {
                        return Err(format!(
                            "duplicate piece list entry for {:?} {:?} at {}",
                            color, piece_type, square
                        ));
                    }
                    seen |= square.bit();
                }
            }
        }

        if let Some(en_passant) = self.en_passant {
            if self.piece_at(en_passant).is_some() {
                return Err("en passant square must be empty".to_owned());
            }
            let expected_rank = match self.side_to_move {
                Color::White => 5,
                Color::Black => 2,
            };
            if en_passant.rank() != expected_rank {
                return Err("en passant square rank does not match side to move".to_owned());
            }
        }

        if self.fullmove_number == 0 {
            return Err("fullmove number must be at least one".to_owned());
        }

        if self.zobrist_key != self.recompute_zobrist() {
            return Err("incremental zobrist key does not match recomputed zobrist".to_owned());
        }

        if self.repetition_history.len() == 0 {
            return Err("repetition history must contain the current position key".to_owned());
        }

        if self.repetition_history.current() != Some(self.zobrist_key) {
            return Err(
                "repetition history tail does not match the current zobrist key".to_owned(),
            );
        }

        Ok(())
    }

    pub fn is_square_attacked(&self, square: Square, by_color: Color) -> bool {
        self.attackers_to(square, by_color) != 0
    }

    pub fn is_in_check(&self, color: Color) -> bool {
        self.attackers_to(self.king_squares[color.index()], color.opposite()) != 0
    }

    /// Generates legal moves into a fixed-capacity buffer.
    ///
    /// This requires mutable access even though legal generation is logically read-only because
    /// en passant still needs temporary `make_move` / `unmake_move` validation. That edge case can
    /// uncover a rook, bishop, or queen attack on the king after both pawns disappear from their
    /// original file/rank relationship, and Phase 2 intentionally reuses the authoritative move
    /// application path for that check instead of maintaining a second special-case legality path.
    pub fn generate_legal_moves(&mut self, moves: &mut MoveList) {
        moves.clear();

        let info = self.check_info();
        let mut pseudo_legal = MoveList::new();

        if info.checkers.count_ones() >= 2 {
            self.generate_king_moves(MoveGenStage::Captures, ALL_TARGETS, &mut pseudo_legal);
            self.generate_king_moves(MoveGenStage::Quiets, ALL_TARGETS, &mut pseudo_legal);
        } else if info.checkers != 0 {
            self.generate_evasions(&info, &mut pseudo_legal);
        } else {
            self.generate_pseudo_legal(MoveGenStage::All, &mut pseudo_legal);
        }

        for mv in pseudo_legal.as_slice().iter().copied() {
            if self.is_legal_fast(mv, &info) {
                moves.push(mv);
            }
        }
    }

    pub fn select_placeholder_bestmove(&mut self) -> Option<Move> {
        let mut legal_moves = MoveList::new();
        self.generate_legal_moves(&mut legal_moves);
        legal_moves.as_slice().first().copied()
    }

    pub fn apply_uci_move(&mut self, text: &str) -> Result<Move, MoveError> {
        let parsed =
            ParsedMove::parse(text).map_err(|_| MoveError::InvalidUciMove(text.to_owned()))?;
        let mut legal_moves = MoveList::new();
        self.generate_legal_moves(&mut legal_moves);

        for mv in legal_moves.as_slice().iter().copied() {
            if mv.matches_parsed(parsed) {
                self.make_move(mv)?;
                return Ok(mv);
            }
        }

        Err(MoveError::IllegalMove(text.to_owned()))
    }

    pub fn make_move(&mut self, mv: Move) -> Result<UndoState, MoveError> {
        self.make_move_with_history(mv, HistoryMode::Persistent)
    }

    #[cfg(test)]
    pub(crate) fn make_null_move(&mut self) -> Result<NullMoveState, MoveError> {
        let moving_color = self.side_to_move;
        let undo = NullMoveState {
            previous_en_passant: self.en_passant,
            previous_halfmove_clock: self.halfmove_clock,
            previous_fullmove_number: self.fullmove_number,
            previous_zobrist_key: self.zobrist_key,
        };

        self.set_en_passant_hashed(None);
        self.halfmove_clock = self.halfmove_clock.saturating_add(1);
        if moving_color == Color::Black {
            self.fullmove_number = self.fullmove_number.saturating_add(1);
        }
        self.set_side_to_move_hashed(moving_color.opposite());

        if self.repetition_history.push(self.zobrist_key).is_err() {
            self.side_to_move = moving_color;
            self.en_passant = undo.previous_en_passant;
            self.halfmove_clock = undo.previous_halfmove_clock;
            self.fullmove_number = undo.previous_fullmove_number;
            self.zobrist_key = undo.previous_zobrist_key;
            return Err(MoveError::HistoryOverflow);
        }

        debug_assert_eq!(self.zobrist_key, self.recompute_zobrist());
        debug_assert_eq!(self.repetition_history.current(), Some(self.zobrist_key));
        Ok(undo)
    }

    #[cfg(test)]
    pub(crate) fn unmake_null_move(&mut self, undo: NullMoveState) {
        self.side_to_move = self.side_to_move.opposite();
        self.en_passant = undo.previous_en_passant;
        self.halfmove_clock = undo.previous_halfmove_clock;
        self.fullmove_number = undo.previous_fullmove_number;
        self.zobrist_key = undo.previous_zobrist_key;
        self.repetition_history.pop();

        debug_assert_eq!(self.zobrist_key, self.recompute_zobrist());
        debug_assert_eq!(self.repetition_history.current(), Some(self.zobrist_key));
    }

    fn make_move_with_history(
        &mut self,
        mv: Move,
        history_mode: HistoryMode,
    ) -> Result<UndoState, MoveError> {
        let moving_color = self.side_to_move;
        let undo = self.make_move_unchecked(mv)?;
        if history_mode == HistoryMode::Persistent
            && self.repetition_history.push(self.zobrist_key).is_err()
        {
            self.unmake_move_internal(mv, undo, HistoryMode::Transient);
            return Err(MoveError::HistoryOverflow);
        }
        if self.is_in_check(moving_color) {
            self.unmake_move_internal(mv, undo, history_mode);
            return Err(MoveError::LeavesKingInCheck);
        }
        debug_assert_eq!(self.zobrist_key, self.recompute_zobrist());
        Ok(undo)
    }

    pub fn unmake_move(&mut self, mv: Move, undo: UndoState) {
        self.unmake_move_internal(mv, undo, HistoryMode::Persistent);
    }

    fn unmake_move_internal(&mut self, mv: Move, undo: UndoState, history_mode: HistoryMode) {
        let moving_color = self.side_to_move.opposite();
        let from = mv.from();
        let to = mv.to();

        if mv.is_castle() {
            self.remove_piece_unchecked(to)
                .expect("castled king must exist on destination");
            self.add_piece_unchecked(from, undo.moved_piece);
            let (rook_from, rook_to) = castle_rook_squares(to);
            let rook = self
                .remove_piece_unchecked(rook_to)
                .expect("castled rook must exist on intermediate square");
            self.add_piece_unchecked(rook_from, rook);
        } else {
            self.remove_piece_unchecked(to)
                .expect("moved piece must exist on destination during unmake");
            self.add_piece_unchecked(from, undo.moved_piece);

            if let Some(captured_piece) = undo.captured_piece {
                if mv.is_en_passant() {
                    let captured_square = match moving_color {
                        Color::White => to
                            .offset(0, -1)
                            .expect("white en passant capture square must exist"),
                        Color::Black => to
                            .offset(0, 1)
                            .expect("black en passant capture square must exist"),
                    };
                    self.add_piece_unchecked(captured_square, captured_piece);
                } else {
                    self.add_piece_unchecked(to, captured_piece);
                }
            }
        }

        self.side_to_move = moving_color;
        self.castling_rights = undo.previous_castling_rights;
        self.en_passant = undo.previous_en_passant;
        self.halfmove_clock = undo.previous_halfmove_clock;
        self.fullmove_number = undo.previous_fullmove_number;
        self.zobrist_key = undo.previous_zobrist_key;

        if history_mode == HistoryMode::Persistent {
            self.repetition_history.pop();
        }

        debug_assert_eq!(self.zobrist_key, self.recompute_zobrist());
        debug_assert_eq!(self.repetition_history.current(), Some(self.zobrist_key));
    }

    pub(crate) fn attackers_to(&self, square: Square, by_color: Color) -> u64 {
        self.attackers_to_with_occupancy(square, by_color, self.occupancies[OCCUPANCY_ALL], 0)
    }

    pub(crate) fn king_square(&self, color: Color) -> Square {
        self.king_squares[color.index()]
    }

    pub(crate) fn occupancy(&self) -> u64 {
        self.occupancies[OCCUPANCY_ALL]
    }

    pub(crate) fn occupancy_by(&self, color: Color) -> u64 {
        self.occupancies[color.index()]
    }

    pub(crate) fn check_info(&self) -> CheckInfo {
        let us = self.side_to_move;
        let them = us.opposite();
        let king_square = self.king_squares[us.index()];
        let checkers = self.attackers_to(king_square, them);
        let mut pinned = 0u64;

        let diagonal_sliders =
            self.pieces(them, PieceType::Bishop) | self.pieces(them, PieceType::Queen);
        let mut diagonal = diagonal_sliders;
        while let Some(slider_square) = pop_lsb(&mut diagonal) {
            if !is_diagonal_alignment(king_square, slider_square) {
                continue;
            }
            let between = attacks::between(king_square, slider_square);
            let blockers = between & self.occupancies[OCCUPANCY_ALL];
            if blockers.count_ones() == 1 && blockers & self.occupancies[us.index()] != 0 {
                pinned |= blockers;
            }
        }

        let orthogonal_sliders =
            self.pieces(them, PieceType::Rook) | self.pieces(them, PieceType::Queen);
        let mut orthogonal = orthogonal_sliders;
        while let Some(slider_square) = pop_lsb(&mut orthogonal) {
            if !is_orthogonal_alignment(king_square, slider_square) {
                continue;
            }
            let between = attacks::between(king_square, slider_square);
            let blockers = between & self.occupancies[OCCUPANCY_ALL];
            if blockers.count_ones() == 1 && blockers & self.occupancies[us.index()] != 0 {
                pinned |= blockers;
            }
        }

        let block_mask = if checkers.count_ones() == 1 {
            let checker_square = Square::from_index_unchecked(checkers.trailing_zeros() as u8);
            let checker_piece = self
                .piece_at(checker_square)
                .expect("attacker bitboard must point to an occupied square");
            match checker_piece.piece_type() {
                PieceType::Bishop | PieceType::Rook | PieceType::Queen => {
                    attacks::between(king_square, checker_square) | checker_square.bit()
                }
                PieceType::Pawn | PieceType::Knight | PieceType::King => checker_square.bit(),
            }
        } else if checkers == 0 {
            ALL_TARGETS
        } else {
            0
        };

        CheckInfo {
            checkers,
            pinned,
            block_mask,
        }
    }

    pub(crate) fn generate_pseudo_legal(&self, stage: MoveGenStage, moves: &mut MoveList) {
        moves.clear();
        match stage {
            MoveGenStage::All => {
                self.generate_non_king_moves(MoveGenStage::Captures, ALL_TARGETS, moves);
                self.generate_king_moves(MoveGenStage::Captures, ALL_TARGETS, moves);
                self.generate_non_king_moves(MoveGenStage::Quiets, ALL_TARGETS, moves);
                self.generate_king_moves(MoveGenStage::Quiets, ALL_TARGETS, moves);
            }
            MoveGenStage::Captures | MoveGenStage::Quiets => {
                self.generate_non_king_moves(stage, ALL_TARGETS, moves);
                self.generate_king_moves(stage, ALL_TARGETS, moves);
            }
            MoveGenStage::Evasions => {}
        }
    }

    pub(crate) fn generate_evasions(&self, info: &CheckInfo, moves: &mut MoveList) {
        let _stage = MoveGenStage::Evasions;
        moves.clear();
        self.generate_king_moves(MoveGenStage::Captures, ALL_TARGETS, moves);
        self.generate_king_moves(MoveGenStage::Quiets, ALL_TARGETS, moves);

        if info.checkers.count_ones() >= 2 {
            return;
        }

        self.generate_non_king_moves(MoveGenStage::Captures, info.block_mask, moves);
        self.generate_non_king_moves(MoveGenStage::Quiets, info.block_mask, moves);
    }

    pub(crate) fn is_legal_fast(&mut self, mv: Move, info: &CheckInfo) -> bool {
        let moving_piece = match self.piece_at(mv.from()) {
            Some(piece) => piece,
            None => return false,
        };

        if mv.is_en_passant() {
            return self.validate_with_make_unmake(mv);
        }

        if moving_piece.piece_type() == PieceType::King {
            if mv.is_castle() {
                return true;
            }
            return self.is_king_move_legal(mv);
        }

        if info.checkers != 0 && mv.to().bit() & info.block_mask == 0 {
            return false;
        }

        if info.pinned & mv.from().bit() != 0 {
            let king_square = self.king_squares[self.side_to_move.index()];
            if attacks::line(king_square, mv.from()) & mv.to().bit() == 0 {
                return false;
            }
        }

        true
    }

    fn generate_non_king_moves(&self, stage: MoveGenStage, target_mask: u64, moves: &mut MoveList) {
        let color = self.side_to_move;

        let mut pawns = self.pieces(color, PieceType::Pawn);
        while let Some(from) = pop_lsb(&mut pawns) {
            self.generate_pawn_moves(from, color, stage, target_mask, moves);
        }

        let mut knights = self.pieces(color, PieceType::Knight);
        while let Some(from) = pop_lsb(&mut knights) {
            self.generate_jump_moves(from, color, PieceType::Knight, stage, target_mask, moves);
        }

        let mut bishops = self.pieces(color, PieceType::Bishop);
        while let Some(from) = pop_lsb(&mut bishops) {
            self.generate_slider_moves(from, color, PieceType::Bishop, stage, target_mask, moves);
        }

        let mut rooks = self.pieces(color, PieceType::Rook);
        while let Some(from) = pop_lsb(&mut rooks) {
            self.generate_slider_moves(from, color, PieceType::Rook, stage, target_mask, moves);
        }

        let mut queens = self.pieces(color, PieceType::Queen);
        while let Some(from) = pop_lsb(&mut queens) {
            self.generate_slider_moves(from, color, PieceType::Queen, stage, target_mask, moves);
        }
    }

    fn generate_king_moves(&self, stage: MoveGenStage, target_mask: u64, moves: &mut MoveList) {
        let color = self.side_to_move;
        let from = self.king_squares[color.index()];
        let occupied = self.occupancies[OCCUPANCY_ALL];
        let friendly = self.occupancies[color.index()];
        let enemy = self.occupancies[color.opposite().index()];
        let attacks = attacks::king_attacks(from);

        match stage {
            MoveGenStage::Captures => {
                let mut targets = attacks & enemy & target_mask;
                while let Some(to) = pop_lsb(&mut targets) {
                    moves.push(Move::new(from, to).with_flags(FLAG_CAPTURE));
                }
            }
            MoveGenStage::Quiets => {
                let mut targets = attacks & !occupied & target_mask;
                while let Some(to) = pop_lsb(&mut targets) {
                    moves.push(Move::new(from, to));
                }
                self.generate_castling_moves(color, moves);
            }
            MoveGenStage::All | MoveGenStage::Evasions => {
                debug_assert!(
                    false,
                    "king move generation expects captures or quiets stage"
                );
            }
        }

        let _ = friendly;
    }

    fn generate_pawn_moves(
        &self,
        from: Square,
        color: Color,
        stage: MoveGenStage,
        target_mask: u64,
        moves: &mut MoveList,
    ) {
        let occupied = self.occupancies[OCCUPANCY_ALL];
        let enemy = self.occupancies[color.opposite().index()];
        let promotion_rank = match color {
            Color::White => 7,
            Color::Black => 0,
        };

        match stage {
            MoveGenStage::Captures => {
                let mut captures = attacks::pawn_attacks(color, from) & enemy & target_mask;
                while let Some(to) = pop_lsb(&mut captures) {
                    if to.rank() == promotion_rank {
                        for promotion_piece in PieceType::promotion_pieces() {
                            moves.push(
                                Move::new(from, to)
                                    .with_flags(FLAG_CAPTURE)
                                    .with_promotion(promotion_piece),
                            );
                        }
                    } else {
                        moves.push(Move::new(from, to).with_flags(FLAG_CAPTURE));
                    }
                }

                if let Some(en_passant) = self.en_passant
                    && attacks::pawn_attacks(color, from) & en_passant.bit() != 0
                {
                    moves.push(
                        Move::new(from, en_passant).with_flags(FLAG_CAPTURE | FLAG_EN_PASSANT),
                    );
                }
            }
            MoveGenStage::Quiets => {
                if let Some(one_step) = from.offset(0, color.pawn_direction())
                    && occupied & one_step.bit() == 0
                {
                    if target_mask & one_step.bit() != 0 {
                        if one_step.rank() == promotion_rank {
                            for promotion_piece in PieceType::promotion_pieces() {
                                moves.push(
                                    Move::new(from, one_step).with_promotion(promotion_piece),
                                );
                            }
                        } else {
                            moves.push(Move::new(from, one_step));
                        }
                    }

                    if one_step.rank() != promotion_rank
                        && from.rank() == color.pawn_start_rank()
                        && let Some(two_step) = from.offset(0, color.pawn_direction() * 2)
                        && occupied & two_step.bit() == 0
                        && target_mask & two_step.bit() != 0
                    {
                        moves.push(Move::new(from, two_step).with_flags(FLAG_DOUBLE_PAWN_PUSH));
                    }
                }
            }
            MoveGenStage::All | MoveGenStage::Evasions => {
                debug_assert!(
                    false,
                    "pawn move generation expects captures or quiets stage"
                );
            }
        }
    }

    fn generate_jump_moves(
        &self,
        from: Square,
        color: Color,
        piece_type: PieceType,
        stage: MoveGenStage,
        target_mask: u64,
        moves: &mut MoveList,
    ) {
        let attack_mask = match piece_type {
            PieceType::Knight => attacks::knight_attacks(from),
            PieceType::King => attacks::king_attacks(from),
            _ => unreachable!("jump move generation only supports knights and kings"),
        };

        let occupied = self.occupancies[OCCUPANCY_ALL];
        let enemy = self.occupancies[color.opposite().index()];
        let targets = match stage {
            MoveGenStage::Captures => attack_mask & enemy & target_mask,
            MoveGenStage::Quiets => attack_mask & !occupied & target_mask,
            MoveGenStage::All | MoveGenStage::Evasions => {
                debug_assert!(
                    false,
                    "jump move generation expects captures or quiets stage"
                );
                0
            }
        };

        let mut targets = targets;
        while let Some(to) = pop_lsb(&mut targets) {
            let mut mv = Move::new(from, to);
            if stage == MoveGenStage::Captures {
                mv = mv.with_flags(FLAG_CAPTURE);
            }
            moves.push(mv);
        }
    }

    fn generate_slider_moves(
        &self,
        from: Square,
        color: Color,
        piece_type: PieceType,
        stage: MoveGenStage,
        target_mask: u64,
        moves: &mut MoveList,
    ) {
        let occupied = self.occupancies[OCCUPANCY_ALL];
        let enemy = self.occupancies[color.opposite().index()];
        let attack_mask = match piece_type {
            PieceType::Bishop => attacks::bishop_attacks(from, occupied),
            PieceType::Rook => attacks::rook_attacks(from, occupied),
            PieceType::Queen => attacks::queen_attacks(from, occupied),
            _ => unreachable!("slider move generation only supports bishops, rooks, and queens"),
        };

        let targets = match stage {
            MoveGenStage::Captures => attack_mask & enemy & target_mask,
            MoveGenStage::Quiets => attack_mask & !occupied & target_mask,
            MoveGenStage::All | MoveGenStage::Evasions => {
                debug_assert!(
                    false,
                    "slider move generation expects captures or quiets stage"
                );
                0
            }
        };

        let mut targets = targets;
        while let Some(to) = pop_lsb(&mut targets) {
            let mut mv = Move::new(from, to);
            if stage == MoveGenStage::Captures {
                mv = mv.with_flags(FLAG_CAPTURE);
            }
            moves.push(mv);
        }
    }

    fn generate_castling_moves(&self, color: Color, moves: &mut MoveList) {
        let home_rank = match color {
            Color::White => 0,
            Color::Black => 7,
        };
        let king_start = Square::from_coords(4, home_rank).expect("home king square must exist");
        if self.piece_at(king_start) != Some(Piece::from_parts(color, PieceType::King)) {
            return;
        }
        if self.is_in_check(color) {
            return;
        }

        if self.castling_rights.has_kingside(color) {
            let rook_square =
                Square::from_coords(7, home_rank).expect("home rook square must exist");
            let f_square = Square::from_coords(5, home_rank).expect("home f square must exist");
            let g_square = Square::from_coords(6, home_rank).expect("home g square must exist");
            if self.piece_at(rook_square) == Some(Piece::from_parts(color, PieceType::Rook))
                && self.piece_at(f_square).is_none()
                && self.piece_at(g_square).is_none()
                && !self.is_square_attacked(f_square, color.opposite())
                && !self.is_square_attacked(g_square, color.opposite())
            {
                moves.push(Move::new(king_start, g_square).with_flags(FLAG_CASTLE));
            }
        }

        if self.castling_rights.has_queenside(color) {
            let rook_square =
                Square::from_coords(0, home_rank).expect("home rook square must exist");
            let b_square = Square::from_coords(1, home_rank).expect("home b square must exist");
            let c_square = Square::from_coords(2, home_rank).expect("home c square must exist");
            let d_square = Square::from_coords(3, home_rank).expect("home d square must exist");
            if self.piece_at(rook_square) == Some(Piece::from_parts(color, PieceType::Rook))
                && self.piece_at(b_square).is_none()
                && self.piece_at(c_square).is_none()
                && self.piece_at(d_square).is_none()
                && !self.is_square_attacked(c_square, color.opposite())
                && !self.is_square_attacked(d_square, color.opposite())
            {
                moves.push(Move::new(king_start, c_square).with_flags(FLAG_CASTLE));
            }
        }
    }

    fn make_move_unchecked(&mut self, mv: Move) -> Result<UndoState, MoveError> {
        let from = mv.from();
        let to = mv.to();
        let moving_piece = self.piece_at(from).ok_or(MoveError::NoPieceOnSource)?;
        let moving_color = self.side_to_move;
        if moving_piece.color() != moving_color {
            return Err(MoveError::WrongSideToMove);
        }

        let promotion_piece = mv.promotion();
        if let Some(piece_type) = promotion_piece {
            let _ = piece_type;
            if moving_piece.piece_type() != PieceType::Pawn || (to.rank() != 0 && to.rank() != 7) {
                return Err(MoveError::InvalidPromotion);
            }
        }

        let castle_rook = if mv.is_castle() {
            if moving_piece.piece_type() != PieceType::King {
                return Err(MoveError::IllegalCastle);
            }
            let (rook_from, rook_to) = castle_rook_squares(to);
            let rook = self.piece_at(rook_from).ok_or(MoveError::IllegalCastle)?;
            if rook != Piece::from_parts(moving_color, PieceType::Rook) {
                return Err(MoveError::IllegalCastle);
            }
            Some((rook_from, rook_to, rook))
        } else {
            None
        };

        if !mv.is_castle()
            && let Some(target_piece) = self.piece_at(to)
            && target_piece.color() == moving_color
        {
            return Err(MoveError::DestinationOccupiedByOwnPiece);
        }

        let mut undo = UndoState {
            captured_piece: None,
            moved_piece: moving_piece,
            previous_castling_rights: self.castling_rights,
            previous_en_passant: self.en_passant,
            previous_halfmove_clock: self.halfmove_clock,
            previous_fullmove_number: self.fullmove_number,
            previous_zobrist_key: self.zobrist_key,
        };

        self.set_en_passant_hashed(None);
        self.halfmove_clock = self.halfmove_clock.saturating_add(1);
        if moving_color == Color::Black {
            self.fullmove_number = self.fullmove_number.saturating_add(1);
        }
        if moving_piece.piece_type() == PieceType::Pawn {
            self.halfmove_clock = 0;
        }

        let captured_piece = if mv.is_en_passant() {
            if moving_piece.piece_type() != PieceType::Pawn || Some(to) != undo.previous_en_passant
            {
                return Err(MoveError::IllegalEnPassant);
            }
            let captured_square = match moving_color {
                Color::White => to
                    .offset(0, -1)
                    .expect("white en passant capture square must exist"),
                Color::Black => to
                    .offset(0, 1)
                    .expect("black en passant capture square must exist"),
            };
            match self.remove_piece_unchecked(captured_square) {
                Some(piece)
                    if piece == Piece::from_parts(moving_color.opposite(), PieceType::Pawn) =>
                {
                    Some(piece)
                }
                Some(piece) => {
                    self.add_piece_unchecked(captured_square, piece);
                    return Err(MoveError::IllegalEnPassant);
                }
                None => return Err(MoveError::IllegalEnPassant),
            }
        } else {
            self.remove_piece_unchecked(to)
        };

        if captured_piece.is_some() {
            self.halfmove_clock = 0;
        }

        undo.captured_piece = captured_piece;
        self.update_castling_rights(from, to, moving_piece, captured_piece);

        if let Some((rook_from, rook_to, rook)) = castle_rook {
            self.remove_piece_unchecked(rook_from)
                .expect("castle rook must still exist on source square");
            self.remove_piece_unchecked(from)
                .expect("moving king must exist on source square");
            self.add_piece_unchecked(to, moving_piece);
            self.add_piece_unchecked(rook_to, rook);
        } else {
            self.remove_piece_unchecked(from)
                .expect("moving piece must exist on source square");
            let placed_piece = if let Some(promotion_piece) = promotion_piece {
                Piece::from_parts(moving_color, promotion_piece)
            } else {
                moving_piece
            };
            self.add_piece_unchecked(to, placed_piece);
            if moving_piece.piece_type() == PieceType::Pawn && mv.is_double_pawn_push() {
                let en_passant_square = match moving_color {
                    Color::White => from
                        .offset(0, 1)
                        .expect("white en passant square must exist"),
                    Color::Black => from
                        .offset(0, -1)
                        .expect("black en passant square must exist"),
                };
                self.set_en_passant_hashed(Some(en_passant_square));
            }
        }

        self.set_side_to_move_hashed(moving_color.opposite());
        debug_assert_eq!(self.zobrist_key, self.recompute_zobrist());
        Ok(undo)
    }

    fn update_castling_rights(
        &mut self,
        from: Square,
        to: Square,
        moving_piece: Piece,
        captured_piece: Option<Piece>,
    ) {
        let mut rights = self.castling_rights;
        if moving_piece.piece_type() == PieceType::King {
            rights.remove_color(moving_piece.color());
        }

        if moving_piece.piece_type() == PieceType::Rook {
            clear_castling_right_for_square(&mut rights, from);
        }

        if let Some(captured_piece) = captured_piece
            && captured_piece.piece_type() == PieceType::Rook
        {
            clear_castling_right_for_square(&mut rights, to);
        }

        self.set_castling_rights_hashed(rights);
    }

    fn add_piece_unchecked(&mut self, square: Square, piece: Piece) {
        let color_index = piece.color().index();
        let piece_index = piece.piece_type().index();

        self.board[square.index()] = Some(piece);
        self.zobrist_key ^= zobrist::piece_square(piece, square);
        self.piece_bitboards[color_index][piece_index] |= square.bit();
        self.occupancies[color_index] |= square.bit();
        self.occupancies[OCCUPANCY_ALL] |= square.bit();
        self.piece_lists[color_index][piece_index].push(square);

        if piece.piece_type() == PieceType::King {
            self.king_squares[color_index] = square;
        }
    }

    fn remove_piece_unchecked(&mut self, square: Square) -> Option<Piece> {
        let piece = self.board[square.index()]?;
        let color_index = piece.color().index();
        let piece_index = piece.piece_type().index();

        self.board[square.index()] = None;
        self.zobrist_key ^= zobrist::piece_square(piece, square);
        self.piece_bitboards[color_index][piece_index] &= !square.bit();
        self.occupancies[color_index] &= !square.bit();
        self.occupancies[OCCUPANCY_ALL] &= !square.bit();
        let removed = self.piece_lists[color_index][piece_index].remove(square);
        debug_assert!(removed, "piece list entry must exist for removed piece");

        Some(piece)
    }

    pub(crate) fn pieces(&self, color: Color, piece_type: PieceType) -> u64 {
        self.piece_bitboards[color.index()][piece_type.index()]
    }

    fn attackers_to_with_occupancy(
        &self,
        square: Square,
        by_color: Color,
        occupied: u64,
        ignored_squares: u64,
    ) -> u64 {
        let enemy_pawns = self.pieces(by_color, PieceType::Pawn) & !ignored_squares;
        let enemy_knights = self.pieces(by_color, PieceType::Knight) & !ignored_squares;
        let enemy_bishops = self.pieces(by_color, PieceType::Bishop) & !ignored_squares;
        let enemy_rooks = self.pieces(by_color, PieceType::Rook) & !ignored_squares;
        let enemy_queens = self.pieces(by_color, PieceType::Queen) & !ignored_squares;
        let enemy_king = self.pieces(by_color, PieceType::King) & !ignored_squares;

        (attacks::pawn_attackers_to(square, by_color) & enemy_pawns)
            | (attacks::knight_attacks(square) & enemy_knights)
            | (attacks::bishop_attacks(square, occupied) & (enemy_bishops | enemy_queens))
            | (attacks::rook_attacks(square, occupied) & (enemy_rooks | enemy_queens))
            | (attacks::king_attacks(square) & enemy_king)
    }

    fn see_internal(&self, mv: Move) -> Value {
        let from = mv.from();
        let to = mv.to();
        let moving_piece = match self.piece_at(from) {
            Some(piece) => piece,
            None => return Value(0),
        };

        if !mv.is_capture() {
            return mv.promotion().map(see::promotion_gain).unwrap_or_default();
        }

        let us = self.side_to_move;
        let them = us.opposite();
        let promotion_piece = mv.promotion();
        let promotion_gain = promotion_piece.map_or(0, |piece| see::promotion_gain(piece).0);

        let captured_piece = if mv.is_en_passant() {
            Some(Piece::from_parts(them, PieceType::Pawn))
        } else {
            self.piece_at(to)
        };
        let Some(captured_piece) = captured_piece else {
            return Value(0);
        };

        let mut piece_bitboards = self.piece_bitboards;
        let mut occupancies = self.occupancies;
        let mut gains = [0i16; 32];
        gains[0] = see::piece_value(captured_piece.piece_type()).0 + promotion_gain;

        remove_local_piece(&mut piece_bitboards, &mut occupancies, from, moving_piece);
        if mv.is_en_passant() {
            let captured_square = match us {
                Color::White => to
                    .offset(0, -1)
                    .expect("white en passant capture square must exist"),
                Color::Black => to
                    .offset(0, 1)
                    .expect("black en passant capture square must exist"),
            };
            remove_local_piece(
                &mut piece_bitboards,
                &mut occupancies,
                captured_square,
                captured_piece,
            );
        } else {
            remove_local_piece(&mut piece_bitboards, &mut occupancies, to, captured_piece);
        }

        let mut occupying_piece_type = promotion_piece.unwrap_or(moving_piece.piece_type());
        add_local_piece(
            &mut piece_bitboards,
            &mut occupancies,
            to,
            Piece::from_parts(us, occupying_piece_type),
        );

        let mut side = them;
        let mut depth = 0usize;
        while let Some((attacker_square, attacker_piece_type)) =
            least_valuable_attacker(to, side, &piece_bitboards, occupancies[OCCUPANCY_ALL])
        {
            depth += 1;
            gains[depth] = see::piece_value(occupying_piece_type).0 - gains[depth - 1];

            let attacker_piece = Piece::from_parts(side, attacker_piece_type);
            remove_local_piece(
                &mut piece_bitboards,
                &mut occupancies,
                attacker_square,
                attacker_piece,
            );

            occupying_piece_type = if attacker_piece_type == PieceType::Pawn
                && to.rank()
                    == match side {
                        Color::White => 7,
                        Color::Black => 0,
                    } {
                PieceType::Queen
            } else {
                attacker_piece_type
            };

            add_local_piece(
                &mut piece_bitboards,
                &mut occupancies,
                to,
                Piece::from_parts(side, occupying_piece_type),
            );
            side = side.opposite();
        }

        while depth > 0 {
            gains[depth - 1] = -std::cmp::max(-gains[depth - 1], gains[depth]);
            depth -= 1;
        }

        Value(gains[0])
    }

    fn is_king_move_legal(&self, mv: Move) -> bool {
        let us = self.side_to_move;
        let them = us.opposite();
        let from = mv.from();
        let to = mv.to();
        let occupied = self.occupancies[OCCUPANCY_ALL];
        let ignored_squares = if let Some(captured_piece) = self.piece_at(to) {
            if captured_piece.color() == them {
                to.bit()
            } else {
                0
            }
        } else {
            0
        };
        let occupied_after = (occupied & !from.bit() & !ignored_squares) | to.bit();

        self.attackers_to_with_occupancy(to, them, occupied_after, ignored_squares) == 0
    }

    fn validate_with_make_unmake(&mut self, mv: Move) -> bool {
        match self.make_move_with_history(mv, HistoryMode::Transient) {
            Ok(undo) => {
                self.unmake_move_internal(mv, undo, HistoryMode::Transient);
                true
            }
            Err(_) => false,
        }
    }
}

fn remove_local_piece(
    piece_bitboards: &mut [[u64; 6]; 2],
    occupancies: &mut [u64; 3],
    square: Square,
    piece: Piece,
) {
    let color_index = piece.color().index();
    let piece_index = piece.piece_type().index();
    piece_bitboards[color_index][piece_index] &= !square.bit();
    occupancies[color_index] &= !square.bit();
    occupancies[OCCUPANCY_ALL] &= !square.bit();
}

fn add_local_piece(
    piece_bitboards: &mut [[u64; 6]; 2],
    occupancies: &mut [u64; 3],
    square: Square,
    piece: Piece,
) {
    let color_index = piece.color().index();
    let piece_index = piece.piece_type().index();
    piece_bitboards[color_index][piece_index] |= square.bit();
    occupancies[color_index] |= square.bit();
    occupancies[OCCUPANCY_ALL] |= square.bit();
}

fn least_valuable_attacker(
    square: Square,
    color: Color,
    piece_bitboards: &[[u64; 6]; 2],
    occupied: u64,
) -> Option<(Square, PieceType)> {
    let color_index = color.index();
    let candidates = [
        (
            PieceType::Pawn,
            attacks::pawn_attackers_to(square, color)
                & piece_bitboards[color_index][PieceType::Pawn.index()],
        ),
        (
            PieceType::Knight,
            attacks::knight_attacks(square)
                & piece_bitboards[color_index][PieceType::Knight.index()],
        ),
        (
            PieceType::Bishop,
            attacks::bishop_attacks(square, occupied)
                & piece_bitboards[color_index][PieceType::Bishop.index()],
        ),
        (
            PieceType::Rook,
            attacks::rook_attacks(square, occupied)
                & piece_bitboards[color_index][PieceType::Rook.index()],
        ),
        (
            PieceType::Queen,
            attacks::queen_attacks(square, occupied)
                & piece_bitboards[color_index][PieceType::Queen.index()],
        ),
    ];

    for (piece_type, attackers) in candidates {
        if attackers != 0 {
            let attacker_square = Square::from_index_unchecked(attackers.trailing_zeros() as u8);
            return Some((attacker_square, piece_type));
        }
    }

    None
}

#[cfg(test)]
impl Position {
    fn generate_legal_moves_slow(&mut self, moves: &mut MoveList) {
        moves.clear();
        let mut pseudo_legal = MoveList::new();
        self.generate_pseudo_legal(MoveGenStage::All, &mut pseudo_legal);

        for mv in pseudo_legal.as_slice().iter().copied() {
            if let Ok(undo) = self.make_move(mv) {
                moves.push(mv);
                self.unmake_move(mv, undo);
            }
        }
    }
}

fn pop_lsb(bitboard: &mut u64) -> Option<Square> {
    if *bitboard == 0 {
        return None;
    }

    let square = Square::from_index_unchecked(bitboard.trailing_zeros() as u8);
    *bitboard &= *bitboard - 1;
    Some(square)
}

fn mix_u64(mut value: u64) -> u64 {
    value ^= value >> 33;
    value = value.wrapping_mul(0xff51_afd7_ed55_8ccd);
    value ^= value >> 33;
    value = value.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    value ^= value >> 33;
    value
}

fn clear_castling_right_for_square(rights: &mut CastlingRights, square: Square) {
    match square {
        Square::A1 => rights.remove(CastlingRights::WHITE_QUEENSIDE),
        Square::H1 => rights.remove(CastlingRights::WHITE_KINGSIDE),
        Square::A8 => rights.remove(CastlingRights::BLACK_QUEENSIDE),
        Square::H8 => rights.remove(CastlingRights::BLACK_KINGSIDE),
        _ => {}
    }
}

fn is_diagonal_alignment(from: Square, to: Square) -> bool {
    from.file().abs_diff(to.file()) == from.rank().abs_diff(to.rank())
}

fn is_orthogonal_alignment(from: Square, to: Square) -> bool {
    from.file() == to.file() || from.rank() == to.rank()
}

fn castle_rook_squares(king_destination: Square) -> (Square, Square) {
    match king_destination {
        Square::G1 => (Square::H1, Square::F1),
        Square::C1 => (Square::A1, Square::D1),
        Square::G8 => (Square::H8, Square::F8),
        Square::C8 => (Square::A8, Square::D8),
        _ => panic!("invalid castling destination"),
    }
}

#[cfg(test)]
mod tests {
    use super::{MoveList, Position, PositionStatus};
    use crate::core::{MoveError, ParsedMove, STARTPOS_FEN, repetition::MAX_REPETITION_HISTORY};

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct PersistentSnapshot {
        side_to_move: super::Color,
        castling_rights: super::CastlingRights,
        en_passant: Option<super::Square>,
        halfmove_clock: u16,
        fullmove_number: u16,
        zobrist_key: u64,
        repetition_history: Vec<u64>,
        fen: String,
    }

    fn snapshot(position: &Position) -> PersistentSnapshot {
        PersistentSnapshot {
            side_to_move: position.side_to_move,
            castling_rights: position.castling_rights,
            en_passant: position.en_passant,
            halfmove_clock: position.halfmove_clock,
            fullmove_number: position.fullmove_number,
            zobrist_key: position.zobrist_key,
            repetition_history: position.repetition_history.as_slice().to_vec(),
            fen: position.to_fen(),
        }
    }

    fn legal_move(position: &mut Position, uci: &str) -> super::Move {
        let parsed = ParsedMove::parse(uci).expect("UCI move parse must succeed");
        let mut legal_moves = MoveList::new();
        position.generate_legal_moves(&mut legal_moves);
        legal_moves
            .as_slice()
            .iter()
            .copied()
            .find(|mv| mv.matches_parsed(parsed))
            .unwrap_or_else(|| panic!("expected legal move {uci}"))
    }

    fn next_test_u64(seed: &mut u64) -> u64 {
        *seed ^= *seed >> 12;
        *seed ^= *seed << 25;
        *seed ^= *seed >> 27;
        *seed = (*seed).wrapping_mul(0x2545_f491_4f6c_dd1d);
        *seed
    }

    fn fill_history_to_capacity(position: &mut Position) {
        position
            .repetition_history
            .clear_and_seed(position.zobrist_key);
        while position.repetition_history.len() < MAX_REPETITION_HISTORY {
            position
                .repetition_history
                .push(position.zobrist_key)
                .expect("test setup must be able to fill history to capacity");
        }
    }

    fn assert_fast_matches_slow(position: &mut Position, label: &str) {
        let mut fast = MoveList::new();
        let mut slow = MoveList::new();
        position.generate_legal_moves(&mut fast);
        position.generate_legal_moves_slow(&mut slow);

        let fast_moves: Vec<String> = fast.as_slice().iter().map(|mv| mv.to_string()).collect();
        let slow_moves: Vec<String> = slow.as_slice().iter().map(|mv| mv.to_string()).collect();
        assert_eq!(fast_moves, slow_moves, "{label}");
    }

    fn assert_fast_matches_slow_recursive(
        position: &mut Position,
        remaining_depth: u8,
        path: &mut Vec<String>,
    ) {
        let label = if path.is_empty() {
            "root".to_owned()
        } else {
            path.join(" ")
        };
        assert_fast_matches_slow(position, &label);
        if remaining_depth == 0 {
            return;
        }

        let mut moves = MoveList::new();
        position.generate_legal_moves(&mut moves);
        for mv in moves.as_slice().iter().copied() {
            let undo = position
                .make_move(mv)
                .expect("recursive move must be legal");
            path.push(mv.to_string());
            assert_fast_matches_slow_recursive(position, remaining_depth - 1, path);
            path.pop();
            position.unmake_move(mv, undo);
        }
    }

    #[test]
    fn fast_and_slow_legal_generation_match_startpos_two_plies() {
        let mut root = Position::startpos();
        assert_fast_matches_slow(&mut root, "startpos root");

        let mut root_moves = MoveList::new();
        root.generate_legal_moves(&mut root_moves);
        for mv in root_moves.as_slice().iter().copied() {
            let undo = root.make_move(mv).expect("root move must be legal");
            assert_fast_matches_slow(&mut root, &format!("startpos after {mv}"));
            root.unmake_move(mv, undo);
        }
    }

    #[test]
    fn fast_and_slow_legal_generation_match_kiwipete_after_e1c1() {
        let mut position = Position::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .expect("FEN parse must succeed");
        position.apply_uci_move("e1c1").expect("move must be legal");

        assert_fast_matches_slow(&mut position, "kiwipete after e1c1");

        let mut black_moves = MoveList::new();
        position.generate_legal_moves(&mut black_moves);
        for mv in black_moves.as_slice().iter().copied() {
            let undo = position.make_move(mv).expect("reply must be legal");
            assert_fast_matches_slow(&mut position, &format!("kiwipete after e1c1 {mv}"));
            position.unmake_move(mv, undo);
        }
    }

    #[test]
    fn fast_and_slow_legal_generation_match_startpos_three_plies_deep() {
        let mut position = Position::startpos();
        let mut path = Vec::new();
        assert_fast_matches_slow_recursive(&mut position, 3, &mut path);
    }

    #[test]
    fn fast_and_slow_legal_generation_match_position6_root() {
        let mut position = Position::from_fen(
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        )
        .expect("FEN parse must succeed");

        assert_fast_matches_slow(&mut position, "position6 root");
    }

    #[test]
    fn make_unmake_restores_persistent_state_across_move_classes() {
        let cases = [
            (STARTPOS_FEN, "e2e4"),
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
            ("4k3/8/8/8/2p5/3B4/8/4K3 w - - 0 1", "d3c4"),
            ("7k/P7/8/8/8/8/8/K7 w - - 0 1", "a7a8q"),
            ("4k2r/6P1/8/8/8/8/8/4K3 w - - 0 1", "g7h8q"),
        ];

        for (fen, uci) in cases {
            let mut position = Position::from_fen(fen).expect("FEN parse must succeed");
            let before = snapshot(&position);
            let mv = legal_move(&mut position, uci);
            let undo = position.make_move(mv).expect("move must be legal");
            position.unmake_move(mv, undo);
            assert_eq!(snapshot(&position), before, "round-trip must restore {uci}");
        }
    }

    #[test]
    fn null_move_round_trip_restores_persistent_state() {
        let mut position = Position::from_fen("r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 7 12")
            .expect("FEN parse must succeed");
        let before = snapshot(&position);

        let undo = position.make_null_move().expect("null move must succeed");
        assert_eq!(position.side_to_move(), super::Color::Black);
        assert_eq!(position.halfmove_clock(), before.halfmove_clock + 1);
        assert_eq!(
            position.to_fen(),
            "r3k2r/8/8/3pP3/8/8/8/R3K2R b KQkq - 8 12"
        );

        position.unmake_null_move(undo);
        assert_eq!(snapshot(&position), before);
    }

    #[test]
    fn generate_legal_moves_leaves_persistent_state_unchanged() {
        let mut position = Position::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
            .expect("FEN parse must succeed");
        let before = snapshot(&position);

        let mut moves = MoveList::new();
        position.generate_legal_moves(&mut moves);

        assert_eq!(snapshot(&position), before);
    }

    #[test]
    fn randomized_make_unmake_sequences_restore_root_state() {
        let mut position = Position::startpos();
        let root = snapshot(&position);
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut played = Vec::new();

        for _ in 0..256 {
            let mut moves = MoveList::new();
            position.generate_legal_moves(&mut moves);
            if moves.is_empty() {
                break;
            }
            let index = next_test_u64(&mut seed) as usize % moves.len();
            let mv = moves.as_slice()[index];
            let undo = position.make_move(mv).expect("selected move must be legal");
            assert_eq!(position.zobrist_key, position.recompute_zobrist());
            played.push((mv, undo));
        }

        while let Some((mv, undo)) = played.pop() {
            position.unmake_move(mv, undo);
        }

        assert_eq!(snapshot(&position), root);
    }

    #[test]
    fn repetition_and_status_helpers_work_on_actual_cycles() {
        let mut position = Position::startpos();
        for mv in [
            "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8",
        ] {
            position
                .apply_uci_move(mv)
                .expect("cycle move must be legal");
        }

        assert!(position.is_draw_by_repetition());
        assert_eq!(position.status(), PositionStatus::DrawByRepetition);
    }

    #[test]
    fn make_unmake_keeps_zobrist_and_history_in_sync_with_validate() {
        let mut position = Position::startpos();
        let mut moves = MoveList::new();
        position.generate_legal_moves(&mut moves);
        for mv in moves.as_slice().iter().copied() {
            let before = snapshot(&position);
            let undo = position.make_move(mv).expect("move must be legal");
            position.validate().expect("post-move state must validate");
            position.unmake_move(mv, undo);
            position
                .validate()
                .expect("post-unmake state must validate");
            assert_eq!(snapshot(&position), before);
        }
    }

    #[test]
    fn persistent_move_application_returns_history_overflow_at_capacity() {
        let mut position = Position::startpos();
        fill_history_to_capacity(&mut position);
        let before = snapshot(&position);

        let mv = legal_move(&mut position, "e2e4");
        let error = position
            .make_move(mv)
            .expect_err("persistent move must fail when history is full");

        assert_eq!(error, MoveError::HistoryOverflow);
        assert_eq!(snapshot(&position), before);
    }

    #[test]
    fn temporary_legality_validation_does_not_consume_history_capacity() {
        let mut position = Position::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
            .expect("FEN parse must succeed");
        fill_history_to_capacity(&mut position);
        let before = snapshot(&position);

        let mut moves = MoveList::new();
        position.generate_legal_moves(&mut moves);

        assert!(
            moves.as_slice().iter().any(|mv| mv.to_string() == "e5d6"),
            "en passant legality validation should still succeed at capacity"
        );
        assert_eq!(snapshot(&position), before);
    }
}
