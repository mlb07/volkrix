use std::{error::Error, fmt};

use super::{
    chess_move::{
        FLAG_CAPTURE, FLAG_CASTLE, FLAG_DOUBLE_PAWN_PUSH, FLAG_EN_PASSANT, Move, ParsedMove,
    },
    movelist::MoveList,
    piece::{Piece, PieceType},
    square::Square,
    types::{CastlingRights, Color},
};

const PIECE_LIST_CAPACITY: usize = 16;
const OCCUPANCY_WHITE: usize = 0;
const OCCUPANCY_BLACK: usize = 1;
const OCCUPANCY_ALL: usize = 2;
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
const BISHOP_DIRECTIONS: [(i8, i8); 4] = [(-1, -1), (-1, 1), (1, -1), (1, 1)];
const ROOK_DIRECTIONS: [(i8, i8); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

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
        let mut index = 0;
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
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UndoState {
    pub captured_piece: Option<Piece>,
    pub moved_piece: Piece,
    pub previous_castling_rights: CastlingRights,
    pub previous_en_passant: Option<Square>,
    pub previous_halfmove_clock: u16,
    pub previous_fullmove_number: u16,
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
            Self::IllegalMove(value) => write!(f, "illegal move: {value}"),
        }
    }
}

impl Error for MoveError {}

impl Position {
    pub fn empty() -> Self {
        Self {
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
        }
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

        Ok(())
    }

    pub fn is_square_attacked(&self, square: Square, by_color: Color) -> bool {
        let pawn_attack_offsets = match by_color {
            Color::White => [(-1, -1), (1, -1)],
            Color::Black => [(-1, 1), (1, 1)],
        };

        for (file_delta, rank_delta) in pawn_attack_offsets {
            if let Some(attacker_square) = square.offset(file_delta, rank_delta)
                && self.piece_at(attacker_square)
                    == Some(Piece::from_parts(by_color, PieceType::Pawn))
            {
                return true;
            }
        }

        for (file_delta, rank_delta) in KNIGHT_OFFSETS {
            if let Some(attacker_square) = square.offset(file_delta, rank_delta)
                && self.piece_at(attacker_square)
                    == Some(Piece::from_parts(by_color, PieceType::Knight))
            {
                return true;
            }
        }

        for (file_delta, rank_delta) in KING_OFFSETS {
            if let Some(attacker_square) = square.offset(file_delta, rank_delta)
                && self.piece_at(attacker_square)
                    == Some(Piece::from_parts(by_color, PieceType::King))
            {
                return true;
            }
        }

        if self.sliding_attack(
            square,
            by_color,
            &BISHOP_DIRECTIONS,
            &[PieceType::Bishop, PieceType::Queen],
        ) {
            return true;
        }

        self.sliding_attack(
            square,
            by_color,
            &ROOK_DIRECTIONS,
            &[PieceType::Rook, PieceType::Queen],
        )
    }

    pub fn is_in_check(&self, color: Color) -> bool {
        self.is_square_attacked(self.king_squares[color.index()], color.opposite())
    }

    pub fn generate_legal_moves(&mut self, moves: &mut MoveList) {
        moves.clear();

        let mut pseudo_legal = MoveList::new();
        self.generate_pseudo_legal_moves(&mut pseudo_legal);

        for mv in pseudo_legal.as_slice().iter().copied() {
            if let Ok(undo) = self.make_move(mv) {
                moves.push(mv);
                self.unmake_move(mv, undo);
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
        let moving_color = self.side_to_move;
        let undo = self.make_move_unchecked(mv)?;
        if self.is_in_check(moving_color) {
            self.unmake_move(mv, undo);
            return Err(MoveError::LeavesKingInCheck);
        }
        Ok(undo)
    }

    pub fn unmake_move(&mut self, mv: Move, undo: UndoState) {
        self.side_to_move = self.side_to_move.opposite();
        self.castling_rights = undo.previous_castling_rights;
        self.en_passant = undo.previous_en_passant;
        self.halfmove_clock = undo.previous_halfmove_clock;
        self.fullmove_number = undo.previous_fullmove_number;

        let moving_color = self.side_to_move;
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
            return;
        }

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

    fn generate_pseudo_legal_moves(&self, moves: &mut MoveList) {
        moves.clear();
        let color = self.side_to_move;

        for from in self.piece_lists[color.index()][PieceType::Pawn.index()].iter() {
            self.generate_pawn_moves(from, color, moves);
        }
        for from in self.piece_lists[color.index()][PieceType::Knight.index()].iter() {
            self.generate_jump_moves(from, color, PieceType::Knight, &KNIGHT_OFFSETS, moves);
        }
        for from in self.piece_lists[color.index()][PieceType::Bishop.index()].iter() {
            self.generate_slider_moves(from, color, &BISHOP_DIRECTIONS, moves);
        }
        for from in self.piece_lists[color.index()][PieceType::Rook.index()].iter() {
            self.generate_slider_moves(from, color, &ROOK_DIRECTIONS, moves);
        }
        for from in self.piece_lists[color.index()][PieceType::Queen.index()].iter() {
            self.generate_slider_moves(from, color, &BISHOP_DIRECTIONS, moves);
            self.generate_slider_moves(from, color, &ROOK_DIRECTIONS, moves);
        }
        for from in self.piece_lists[color.index()][PieceType::King.index()].iter() {
            self.generate_jump_moves(from, color, PieceType::King, &KING_OFFSETS, moves);
            self.generate_castling_moves(color, moves);
        }
    }

    fn generate_pawn_moves(&self, from: Square, color: Color, moves: &mut MoveList) {
        let forward_rank_delta = color.pawn_direction();
        if let Some(one_step) = from.offset(0, forward_rank_delta)
            && self.piece_at(one_step).is_none()
        {
            if from.rank() == color.promotion_from_rank() {
                for promotion_piece in PieceType::promotion_pieces() {
                    moves.push(Move::new(from, one_step).with_promotion(promotion_piece));
                }
            } else {
                moves.push(Move::new(from, one_step));
                if from.rank() == color.pawn_start_rank()
                    && let Some(two_step) = from.offset(0, forward_rank_delta * 2)
                    && self.piece_at(two_step).is_none()
                {
                    moves.push(Move::new(from, two_step).with_flags(FLAG_DOUBLE_PAWN_PUSH));
                }
            }
        }

        for file_delta in [-1, 1] {
            if let Some(target) = from.offset(file_delta, forward_rank_delta) {
                if let Some(target_piece) = self.piece_at(target) {
                    if target_piece.color() != color {
                        if from.rank() == color.promotion_from_rank() {
                            for promotion_piece in PieceType::promotion_pieces() {
                                moves.push(
                                    Move::new(from, target)
                                        .with_flags(FLAG_CAPTURE)
                                        .with_promotion(promotion_piece),
                                );
                            }
                        } else {
                            moves.push(Move::new(from, target).with_flags(FLAG_CAPTURE));
                        }
                    }
                } else if Some(target) == self.en_passant {
                    moves.push(Move::new(from, target).with_flags(FLAG_CAPTURE | FLAG_EN_PASSANT));
                }
            }
        }
    }

    fn generate_jump_moves(
        &self,
        from: Square,
        color: Color,
        piece_type: PieceType,
        offsets: &[(i8, i8)],
        moves: &mut MoveList,
    ) {
        debug_assert!(matches!(piece_type, PieceType::Knight | PieceType::King));

        for &(file_delta, rank_delta) in offsets {
            if let Some(target) = from.offset(file_delta, rank_delta) {
                match self.piece_at(target) {
                    Some(target_piece) if target_piece.color() == color => {}
                    Some(_) => moves.push(Move::new(from, target).with_flags(FLAG_CAPTURE)),
                    None => moves.push(Move::new(from, target)),
                }
            }
        }
    }

    fn generate_slider_moves(
        &self,
        from: Square,
        color: Color,
        directions: &[(i8, i8)],
        moves: &mut MoveList,
    ) {
        for &(file_delta, rank_delta) in directions {
            let mut current = from;
            while let Some(target) = current.offset(file_delta, rank_delta) {
                match self.piece_at(target) {
                    Some(target_piece) if target_piece.color() == color => break,
                    Some(_) => {
                        moves.push(Move::new(from, target).with_flags(FLAG_CAPTURE));
                        break;
                    }
                    None => moves.push(Move::new(from, target)),
                }
                current = target;
            }
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
        };

        self.en_passant = None;
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

        if mv.is_castle() {
            if moving_piece.piece_type() != PieceType::King {
                return Err(MoveError::IllegalCastle);
            }
            let (rook_from, rook_to) = castle_rook_squares(to);
            let rook = self
                .remove_piece_unchecked(rook_from)
                .ok_or(MoveError::IllegalCastle)?;
            if rook != Piece::from_parts(moving_color, PieceType::Rook) {
                self.add_piece_unchecked(rook_from, rook);
                return Err(MoveError::IllegalCastle);
            }
            self.remove_piece_unchecked(from)
                .expect("moving king must exist on source square");
            self.add_piece_unchecked(to, moving_piece);
            self.add_piece_unchecked(rook_to, rook);
        } else {
            self.remove_piece_unchecked(from)
                .expect("moving piece must exist on source square");
            let placed_piece = if let Some(promotion_piece) = mv.promotion() {
                if moving_piece.piece_type() != PieceType::Pawn {
                    return Err(MoveError::InvalidPromotion);
                }
                if to.rank() != 0 && to.rank() != 7 {
                    return Err(MoveError::InvalidPromotion);
                }
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
                self.en_passant = Some(en_passant_square);
            }
        }

        self.side_to_move = moving_color.opposite();
        Ok(undo)
    }

    fn update_castling_rights(
        &mut self,
        from: Square,
        to: Square,
        moving_piece: Piece,
        captured_piece: Option<Piece>,
    ) {
        if moving_piece.piece_type() == PieceType::King {
            self.castling_rights.remove_color(moving_piece.color());
        }

        if moving_piece.piece_type() == PieceType::Rook {
            self.clear_castling_right_for_square(from);
        }

        if let Some(captured_piece) = captured_piece
            && captured_piece.piece_type() == PieceType::Rook
        {
            self.clear_castling_right_for_square(to);
        }
    }

    fn clear_castling_right_for_square(&mut self, square: Square) {
        match square {
            Square::A1 => self.castling_rights.remove(CastlingRights::WHITE_QUEENSIDE),
            Square::H1 => self.castling_rights.remove(CastlingRights::WHITE_KINGSIDE),
            Square::A8 => self.castling_rights.remove(CastlingRights::BLACK_QUEENSIDE),
            Square::H8 => self.castling_rights.remove(CastlingRights::BLACK_KINGSIDE),
            _ => {}
        }
    }

    fn add_piece_unchecked(&mut self, square: Square, piece: Piece) {
        let color_index = piece.color().index();
        let piece_index = piece.piece_type().index();

        self.board[square.index()] = Some(piece);
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
        self.piece_bitboards[color_index][piece_index] &= !square.bit();
        self.occupancies[color_index] &= !square.bit();
        self.occupancies[OCCUPANCY_ALL] &= !square.bit();
        let removed = self.piece_lists[color_index][piece_index].remove(square);
        debug_assert!(removed, "piece list entry must exist for removed piece");

        Some(piece)
    }

    fn sliding_attack(
        &self,
        square: Square,
        by_color: Color,
        directions: &[(i8, i8)],
        attackers: &[PieceType],
    ) -> bool {
        for &(file_delta, rank_delta) in directions {
            let mut current = square;
            while let Some(next) = current.offset(file_delta, rank_delta) {
                if let Some(piece) = self.piece_at(next) {
                    if piece.color() == by_color && attackers.contains(&piece.piece_type()) {
                        return true;
                    }
                    break;
                }
                current = next;
            }
        }
        false
    }
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
