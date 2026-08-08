use std::{cell::RefCell, io::Read, sync::Arc};

#[cfg(test)]
use crate::nnue_rs::Arch;
use crate::nnue_rs::{
    Accumulator, Board as NnueBoard, BoardDelta, Color as NnueColor, Evaluation, Network,
    Piece as NnuePiece, PieceKind as NnuePieceKind,
};

use crate::core::{Color, Move, Piece, PieceType, Position, Square, UndoState};

use super::root::MAX_PLY;

/// A validated Stockfish-format NNUE network. Network files are external data;
/// Volkrix deliberately does not bundle a 100+ MiB evaluator in the source tree.
pub(crate) struct StockfishNnueService {
    network: Network,
}

impl StockfishNnueService {
    pub(crate) fn from_reader(path: &str, reader: &mut impl Read) -> Result<Arc<Self>, String> {
        let network = Network::from_reader(reader)
            .map_err(|error| format!("failed to load Stockfish NNUE file '{path}': {error}"))?;
        Ok(Arc::new(Self { network }))
    }

    #[cfg(test)]
    pub(crate) fn evaluate_fresh(&self, position: &Position) -> i32 {
        self.network.evaluate(&PositionBoard(position))
    }

    #[cfg(test)]
    pub(crate) fn network_architecture(&self) -> Arch {
        self.network.arch()
    }
}

/// Search-local accumulator storage for an external Stockfish-format network.
/// Every ply owns reusable accumulator buffers, so ordinary make/unmake does
/// not allocate after a depth has been visited once.
pub(crate) struct StockfishNnueState {
    service: Arc<StockfishNnueService>,
    frames: RefCell<Vec<StockfishNnueFrame>>,
    active: usize,
}

struct StockfishNnueFrame {
    accumulator: Accumulator,
    delta: Option<BoardDelta>,
    computed: bool,
}

impl StockfishNnueState {
    pub(crate) fn new(service: Arc<StockfishNnueService>) -> Self {
        let mut frames = Vec::with_capacity(MAX_PLY + 1);
        for _ in 0..=MAX_PLY {
            frames.push(StockfishNnueFrame {
                accumulator: service.network.empty_accumulator(),
                delta: None,
                computed: false,
            });
        }
        Self {
            service,
            frames: RefCell::new(frames),
            active: 0,
        }
    }

    pub(crate) fn reset(&mut self, position: &Position) {
        let mut frames = self.frames.borrow_mut();
        self.service
            .network
            .refresh(&PositionBoard(position), &mut frames[0].accumulator);
        frames[0].delta = None;
        frames[0].computed = true;
        self.active = 1;
    }

    pub(crate) fn push_child(&mut self, child_position: &Position, mv: Move, undo: UndoState) {
        assert!(
            self.active > 0,
            "Stockfish NNUE state must be reset before use"
        );
        assert!(
            self.active < self.frames.borrow().len(),
            "Stockfish NNUE stack overflow"
        );

        let child_index = self.active;
        let mut frames = self.frames.borrow_mut();
        frames[child_index].delta = Some(move_delta(child_position, mv, undo));
        frames[child_index].computed = false;
        self.active += 1;
    }

    pub(crate) fn pop(&mut self) {
        assert!(self.active > 1, "cannot pop the root Stockfish NNUE frame");
        self.active -= 1;
    }

    #[cfg(test)]
    pub(crate) fn evaluate(&self, position: &Position) -> i32 {
        self.evaluate_components(position).total()
    }

    pub(crate) fn evaluate_components(&self, position: &Position) -> Evaluation {
        assert!(
            self.active > 0,
            "Stockfish NNUE state must be reset before use"
        );
        let current = self.active - 1;
        let mut frames = self.frames.borrow_mut();
        if !frames[current].computed {
            if current > 0 && frames[current - 1].computed {
                let (parents, children) = frames.split_at_mut(current);
                let parent = &parents[current - 1].accumulator;
                let child = &mut children[0];
                self.service.network.update_delta(
                    &PositionBoard(position),
                    child
                        .delta
                        .as_ref()
                        .expect("lazy NNUE child frame must retain its move delta"),
                    parent,
                    &mut child.accumulator,
                );
                child.computed = true;
            } else {
                self.service
                    .network
                    .refresh(&PositionBoard(position), &mut frames[current].accumulator);
                frames[current].computed = true;
            }
        }
        self.service.network.evaluate_accumulator_components(
            &frames[current].accumulator,
            nnue_color(position.side_to_move()),
        )
    }

    #[cfg(test)]
    fn computed_frames(&self) -> usize {
        self.frames.borrow()[..self.active]
            .iter()
            .filter(|frame| frame.computed)
            .count()
    }
}

fn move_delta(child: &Position, mv: Move, undo: UndoState) -> BoardDelta {
    let mover = undo.moved_piece;
    let mut parent_kings = [
        child.king_square(Color::White).index() as u8,
        child.king_square(Color::Black).index() as u8,
    ];
    if mover.piece_type() == PieceType::King {
        parent_kings[mover.color().index()] = mv.from().index() as u8;
    }

    let mut delta = BoardDelta::new(parent_kings);
    delta.remove(mv.from().index() as u8, nnue_piece(mover));

    if let Some(captured) = undo.captured_piece {
        let square = if mv.is_en_passant() {
            let rank_delta = match mover.color() {
                Color::White => -1,
                Color::Black => 1,
            };
            mv.to()
                .offset(0, rank_delta)
                .expect("en-passant capture square must exist")
        } else {
            mv.to()
        };
        delta.remove(square.index() as u8, nnue_piece(captured));
    }

    if mv.is_castle() {
        let rank = mv.from().rank();
        let (rook_from, rook_to) = if mv.to().file() == 6 {
            (
                Square::from_coords(7, rank).expect("rook source must exist"),
                Square::from_coords(5, rank).expect("rook destination must exist"),
            )
        } else {
            (
                Square::from_coords(0, rank).expect("rook source must exist"),
                Square::from_coords(3, rank).expect("rook destination must exist"),
            )
        };
        let rook = Piece::from_parts(mover.color(), PieceType::Rook);
        delta.remove(rook_from.index() as u8, nnue_piece(rook));
        delta.add(rook_to.index() as u8, nnue_piece(rook));
    }

    let placed = Piece::from_parts(mover.color(), mv.promotion().unwrap_or(mover.piece_type()));
    delta.add(mv.to().index() as u8, nnue_piece(placed));
    delta
}

struct PositionBoard<'a>(&'a Position);

impl NnueBoard for PositionBoard<'_> {
    fn side_to_move(&self) -> NnueColor {
        nnue_color(self.0.side_to_move())
    }

    fn king_square(&self, color: NnueColor) -> u8 {
        self.0.king_square(core_color(color)).index() as u8
    }

    fn for_each_piece(&self, callback: &mut dyn FnMut(u8, NnuePiece)) {
        for color in Color::ALL {
            for piece_type in PieceType::ALL {
                let mut pieces = self.0.pieces(color, piece_type);
                while pieces != 0 {
                    let square = pieces.trailing_zeros() as u8;
                    pieces &= pieces - 1;
                    callback(square, nnue_piece(Piece::from_parts(color, piece_type)));
                }
            }
        }
    }
}

/// A zero-copy view of the position immediately before `mv`. Search invokes
/// NNUE after making a move, but the external incremental API wants both board
/// states. Reconstructing the handful of changed squares avoids cloning the
/// complete Position and its repetition history at every edge.
#[cfg(test)]
struct ParentBoard<'a> {
    child: &'a Position,
    mv: Move,
    undo: UndoState,
    castle_rook: Option<(Square, Square)>,
    capture_square: Option<Square>,
}

#[cfg(test)]
impl<'a> ParentBoard<'a> {
    fn new(child: &'a Position, mv: Move, undo: UndoState) -> Self {
        let castle_rook = mv.is_castle().then(|| {
            let rank = mv.from().rank();
            if mv.to().file() == 6 {
                (
                    Square::from_coords(7, rank).expect("rook source must exist"),
                    Square::from_coords(5, rank).expect("rook destination must exist"),
                )
            } else {
                (
                    Square::from_coords(0, rank).expect("rook source must exist"),
                    Square::from_coords(3, rank).expect("rook destination must exist"),
                )
            }
        });
        let capture_square = if undo.captured_piece.is_none() {
            None
        } else if mv.is_en_passant() {
            let rank_delta = match undo.moved_piece.color() {
                Color::White => -1,
                Color::Black => 1,
            };
            mv.to().offset(0, rank_delta)
        } else {
            Some(mv.to())
        };
        Self {
            child,
            mv,
            undo,
            castle_rook,
            capture_square,
        }
    }

    fn piece_at(&self, square: Square) -> Option<Piece> {
        if square == self.mv.from() {
            return Some(self.undo.moved_piece);
        }

        if let Some((rook_from, rook_to)) = self.castle_rook {
            if square == rook_from {
                return Some(Piece::from_parts(
                    self.undo.moved_piece.color(),
                    PieceType::Rook,
                ));
            }
            if square == rook_to || square == self.mv.to() {
                return None;
            }
        }

        if Some(square) == self.capture_square {
            return self.undo.captured_piece;
        }
        if square == self.mv.to() {
            return None;
        }

        self.child.piece_at(square)
    }
}

#[cfg(test)]
impl NnueBoard for ParentBoard<'_> {
    fn side_to_move(&self) -> NnueColor {
        nnue_color(self.child.side_to_move().opposite())
    }

    fn king_square(&self, color: NnueColor) -> u8 {
        let color = core_color(color);
        if self.undo.moved_piece.piece_type() == PieceType::King
            && self.undo.moved_piece.color() == color
        {
            self.mv.from().index() as u8
        } else {
            self.child.king_square(color).index() as u8
        }
    }

    fn for_each_piece(&self, callback: &mut dyn FnMut(u8, NnuePiece)) {
        for index in 0..64u8 {
            let square = Square::from_index_unchecked(index);
            if let Some(piece) = self.piece_at(square) {
                callback(index, nnue_piece(piece));
            }
        }
    }
}

const fn nnue_color(color: Color) -> NnueColor {
    match color {
        Color::White => NnueColor::White,
        Color::Black => NnueColor::Black,
    }
}

const fn core_color(color: NnueColor) -> Color {
    match color {
        NnueColor::White => Color::White,
        NnueColor::Black => Color::Black,
    }
}

fn nnue_piece(piece: Piece) -> NnuePiece {
    let kind = match piece.piece_type() {
        PieceType::Pawn => NnuePieceKind::Pawn,
        PieceType::Knight => NnuePieceKind::Knight,
        PieceType::Bishop => NnuePieceKind::Bishop,
        PieceType::Rook => NnuePieceKind::Rook,
        PieceType::Queen => NnuePieceKind::Queen,
        PieceType::King => NnuePieceKind::King,
    };
    NnuePiece::new(nnue_color(piece.color()), kind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{MoveList, ParsedMove};
    use std::{fs::File, io::BufReader, path::PathBuf};

    const DEFAULT_SFNNV10_TEST_NET: &str = "/tmp/nn-c288c895ea92.nnue";

    fn test_net_path() -> Option<PathBuf> {
        let path = std::env::var_os("VOLKRIX_SFNNUE_TEST_NET")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_SFNNV10_TEST_NET));
        path.is_file().then_some(path)
    }

    fn open_test_service(path: &std::path::Path) -> Arc<StockfishNnueService> {
        let path_text = path.to_str().expect("test net path must be UTF-8");
        let mut reader = BufReader::new(File::open(path).expect("test network must open"));
        StockfishNnueService::from_reader(path_text, &mut reader).expect("test network must load")
    }

    fn legal_move(position: &mut Position, text: &str) -> Move {
        let parsed = ParsedMove::parse(text).expect("move must parse");
        let mut moves = MoveList::new();
        position.generate_legal_moves(&mut moves);
        moves
            .iter()
            .copied()
            .find(|mv| mv.matches_parsed(parsed))
            .expect("move must be legal")
    }

    fn board_pieces(board: &impl NnueBoard) -> Vec<(u8, NnuePiece)> {
        let mut pieces = Vec::new();
        board.for_each_piece(&mut |square, piece| pieces.push((square, piece)));
        pieces.sort_by_key(|(square, _)| *square);
        pieces
    }

    fn assert_parent_view(fen: &str, move_text: &str) {
        let mut child = Position::from_fen(fen).expect("FEN must parse");
        let parent_pieces = board_pieces(&PositionBoard(&child));
        let parent_side = child.side_to_move();
        let parent_kings = [
            child.king_square(Color::White),
            child.king_square(Color::Black),
        ];
        let mv = legal_move(&mut child, move_text);
        let undo = child.make_move(mv).expect("move must apply");
        let view = ParentBoard::new(&child, mv, undo);

        assert_eq!(board_pieces(&view), parent_pieces);
        assert_eq!(view.side_to_move(), nnue_color(parent_side));
        assert_eq!(
            view.king_square(NnueColor::White),
            parent_kings[0].index() as u8
        );
        assert_eq!(
            view.king_square(NnueColor::Black),
            parent_kings[1].index() as u8
        );
    }

    #[test]
    fn parent_view_covers_every_special_move_shape() {
        assert_parent_view("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1");
        assert_parent_view("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1");
        assert_parent_view("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8g8");
        assert_parent_view("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8");
        assert_parent_view("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6");
        assert_parent_view("4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1", "e4d3");
        assert_parent_view("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q");
        assert_parent_view("4k3/8/8/8/8/8/7p/4K3 b - - 0 1", "h2h1n");
        assert_parent_view("r3k3/1P6/8/8/8/8/8/4K3 w - - 0 1", "b7a8q");
        assert_parent_view("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5");
        assert_parent_view(crate::core::STARTPOS_FEN, "g1f3");
    }

    #[test]
    fn stockfish_incremental_state_matches_refresh_for_all_move_shapes() {
        let Some(path) = test_net_path() else {
            return;
        };
        let service = open_test_service(&path);
        assert_eq!(service.network.arch(), Arch::Sfnnv10);

        for (fen, moves) in [
            (
                crate::core::STARTPOS_FEN,
                &["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"][..],
            ),
            (
                "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
                &["e1g1", "e8c8"][..],
            ),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", &["e5d6"]),
            ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", &["a7a8q"]),
            ("r3k3/1P6/8/8/8/8/8/4K3 w - - 0 1", &["b7a8q"]),
        ] {
            let mut position = Position::from_fen(fen).expect("test FEN must parse");
            let mut state = StockfishNnueState::new(Arc::clone(&service));
            state.reset(&position);
            assert_eq!(state.computed_frames(), 1);
            assert_eq!(state.evaluate(&position), service.evaluate_fresh(&position));

            for move_text in moves {
                let mv = legal_move(&mut position, move_text);
                let undo = position.make_move(mv).expect("move must apply");
                let computed_before_push = state.computed_frames();
                state.push_child(&position, mv, undo);
                assert_eq!(
                    state.computed_frames(),
                    computed_before_push,
                    "pushing {move_text} must only record a lazy compact delta"
                );
                assert_eq!(
                    state.evaluate(&position),
                    service.evaluate_fresh(&position),
                    "incremental mismatch after {move_text} from {fen}"
                );
                assert_eq!(state.computed_frames(), computed_before_push + 1);
            }
        }
    }

    #[test]
    fn unevaluated_delta_chain_refreshes_only_the_requested_leaf() {
        let Some(path) = test_net_path() else {
            return;
        };
        let service = open_test_service(&path);
        let mut position = Position::startpos();
        let mut state = StockfishNnueState::new(Arc::clone(&service));
        state.reset(&position);

        for move_text in ["e2e4", "e7e5", "g1f3", "b8c6"] {
            let mv = legal_move(&mut position, move_text);
            let undo = position.make_move(mv).expect("move must apply");
            state.push_child(&position, mv, undo);
        }

        assert_eq!(state.computed_frames(), 1);
        assert_eq!(state.evaluate(&position), service.evaluate_fresh(&position));
        assert_eq!(
            state.computed_frames(),
            2,
            "lazy materialization must not fill unused intermediate frames"
        );
    }

    #[test]
    #[ignore = "manual release profile using the external 104 MiB SFNNv10 network"]
    fn stockfish_sfnnv10_runtime_profile_report() {
        let Some(path) = test_net_path() else {
            return;
        };
        let load_started = std::time::Instant::now();
        let service = open_test_service(&path);
        let load_ms = load_started.elapsed().as_millis();

        let mut position = Position::startpos();
        let state_started = std::time::Instant::now();
        let mut state = StockfishNnueState::new(Arc::clone(&service));
        state.reset(&position);
        let state_init_us = state_started.elapsed().as_micros();
        let mv = legal_move(&mut position, "e2e4");
        let undo = position.make_move(mv).expect("profile move must apply");

        let profile_started = std::time::Instant::now();
        let mut checksum = 0i64;
        for _ in 0..10_000 {
            state.push_child(std::hint::black_box(&position), mv, undo);
            checksum = checksum.wrapping_add(i64::from(state.evaluate(&position)));
            state.pop();
        }
        let elapsed_us = profile_started.elapsed().as_micros();

        #[cfg(target_arch = "aarch64")]
        let simd = if std::arch::is_aarch64_feature_detected!("dotprod") {
            "AArch64 NEON + DotProd"
        } else {
            "AArch64 NEON"
        };
        #[cfg(target_arch = "x86_64")]
        let simd = if std::is_x86_feature_detected!("avx2") {
            "AVX2"
        } else {
            "scalar"
        };
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let simd = "scalar";

        println!(
            "sfnnv10_runtime: load_ms {load_ms} state_init_us {state_init_us} operations 10000 elapsed_us {elapsed_us} checksum {checksum} simd {simd}"
        );
    }
}
