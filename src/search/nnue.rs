use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::core::{Color, Move, PieceType, Position, Score, Square, UndoState};

pub(crate) const NNUE_MAGIC: &[u8; 8] = b"VOLKNNUE";
pub(crate) const NNUE_VERSION: u32 = 1;
pub(crate) const NNUE_TOPOLOGY_HALFKP_128X2: u32 = 1;
pub(crate) const NNUE_TOPOLOGY_HALFKP_256X2: u32 = 2;
pub(crate) const NNUE_FEATURE_BUCKETS: usize = 10;
pub(crate) const NNUE_FEATURE_COUNT: usize = 64 * NNUE_FEATURE_BUCKETS * 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct HalfkpTopology {
    pub(crate) id: u32,
    pub(crate) name: &'static str,
    pub(crate) hidden_size: usize,
}

impl HalfkpTopology {
    pub(crate) const fn output_inputs(self) -> usize {
        self.hidden_size * 2
    }

    pub(crate) const fn input_weight_count(self) -> usize {
        NNUE_FEATURE_COUNT * self.hidden_size
    }
}

pub(crate) const HALFKP_128X2: HalfkpTopology = HalfkpTopology {
    id: NNUE_TOPOLOGY_HALFKP_128X2,
    name: "HalfKP128x2",
    hidden_size: 128,
};

pub(crate) const HALFKP_256X2: HalfkpTopology = HalfkpTopology {
    id: NNUE_TOPOLOGY_HALFKP_256X2,
    name: "HalfKP256x2",
    hidden_size: 256,
};

#[cfg_attr(not(feature = "offline-tools"), allow(dead_code))]
pub(crate) const RETAINED_PRODUCTION_TOPOLOGY: HalfkpTopology = HALFKP_256X2;

// Retained HalfKP-like bucket order:
// own pawn, own knight, own bishop, own rook, own queen,
// enemy pawn, enemy knight, enemy bishop, enemy rook, enemy queen.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const FEATURE_BUCKET_LABELS: [&str; NNUE_FEATURE_BUCKETS] = [
    "own_pawn",
    "own_knight",
    "own_bishop",
    "own_rook",
    "own_queen",
    "enemy_pawn",
    "enemy_knight",
    "enemy_bishop",
    "enemy_rook",
    "enemy_queen",
];

#[derive(Clone, Copy)]
struct PieceFeature {
    color: Color,
    piece_type: PieceType,
    square: Square,
}

#[cfg(feature = "offline-tools")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SparseFeaturePair {
    pub(crate) active: Vec<u16>,
    pub(crate) passive: Vec<u16>,
}

const EXPECTED_HEADER_BYTES: usize = 8 + 4 + 4 + 4 + 4 + 4 + 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct NnueMetadata {
    pub(crate) version: u32,
    pub(crate) topology: u32,
    pub(crate) feature_count: u32,
    pub(crate) hidden_size: u32,
    pub(crate) output_inputs: u32,
    pub(crate) output_scale: i32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct AccumulatorPair {
    perspectives: [Vec<i32>; 2],
}

impl AccumulatorPair {
    fn from_biases(hidden_biases: &[i16]) -> Self {
        let lanes = hidden_biases
            .iter()
            .map(|bias| i32::from(*bias))
            .collect::<Vec<_>>();
        let perspectives = [lanes.clone(), lanes];
        Self { perspectives }
    }

    fn perspective(&self, color: Color) -> &[i32] {
        &self.perspectives[color.index()]
    }

    fn perspective_mut(&mut self, color: Color) -> &mut [i32] {
        &mut self.perspectives[color.index()]
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct AccumulatorStack {
    frames: Vec<AccumulatorPair>,
}

impl AccumulatorStack {
    pub(crate) fn reset(&mut self, root: AccumulatorPair) {
        self.frames.clear();
        self.frames.push(root);
    }

    pub(crate) fn push(&mut self, frame: AccumulatorPair) {
        self.frames.push(frame);
    }

    pub(crate) fn pop(&mut self) {
        assert!(
            self.frames.len() > 1,
            "cannot pop the root accumulator frame"
        );
        self.frames.pop();
    }

    pub(crate) fn current(&self) -> &AccumulatorPair {
        self.frames
            .last()
            .expect("NNUE accumulator stack must contain at least one frame")
    }
}

#[derive(Clone)]
pub(crate) struct NnueSearchState {
    service: Arc<NnueService>,
    stack: AccumulatorStack,
}

impl NnueSearchState {
    pub(crate) fn new(service: Arc<NnueService>) -> Self {
        Self {
            service,
            stack: AccumulatorStack::default(),
        }
    }

    pub(crate) fn reset(&mut self, position: &Position) {
        self.stack.reset(self.service.build_accumulator(position));
    }

    pub(crate) fn push_child(&mut self, child_position: &Position, mv: Move, undo: UndoState) {
        let child =
            self.service
                .derive_child_accumulator(self.stack.current(), child_position, mv, undo);
        self.stack.push(child);
    }

    pub(crate) fn pop(&mut self) {
        self.stack.pop();
    }

    pub(crate) fn evaluate(&self, position: &Position) -> Score {
        self.service.evaluate(position, self.stack.current())
    }
}

#[derive(Clone)]
pub(crate) struct NnueService {
    network: Arc<NnueNetwork>,
}

impl NnueService {
    pub(crate) fn open_eval_file(path: &str) -> Result<Arc<Self>, String> {
        let path = path.trim();
        if path.is_empty() {
            return Err("EvalFile requires a non-empty path".to_owned());
        }

        let bytes =
            fs::read(path).map_err(|error| format!("failed to read EvalFile '{path}': {error}"))?;
        let network = NnueNetwork::parse(&bytes)
            .map_err(|error| format!("failed to load EvalFile '{path}': {error}"))?;
        Ok(Arc::new(Self {
            network: Arc::new(network),
        }))
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn metadata(&self) -> NnueMetadata {
        self.network.metadata
    }

    pub(crate) fn build_accumulator(&self, position: &Position) -> AccumulatorPair {
        self.network.build_accumulator(position)
    }

    pub(crate) fn derive_child_accumulator(
        &self,
        parent: &AccumulatorPair,
        child_position: &Position,
        mv: Move,
        undo: UndoState,
    ) -> AccumulatorPair {
        self.network
            .derive_child_accumulator(parent, child_position, mv, undo)
    }

    pub(crate) fn evaluate(&self, position: &Position, accumulators: &AccumulatorPair) -> Score {
        self.network.evaluate(position, accumulators)
    }
}

#[cfg(feature = "offline-tools")]
pub(crate) fn sparse_features_for_side_to_move(position: &Position) -> SparseFeaturePair {
    let active = sparse_features_for_perspective(position, position.side_to_move());
    let passive = sparse_features_for_perspective(position, position.side_to_move().opposite());
    SparseFeaturePair { active, passive }
}

#[cfg(feature = "offline-tools")]
fn sparse_features_for_perspective(position: &Position, perspective: Color) -> Vec<u16> {
    let mut features = Vec::new();
    let king_square = position.king_square(perspective);
    for piece_color in Color::ALL {
        for piece_type in [
            PieceType::Pawn,
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
        ] {
            let mut pieces = position.pieces(piece_color, piece_type);
            while let Some(square) = pop_lsb(&mut pieces) {
                let Some(bucket) = feature_bucket(perspective, piece_color, piece_type) else {
                    continue;
                };
                let feature = feature_index(perspective, king_square, bucket, square);
                features.push(feature as u16);
            }
        }
    }
    features.sort_unstable();
    features
}

#[cfg(feature = "offline-tools")]
pub(crate) fn encode_volknnue(
    topology: HalfkpTopology,
    hidden_biases: &[i16],
    input_weights: &[i16],
    output_bias: i32,
    output_weights: &[i16],
    output_scale: i32,
) -> Result<Vec<u8>, String> {
    if hidden_biases.len() != topology.hidden_size {
        return Err(format!(
            "hidden bias count {} did not match topology {} hidden size {}",
            hidden_biases.len(),
            topology.name,
            topology.hidden_size
        ));
    }
    let input_weight_count = topology.input_weight_count();
    if input_weights.len() != input_weight_count {
        return Err(format!(
            "input weight count {} did not match topology {} count {}",
            input_weights.len(),
            topology.name,
            input_weight_count
        ));
    }
    if output_weights.len() != topology.output_inputs() {
        return Err(format!(
            "output weight count {} did not match topology {} count {}",
            output_weights.len(),
            topology.name,
            topology.output_inputs()
        ));
    }
    if output_scale <= 0 {
        return Err("VOLKNNUE output scale must be positive".to_owned());
    }

    let mut bytes = Vec::with_capacity(
        EXPECTED_HEADER_BYTES
            + hidden_biases.len() * std::mem::size_of::<i16>()
            + std::mem::size_of_val(input_weights)
            + std::mem::size_of::<i32>()
            + output_weights.len() * std::mem::size_of::<i16>(),
    );
    bytes.extend_from_slice(NNUE_MAGIC);
    bytes.extend_from_slice(&NNUE_VERSION.to_le_bytes());
    bytes.extend_from_slice(&topology.id.to_le_bytes());
    bytes.extend_from_slice(&(NNUE_FEATURE_COUNT as u32).to_le_bytes());
    bytes.extend_from_slice(&(topology.hidden_size as u32).to_le_bytes());
    bytes.extend_from_slice(&(topology.output_inputs() as u32).to_le_bytes());
    bytes.extend_from_slice(&output_scale.to_le_bytes());
    for bias in hidden_biases {
        bytes.extend_from_slice(&bias.to_le_bytes());
    }
    for weight in input_weights {
        bytes.extend_from_slice(&weight.to_le_bytes());
    }
    bytes.extend_from_slice(&output_bias.to_le_bytes());
    for weight in output_weights {
        bytes.extend_from_slice(&weight.to_le_bytes());
    }

    NnueNetwork::parse(&bytes)?;
    Ok(bytes)
}

struct NnueNetwork {
    topology: HalfkpTopology,
    metadata: NnueMetadata,
    hidden_biases: Box<[i16]>,
    input_weights: Box<[i16]>,
    output_bias: i32,
    output_weights: Box<[i16]>,
}

impl NnueNetwork {
    fn parse(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < EXPECTED_HEADER_BYTES {
            return Err("file is too small to contain a VOLKNNUE header".to_owned());
        }

        if &bytes[..NNUE_MAGIC.len()] != NNUE_MAGIC {
            return Err("missing VOLKNNUE magic header".to_owned());
        }

        let mut cursor = NNUE_MAGIC.len();
        let metadata = NnueMetadata {
            version: read_u32_le(bytes, &mut cursor)?,
            topology: read_u32_le(bytes, &mut cursor)?,
            feature_count: read_u32_le(bytes, &mut cursor)?,
            hidden_size: read_u32_le(bytes, &mut cursor)?,
            output_inputs: read_u32_le(bytes, &mut cursor)?,
            output_scale: read_i32_le(bytes, &mut cursor)?,
        };

        if metadata.version != NNUE_VERSION {
            return Err(format!(
                "unsupported VOLKNNUE version {} (expected {})",
                metadata.version, NNUE_VERSION
            ));
        }
        let topology = supported_halfkp_topology(
            metadata.topology,
            metadata.hidden_size as usize,
            metadata.output_inputs as usize,
        )?;
        if metadata.feature_count as usize != NNUE_FEATURE_COUNT {
            return Err(format!(
                "VOLKNNUE feature count {} did not match retained HalfKP feature count {}",
                metadata.feature_count, NNUE_FEATURE_COUNT
            ));
        }
        if metadata.output_scale <= 0 {
            return Err("VOLKNNUE output scale must be positive".to_owned());
        }

        let mut hidden_biases = Vec::with_capacity(topology.hidden_size);
        for _ in 0..topology.hidden_size {
            hidden_biases.push(read_i16_le(bytes, &mut cursor)?);
        }

        let input_weight_count = topology.input_weight_count();
        let mut input_weights = Vec::with_capacity(input_weight_count);
        for _ in 0..input_weight_count {
            input_weights.push(read_i16_le(bytes, &mut cursor)?);
        }

        let output_bias = read_i32_le(bytes, &mut cursor)?;
        let mut output_weights = Vec::with_capacity(topology.output_inputs());
        for _ in 0..topology.output_inputs() {
            output_weights.push(read_i16_le(bytes, &mut cursor)?);
        }

        if cursor != bytes.len() {
            return Err(format!(
                "VOLKNNUE payload length mismatch: parsed {} bytes, file contained {} bytes",
                cursor,
                bytes.len()
            ));
        }

        Ok(Self {
            topology,
            metadata,
            hidden_biases: hidden_biases.into_boxed_slice(),
            input_weights: input_weights.into_boxed_slice(),
            output_bias,
            output_weights: output_weights.into_boxed_slice(),
        })
    }

    fn build_accumulator(&self, position: &Position) -> AccumulatorPair {
        let mut accumulators = AccumulatorPair::from_biases(&self.hidden_biases);
        for perspective in Color::ALL {
            let king_square = position.king_square(perspective);
            for piece_color in Color::ALL {
                for piece_type in [
                    PieceType::Pawn,
                    PieceType::Knight,
                    PieceType::Bishop,
                    PieceType::Rook,
                    PieceType::Queen,
                ] {
                    let mut pieces = position.pieces(piece_color, piece_type);
                    while let Some(square) = pop_lsb(&mut pieces) {
                        self.apply_piece_delta(
                            accumulators.perspective_mut(perspective),
                            perspective,
                            king_square,
                            PieceFeature {
                                color: piece_color,
                                piece_type,
                                square,
                            },
                            1,
                        );
                    }
                }
            }
        }
        accumulators
    }

    fn derive_child_accumulator(
        &self,
        parent: &AccumulatorPair,
        child_position: &Position,
        mv: Move,
        undo: UndoState,
    ) -> AccumulatorPair {
        let mut child = parent.clone();
        let moving_color = undo.moved_piece.color();
        let moving_piece_type = undo.moved_piece.piece_type();
        let capture_square = capture_square(mv, moving_color);

        for perspective in Color::ALL {
            if moving_piece_type == PieceType::King && moving_color == perspective {
                child.perspectives[perspective.index()] =
                    self.build_perspective(child_position, perspective);
                continue;
            }

            let king_square = child_position.king_square(perspective);
            let lanes = child.perspective_mut(perspective);

            if moving_piece_type != PieceType::King {
                self.apply_piece_delta(
                    lanes,
                    perspective,
                    king_square,
                    PieceFeature {
                        color: moving_color,
                        piece_type: moving_piece_type,
                        square: mv.from(),
                    },
                    -1,
                );
                self.apply_piece_delta(
                    lanes,
                    perspective,
                    king_square,
                    PieceFeature {
                        color: moving_color,
                        piece_type: mv.promotion().unwrap_or(moving_piece_type),
                        square: mv.to(),
                    },
                    1,
                );
            } else if mv.is_castle() {
                let (rook_from, rook_to) = castle_rook_squares(mv.to());
                self.apply_piece_delta(
                    lanes,
                    perspective,
                    king_square,
                    PieceFeature {
                        color: moving_color,
                        piece_type: PieceType::Rook,
                        square: rook_from,
                    },
                    -1,
                );
                self.apply_piece_delta(
                    lanes,
                    perspective,
                    king_square,
                    PieceFeature {
                        color: moving_color,
                        piece_type: PieceType::Rook,
                        square: rook_to,
                    },
                    1,
                );
            }

            if let Some(captured_piece) = undo.captured_piece {
                self.apply_piece_delta(
                    lanes,
                    perspective,
                    king_square,
                    PieceFeature {
                        color: captured_piece.color(),
                        piece_type: captured_piece.piece_type(),
                        square: capture_square,
                    },
                    -1,
                );
            }
        }

        child
    }

    fn build_perspective(&self, position: &Position, perspective: Color) -> Vec<i32> {
        let mut lanes = self
            .hidden_biases
            .iter()
            .map(|bias| i32::from(*bias))
            .collect::<Vec<_>>();

        let king_square = position.king_square(perspective);
        for piece_color in Color::ALL {
            for piece_type in [
                PieceType::Pawn,
                PieceType::Knight,
                PieceType::Bishop,
                PieceType::Rook,
                PieceType::Queen,
            ] {
                let mut pieces = position.pieces(piece_color, piece_type);
                while let Some(square) = pop_lsb(&mut pieces) {
                    self.apply_piece_delta(
                        &mut lanes,
                        perspective,
                        king_square,
                        PieceFeature {
                            color: piece_color,
                            piece_type,
                            square,
                        },
                        1,
                    );
                }
            }
        }

        lanes
    }

    fn evaluate(&self, position: &Position, accumulators: &AccumulatorPair) -> Score {
        let (active, passive) = match position.side_to_move() {
            Color::White => (
                accumulators.perspective(Color::White),
                accumulators.perspective(Color::Black),
            ),
            Color::Black => (
                accumulators.perspective(Color::Black),
                accumulators.perspective(Color::White),
            ),
        };

        let mut output = self.output_bias;
        let (active_weights, passive_weights) =
            self.output_weights.split_at(self.topology.hidden_size);
        for (lane, weight) in active.iter().zip(active_weights.iter()) {
            output += clipped_relu(*lane) * i32::from(*weight);
        }
        for (lane, weight) in passive.iter().zip(passive_weights.iter()) {
            output += clipped_relu(*lane) * i32::from(*weight);
        }

        // Final NNUE score orientation matches the engine's static-eval convention:
        // positive scores favor the side to move, negative scores favor the opponent.
        Score(output / self.metadata.output_scale)
    }

    fn apply_piece_delta(
        &self,
        lanes: &mut [i32],
        perspective: Color,
        king_square: Square,
        feature: PieceFeature,
        delta: i32,
    ) {
        let Some(bucket) = feature_bucket(perspective, feature.color, feature.piece_type) else {
            return;
        };
        let feature_index = feature_index(perspective, king_square, bucket, feature.square);
        let weights_offset = feature_index * self.topology.hidden_size;
        let weights =
            &self.input_weights[weights_offset..weights_offset + self.topology.hidden_size];
        for (lane, weight) in lanes.iter_mut().zip(weights.iter()) {
            *lane += delta * *weight as i32;
        }
    }
}

pub(crate) fn supported_halfkp_topology(
    topology_id: u32,
    hidden_size: usize,
    output_inputs: usize,
) -> Result<HalfkpTopology, String> {
    if output_inputs != hidden_size * 2 {
        return Err(format!(
            "VOLKNNUE output input count {} did not equal 2 * hidden size {}",
            output_inputs, hidden_size
        ));
    }

    let topology = match (topology_id, hidden_size) {
        (NNUE_TOPOLOGY_HALFKP_128X2, 128) => HALFKP_128X2,
        (NNUE_TOPOLOGY_HALFKP_256X2, 256) => HALFKP_256X2,
        _ => {
            return Err(format!(
                "unsupported VOLKNNUE HalfKP topology {} with hidden size {} and output inputs {}",
                topology_id, hidden_size, output_inputs
            ));
        }
    };
    if output_inputs != topology.output_inputs() {
        return Err(format!(
            "VOLKNNUE output input count {} did not match topology {} count {}",
            output_inputs,
            topology.name,
            topology.output_inputs()
        ));
    }
    Ok(topology)
}

fn read_u32_le(bytes: &[u8], cursor: &mut usize) -> Result<u32, String> {
    if *cursor + 4 > bytes.len() {
        return Err("unexpected EOF while reading VOLKNNUE header".to_owned());
    }
    let value = u32::from_le_bytes(bytes[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    Ok(value)
}

fn read_i32_le(bytes: &[u8], cursor: &mut usize) -> Result<i32, String> {
    read_u32_le(bytes, cursor).map(|value| value as i32)
}

fn read_i16_le(bytes: &[u8], cursor: &mut usize) -> Result<i16, String> {
    if *cursor + 2 > bytes.len() {
        return Err("unexpected EOF while reading VOLKNNUE payload".to_owned());
    }
    let value = i16::from_le_bytes(bytes[*cursor..*cursor + 2].try_into().unwrap());
    *cursor += 2;
    Ok(value)
}

fn feature_bucket(perspective: Color, piece_color: Color, piece_type: PieceType) -> Option<usize> {
    let base = match piece_type {
        PieceType::Pawn => 0,
        PieceType::Knight => 1,
        PieceType::Bishop => 2,
        PieceType::Rook => 3,
        PieceType::Queen => 4,
        PieceType::King => return None,
    };
    Some(if piece_color == perspective {
        base
    } else {
        base + 5
    })
}

fn feature_index(
    perspective: Color,
    king_square: Square,
    bucket: usize,
    piece_square: Square,
) -> usize {
    let king_index = normalize_square(perspective, king_square);
    let piece_index = normalize_square(perspective, piece_square);
    (king_index * NNUE_FEATURE_BUCKETS + bucket) * 64 + piece_index
}

fn normalize_square(perspective: Color, square: Square) -> usize {
    match perspective {
        Color::White => square.index(),
        Color::Black => square.index() ^ 56,
    }
}

fn clipped_relu(value: i32) -> i32 {
    value.clamp(0, 255)
}

fn capture_square(mv: Move, moving_color: Color) -> Square {
    if mv.is_en_passant() {
        mv.to()
            .offset(0, -moving_color.pawn_direction())
            .expect("en passant capture square must stay on the board")
    } else {
        mv.to()
    }
}

fn castle_rook_squares(king_to: Square) -> (Square, Square) {
    match king_to {
        Square::G1 => (Square::H1, Square::F1),
        Square::C1 => (Square::A1, Square::D1),
        Square::G8 => (Square::H8, Square::F8),
        Square::C8 => (Square::A8, Square::D8),
        _ => panic!("castle rook squares requested for non-castle king destination"),
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

#[cfg(any(test, debug_assertions, feature = "internal-testing"))]
pub(crate) fn tiny_test_evalfile_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join("nnue")
        .join("volkrix-halfkp128x2-test.volknnue")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{MoveList, ParsedMove};

    fn tiny_test_service() -> Arc<NnueService> {
        NnueService::open_eval_file(
            tiny_test_evalfile_path()
                .to_str()
                .expect("tiny test eval file path must be UTF-8"),
        )
        .expect("tiny deterministic NNUE test net must load")
    }

    fn find_legal_move(position: &mut Position, uci: &str) -> Move {
        let parsed = ParsedMove::parse(uci).expect("test move must parse");
        let mut legal_moves = MoveList::new();
        position.generate_legal_moves(&mut legal_moves);
        legal_moves
            .iter()
            .copied()
            .find(|mv| mv.matches_parsed(parsed))
            .expect("test move must be legal")
    }

    fn assert_incremental_sequence_matches_full_rebuild(fen: &str, moves: &[&str]) {
        let service = tiny_test_service();
        let mut position = Position::from_fen(fen).expect("FEN parse must succeed");
        let mut stack = AccumulatorStack::default();
        stack.reset(service.build_accumulator(&position));
        assert_eq!(
            *stack.current(),
            service.build_accumulator(&position),
            "root accumulator mismatch"
        );

        let mut undos = Vec::new();
        for uci in moves {
            let mv = find_legal_move(&mut position, uci);
            let undo = position.make_move(mv).expect("test move must be legal");
            let child = service.derive_child_accumulator(stack.current(), &position, mv, undo);
            let rebuilt = service.build_accumulator(&position);
            assert_eq!(
                child, rebuilt,
                "incremental child accumulator mismatch after {uci}"
            );
            stack.push(child);
            undos.push((mv, undo));
        }

        while let Some((mv, undo)) = undos.pop() {
            stack.pop();
            position.unmake_move(mv, undo);
            let rebuilt = service.build_accumulator(&position);
            assert_eq!(
                *stack.current(),
                rebuilt,
                "unmake must restore the previous accumulator frame exactly"
            );
        }
    }

    #[test]
    fn tiny_test_net_metadata_matches_supported_compatibility_topology() {
        let service = tiny_test_service();
        let metadata = service.metadata();

        assert_eq!(metadata.version, NNUE_VERSION);
        assert_eq!(metadata.topology, HALFKP_128X2.id);
        assert_eq!(metadata.feature_count as usize, NNUE_FEATURE_COUNT);
        assert_eq!(metadata.hidden_size as usize, HALFKP_128X2.hidden_size);
        assert_eq!(
            metadata.output_inputs as usize,
            HALFKP_128X2.output_inputs()
        );
        assert!(metadata.output_scale > 0);
    }

    #[test]
    fn feature_bucket_layout_is_explicit_and_stable() {
        assert_eq!(
            FEATURE_BUCKET_LABELS,
            [
                "own_pawn",
                "own_knight",
                "own_bishop",
                "own_rook",
                "own_queen",
                "enemy_pawn",
                "enemy_knight",
                "enemy_bishop",
                "enemy_rook",
                "enemy_queen",
            ]
        );
    }

    #[test]
    fn retained_production_topology_is_halfkp256x2() {
        assert_eq!(RETAINED_PRODUCTION_TOPOLOGY, HALFKP_256X2);
        assert_eq!(RETAINED_PRODUCTION_TOPOLOGY.hidden_size, 256);
        assert_eq!(RETAINED_PRODUCTION_TOPOLOGY.output_inputs(), 512);
    }

    #[test]
    fn full_accumulator_build_is_deterministic() {
        let service = tiny_test_service();
        let position = Position::from_fen(
            "r2q1rk1/ppp2ppp/2npbn2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8",
        )
        .expect("FEN parse must succeed");

        let first = service.build_accumulator(&position);
        let second = service.build_accumulator(&position);
        assert_eq!(first, second);
    }

    #[test]
    fn incremental_updates_match_full_recomputation_for_ordinary_moves_and_captures() {
        assert_incremental_sequence_matches_full_rebuild(
            crate::core::STARTPOS_FEN,
            &["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5e6"],
        );
    }

    #[test]
    fn incremental_updates_match_full_recomputation_for_castling() {
        assert_incremental_sequence_matches_full_rebuild(
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            &["e1g1", "e8c8"],
        );
    }

    #[test]
    fn incremental_updates_match_full_recomputation_for_en_passant() {
        assert_incremental_sequence_matches_full_rebuild(
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
            &["e5d6"],
        );
    }

    #[test]
    fn incremental_updates_match_full_recomputation_for_promotions() {
        assert_incremental_sequence_matches_full_rebuild(
            "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
            &["a7a8q"],
        );
        assert_incremental_sequence_matches_full_rebuild(
            "4k3/8/8/8/8/8/7p/4K3 b - - 0 1",
            &["h2h1q"],
        );
    }

    #[test]
    fn score_orientation_is_side_to_move_relative() {
        let service = tiny_test_service();
        let white_to_move =
            Position::from_fen("4k3/8/8/8/8/8/3Q4/4K3 w - - 0 1").expect("FEN parse must succeed");
        let black_to_move =
            Position::from_fen("4k3/8/8/8/8/8/3Q4/4K3 b - - 0 1").expect("FEN parse must succeed");

        let white_score = service
            .evaluate(&white_to_move, &service.build_accumulator(&white_to_move))
            .0;
        let black_score = service
            .evaluate(&black_to_move, &service.build_accumulator(&black_to_move))
            .0;

        assert!(white_score > 0);
        assert!(black_score < 0);
    }

    #[test]
    fn parser_rejects_malformed_network() {
        let mut bytes = Vec::from(&NNUE_MAGIC[..]);
        bytes.extend_from_slice(&NNUE_VERSION.to_le_bytes());
        bytes.extend_from_slice(&NNUE_TOPOLOGY_HALFKP_128X2.to_le_bytes());
        bytes.extend_from_slice(&(NNUE_FEATURE_COUNT as u32).to_le_bytes());
        bytes.extend_from_slice(&(HALFKP_128X2.hidden_size as u32).to_le_bytes());
        bytes.extend_from_slice(&(HALFKP_128X2.output_inputs() as u32).to_le_bytes());
        bytes.extend_from_slice(&1i32.to_le_bytes());
        assert!(NnueNetwork::parse(&bytes).is_err());
    }
}
