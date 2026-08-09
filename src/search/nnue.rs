#[cfg(any(test, debug_assertions, feature = "internal-testing"))]
use std::path::{Path, PathBuf};
use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::core::{Color, Move, PieceType, Position, Score, Square, UndoState};

use super::stockfish_nnue::{StockfishNnueService, StockfishNnueState};

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

// Keep each standalone perspective on its own cache-line boundary. Search uses
// topology-sized slabs below, while these fixed values make root construction,
// full refreshes, and exact incremental-update tests allocation-free.
#[repr(align(64))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AlignedAccumulator<const HIDDEN_SIZE: usize>([i32; HIDDEN_SIZE]);

impl<const HIDDEN_SIZE: usize> Default for AlignedAccumulator<HIDDEN_SIZE> {
    fn default() -> Self {
        Self([0; HIDDEN_SIZE])
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AccumulatorStorage<const HIDDEN_SIZE: usize> {
    perspectives: [AlignedAccumulator<HIDDEN_SIZE>; 2],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
// Boxing the larger topology would reintroduce heap allocation into standalone
// accumulator construction; the search stack itself stores active lanes in
// topology-sized slabs and does not copy this enum per edge.
#[allow(clippy::large_enum_variant)]
enum AccumulatorPairStorage {
    Halfkp128(AccumulatorStorage<128>),
    Halfkp256(AccumulatorStorage<256>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AccumulatorPair(AccumulatorPairStorage);

impl AccumulatorPair {
    fn from_biases(hidden_biases: &[i16]) -> Self {
        match hidden_biases.len() {
            128 => Self(AccumulatorPairStorage::Halfkp128(
                AccumulatorStorage::from_biases(hidden_biases),
            )),
            256 => Self(AccumulatorPairStorage::Halfkp256(
                AccumulatorStorage::from_biases(hidden_biases),
            )),
            hidden_size => panic!("unsupported accumulator hidden size {hidden_size}"),
        }
    }

    fn hidden_size(&self) -> usize {
        match &self.0 {
            AccumulatorPairStorage::Halfkp128(_) => 128,
            AccumulatorPairStorage::Halfkp256(_) => 256,
        }
    }

    fn perspective(&self, color: Color) -> &[i32] {
        match &self.0 {
            AccumulatorPairStorage::Halfkp128(storage) => &storage.perspectives[color.index()].0,
            AccumulatorPairStorage::Halfkp256(storage) => &storage.perspectives[color.index()].0,
        }
    }

    fn perspective_mut(&mut self, color: Color) -> &mut [i32] {
        match &mut self.0 {
            AccumulatorPairStorage::Halfkp128(storage) => {
                &mut storage.perspectives[color.index()].0
            }
            AccumulatorPairStorage::Halfkp256(storage) => {
                &mut storage.perspectives[color.index()].0
            }
        }
    }

    #[cfg(test)]
    fn perspectives_mut(&mut self) -> (&mut [i32], &mut [i32]) {
        match &mut self.0 {
            AccumulatorPairStorage::Halfkp128(storage) => {
                let [white, black] = &mut storage.perspectives;
                (&mut white.0, &mut black.0)
            }
            AccumulatorPairStorage::Halfkp256(storage) => {
                let [white, black] = &mut storage.perspectives;
                (&mut white.0, &mut black.0)
            }
        }
    }

    #[cfg(test)]
    fn from_perspectives(white: &[i32], black: &[i32]) -> Self {
        debug_assert_eq!(white.len(), black.len());
        match white.len() {
            128 => Self(AccumulatorPairStorage::Halfkp128(
                AccumulatorStorage::from_perspectives(white, black),
            )),
            256 => Self(AccumulatorPairStorage::Halfkp256(
                AccumulatorStorage::from_perspectives(white, black),
            )),
            hidden_size => panic!("unsupported accumulator hidden size {hidden_size}"),
        }
    }
}

impl<const HIDDEN_SIZE: usize> AccumulatorStorage<HIDDEN_SIZE> {
    fn from_biases(hidden_biases: &[i16]) -> Self {
        debug_assert_eq!(hidden_biases.len(), HIDDEN_SIZE);
        let mut lanes = AlignedAccumulator::default();
        for (lane, bias) in lanes.0.iter_mut().zip(hidden_biases) {
            *lane = i32::from(*bias);
        }
        Self {
            perspectives: [lanes; 2],
        }
    }

    #[cfg(test)]
    fn from_perspectives(white: &[i32], black: &[i32]) -> Self {
        debug_assert_eq!(white.len(), HIDDEN_SIZE);
        debug_assert_eq!(black.len(), HIDDEN_SIZE);
        let mut white_lanes = AlignedAccumulator::default();
        let mut black_lanes = AlignedAccumulator::default();
        white_lanes.0.copy_from_slice(white);
        black_lanes.0.copy_from_slice(black);
        Self {
            perspectives: [white_lanes, black_lanes],
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct AccumulatorStack {
    perspectives: [Vec<i32>; 2],
    hidden_size: usize,
    frame_count: usize,
}

impl AccumulatorStack {
    fn with_hidden_size(hidden_size: usize) -> Self {
        let lane_capacity = hidden_size * (super::root::MAX_PLY + 1);
        Self {
            perspectives: [
                Vec::with_capacity(lane_capacity),
                Vec::with_capacity(lane_capacity),
            ],
            hidden_size,
            frame_count: 0,
        }
    }

    pub(crate) fn reset(&mut self, root: AccumulatorPair) {
        if self.hidden_size != root.hidden_size() {
            *self = Self::with_hidden_size(root.hidden_size());
        }
        for color in Color::ALL {
            let lanes = &mut self.perspectives[color.index()];
            lanes.clear();
            lanes.extend_from_slice(root.perspective(color));
        }
        self.frame_count = 1;
    }

    #[cfg(test)]
    pub(crate) fn push(&mut self, frame: AccumulatorPair) {
        assert!(
            self.frame_count <= super::root::MAX_PLY,
            "NNUE accumulator stack overflow"
        );
        debug_assert_eq!(frame.hidden_size(), self.hidden_size);
        for color in Color::ALL {
            self.perspectives[color.index()].extend_from_slice(frame.perspective(color));
        }
        self.frame_count += 1;
    }

    fn push_current(&mut self) {
        assert!(
            self.frame_count > 0,
            "cannot copy an empty accumulator stack"
        );
        assert!(
            self.frame_count <= super::root::MAX_PLY,
            "NNUE accumulator stack overflow"
        );
        let start = (self.frame_count - 1) * self.hidden_size;
        let end = start + self.hidden_size;
        for lanes in &mut self.perspectives {
            lanes.extend_from_within(start..end);
        }
        self.frame_count += 1;
    }

    pub(crate) fn pop(&mut self) {
        assert!(
            self.frame_count > 1,
            "cannot pop the root accumulator frame"
        );
        self.frame_count -= 1;
        let new_len = self.frame_count * self.hidden_size;
        for lanes in &mut self.perspectives {
            lanes.truncate(new_len);
        }
    }

    fn current(&self, color: Color) -> &[i32] {
        assert!(self.frame_count > 0, "NNUE accumulator stack is empty");
        let start = (self.frame_count - 1) * self.hidden_size;
        &self.perspectives[color.index()][start..start + self.hidden_size]
    }

    fn current_mut(&mut self) -> (&mut [i32], &mut [i32]) {
        assert!(self.frame_count > 0, "NNUE accumulator stack is empty");
        let start = (self.frame_count - 1) * self.hidden_size;
        let end = start + self.hidden_size;
        let [white, black] = &mut self.perspectives;
        (&mut white[start..end], &mut black[start..end])
    }

    #[cfg(test)]
    fn current_pair(&self) -> AccumulatorPair {
        AccumulatorPair::from_perspectives(self.current(Color::White), self.current(Color::Black))
    }
}

#[allow(private_interfaces)]
pub(crate) enum NnueSearchBackend {
    Volk {
        network: Arc<NnueNetwork>,
        stack: AccumulatorStack,
    },
    Stockfish(StockfishNnueState),
}

pub(crate) enum NnueSearchState {
    Single(NnueSearchBackend),
    Dual {
        big: Box<NnueSearchState>,
        small: Box<NnueSearchState>,
        ambiguity_threshold: i32,
        counters: Arc<DualEvalCounters>,
    },
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct DualEvalCounterSnapshot {
    pub(crate) small_selected: u64,
    pub(crate) big_fallbacks: u64,
}

#[derive(Default)]
pub(crate) struct DualEvalCounters {
    small_selected: AtomicU64,
    big_fallbacks: AtomicU64,
}

impl DualEvalCounters {
    pub(crate) fn snapshot(&self) -> DualEvalCounterSnapshot {
        DualEvalCounterSnapshot {
            small_selected: self.small_selected.load(Ordering::Relaxed),
            big_fallbacks: self.big_fallbacks.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct NnueEvaluationComponents {
    pub(crate) psqt: i32,
    pub(crate) positional: i32,
}

impl NnueEvaluationComponents {
    pub(crate) const fn total(self) -> Score {
        Score(self.psqt + self.positional)
    }

    #[allow(dead_code)] // Staged for the root score-policy integration.
    pub(crate) fn scaled(self, psqt_weight: i32, positional_weight: i32, divisor: i32) -> Score {
        assert!(divisor > 0, "NNUE component scale divisor must be positive");
        let weighted = i64::from(self.psqt) * i64::from(psqt_weight)
            + i64::from(self.positional) * i64::from(positional_weight);
        Score((weighted / i64::from(divisor)).clamp(i32::MIN as i64, i32::MAX as i64) as i32)
    }
}

impl NnueSearchState {
    pub(crate) fn new(service: Arc<NnueService>) -> Self {
        match &service.backend {
            NnueBackend::Volk(network) => Self::Single(NnueSearchBackend::Volk {
                network: Arc::clone(network),
                stack: AccumulatorStack::with_hidden_size(network.topology.hidden_size),
            }),
            NnueBackend::Stockfish(stockfish) => Self::Single(NnueSearchBackend::Stockfish(
                StockfishNnueState::new(Arc::clone(stockfish)),
            )),
            NnueBackend::Dual {
                big,
                small,
                ambiguity_threshold,
                counters,
            } => Self::Dual {
                big: Box::new(Self::new(Arc::clone(big))),
                small: Box::new(Self::new(Arc::clone(small))),
                ambiguity_threshold: *ambiguity_threshold,
                counters: Arc::clone(counters),
            },
        }
    }

    pub(crate) fn reset(&mut self, position: &Position) {
        match self {
            Self::Dual { big, small, .. } => {
                big.reset(position);
                small.reset(position);
            }
            Self::Single(backend) => match backend {
                NnueSearchBackend::Volk { network, stack } => {
                    stack.reset(network.build_accumulator(position));
                }
                NnueSearchBackend::Stockfish(state) => state.reset(position),
            },
        }
    }

    pub(crate) fn push_child(&mut self, child_position: &Position, mv: Move, undo: UndoState) {
        match self {
            Self::Dual { big, small, .. } => {
                big.push_child(child_position, mv, undo);
                small.push_child(child_position, mv, undo);
            }
            Self::Single(backend) => match backend {
                NnueSearchBackend::Volk { network, stack } => {
                    stack.push_current();
                    let (white, black) = stack.current_mut();
                    network.update_child_perspectives(white, black, child_position, mv, undo);
                }
                NnueSearchBackend::Stockfish(state) => state.push_child(child_position, mv, undo),
            },
        }
    }

    pub(crate) fn pop(&mut self) {
        match self {
            Self::Dual { big, small, .. } => {
                small.pop();
                big.pop();
            }
            Self::Single(backend) => match backend {
                NnueSearchBackend::Volk { stack, .. } => stack.pop(),
                NnueSearchBackend::Stockfish(state) => state.pop(),
            },
        }
    }

    pub(crate) fn evaluate(&self, position: &Position) -> Score {
        self.evaluate_components(position).total()
    }

    pub(crate) fn evaluate_components(&self, position: &Position) -> NnueEvaluationComponents {
        match self {
            Self::Dual {
                big,
                small,
                ambiguity_threshold,
                counters,
            } => {
                let small_evaluation = small.evaluate_components(position);
                if small_evaluation.total().0.abs() < *ambiguity_threshold {
                    counters.big_fallbacks.fetch_add(1, Ordering::Relaxed);
                    big.evaluate_components(position)
                } else {
                    counters.small_selected.fetch_add(1, Ordering::Relaxed);
                    small_evaluation
                }
            }
            Self::Single(backend) => match backend {
                NnueSearchBackend::Volk { network, stack } => NnueEvaluationComponents {
                    psqt: 0,
                    positional: network
                        .evaluate_perspectives(
                            position,
                            stack.current(Color::White),
                            stack.current(Color::Black),
                        )
                        .0,
                },
                NnueSearchBackend::Stockfish(state) => {
                    let evaluation = state.evaluate_components(position);
                    NnueEvaluationComponents {
                        psqt: evaluation.psqt,
                        positional: evaluation.positional,
                    }
                }
            },
        }
    }

    #[cfg(test)]
    fn volk_stack(&self) -> &AccumulatorStack {
        match self {
            Self::Dual { .. } => panic!("VOLKNNUE stack requested for a dual evaluator"),
            Self::Single(backend) => match backend {
                NnueSearchBackend::Volk { stack, .. } => stack,
                NnueSearchBackend::Stockfish(_) => {
                    panic!("VOLKNNUE stack requested for Stockfish backend")
                }
            },
        }
    }
}

/// Search-local state for an optional big/small evaluator pair. Selection is
/// intentionally caller-driven: engine search and time-control tuning decide
/// when a cheaper net is appropriate, while this type keeps both lazy stacks
/// synchronized and provides an ambiguity fallback to the big evaluator.
#[allow(dead_code)]
pub(crate) struct DualNnueSearchState {
    big: NnueSearchState,
    small: NnueSearchState,
}

#[allow(dead_code)]
impl DualNnueSearchState {
    pub(crate) fn new(big: Arc<NnueService>, small: Arc<NnueService>) -> Self {
        Self {
            big: NnueSearchState::new(big),
            small: NnueSearchState::new(small),
        }
    }

    pub(crate) fn reset(&mut self, position: &Position) {
        self.big.reset(position);
        self.small.reset(position);
    }

    pub(crate) fn push_child(&mut self, position: &Position, mv: Move, undo: UndoState) {
        self.big.push_child(position, mv, undo);
        self.small.push_child(position, mv, undo);
    }

    pub(crate) fn pop(&mut self) {
        self.small.pop();
        self.big.pop();
    }

    pub(crate) fn evaluate(
        &self,
        position: &Position,
        prefer_small: bool,
        small_ambiguity_threshold: i32,
    ) -> (NnueEvaluationComponents, bool) {
        if !prefer_small {
            return (self.big.evaluate_components(position), false);
        }
        let small = self.small.evaluate_components(position);
        if small.total().0.abs() < small_ambiguity_threshold.max(0) {
            (self.big.evaluate_components(position), false)
        } else {
            (small, true)
        }
    }
}

#[derive(Clone)]
enum NnueBackend {
    Volk(Arc<NnueNetwork>),
    Stockfish(Arc<StockfishNnueService>),
    Dual {
        big: Arc<NnueService>,
        small: Arc<NnueService>,
        ambiguity_threshold: i32,
        counters: Arc<DualEvalCounters>,
    },
}

#[derive(Clone)]
pub(crate) struct NnueService {
    backend: NnueBackend,
}

impl NnueService {
    pub(crate) fn open_eval_file(path: &str) -> Result<Arc<Self>, String> {
        let path = path.trim();
        if path.is_empty() {
            return Err("EvalFile requires a non-empty path".to_owned());
        }

        let file = File::open(path)
            .map_err(|error| format!("failed to read EvalFile '{path}': {error}"))?;
        let mut reader = BufReader::new(file);
        let volk_format = reader
            .fill_buf()
            .map_err(|error| format!("failed to read EvalFile '{path}': {error}"))?
            .starts_with(NNUE_MAGIC);

        Self::from_reader(path, &mut reader, volk_format)
    }

    #[cfg(volkrix_embedded_nnue)]
    pub(crate) fn open_embedded_eval() -> Result<Arc<Self>, String> {
        let bytes = include_bytes!(env!("VOLKRIX_EMBEDDED_NNUE"));
        let label = format!(
            "<embedded:{}:{}>",
            env!("VOLKRIX_EMBEDDED_NNUE_SHA256"),
            env!("VOLKRIX_EMBEDDED_NNUE_SIZE")
        );
        let mut reader = std::io::Cursor::new(bytes.as_slice());
        Self::from_reader(&label, &mut reader, bytes.starts_with(NNUE_MAGIC))
    }

    fn from_reader(
        path: &str,
        reader: &mut impl Read,
        volk_format: bool,
    ) -> Result<Arc<Self>, String> {
        let backend = if volk_format {
            let mut bytes = Vec::new();
            reader
                .read_to_end(&mut bytes)
                .map_err(|error| format!("failed to read EvalFile '{path}': {error}"))?;
            let network = NnueNetwork::parse(&bytes)
                .map_err(|error| format!("failed to load EvalFile '{path}': {error}"))?;
            NnueBackend::Volk(Arc::new(network))
        } else {
            NnueBackend::Stockfish(StockfishNnueService::from_reader(path, reader)?)
        };
        Ok(Arc::new(Self { backend }))
    }

    pub(crate) fn with_small_fallback(
        big: Arc<Self>,
        small: Arc<Self>,
        ambiguity_threshold: i32,
        counters: Arc<DualEvalCounters>,
    ) -> Arc<Self> {
        assert!(
            !matches!(big.backend, NnueBackend::Dual { .. })
                && !matches!(small.backend, NnueBackend::Dual { .. }),
            "dual evaluator inputs must be standalone networks"
        );
        Arc::new(Self {
            backend: NnueBackend::Dual {
                big,
                small,
                ambiguity_threshold: ambiguity_threshold.max(0),
                counters,
            },
        })
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn metadata(&self) -> NnueMetadata {
        self.volk_network().metadata
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    pub(crate) fn build_accumulator(&self, position: &Position) -> AccumulatorPair {
        self.volk_network().build_accumulator(position)
    }

    #[cfg(test)]
    pub(crate) fn derive_child_accumulator(
        &self,
        parent: &AccumulatorPair,
        child_position: &Position,
        mv: Move,
        undo: UndoState,
    ) -> AccumulatorPair {
        self.volk_network()
            .derive_child_accumulator(parent, child_position, mv, undo)
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    pub(crate) fn evaluate(&self, position: &Position, accumulators: &AccumulatorPair) -> Score {
        self.volk_network().evaluate(position, accumulators)
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    fn volk_network(&self) -> &NnueNetwork {
        match &self.backend {
            NnueBackend::Volk(network) => network,
            NnueBackend::Stockfish(_) => {
                panic!("legacy VOLKNNUE accumulator API used with a Stockfish NNUE network")
            }
            NnueBackend::Dual { .. } => {
                panic!("legacy VOLKNNUE accumulator API used with a dual NNUE network")
            }
        }
    }

    #[cfg(test)]
    fn is_stockfish(&self) -> bool {
        matches!(&self.backend, NnueBackend::Stockfish(_))
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
            + std::mem::size_of_val(hidden_biases)
            + std::mem::size_of_val(input_weights)
            + std::mem::size_of::<i32>()
            + std::mem::size_of_val(output_weights),
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

    #[cfg(test)]
    fn derive_child_accumulator(
        &self,
        parent: &AccumulatorPair,
        child_position: &Position,
        mv: Move,
        undo: UndoState,
    ) -> AccumulatorPair {
        let mut child = *parent;
        self.update_child_accumulator(&mut child, child_position, mv, undo);
        child
    }

    #[cfg(test)]
    fn update_child_accumulator(
        &self,
        child: &mut AccumulatorPair,
        child_position: &Position,
        mv: Move,
        undo: UndoState,
    ) {
        debug_assert_eq!(child.hidden_size(), self.topology.hidden_size);
        let (white, black) = child.perspectives_mut();
        self.update_child_perspectives(white, black, child_position, mv, undo);
    }

    fn update_child_perspectives(
        &self,
        white: &mut [i32],
        black: &mut [i32],
        child_position: &Position,
        mv: Move,
        undo: UndoState,
    ) {
        debug_assert_eq!(white.len(), self.topology.hidden_size);
        debug_assert_eq!(black.len(), self.topology.hidden_size);
        let moving_color = undo.moved_piece.color();
        let moving_piece_type = undo.moved_piece.piece_type();
        let capture_square = capture_square(mv, moving_color);

        self.update_child_perspective(
            white,
            Color::White,
            child_position,
            mv,
            moving_color,
            moving_piece_type,
            undo,
            capture_square,
        );
        self.update_child_perspective(
            black,
            Color::Black,
            child_position,
            mv,
            moving_color,
            moving_piece_type,
            undo,
            capture_square,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn update_child_perspective(
        &self,
        lanes: &mut [i32],
        perspective: Color,
        child_position: &Position,
        mv: Move,
        moving_color: Color,
        moving_piece_type: PieceType,
        undo: UndoState,
        capture_square: Square,
    ) {
        if moving_piece_type == PieceType::King && moving_color == perspective {
            self.build_perspective_into(child_position, perspective, lanes);
            return;
        }

        let king_square = child_position.king_square(perspective);
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

    fn build_perspective_into(&self, position: &Position, perspective: Color, lanes: &mut [i32]) {
        debug_assert_eq!(lanes.len(), self.topology.hidden_size);
        for (lane, bias) in lanes.iter_mut().zip(self.hidden_biases.iter()) {
            *lane = i32::from(*bias);
        }

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
                        lanes,
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

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    fn evaluate(&self, position: &Position, accumulators: &AccumulatorPair) -> Score {
        self.evaluate_perspectives(
            position,
            accumulators.perspective(Color::White),
            accumulators.perspective(Color::Black),
        )
    }

    fn evaluate_perspectives(&self, position: &Position, white: &[i32], black: &[i32]) -> Score {
        let (active, passive) = match position.side_to_move() {
            Color::White => (white, black),
            Color::Black => (black, white),
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
        debug_assert!(matches!(delta, -1 | 1));
        let Some(bucket) = feature_bucket(perspective, feature.color, feature.piece_type) else {
            return;
        };
        let feature_index = feature_index(perspective, king_square, bucket, feature.square);
        let weights_offset = feature_index * self.topology.hidden_size;
        let weights =
            &self.input_weights[weights_offset..weights_offset + self.topology.hidden_size];
        if delta > 0 {
            for (lane, weight) in lanes.iter_mut().zip(weights.iter()) {
                *lane += i32::from(*weight);
            }
        } else {
            for (lane, weight) in lanes.iter_mut().zip(weights.iter()) {
                *lane -= i32::from(*weight);
            }
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
    use crate::nnue_rs::Arch;
    use crate::search::{
        SearchLimits,
        service::{SearchRequest, UciSearchService},
    };
    use std::{
        mem::{align_of, size_of},
        path::PathBuf,
    };

    const DEFAULT_SFNNV10_TEST_NET: &str = "/tmp/nn-c288c895ea92.nnue";
    const DEFAULT_STOCKFISH_SMALL_TEST_NET: &str = "/tmp/nn-37f18f62d772.nnue";

    fn stockfish_test_net_path() -> Option<PathBuf> {
        let path = std::env::var_os("VOLKRIX_SFNNUE_TEST_NET")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_SFNNV10_TEST_NET));
        path.is_file().then_some(path)
    }

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
            stack.current_pair(),
            service.build_accumulator(&position),
            "root accumulator mismatch"
        );

        let mut undos = Vec::new();
        for uci in moves {
            let mv = find_legal_move(&mut position, uci);
            let undo = position.make_move(mv).expect("test move must be legal");
            let parent = stack.current_pair();
            let child = service.derive_child_accumulator(&parent, &position, mv, undo);
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
                stack.current_pair(),
                rebuilt,
                "unmake must restore the previous accumulator frame exactly"
            );
        }
    }

    #[test]
    fn tiny_test_net_metadata_matches_supported_compatibility_topology() {
        let service = tiny_test_service();
        assert!(!service.is_stockfish());
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
    fn accumulator_storage_is_fixed_aligned_and_preallocated_for_the_search_horizon() {
        let service = tiny_test_service();
        let position = Position::startpos();
        let root = service.build_accumulator(&position);

        assert_eq!(align_of::<AccumulatorPair>(), 64);
        assert!(size_of::<AccumulatorPair>() >= 2 * HALFKP_256X2.hidden_size * size_of::<i32>());

        let mut stack = AccumulatorStack::default();
        stack.reset(root);
        let initial_capacities = stack.perspectives.each_ref().map(|lanes| lanes.capacity());
        assert!(
            initial_capacities
                .iter()
                .all(|capacity| *capacity > super::super::root::MAX_PLY * stack.hidden_size)
        );
        for _ in 0..super::super::root::MAX_PLY {
            stack.push_current();
            assert_eq!(
                stack.perspectives.each_ref().map(|lanes| lanes.capacity()),
                initial_capacities
            );
        }
    }

    #[test]
    fn retained_256_topology_uses_the_same_allocation_free_search_stack() {
        let biases = (0..HALFKP_256X2.hidden_size)
            .map(|lane| lane as i16 - 128)
            .collect::<Vec<_>>();
        let root = AccumulatorPair::from_biases(&biases);
        assert_eq!(root.hidden_size(), HALFKP_256X2.hidden_size);
        assert_eq!(root.perspective(Color::White).len(), 256);

        let mut stack = AccumulatorStack::with_hidden_size(HALFKP_256X2.hidden_size);
        stack.reset(root);
        let initial_capacities = stack.perspectives.each_ref().map(|lanes| lanes.capacity());
        for _ in 0..super::super::root::MAX_PLY {
            stack.push_current();
        }
        assert_eq!(stack.current_pair(), root);
        assert_eq!(
            stack.perspectives.each_ref().map(|lanes| lanes.capacity()),
            initial_capacities
        );
    }

    #[test]
    fn long_deterministic_legal_walk_matches_full_refresh_at_every_ply() {
        let service = tiny_test_service();
        let mut position = Position::startpos();
        let mut state = NnueSearchState::new(Arc::clone(&service));
        state.reset(&position);
        let initial_capacities = state
            .volk_stack()
            .perspectives
            .each_ref()
            .map(|lanes| lanes.capacity());
        let mut undos = Vec::new();
        let mut random_state = 0x9e37_79b9_7f4a_7c15u64;

        for ply in 0..96 {
            let mut legal_moves = MoveList::new();
            position.generate_legal_moves(&mut legal_moves);
            if legal_moves.is_empty() {
                break;
            }

            random_state ^= random_state << 13;
            random_state ^= random_state >> 7;
            random_state ^= random_state << 17;
            let mv = legal_moves.get((random_state as usize) % legal_moves.len());
            let undo = position
                .make_move(mv)
                .expect("generated move must remain legal");
            state.push_child(&position, mv, undo);

            let rebuilt = service.build_accumulator(&position);
            assert_eq!(
                state.volk_stack().current_pair(),
                rebuilt,
                "incremental accumulator mismatch at deterministic ply {ply} after {mv}"
            );
            assert_eq!(
                state.evaluate(&position),
                service.evaluate(&position, &rebuilt),
                "incremental evaluation mismatch at deterministic ply {ply} after {mv}"
            );
            assert_eq!(
                state
                    .volk_stack()
                    .perspectives
                    .each_ref()
                    .map(|lanes| lanes.capacity()),
                initial_capacities
            );
            undos.push((mv, undo));
        }

        while let Some((mv, undo)) = undos.pop() {
            state.pop();
            position.unmake_move(mv, undo);
            assert_eq!(
                state.volk_stack().current_pair(),
                service.build_accumulator(&position)
            );
        }
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
    fn component_scaling_is_explicit_and_overflow_safe() {
        let components = NnueEvaluationComponents {
            psqt: 320,
            positional: -96,
        };
        assert_eq!(components.total(), Score(224));
        assert_eq!(components.scaled(125, 131, 128), Score(214));

        let extreme = NnueEvaluationComponents {
            psqt: i32::MAX,
            positional: i32::MAX,
        };
        assert_eq!(extreme.scaled(i32::MAX, i32::MAX, 1), Score(i32::MAX));
    }

    #[test]
    fn dual_state_selects_small_and_falls_back_to_big_when_ambiguous() {
        let Some(big_path) = stockfish_test_net_path() else {
            return;
        };
        let small_path = Path::new(DEFAULT_STOCKFISH_SMALL_TEST_NET);
        if !small_path.is_file() {
            return;
        }
        let big =
            NnueService::open_eval_file(big_path.to_str().expect("big net path must be UTF-8"))
                .expect("big net must load");
        let small =
            NnueService::open_eval_file(small_path.to_str().expect("small net path must be UTF-8"))
                .expect("small net must load");
        let mut state = DualNnueSearchState::new(big, small);
        let mut position = Position::startpos();
        state.reset(&position);

        let (selected_small, used_small) = state.evaluate(&position, true, 200);
        assert!(used_small);
        assert_eq!(selected_small.total(), Score(251));

        let (fallback_big, used_small) = state.evaluate(&position, true, 300);
        assert!(!used_small);
        assert_eq!(fallback_big.total(), Score(44));

        let mv = find_legal_move(&mut position, "e2e4");
        let undo = position.make_move(mv).expect("test move must apply");
        state.push_child(&position, mv, undo);
        let (forced_big, used_small) = state.evaluate(&position, false, 0);
        assert!(!used_small);
        assert_eq!(forced_big.total(), Score(-65));
        state.pop();
        position.unmake_move(mv, undo);
    }

    #[test]
    fn production_dual_service_tracks_selection_and_fallback_counters() {
        let Some(big_path) = stockfish_test_net_path() else {
            return;
        };
        let small_path = Path::new(DEFAULT_STOCKFISH_SMALL_TEST_NET);
        if !small_path.is_file() {
            return;
        }
        let big =
            NnueService::open_eval_file(big_path.to_str().expect("big net path must be UTF-8"))
                .expect("big net must load");
        let small =
            NnueService::open_eval_file(small_path.to_str().expect("small net path must be UTF-8"))
                .expect("small net must load");
        let counters = Arc::new(DualEvalCounters::default());
        let dual = NnueService::with_small_fallback(big, small, 300, Arc::clone(&counters));
        let mut state = NnueSearchState::new(dual);
        let mut position = Position::startpos();
        state.reset(&position);

        let fallback_score = state.evaluate(&position);
        assert_ne!(fallback_score, Score(251));
        assert_eq!(
            counters.snapshot(),
            DualEvalCounterSnapshot {
                small_selected: 0,
                big_fallbacks: 1,
            }
        );

        let mut legal_moves = MoveList::new();
        position.generate_legal_moves(&mut legal_moves);
        let mv = legal_moves
            .iter()
            .copied()
            .find(|mv| mv.to_string() == "e2e4")
            .expect("e2e4 must be legal");
        let undo = position.make_move(mv).expect("e2e4 must apply");
        state.push_child(&position, mv, undo);
        let _ = state.evaluate(&position);
        state.pop();
        position.unmake_move(mv, undo);
        assert_eq!(state.evaluate(&position), fallback_score);
        assert_eq!(counters.snapshot().big_fallbacks, 3);
    }

    #[test]
    fn evalfile_auto_detects_sfnnv10_and_runs_incremental_search() {
        let Some(path) = stockfish_test_net_path() else {
            return;
        };
        let path_text = path.to_str().expect("test net path must be UTF-8");
        let service = NnueService::open_eval_file(path_text).expect("SFNNv10 network must load");
        assert!(service.is_stockfish());

        let stockfish = match &service.backend {
            NnueBackend::Stockfish(stockfish) => stockfish,
            NnueBackend::Volk(_) => panic!("Stockfish network was misdetected as VOLKNNUE"),
            NnueBackend::Dual { .. } => panic!("standalone network unexpectedly became dual"),
        };
        assert_eq!(stockfish.network_architecture(), Arch::Sfnnv10);

        let mut position = Position::startpos();
        let mut state = NnueSearchState::new(Arc::clone(&service));
        state.reset(&position);
        let start_score = state.evaluate(&position).0;
        assert_eq!(start_score, stockfish.evaluate_fresh(&position));
        if path == Path::new(DEFAULT_SFNNV10_TEST_NET) {
            assert_eq!(start_score, 44, "pinned SFNNv10 start-position oracle");
        }
        let mv = find_legal_move(&mut position, "e2e4");
        let undo = position.make_move(mv).expect("test move must apply");
        state.push_child(&position, mv, undo);
        let e4_score = state.evaluate(&position).0;
        assert_eq!(e4_score, stockfish.evaluate_fresh(&position));
        if path == Path::new(DEFAULT_SFNNV10_TEST_NET) {
            assert_eq!(e4_score, -65, "pinned SFNNv10 e2e4 oracle");
        }
        state.pop();
        position.unmake_move(mv, undo);

        let mut search_service = UciSearchService::new();
        search_service.debug_install_nnue(path_text, service);
        let result = search_service.search(
            &mut position,
            SearchRequest {
                limits: SearchLimits::new(2),
                soft_deadline: None,
                hard_deadline: None,
                stop_flag: None,
                root_moves: None,
            },
        );
        assert_eq!(result.depth, 2);
        assert!(result.best_move.is_some());
        assert!(result.nodes > 0);
    }

    #[test]
    fn evalfile_accepts_stockfish_small_halfkav2_hm_network() {
        let path = Path::new(DEFAULT_STOCKFISH_SMALL_TEST_NET);
        if !path.is_file() {
            return;
        }
        let path_text = path.to_str().expect("test net path must be UTF-8");
        let service = NnueService::open_eval_file(path_text).expect("small network must load");
        let stockfish = match &service.backend {
            NnueBackend::Stockfish(stockfish) => stockfish,
            NnueBackend::Volk(_) => panic!("Stockfish network was misdetected as VOLKNNUE"),
            NnueBackend::Dual { .. } => panic!("standalone network unexpectedly became dual"),
        };
        assert_eq!(stockfish.network_architecture(), Arch::HalfKAv2Hm);

        let mut position = Position::startpos();
        let mut state = NnueSearchState::new(Arc::clone(&service));
        state.reset(&position);
        assert_eq!(state.evaluate(&position).0, 251, "small-net start oracle");
        let mv = find_legal_move(&mut position, "e2e4");
        let undo = position.make_move(mv).expect("test move must apply");
        state.push_child(&position, mv, undo);
        assert_eq!(state.evaluate(&position).0, 49, "small-net e2e4 oracle");
        assert_eq!(
            state.evaluate(&position).0,
            stockfish.evaluate_fresh(&position)
        );
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

    #[test]
    #[ignore = "manual release profile for incremental NNUE accumulator updates"]
    fn incremental_accumulator_update_profile_report() {
        let service = tiny_test_service();
        let mut position = Position::startpos();
        let mut state = NnueSearchState::new(Arc::clone(&service));
        state.reset(&position);
        let mv = find_legal_move(&mut position, "e2e4");
        let undo = position.make_move(mv).expect("profile move must be legal");
        let started = std::time::Instant::now();
        let mut checksum = 0i64;

        for _ in 0..100_000 {
            state.push_child(std::hint::black_box(&position), mv, undo);
            checksum = checksum.wrapping_add(i64::from(state.evaluate(&position).0));
            state.pop();
        }

        println!(
            "incremental_accumulator_updates: count 100000 checksum {checksum} time_us {}",
            started.elapsed().as_micros()
        );
    }
}
