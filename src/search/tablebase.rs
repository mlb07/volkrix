use std::{
    ffi::CString,
    fmt,
    os::raw::{c_char, c_uint},
    ptr,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::core::{Color, Move, MoveList, ParsedMove, PieceType, Position, Square};

#[cfg(test)]
use std::collections::BTreeMap;

use super::root::MATE_SCORE;

pub(crate) const TABLEBASE_SCORE_BAND: i32 = 20_000;

const TB_RESULT_FAILED: u32 = 0xFFFF_FFFF;
const TB_RESULT_WDL_MASK: u32 = 0x0000_000F;
const TB_RESULT_TO_MASK: u32 = 0x0000_03F0;
const TB_RESULT_FROM_MASK: u32 = 0x0000_FC00;
const TB_RESULT_PROMOTES_MASK: u32 = 0x0007_0000;
const TB_RESULT_DTZ_MASK: u32 = 0xFFF0_0000;
const TB_RESULT_WDL_SHIFT: u32 = 0;
const TB_RESULT_TO_SHIFT: u32 = 4;
const TB_RESULT_FROM_SHIFT: u32 = 10;
const TB_RESULT_PROMOTES_SHIFT: u32 = 16;
const TB_RESULT_DTZ_SHIFT: u32 = 20;

const TB_LOSS: u32 = 0;
const TB_BLESSED_LOSS: u32 = 1;
const TB_DRAW: u32 = 2;
const TB_CURSED_WIN: u32 = 3;
const TB_WIN: u32 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WdlOutcome {
    Win,
    CursedWin,
    Draw,
    BlessedLoss,
    Loss,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RootProbe {
    pub(crate) best_move: Move,
    pub(crate) wdl: WdlOutcome,
    pub(crate) dtz: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProbeError(String);

impl ProbeError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
pub(crate) trait TablebaseBackend: Send + Sync {
    fn supports_root(&self, position: &Position) -> bool;
    fn supports_non_root(&self, position: &Position) -> bool;
    fn probe_wdl(&self, position: &Position) -> Result<Option<WdlOutcome>, ProbeError>;
    fn probe_root(
        &self,
        position: &Position,
        legal_moves: &MoveList,
    ) -> Result<Option<RootProbe>, ProbeError>;
}

enum BackendKind {
    Fathom(FathomBackend),
    #[cfg(test)]
    Mock(Arc<dyn TablebaseBackend>),
}

pub(crate) struct TablebaseService {
    path: String,
    backend: BackendKind,
}

impl TablebaseService {
    pub(crate) fn open_syzygy_path(
        path: &str,
        previous: Option<&Arc<Self>>,
    ) -> Result<Arc<Self>, String> {
        let path = path.trim();
        if path.is_empty() {
            return Err("SyzygyPath requires a non-empty path".to_owned());
        }

        let previous_fathom = previous.and_then(|service| service.fathom_identity());
        let service_id = FathomBackend::initialize(path, previous_fathom.as_ref())?;
        Ok(Arc::new(Self {
            path: path.to_owned(),
            backend: BackendKind::Fathom(FathomBackend { service_id }),
        }))
    }

    #[cfg(test)]
    pub(crate) fn from_backend_for_tests(
        path: impl Into<String>,
        backend: Arc<dyn TablebaseBackend>,
    ) -> Arc<Self> {
        Arc::new(Self {
            path: path.into(),
            backend: BackendKind::Mock(backend),
        })
    }

    pub(crate) fn supports_root(&self, position: &Position) -> bool {
        self.is_within_retained_scope(position)
            && match &self.backend {
                BackendKind::Fathom(backend) => backend.supports_root(position),
                #[cfg(test)]
                BackendKind::Mock(backend) => backend.supports_root(position),
            }
    }

    pub(crate) fn supports_non_root(&self, position: &Position) -> bool {
        self.is_within_retained_scope(position)
            && match &self.backend {
                BackendKind::Fathom(backend) => backend.supports_non_root(position),
                #[cfg(test)]
                BackendKind::Mock(backend) => backend.supports_non_root(position),
            }
    }

    pub(crate) fn probe_wdl(&self, position: &Position) -> Result<Option<WdlOutcome>, ProbeError> {
        if !self.supports_non_root(position) {
            return Ok(None);
        }
        match &self.backend {
            BackendKind::Fathom(backend) => backend.probe_wdl(position),
            #[cfg(test)]
            BackendKind::Mock(backend) => backend.probe_wdl(position),
        }
    }

    pub(crate) fn probe_root(
        &self,
        position: &Position,
        legal_moves: &MoveList,
    ) -> Result<Option<RootProbe>, ProbeError> {
        if !self.supports_root(position) {
            return Ok(None);
        }
        let probe = match &self.backend {
            BackendKind::Fathom(backend) => backend.probe_root(position, legal_moves),
            #[cfg(test)]
            BackendKind::Mock(backend) => backend.probe_root(position, legal_moves),
        }?;
        if let Some(root_probe) = probe
            && !move_list_contains(legal_moves, root_probe.best_move)
        {
            return Err(ProbeError::new(
                "tablebase root probe returned an illegal best move",
            ));
        }
        Ok(probe)
    }

    fn is_within_retained_scope(&self, position: &Position) -> bool {
        position.castling_rights().is_empty() && position.occupancy().count_ones() <= 6
    }

    fn fathom_identity(&self) -> Option<FathomIdentity> {
        match self.backend {
            BackendKind::Fathom(FathomBackend { service_id }) => Some(FathomIdentity {
                service_id,
                path: self.path.clone(),
            }),
            #[cfg(test)]
            BackendKind::Mock(_) => None,
        }
    }
}

impl Drop for TablebaseService {
    fn drop(&mut self) {
        let service_id = match &self.backend {
            BackendKind::Fathom(FathomBackend { service_id }) => *service_id,
            #[cfg(test)]
            BackendKind::Mock(_) => return,
        };

        let mut state = fathom_state().lock().expect("Fathom state lock poisoned");
        if state.current_service_id == Some(service_id) {
            unsafe {
                tb_free();
            }
            state.current_service_id = None;
            state.current_path = None;
        }
    }
}

pub(crate) fn score_from_wdl(outcome: WdlOutcome, ply: usize) -> i32 {
    debug_assert!(TABLEBASE_SCORE_BAND < MATE_SCORE - super::root::MAX_PLY as i32);
    match outcome {
        WdlOutcome::Win => TABLEBASE_SCORE_BAND - ply as i32,
        WdlOutcome::CursedWin | WdlOutcome::Draw | WdlOutcome::BlessedLoss => 0,
        WdlOutcome::Loss => -TABLEBASE_SCORE_BAND + ply as i32,
    }
}

fn move_list_contains(legal_moves: &MoveList, target: Move) -> bool {
    (0..legal_moves.len()).any(|index| legal_moves.get(index) == target)
}

struct FathomBackend {
    service_id: u64,
}

#[derive(Clone)]
struct FathomIdentity {
    service_id: u64,
    path: String,
}

#[derive(Default)]
struct FathomGlobalState {
    current_service_id: Option<u64>,
    current_path: Option<String>,
}

impl FathomBackend {
    fn initialize(path: &str, previous: Option<&FathomIdentity>) -> Result<u64, String> {
        let c_path = CString::new(path)
            .map_err(|_| "SyzygyPath must not contain interior NUL bytes".to_owned())?;
        let mut state = fathom_state().lock().expect("Fathom state lock poisoned");

        let success = unsafe { tb_init(c_path.as_ptr()) };
        let largest = unsafe { TB_LARGEST };
        if !success || largest == 0 {
            unsafe {
                tb_free();
            }
            restore_previous_fathom(previous, &mut state)?;
            return Err(if !success {
                "SyzygyPath failed to initialize the approved Fathom backend".to_owned()
            } else {
                "SyzygyPath did not load any supported Syzygy tablebase files".to_owned()
            });
        }

        let service_id = NEXT_FATHOM_SERVICE_ID.fetch_add(1, Ordering::Relaxed);
        state.current_service_id = Some(service_id);
        state.current_path = Some(path.to_owned());
        Ok(service_id)
    }

    fn supports_root(&self, position: &Position) -> bool {
        fathom_supports_loaded_cardinality(position)
    }

    fn supports_non_root(&self, position: &Position) -> bool {
        position.halfmove_clock() == 0 && fathom_supports_loaded_cardinality(position)
    }

    fn probe_wdl(&self, position: &Position) -> Result<Option<WdlOutcome>, ProbeError> {
        let probe = unsafe {
            tb_probe_wdl_impl(
                position.occupancy_by(Color::White),
                position.occupancy_by(Color::Black),
                piece_mask(position, PieceType::King),
                piece_mask(position, PieceType::Queen),
                piece_mask(position, PieceType::Rook),
                piece_mask(position, PieceType::Bishop),
                piece_mask(position, PieceType::Knight),
                piece_mask(position, PieceType::Pawn),
                en_passant_square(position),
                position.side_to_move() == Color::White,
            )
        };

        if probe == TB_RESULT_FAILED {
            return Ok(None);
        }

        decode_wdl(probe).map(Some)
    }

    fn probe_root(
        &self,
        position: &Position,
        legal_moves: &MoveList,
    ) -> Result<Option<RootProbe>, ProbeError> {
        let _root_lock = fathom_root_probe_lock()
            .lock()
            .expect("Fathom root probe lock poisoned");
        let result = unsafe {
            tb_probe_root_impl(
                position.occupancy_by(Color::White),
                position.occupancy_by(Color::Black),
                piece_mask(position, PieceType::King),
                piece_mask(position, PieceType::Queen),
                piece_mask(position, PieceType::Rook),
                piece_mask(position, PieceType::Bishop),
                piece_mask(position, PieceType::Knight),
                piece_mask(position, PieceType::Pawn),
                position.halfmove_clock() as c_uint,
                en_passant_square(position),
                position.side_to_move() == Color::White,
                ptr::null_mut(),
            )
        };

        if result == TB_RESULT_FAILED
            || result == TB_RESULT_STALEMATE
            || result == TB_RESULT_CHECKMATE
        {
            return Ok(None);
        }

        let parsed = decode_root_move(result)?;
        let best_move = find_legal_move(legal_moves, parsed).ok_or_else(|| {
            ProbeError::new("Fathom root probe returned a move that is not legal in this position")
        })?;

        Ok(Some(RootProbe {
            best_move,
            wdl: decode_wdl(tb_get_wdl(result))?,
            dtz: Some(tb_get_dtz(result)),
        }))
    }
}

fn piece_mask(position: &Position, piece_type: PieceType) -> u64 {
    position.pieces(Color::White, piece_type) | position.pieces(Color::Black, piece_type)
}

fn fathom_supports_loaded_cardinality(position: &Position) -> bool {
    let largest = unsafe { TB_LARGEST };
    largest != 0 && position.occupancy().count_ones() <= largest.min(6)
}

fn en_passant_square(position: &Position) -> c_uint {
    position
        .en_passant()
        .map(|square| square.index() as c_uint)
        .unwrap_or(0)
}

fn decode_wdl(value: u32) -> Result<WdlOutcome, ProbeError> {
    match value {
        TB_WIN => Ok(WdlOutcome::Win),
        TB_CURSED_WIN => Ok(WdlOutcome::CursedWin),
        TB_DRAW => Ok(WdlOutcome::Draw),
        TB_BLESSED_LOSS => Ok(WdlOutcome::BlessedLoss),
        TB_LOSS => Ok(WdlOutcome::Loss),
        _ => Err(ProbeError::new(format!(
            "Fathom returned an unknown WDL value: {value}"
        ))),
    }
}

fn decode_root_move(result: u32) -> Result<ParsedMove, ProbeError> {
    let from = Square::try_from_index(tb_get_from(result) as u8)
        .ok_or_else(|| ProbeError::new("Fathom root probe returned an invalid from-square"))?;
    let to = Square::try_from_index(tb_get_to(result) as u8)
        .ok_or_else(|| ProbeError::new("Fathom root probe returned an invalid to-square"))?;
    let promotion = match tb_get_promotes(result) {
        0 => None,
        1 => Some(PieceType::Queen),
        2 => Some(PieceType::Rook),
        3 => Some(PieceType::Bishop),
        4 => Some(PieceType::Knight),
        other => {
            return Err(ProbeError::new(format!(
                "Fathom root probe returned an unknown promotion code: {other}"
            )));
        }
    };

    let mut value = String::with_capacity(if promotion.is_some() { 5 } else { 4 });
    value.push_str(&from.to_coord());
    value.push_str(&to.to_coord());
    if let Some(piece_type) = promotion {
        value.push(
            piece_type
                .promotion_char()
                .expect("promotion piece must have a promotion character"),
        );
    }
    ParsedMove::parse(&value)
        .map_err(|_| ProbeError::new("Fathom root probe returned an unparsable move"))
}

fn find_legal_move(legal_moves: &MoveList, target: ParsedMove) -> Option<Move> {
    for index in 0..legal_moves.len() {
        let mv = legal_moves.get(index);
        if mv.matches_parsed(target) {
            return Some(mv);
        }
    }
    None
}

fn tb_get_wdl(result: u32) -> u32 {
    (result & TB_RESULT_WDL_MASK) >> TB_RESULT_WDL_SHIFT
}

fn tb_get_to(result: u32) -> u32 {
    (result & TB_RESULT_TO_MASK) >> TB_RESULT_TO_SHIFT
}

fn tb_get_from(result: u32) -> u32 {
    (result & TB_RESULT_FROM_MASK) >> TB_RESULT_FROM_SHIFT
}

fn tb_get_promotes(result: u32) -> u32 {
    (result & TB_RESULT_PROMOTES_MASK) >> TB_RESULT_PROMOTES_SHIFT
}

fn tb_get_dtz(result: u32) -> u32 {
    (result & TB_RESULT_DTZ_MASK) >> TB_RESULT_DTZ_SHIFT
}

fn restore_previous_fathom(
    previous: Option<&FathomIdentity>,
    state: &mut FathomGlobalState,
) -> Result<(), String> {
    let Some(previous) = previous else {
        state.current_service_id = None;
        state.current_path = None;
        return Ok(());
    };

    let c_path = CString::new(previous.path.as_str())
        .map_err(|_| "previous SyzygyPath contained interior NUL bytes".to_owned())?;
    let restored = unsafe { tb_init(c_path.as_ptr()) };
    let largest = unsafe { TB_LARGEST };
    if !restored || largest == 0 {
        unsafe {
            tb_free();
        }
        state.current_service_id = None;
        state.current_path = None;
        return Err(format!(
            "failed to restore previously configured SyzygyPath '{}'",
            previous.path
        ));
    }

    state.current_service_id = Some(previous.service_id);
    state.current_path = Some(previous.path.clone());
    Ok(())
}

fn fathom_state() -> &'static Mutex<FathomGlobalState> {
    static STATE: OnceLock<Mutex<FathomGlobalState>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(FathomGlobalState::default()))
}

fn fathom_root_probe_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

static NEXT_FATHOM_SERVICE_ID: AtomicU64 = AtomicU64::new(1);

unsafe extern "C" {
    static mut TB_LARGEST: c_uint;

    fn tb_init(path: *const c_char) -> bool;
    fn tb_free();
    fn tb_probe_wdl_impl(
        white: u64,
        black: u64,
        kings: u64,
        queens: u64,
        rooks: u64,
        bishops: u64,
        knights: u64,
        pawns: u64,
        ep: c_uint,
        turn: bool,
    ) -> c_uint;
    fn tb_probe_root_impl(
        white: u64,
        black: u64,
        kings: u64,
        queens: u64,
        rooks: u64,
        bishops: u64,
        knights: u64,
        pawns: u64,
        rule50: c_uint,
        ep: c_uint,
        turn: bool,
        results: *mut c_uint,
    ) -> c_uint;
}

const TB_RESULT_STALEMATE: u32 = TB_DRAW;
const TB_RESULT_CHECKMATE: u32 = TB_WIN;

#[cfg(test)]
#[derive(Default)]
pub(crate) struct MockTablebaseBackend {
    root_probes: BTreeMap<String, (String, WdlOutcome, Option<u32>)>,
    wdl_probes: BTreeMap<String, WdlOutcome>,
}

#[cfg(test)]
impl MockTablebaseBackend {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn with_root_probe(
        mut self,
        fen: &str,
        best_move: &str,
        wdl: WdlOutcome,
        dtz: Option<u32>,
    ) -> Self {
        self.root_probes
            .insert(fen.to_owned(), (best_move.to_owned(), wdl, dtz));
        self
    }

    pub(crate) fn with_wdl_probe(mut self, fen: &str, wdl: WdlOutcome) -> Self {
        self.wdl_probes.insert(fen.to_owned(), wdl);
        self
    }
}

#[cfg(test)]
impl TablebaseBackend for MockTablebaseBackend {
    fn supports_root(&self, position: &Position) -> bool {
        self.root_probes.contains_key(&position.to_fen())
    }

    fn supports_non_root(&self, position: &Position) -> bool {
        self.wdl_probes.contains_key(&position.to_fen())
    }

    fn probe_wdl(&self, position: &Position) -> Result<Option<WdlOutcome>, ProbeError> {
        Ok(self.wdl_probes.get(&position.to_fen()).copied())
    }

    fn probe_root(
        &self,
        position: &Position,
        legal_moves: &MoveList,
    ) -> Result<Option<RootProbe>, ProbeError> {
        let Some((best_move, wdl, dtz)) = self.root_probes.get(&position.to_fen()) else {
            return Ok(None);
        };
        let parsed = ParsedMove::parse(best_move)
            .map_err(|_| ProbeError::new("mock tablebase move must parse"))?;
        let best_move = find_legal_move(legal_moves, parsed)
            .ok_or_else(|| ProbeError::new("mock tablebase move is not legal in this position"))?;
        Ok(Some(RootProbe {
            best_move,
            wdl: *wdl,
            dtz: *dtz,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Position;

    #[test]
    fn score_band_stays_below_mate_threshold() {
        assert!(TABLEBASE_SCORE_BAND < MATE_SCORE - super::super::root::MAX_PLY as i32);
        assert!(score_from_wdl(WdlOutcome::Win, 3) > 0);
        assert_eq!(score_from_wdl(WdlOutcome::CursedWin, 3), 0);
        assert_eq!(score_from_wdl(WdlOutcome::BlessedLoss, 3), 0);
        assert!(score_from_wdl(WdlOutcome::Loss, 3) < 0);
    }

    #[test]
    fn retained_scope_requires_no_castling_and_six_or_fewer_pieces() {
        let backend = Arc::new(
            MockTablebaseBackend::new()
                .with_wdl_probe("8/8/8/8/8/3Q4/2K5/k7 w - - 0 1", WdlOutcome::Win)
                .with_root_probe(
                    "8/8/8/8/8/3Q4/2K5/k7 w - - 0 1",
                    "d3d7",
                    WdlOutcome::Win,
                    Some(1),
                ),
        );
        let service = TablebaseService::from_backend_for_tests("/mock", backend);

        let eligible =
            Position::from_fen("8/8/8/8/8/3Q4/2K5/k7 w - - 0 1").expect("FEN parse must succeed");
        assert!(service.supports_root(&eligible));
        assert!(service.supports_non_root(&eligible));

        let castling = Position::startpos();
        assert!(!service.supports_root(&castling));
        assert!(!service.supports_non_root(&castling));

        let seven_pieces = Position::from_fen("8/8/8/8/8/3Q4/2K1NNBR/k7 w - - 0 1")
            .expect("FEN parse must succeed");
        assert!(!service.supports_root(&seven_pieces));
        assert!(!service.supports_non_root(&seven_pieces));
    }

    #[test]
    fn non_root_probe_scope_rejects_nonzero_halfmove_clock() {
        let service = TablebaseService {
            path: "/fathom".to_owned(),
            backend: BackendKind::Fathom(FathomBackend { service_id: 1 }),
        };
        let position =
            Position::from_fen("8/8/8/8/8/3Q4/2K5/k7 w - - 7 1").expect("FEN parse must succeed");
        assert!(!service.supports_non_root(&position));
    }
}
