use std::{
    ffi::CString,
    fmt,
    os::raw::{c_char, c_uint},
    ptr,
    sync::{
        Arc, Mutex, OnceLock, RwLock,
        atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering},
    },
};

use crate::core::{Color, Move, MoveList, ParsedMove, PieceType, Position, Square};

#[cfg(test)]
use std::collections::BTreeMap;

use super::root::MATE_SCORE;

pub(crate) const TABLEBASE_SCORE_BAND: i32 = 20_000;
pub(crate) const MAX_SYZYGY_PIECES: u8 = 7;

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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct TablebaseProbeStats {
    pub(crate) root_attempts: u64,
    pub(crate) wdl_attempts: u64,
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) errors: u64,
}

impl TablebaseProbeStats {
    pub(crate) fn delta_since(self, earlier: Self) -> Self {
        Self {
            root_attempts: self.root_attempts.saturating_sub(earlier.root_attempts),
            wdl_attempts: self.wdl_attempts.saturating_sub(earlier.wdl_attempts),
            hits: self.hits.saturating_sub(earlier.hits),
            misses: self.misses.saturating_sub(earlier.misses),
            errors: self.errors.saturating_sub(earlier.errors),
        }
    }

    pub(crate) const fn attempts(self) -> u64 {
        self.root_attempts.saturating_add(self.wdl_attempts)
    }
}

#[derive(Default)]
struct ProbeCounters {
    root_attempts: AtomicU64,
    wdl_attempts: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    errors: AtomicU64,
    last_error: Mutex<Option<String>>,
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
    probe_limit: AtomicU8,
    rule50_enabled: AtomicBool,
    counters: ProbeCounters,
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
            backend: BackendKind::Fathom(FathomBackend {
                service_id: service_id.id,
                cardinality: service_id.cardinality,
            }),
            probe_limit: AtomicU8::new(MAX_SYZYGY_PIECES),
            rule50_enabled: AtomicBool::new(true),
            counters: ProbeCounters::default(),
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
            probe_limit: AtomicU8::new(MAX_SYZYGY_PIECES),
            rule50_enabled: AtomicBool::new(true),
            counters: ProbeCounters::default(),
        })
    }

    pub(crate) fn set_probe_limit(&self, limit: u8) {
        self.probe_limit
            .store(limit.min(MAX_SYZYGY_PIECES), Ordering::Relaxed);
    }

    pub(crate) fn probe_limit(&self) -> u8 {
        self.probe_limit.load(Ordering::Relaxed)
    }

    pub(crate) fn set_rule50_enabled(&self, enabled: bool) {
        self.rule50_enabled.store(enabled, Ordering::Relaxed);
    }

    pub(crate) fn rule50_enabled(&self) -> bool {
        self.rule50_enabled.load(Ordering::Relaxed)
    }

    pub(crate) fn loaded_cardinality(&self) -> Option<u8> {
        match &self.backend {
            BackendKind::Fathom(backend) => Some(backend.cardinality),
            #[cfg(test)]
            BackendKind::Mock(_) => None,
        }
    }

    pub(crate) fn probe_stats(&self) -> TablebaseProbeStats {
        TablebaseProbeStats {
            root_attempts: self.counters.root_attempts.load(Ordering::Relaxed),
            wdl_attempts: self.counters.wdl_attempts.load(Ordering::Relaxed),
            hits: self.counters.hits.load(Ordering::Relaxed),
            misses: self.counters.misses.load(Ordering::Relaxed),
            errors: self.counters.errors.load(Ordering::Relaxed),
        }
    }

    pub(crate) fn last_probe_error(&self) -> Option<String> {
        self.counters
            .last_error
            .lock()
            .expect("tablebase diagnostic lock poisoned")
            .clone()
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
            && (!self.rule50_enabled() || position.halfmove_clock() == 0)
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
        self.counters.wdl_attempts.fetch_add(1, Ordering::Relaxed);
        let result = match &self.backend {
            BackendKind::Fathom(backend) => backend.probe_wdl(position),
            #[cfg(test)]
            BackendKind::Mock(backend) => backend.probe_wdl(position),
        }
        .map(|outcome| outcome.map(|outcome| self.apply_rule50_policy(outcome)));
        self.record_probe_result(&result);
        result
    }

    pub(crate) fn probe_root(
        &self,
        position: &Position,
        legal_moves: &MoveList,
    ) -> Result<Option<RootProbe>, ProbeError> {
        if !self.supports_root(position) {
            return Ok(None);
        }
        self.counters.root_attempts.fetch_add(1, Ordering::Relaxed);
        let result = match &self.backend {
            BackendKind::Fathom(backend) => {
                backend.probe_root(position, legal_moves, self.rule50_enabled())
            }
            #[cfg(test)]
            BackendKind::Mock(backend) => backend.probe_root(position, legal_moves),
        }
        .map(|probe| {
            probe.map(|mut probe| {
                probe.wdl = self.apply_rule50_policy(probe.wdl);
                probe
            })
        });
        let result = result.and_then(|probe| {
            if let Some(root_probe) = probe
                && !move_list_contains(legal_moves, root_probe.best_move)
            {
                return Err(ProbeError::new(
                    "tablebase root probe returned a move outside the allowed root move list",
                ));
            }
            Ok(probe)
        });
        self.record_probe_result(&result);
        result
    }

    fn is_within_retained_scope(&self, position: &Position) -> bool {
        let pieces = position.occupancy().count_ones() as u8;
        position_is_within_retained_scope(position) && pieces <= self.probe_limit()
    }

    fn apply_rule50_policy(&self, outcome: WdlOutcome) -> WdlOutcome {
        if self.rule50_enabled() {
            outcome
        } else {
            match outcome {
                WdlOutcome::CursedWin => WdlOutcome::Win,
                WdlOutcome::BlessedLoss => WdlOutcome::Loss,
                other => other,
            }
        }
    }

    fn record_probe_result<T>(&self, result: &Result<Option<T>, ProbeError>) {
        match result {
            Ok(Some(_)) => {
                self.counters.hits.fetch_add(1, Ordering::Relaxed);
            }
            Ok(None) => {
                self.counters.misses.fetch_add(1, Ordering::Relaxed);
            }
            Err(error) => {
                self.counters.errors.fetch_add(1, Ordering::Relaxed);
                *self
                    .counters
                    .last_error
                    .lock()
                    .expect("tablebase diagnostic lock poisoned") = Some(error.to_string());
            }
        }
    }

    fn fathom_identity(&self) -> Option<FathomIdentity> {
        match &self.backend {
            BackendKind::Fathom(FathomBackend { service_id, .. }) => Some(FathomIdentity {
                service_id: *service_id,
                path: self.path.clone(),
            }),
            #[cfg(test)]
            BackendKind::Mock(_) => None,
        }
    }
}

pub(crate) fn position_is_within_retained_scope(position: &Position) -> bool {
    position.castling_rights().is_empty()
        && position.occupancy().count_ones() <= u32::from(MAX_SYZYGY_PIECES)
}

impl Drop for TablebaseService {
    fn drop(&mut self) {
        let service_id = match &self.backend {
            BackendKind::Fathom(FathomBackend { service_id, .. }) => *service_id,
            #[cfg(test)]
            BackendKind::Mock(_) => return,
        };

        let mut state = fathom_state().write().expect("Fathom state lock poisoned");
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
    cardinality: u8,
}

#[derive(Clone)]
struct FathomIdentity {
    service_id: u64,
    path: String,
}

struct FathomInitialization {
    id: u64,
    cardinality: u8,
}

#[derive(Default)]
struct FathomGlobalState {
    current_service_id: Option<u64>,
    current_path: Option<String>,
}

impl FathomBackend {
    fn initialize(
        path: &str,
        previous: Option<&FathomIdentity>,
    ) -> Result<FathomInitialization, String> {
        let c_path = CString::new(path)
            .map_err(|_| "SyzygyPath must not contain interior NUL bytes".to_owned())?;
        let mut state = fathom_state().write().expect("Fathom state lock poisoned");

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
        Ok(FathomInitialization {
            id: service_id,
            cardinality: largest.min(u32::from(MAX_SYZYGY_PIECES)) as u8,
        })
    }

    fn supports_root(&self, position: &Position) -> bool {
        self.is_current() && position.occupancy().count_ones() <= u32::from(self.cardinality)
    }

    fn supports_non_root(&self, position: &Position) -> bool {
        self.supports_root(position)
    }

    fn probe_wdl(&self, position: &Position) -> Result<Option<WdlOutcome>, ProbeError> {
        let _state = self.lock_current()?;
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
            return Err(ProbeError::new(
                "Fathom WDL probe failed for a supported-cardinality position",
            ));
        }

        decode_wdl(probe).map(Some)
    }

    fn probe_root(
        &self,
        position: &Position,
        legal_moves: &MoveList,
        rule50_enabled: bool,
    ) -> Result<Option<RootProbe>, ProbeError> {
        let _root_lock = fathom_root_probe_lock()
            .lock()
            .expect("Fathom root probe lock poisoned");
        let _state = self.lock_current()?;
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
                if rule50_enabled {
                    position.halfmove_clock() as c_uint
                } else {
                    0
                },
                en_passant_square(position),
                position.side_to_move() == Color::White,
                ptr::null_mut(),
            )
        };

        if result == TB_RESULT_FAILED {
            return Err(ProbeError::new(
                "Fathom root DTZ probe failed for a supported-cardinality position",
            ));
        }
        if result == TB_RESULT_STALEMATE || result == TB_RESULT_CHECKMATE {
            return Err(ProbeError::new(
                "Fathom root DTZ probe reported a terminal position despite legal root moves",
            ));
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

    fn is_current(&self) -> bool {
        fathom_state()
            .read()
            .expect("Fathom state lock poisoned")
            .current_service_id
            == Some(self.service_id)
    }

    fn lock_current(
        &self,
    ) -> Result<std::sync::RwLockReadGuard<'static, FathomGlobalState>, ProbeError> {
        let state = fathom_state().read().expect("Fathom state lock poisoned");
        if state.current_service_id != Some(self.service_id) {
            return Err(ProbeError::new(
                "Syzygy service became stale during tablebase reconfiguration",
            ));
        }
        Ok(state)
    }
}

fn piece_mask(position: &Position, piece_type: PieceType) -> u64 {
    position.pieces(Color::White, piece_type) | position.pieces(Color::Black, piece_type)
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

fn fathom_state() -> &'static RwLock<FathomGlobalState> {
    static STATE: OnceLock<RwLock<FathomGlobalState>> = OnceLock::new();
    STATE.get_or_init(|| RwLock::new(FathomGlobalState::default()))
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

    struct MissBackend;

    impl TablebaseBackend for MissBackend {
        fn supports_root(&self, _position: &Position) -> bool {
            false
        }

        fn supports_non_root(&self, _position: &Position) -> bool {
            true
        }

        fn probe_wdl(&self, _position: &Position) -> Result<Option<WdlOutcome>, ProbeError> {
            Ok(None)
        }

        fn probe_root(
            &self,
            _position: &Position,
            _legal_moves: &MoveList,
        ) -> Result<Option<RootProbe>, ProbeError> {
            Ok(None)
        }
    }

    struct ErrorBackend;

    impl TablebaseBackend for ErrorBackend {
        fn supports_root(&self, _position: &Position) -> bool {
            false
        }

        fn supports_non_root(&self, _position: &Position) -> bool {
            true
        }

        fn probe_wdl(&self, _position: &Position) -> Result<Option<WdlOutcome>, ProbeError> {
            Err(ProbeError::new("intentional mock probe failure"))
        }

        fn probe_root(
            &self,
            _position: &Position,
            _legal_moves: &MoveList,
        ) -> Result<Option<RootProbe>, ProbeError> {
            Ok(None)
        }
    }

    #[test]
    fn score_band_stays_below_mate_threshold() {
        assert!(TABLEBASE_SCORE_BAND < MATE_SCORE - super::super::root::MAX_PLY as i32);
        assert!(score_from_wdl(WdlOutcome::Win, 3) > 0);
        assert_eq!(score_from_wdl(WdlOutcome::CursedWin, 3), 0);
        assert_eq!(score_from_wdl(WdlOutcome::BlessedLoss, 3), 0);
        assert!(score_from_wdl(WdlOutcome::Loss, 3) < 0);
    }

    #[test]
    fn retained_scope_requires_no_castling_and_seven_or_fewer_pieces() {
        const SEVEN_PIECES: &str = "8/8/8/8/8/3Q4/2K1NNBR/k7 w - - 0 1";
        let backend = Arc::new(
            MockTablebaseBackend::new()
                .with_wdl_probe("8/8/8/8/8/3Q4/2K5/k7 w - - 0 1", WdlOutcome::Win)
                .with_root_probe(
                    "8/8/8/8/8/3Q4/2K5/k7 w - - 0 1",
                    "d3d7",
                    WdlOutcome::Win,
                    Some(1),
                )
                .with_wdl_probe(SEVEN_PIECES, WdlOutcome::Win)
                .with_root_probe(SEVEN_PIECES, "d3d7", WdlOutcome::Win, Some(1)),
        );
        let service = TablebaseService::from_backend_for_tests("/mock", backend);

        let eligible =
            Position::from_fen("8/8/8/8/8/3Q4/2K5/k7 w - - 0 1").expect("FEN parse must succeed");
        assert!(service.supports_root(&eligible));
        assert!(service.supports_non_root(&eligible));

        let castling = Position::startpos();
        assert!(!service.supports_root(&castling));
        assert!(!service.supports_non_root(&castling));

        let seven_pieces = Position::from_fen(SEVEN_PIECES).expect("FEN parse must succeed");
        assert!(service.supports_root(&seven_pieces));
        assert!(service.supports_non_root(&seven_pieces));

        let eight_pieces = Position::from_fen("8/8/8/8/8/3Q4/2K1NNBR/k6P w - - 0 1")
            .expect("FEN parse must succeed");
        assert!(!service.supports_root(&eight_pieces));
        assert!(!service.supports_non_root(&eight_pieces));
    }

    #[test]
    fn non_root_probe_scope_rejects_nonzero_halfmove_clock() {
        let fen = "8/8/8/8/8/3Q4/2K5/k7 w - - 7 1";
        let service = TablebaseService::from_backend_for_tests(
            "/mock",
            Arc::new(MockTablebaseBackend::new().with_wdl_probe(fen, WdlOutcome::CursedWin)),
        );
        let position = Position::from_fen(fen).expect("FEN parse must succeed");
        assert!(!service.supports_non_root(&position));

        service.set_rule50_enabled(false);
        assert!(service.supports_non_root(&position));
        assert_eq!(
            service.probe_wdl(&position).expect("probe must succeed"),
            Some(WdlOutcome::Win),
            "ignoring the 50-move rule must collapse cursed wins into unconditional wins"
        );
    }

    #[test]
    fn probe_limit_can_reduce_or_disable_tablebase_scope() {
        let fen = "8/8/8/8/8/3Q4/2K5/k7 w - - 0 1";
        let position = Position::from_fen(fen).expect("FEN parse must succeed");
        let service = TablebaseService::from_backend_for_tests(
            "/mock",
            Arc::new(MockTablebaseBackend::new().with_wdl_probe(fen, WdlOutcome::Win)),
        );

        assert!(service.supports_non_root(&position));
        service.set_probe_limit(2);
        assert!(!service.supports_non_root(&position));
        service.set_probe_limit(0);
        assert!(!service.supports_non_root(&position));
        service.set_probe_limit(99);
        assert_eq!(service.probe_limit(), MAX_SYZYGY_PIECES);
        assert!(service.supports_non_root(&position));
    }

    #[test]
    fn probe_diagnostics_distinguish_hits_misses_and_errors() {
        let fen = "8/8/8/8/8/3Q4/2K5/k7 w - - 0 1";
        let position = Position::from_fen(fen).expect("FEN parse must succeed");

        let hit = TablebaseService::from_backend_for_tests(
            "/hit",
            Arc::new(MockTablebaseBackend::new().with_wdl_probe(fen, WdlOutcome::Draw)),
        );
        assert_eq!(
            hit.probe_wdl(&position).expect("probe must succeed"),
            Some(WdlOutcome::Draw)
        );
        assert_eq!(
            hit.probe_stats(),
            TablebaseProbeStats {
                wdl_attempts: 1,
                hits: 1,
                ..TablebaseProbeStats::default()
            }
        );

        let miss = TablebaseService::from_backend_for_tests("/miss", Arc::new(MissBackend));
        assert_eq!(miss.probe_wdl(&position).expect("probe must succeed"), None);
        assert_eq!(miss.probe_stats().misses, 1);

        let error = TablebaseService::from_backend_for_tests("/error", Arc::new(ErrorBackend));
        assert!(error.probe_wdl(&position).is_err());
        assert_eq!(error.probe_stats().errors, 1);
        assert_eq!(
            error.last_probe_error().as_deref(),
            Some("intentional mock probe failure")
        );

        let delta = error
            .probe_stats()
            .delta_since(TablebaseProbeStats::default());
        assert_eq!(delta.attempts(), 1);
    }

    #[test]
    fn root_result_decoder_preserves_move_promotion_wdl_and_dtz() {
        let from = Square::from_coord_text("a7")
            .expect("square must parse")
            .index() as u32;
        let to = Square::from_coord_text("a8")
            .expect("square must parse")
            .index() as u32;
        let result = TB_WIN
            | (to << TB_RESULT_TO_SHIFT)
            | (from << TB_RESULT_FROM_SHIFT)
            | (1 << TB_RESULT_PROMOTES_SHIFT)
            | (37 << TB_RESULT_DTZ_SHIFT);

        assert_eq!(decode_wdl(tb_get_wdl(result)), Ok(WdlOutcome::Win));
        assert_eq!(
            decode_root_move(result),
            ParsedMove::parse("a7a8q").map_err(|_| ProbeError::new("unexpected parse error"))
        );
        assert_eq!(tb_get_dtz(result), 37);
    }

    #[test]
    #[ignore = "requires VOLKRIX_SYZYGY_PATH with real Syzygy files"]
    fn real_fathom_wdl_probes_survive_concurrent_reconfiguration() {
        use std::{
            sync::{Arc, Barrier},
            thread,
        };

        let path = std::env::var("VOLKRIX_SYZYGY_PATH")
            .expect("VOLKRIX_SYZYGY_PATH must be set for real tablebase tests");
        let position =
            Position::from_fen("8/8/8/8/8/3Q4/2K5/k7 w - - 0 1").expect("FEN parse must succeed");
        let original = TablebaseService::open_syzygy_path(&path, None)
            .expect("original Fathom service must initialize");
        let barrier = Arc::new(Barrier::new(5));
        let mut workers = Vec::new();
        for _ in 0..4 {
            let service = Arc::clone(&original);
            let barrier = Arc::clone(&barrier);
            let position = position.clone();
            workers.push(thread::spawn(move || {
                barrier.wait();
                for _ in 0..100 {
                    match service.probe_wdl(&position) {
                        Ok(Some(_)) | Ok(None) => {}
                        Err(error) => assert!(
                            error
                                .to_string()
                                .contains("stale during tablebase reconfiguration"),
                            "unexpected concurrent probe error: {error}"
                        ),
                    }
                }
            }));
        }

        barrier.wait();
        let replacement = TablebaseService::open_syzygy_path(&path, Some(&original))
            .expect("replacement Fathom service must initialize");
        for worker in workers {
            worker.join().expect("probe worker must not panic");
        }

        assert!(!original.supports_non_root(&position));
        assert!(replacement.supports_non_root(&position));
        assert!(
            replacement
                .probe_wdl(&position)
                .expect("replacement probe must succeed")
                .is_some()
        );
    }
}
