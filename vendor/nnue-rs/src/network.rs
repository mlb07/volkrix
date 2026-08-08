use std::cell::RefCell;
use std::fs::File;
use std::io::{self, BufReader, Read};

use crate::error::{Error, Result};
use crate::feature::{active_indices, make_index, Arch};
use crate::leb128;
use crate::threats;
use crate::types::{Board, Color, Piece, PieceKind};

#[derive(Default)]
struct Scratch {
    white: Vec<usize>,
    input: Vec<u8>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

const VERSION_HALFKA: u32 = 0x7AF32F20;
const VERSION_HALFKP: u32 = 0x7AF32F16;
// SFNNv16 introduced PP_3Wide inputs. Keep this as an explicit rejection
// marker: silently treating the format as an older HalfKA layout would consume
// different feature blocks and can yield plausible but incorrect evaluations.
const VERSION_SFNNV16_PP_3WIDE: u32 = 0x6A448AFA;
const FEATURE_HASH_THREATS: u32 = 0x8f234cb8;
const FEATURE_HASH_HM: u32 = 0x7f234cb8;
const FEATURE_HASH_V2: u32 = 0x5f234cb8;
const FEATURE_HASH_HALFKP: u32 = 0x5d69d7b8;
const NETWORK_HASH_HALFKP: u32 = 0x63337156;
const PSQT_BUCKETS: usize = 8;
const LAYER_STACKS: usize = 8;
const WEIGHT_SCALE_BITS: i32 = 6;
const OUTPUT_SCALE: i32 = 16;
const MAX_CHANGED: usize = 4;
// Network descriptions are normally only a few dozen bytes. Keep enough room
// for unusually descriptive, otherwise-valid generators while preventing an
// untrusted four-byte header from requesting a multi-gigabyte allocation.
const MAX_DESCRIPTION_LEN: usize = 1024 * 1024;

/// Compact, caller-supplied description of the pieces changed by one move.
///
/// Supplying this to [`Network::update_delta`] avoids reconstructing and
/// comparing two complete boards. `parent_kings` contains the king squares
/// before the move; the child king squares are read from the child board.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoardDelta {
    removed: [(u8, Piece); MAX_CHANGED],
    added: [(u8, Piece); MAX_CHANGED],
    removed_len: u8,
    added_len: u8,
    parent_kings: [u8; 2],
}

impl BoardDelta {
    /// Construct an empty delta for a board with the given parent king squares.
    pub fn new(parent_kings: [u8; 2]) -> Self {
        let empty = (0, Piece::new(Color::White, PieceKind::Pawn));
        Self {
            removed: [empty; MAX_CHANGED],
            added: [empty; MAX_CHANGED],
            removed_len: 0,
            added_len: 0,
            parent_kings,
        }
    }

    /// Record a piece removed by the move.
    pub fn remove(&mut self, square: u8, piece: Piece) {
        let index = self.removed_len as usize;
        assert!(index < MAX_CHANGED, "NNUE delta removed-piece overflow");
        self.removed[index] = (square, piece);
        self.removed_len += 1;
    }

    /// Record a piece added by the move.
    pub fn add(&mut self, square: u8, piece: Piece) {
        let index = self.added_len as usize;
        assert!(index < MAX_CHANGED, "NNUE delta added-piece overflow");
        self.added[index] = (square, piece);
        self.added_len += 1;
    }

    fn removed(&self) -> &[(u8, Piece)] {
        &self.removed[..self.removed_len as usize]
    }

    fn added(&self) -> &[(u8, Piece)] {
        &self.added[..self.added_len as usize]
    }
}

/// The two independently scaled components produced by a Stockfish network.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Evaluation {
    pub psqt: i32,
    pub positional: i32,
}

impl Evaluation {
    /// Preserve the legacy nnue-rs total, including its component-wise integer
    /// division behavior.
    pub fn total(self) -> i32 {
        self.psqt + self.positional
    }
}

const HM_L2: usize = 15;
const HM_L3: usize = 32;
const HM_FC0_OUT: usize = HM_L2 + 1;
const HM_FC1_IN: usize = HM_L2 * 2;

const V2_FC0_OUT: usize = 16;
const V2_FC1_OUT: usize = 32;
const FC1_PAD: usize = 32;

const HALFKP_HALF_DIMENSIONS: usize = 256;
const HALFKP_FC0_OUT: usize = 32;
const HALFKP_FC1_OUT: usize = 32;

struct Bucket {
    fc0_bias: Vec<i32>,
    fc0_weight: Vec<i8>,
    fc1_bias: Vec<i32>,
    fc1_weight: Vec<i8>,
    fc2_bias: i32,
    fc2_weight: Vec<i8>,
}

enum Layers {
    Sfnnv10 {
        ft_psqt: Vec<i32>,
        threat_weight: Vec<i8>,
        threat_psqt: Vec<i32>,
        buckets: Vec<Bucket>,
    },
    HalfKAv2Hm {
        ft_psqt: Vec<i32>,
        buckets: Vec<Bucket>,
    },
    HalfKAv2 {
        ft_psqt: Vec<i32>,
        buckets: Vec<Bucket>,
    },
    HalfKP(Bucket),
}

/// A loaded NNUE network.
///
/// Load one with [`Network::from_file`], [`Network::from_bytes`] or
/// [`Network::from_reader`], then evaluate positions with [`Network::evaluate`],
/// [`Network::evaluate_fen`], or the incremental accumulator API. The
/// architecture is detected automatically from the file header; see
/// [`Network::arch`].
pub struct Network {
    arch: Arch,
    l1: usize,
    ft_bias: Vec<i16>,
    ft_weight: Vec<i16>,
    layers: Layers,
}

/// A pair of feature-transformer accumulators (one per side-to-move
/// perspective) plus the piece count for the position they describe.
///
/// Obtain one with [`Network::accumulator`] or [`Network::empty_accumulator`],
/// keep it alongside your board, and advance it incrementally with
/// [`Network::update`] as moves are made. Evaluate it with
/// [`Network::evaluate_accumulator`].
#[derive(Clone)]
pub struct Accumulator {
    white: Vec<i16>,
    black: Vec<i16>,
    psqt_white: [i32; PSQT_BUCKETS],
    psqt_black: [i32; PSQT_BUCKETS],
    piece_count: usize,
    threat_pairs: Vec<u32>,
}

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32_raw(reader: &mut impl Read, count: usize) -> io::Result<Vec<i32>> {
    let mut out = Vec::with_capacity(count);
    let mut buf = [0u8; 4];
    for _ in 0..count {
        reader.read_exact(&mut buf)?;
        out.push(i32::from_le_bytes(buf));
    }
    Ok(out)
}

fn read_i16_raw(reader: &mut impl Read, count: usize) -> io::Result<Vec<i16>> {
    let mut out = Vec::with_capacity(count);
    let mut buf = [0u8; 2];
    for _ in 0..count {
        reader.read_exact(&mut buf)?;
        out.push(i16::from_le_bytes(buf));
    }
    Ok(out)
}

fn read_i8_raw(reader: &mut impl Read, count: usize) -> io::Result<Vec<i8>> {
    let mut bytes = vec![0u8; count];
    reader.read_exact(&mut bytes)?;
    Ok(bytes.into_iter().map(|b| b as i8).collect())
}

fn ensure_eof(reader: &mut impl Read) -> Result<()> {
    let mut trailing = [0u8; 1];
    loop {
        match reader.read(&mut trailing) {
            Ok(0) => return Ok(()),
            Ok(_) => return Err(Error::InvalidData("trailing data after network")),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error.into()),
        }
    }
}

fn count_pieces(board: &impl Board) -> usize {
    let mut n = 0;
    board.for_each_piece(&mut |_, _| n += 1);
    n
}

fn diff_boards(
    parent: &impl Board,
    child: &impl Board,
    removed: &mut [(u8, Piece)],
    added: &mut [(u8, Piece)],
) -> (usize, usize) {
    let mut p = [None; 64];
    let mut c = [None; 64];
    parent.for_each_piece(&mut |sq, piece| p[sq as usize] = Some(piece));
    child.for_each_piece(&mut |sq, piece| c[sq as usize] = Some(piece));
    let mut nr = 0;
    let mut na = 0;
    for sq in 0..64u8 {
        let (a, b) = (p[sq as usize], c[sq as usize]);
        if a != b {
            if let Some(piece) = a {
                if nr < removed.len() {
                    removed[nr] = (sq, piece);
                }
                nr += 1;
            }
            if let Some(piece) = b {
                if na < added.len() {
                    added[na] = (sq, piece);
                }
                na += 1;
            }
        }
    }
    (nr.min(removed.len()), na.min(added.len()))
}

impl Network {
    /// Load a network from a `.nnue` file on disk.
    pub fn from_file(path: &str) -> Result<Self> {
        Self::from_reader(&mut BufReader::new(File::open(path)?))
    }

    /// Load a network from an in-memory byte slice (e.g. an embedded net).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_reader(&mut &bytes[..])
    }

    /// Load a network from any reader.
    ///
    /// The architecture is selected from the file's version header.
    pub fn from_reader(reader: &mut impl Read) -> Result<Self> {
        let version = read_u32(reader)?;
        let architecture_hash = read_u32(reader)?;
        let desc_len = read_u32(reader)? as usize;
        if desc_len > MAX_DESCRIPTION_LEN {
            return Err(Error::InvalidData("network description is too large"));
        }
        let mut desc = vec![0u8; desc_len];
        reader.read_exact(&mut desc)?;

        let network = match version {
            VERSION_HALFKA => Self::load_halfka(reader, architecture_hash),
            VERSION_HALFKP => Self::load_halfkp(reader, architecture_hash),
            VERSION_SFNNV16_PP_3WIDE => Err(Error::UnsupportedPp3Wide(version)),
            other => Err(Error::UnsupportedVersion(other)),
        }?;

        ensure_eof(reader)?;

        Ok(network)
    }

    /// The feature-set architecture this network was loaded as.
    pub fn arch(&self) -> Arch {
        self.arch
    }

    fn detect_halfka(ft_hash: u32) -> Result<(Arch, usize)> {
        for &(arch, hash) in &[
            (Arch::Sfnnv10, FEATURE_HASH_THREATS),
            (Arch::HalfKAv2Hm, FEATURE_HASH_HM),
            (Arch::HalfKAv2, FEATURE_HASH_V2),
        ] {
            let x = ft_hash ^ hash;
            if x != 0 && x % 2 == 0 {
                let l1 = (x / 2) as usize;
                if l1 >= 16 && l1 <= 4096 && l1 % 16 == 0 {
                    return Ok((arch, l1));
                }
            }
        }
        Err(Error::InvalidData("bad feature-transformer width"))
    }

    fn read_bucket(
        reader: &mut impl Read,
        fc0_out: usize,
        fc0_in: usize,
        fc1_out: usize,
    ) -> Result<Bucket> {
        Ok(Bucket {
            fc0_bias: read_i32_raw(reader, fc0_out)?,
            fc0_weight: read_i8_raw(reader, fc0_out * fc0_in)?,
            fc1_bias: read_i32_raw(reader, fc1_out)?,
            fc1_weight: read_i8_raw(reader, fc1_out * FC1_PAD)?,
            fc2_bias: read_i32_raw(reader, 1)?[0],
            fc2_weight: read_i8_raw(reader, fc1_out)?,
        })
    }

    fn load_halfka(reader: &mut impl Read, architecture_hash: u32) -> Result<Self> {
        let ft_hash = read_u32(reader)?;
        let (arch, l1) = Self::detect_halfka(ft_hash)?;
        let input_dims = arch.input_dimensions();
        let network_hash = architecture_hash ^ ft_hash;

        if arch == Arch::Sfnnv10 {
            return Self::load_sfnnv10(reader, l1, input_dims, network_hash);
        }

        let mirrored = arch == Arch::HalfKAv2Hm;

        let (ft_bias, ft_weight, ft_psqt) = if mirrored {
            let mut bias = leb128::read_i16(reader, l1)?;
            let mut weight = leb128::read_i16(reader, l1 * input_dims)?;
            let psqt = leb128::read_i32(reader, PSQT_BUCKETS * input_dims)?;
            for b in bias.iter_mut() {
                *b = b.wrapping_mul(2);
            }
            for w in weight.iter_mut() {
                *w = w.wrapping_mul(2);
            }
            (bias, weight, psqt)
        } else {
            let bias = read_i16_raw(reader, l1)?;
            let weight = read_i16_raw(reader, l1 * input_dims)?;
            let psqt = read_i32_raw(reader, PSQT_BUCKETS * input_dims)?;
            (bias, weight, psqt)
        };

        let (fc0_out, fc0_in, fc1_out) = if mirrored {
            (HM_FC0_OUT, l1, HM_L3)
        } else {
            (V2_FC0_OUT, 2 * l1, V2_FC1_OUT)
        };

        let mut buckets = Vec::with_capacity(LAYER_STACKS);
        for _ in 0..LAYER_STACKS {
            let bucket_hash = read_u32(reader)?;
            if bucket_hash != network_hash {
                return Err(Error::InvalidData(
                    "architecture and network bucket hashes do not match",
                ));
            }
            buckets.push(Self::read_bucket(reader, fc0_out, fc0_in, fc1_out)?);
        }

        let layers = if mirrored {
            Layers::HalfKAv2Hm { ft_psqt, buckets }
        } else {
            Layers::HalfKAv2 { ft_psqt, buckets }
        };

        Ok(Self {
            arch,
            l1,
            ft_bias,
            ft_weight,
            layers,
        })
    }

    fn load_sfnnv10(
        reader: &mut impl Read,
        l1: usize,
        input_dims: usize,
        network_hash: u32,
    ) -> Result<Self> {
        let ft_bias = leb128::read_i16(reader, l1)?;
        let threat_weight = read_i8_raw(reader, threats::THREAT_DIMENSIONS * l1)?;
        let ft_weight = leb128::read_i16(reader, l1 * input_dims)?;
        let mut threat_psqt = leb128::read_i32(
            reader,
            (threats::THREAT_DIMENSIONS + input_dims) * PSQT_BUCKETS,
        )?;
        let ft_psqt = threat_psqt.split_off(threats::THREAT_DIMENSIONS * PSQT_BUCKETS);

        let mut buckets = Vec::with_capacity(LAYER_STACKS);
        for _ in 0..LAYER_STACKS {
            let bucket_hash = read_u32(reader)?;
            if bucket_hash != network_hash {
                return Err(Error::InvalidData(
                    "architecture and network bucket hashes do not match",
                ));
            }
            buckets.push(Self::read_bucket(reader, HM_FC0_OUT, l1, HM_L3)?);
        }

        Ok(Self {
            arch: Arch::Sfnnv10,
            l1,
            ft_bias,
            ft_weight,
            layers: Layers::Sfnnv10 {
                ft_psqt,
                threat_weight,
                threat_psqt,
                buckets,
            },
        })
    }

    fn load_halfkp(reader: &mut impl Read, architecture_hash: u32) -> Result<Self> {
        let arch = Arch::HalfKP;
        let l1 = HALFKP_HALF_DIMENSIONS;
        let input_dims = arch.input_dimensions();

        if architecture_hash != FEATURE_HASH_HALFKP ^ NETWORK_HASH_HALFKP {
            return Err(Error::InvalidData("incompatible HalfKP architecture hash"));
        }

        let ft_hash = read_u32(reader)?;
        if ft_hash != FEATURE_HASH_HALFKP {
            return Err(Error::InvalidData(
                "incompatible HalfKP feature-transformer hash",
            ));
        }
        let ft_bias = read_i16_raw(reader, l1)?;
        let ft_weight = read_i16_raw(reader, l1 * input_dims)?;

        let network_hash = read_u32(reader)?;
        if network_hash != NETWORK_HASH_HALFKP {
            return Err(Error::InvalidData("incompatible HalfKP network hash"));
        }
        let bucket = Self::read_bucket(reader, HALFKP_FC0_OUT, 2 * l1, HALFKP_FC1_OUT)?;

        Ok(Self {
            arch,
            l1,
            ft_bias,
            ft_weight,
            layers: Layers::HalfKP(bucket),
        })
    }

    fn ft_psqt(&self) -> &[i32] {
        match &self.layers {
            Layers::Sfnnv10 { ft_psqt, .. }
            | Layers::HalfKAv2Hm { ft_psqt, .. }
            | Layers::HalfKAv2 { ft_psqt, .. } => ft_psqt,
            Layers::HalfKP(_) => &[],
        }
    }

    fn threat_tables(&self) -> (&[i8], &[i32]) {
        match &self.layers {
            Layers::Sfnnv10 {
                threat_weight,
                threat_psqt,
                ..
            } => (threat_weight, threat_psqt),
            _ => (&[], &[]),
        }
    }

    fn accumulate(&self, indices: &[usize], acc: &mut [i16], psqt: &mut [i32; PSQT_BUCKETS]) {
        acc.copy_from_slice(&self.ft_bias);
        *psqt = [0i32; PSQT_BUCKETS];
        let ft_psqt = self.ft_psqt();
        let has_psqt = !ft_psqt.is_empty();
        for &feat in indices {
            let base = feat * self.l1;
            crate::simd::add_i16(acc, &self.ft_weight[base..base + self.l1]);
            if has_psqt {
                let pbase = feat * PSQT_BUCKETS;
                for b in 0..PSQT_BUCKETS {
                    psqt[b] += ft_psqt[pbase + b];
                }
            }
        }
    }

    /// Allocate a zeroed accumulator sized for this network.
    ///
    /// Useful for pre-allocating a stack of accumulators in a search; fill each
    /// with [`Network::refresh`] or [`Network::update`] before evaluating.
    pub fn empty_accumulator(&self) -> Accumulator {
        Accumulator {
            white: vec![0i16; self.l1],
            black: vec![0i16; self.l1],
            psqt_white: [0i32; PSQT_BUCKETS],
            psqt_black: [0i32; PSQT_BUCKETS],
            piece_count: 0,
            threat_pairs: Vec::new(),
        }
    }

    /// Allocate and fully compute an accumulator for `board`.
    pub fn accumulator(&self, board: &impl Board) -> Accumulator {
        let mut acc = self.empty_accumulator();
        self.refresh(board, &mut acc);
        acc
    }

    fn refresh_side(
        &self,
        board: &impl Board,
        color: Color,
        acc: &mut [i16],
        psqt: &mut [i32; PSQT_BUCKETS],
    ) {
        SCRATCH.with(|cell| {
            let mut s = cell.borrow_mut();
            active_indices(self.arch, board, color, &mut s.white);
            self.accumulate(&s.white, acc, psqt);
        });
    }

    /// Recompute `acc` from scratch for `board`.
    ///
    /// Call at the root of a search; [`Network::update`] calls it internally
    /// when a king moves.
    pub fn refresh(&self, board: &impl Board, acc: &mut Accumulator) {
        if self.arch == Arch::Sfnnv10 {
            let pos = threats::PosInfo::from_board(board);
            threats::threat_pairs(&pos, &mut acc.threat_pairs);
            let Accumulator {
                white,
                black,
                psqt_white,
                psqt_black,
                threat_pairs,
                ..
            } = acc;
            self.refresh_side_v10(&pos, Color::White, threat_pairs, white, psqt_white);
            self.refresh_side_v10(&pos, Color::Black, threat_pairs, black, psqt_black);
            acc.piece_count = pos.occ.count_ones() as usize;
            return;
        }
        self.refresh_side(board, Color::White, &mut acc.white, &mut acc.psqt_white);
        self.refresh_side(board, Color::Black, &mut acc.black, &mut acc.psqt_black);
        acc.piece_count = count_pieces(board);
    }

    fn refresh_side_v10(
        &self,
        pos: &threats::PosInfo,
        color: Color,
        pairs: &[u32],
        acc: &mut [i16],
        psqt: &mut [i32; PSQT_BUCKETS],
    ) {
        let ksq = pos.kings[color.index()];
        acc.copy_from_slice(&self.ft_bias);
        *psqt = [0i32; PSQT_BUCKETS];
        let ft_psqt = self.ft_psqt();

        let mut bb = pos.occ;
        while bb != 0 {
            let sq = bb.trailing_zeros() as u8;
            let code = pos.piece_on[sq as usize] as usize;
            let feat = make_index(self.arch, color, sq, code, ksq);
            let base = feat * self.l1;
            crate::simd::add_i16(acc, &self.ft_weight[base..base + self.l1]);
            let pbase = feat * PSQT_BUCKETS;
            for b in 0..PSQT_BUCKETS {
                psqt[b] += ft_psqt[pbase + b];
            }
            bb &= bb - 1;
        }

        let (threat_weight, threat_psqt) = self.threat_tables();
        for &pair in pairs {
            let (from, to, attacker, attacked) = threats::unpack_pair(pair);
            let idx = threats::threat_index(color, attacker, from, to, attacked, ksq);
            if idx >= threats::EXCLUDED {
                continue;
            }
            let base = idx as usize * self.l1;
            crate::simd::add_i8_i16(acc, &threat_weight[base..base + self.l1]);
            let pbase = idx as usize * PSQT_BUCKETS;
            for b in 0..PSQT_BUCKETS {
                psqt[b] += threat_psqt[pbase + b];
            }
        }
    }

    fn apply_threat_diff(
        &self,
        color: Color,
        ksq: u8,
        parent_pairs: &[u32],
        child_pairs: &[u32],
        acc: &mut [i16],
        psqt: &mut [i32; PSQT_BUCKETS],
    ) {
        let (threat_weight, threat_psqt) = self.threat_tables();
        let mut apply = |pair: u32, add: bool| {
            let (from, to, attacker, attacked) = threats::unpack_pair(pair);
            let idx = threats::threat_index(color, attacker, from, to, attacked, ksq);
            if idx >= threats::EXCLUDED {
                return;
            }
            let base = idx as usize * self.l1;
            let row = &threat_weight[base..base + self.l1];
            let pbase = idx as usize * PSQT_BUCKETS;
            if add {
                crate::simd::add_i8_i16(acc, row);
                for b in 0..PSQT_BUCKETS {
                    psqt[b] += threat_psqt[pbase + b];
                }
            } else {
                crate::simd::sub_i8_i16(acc, row);
                for b in 0..PSQT_BUCKETS {
                    psqt[b] -= threat_psqt[pbase + b];
                }
            }
        };

        let (mut i, mut j) = (0usize, 0usize);
        while i < parent_pairs.len() && j < child_pairs.len() {
            let p = parent_pairs[i];
            let c = child_pairs[j];
            if p == c {
                i += 1;
                j += 1;
            } else if p < c {
                apply(p, false);
                i += 1;
            } else {
                apply(c, true);
                j += 1;
            }
        }
        while i < parent_pairs.len() {
            apply(parent_pairs[i], false);
            i += 1;
        }
        while j < child_pairs.len() {
            apply(child_pairs[j], true);
            j += 1;
        }
    }

    fn apply_side(
        &self,
        parent: &[i16],
        parent_psqt: &[i32; PSQT_BUCKETS],
        child: &mut [i16],
        child_psqt: &mut [i32; PSQT_BUCKETS],
        king_square: u8,
        color: Color,
        removed: &[(u8, Piece)],
        added: &[(u8, Piece)],
    ) {
        child.copy_from_slice(parent);
        *child_psqt = *parent_psqt;
        let ft_psqt = self.ft_psqt();
        let has_psqt = !ft_psqt.is_empty();
        let kings_are_features = self.arch.kings_are_features();
        for &(sq, piece) in removed {
            if !kings_are_features && piece.kind == PieceKind::King {
                continue;
            }
            let feat = make_index(self.arch, color, sq, piece.sf_index(), king_square);
            let base = feat * self.l1;
            crate::simd::sub_i16(child, &self.ft_weight[base..base + self.l1]);
            if has_psqt {
                let pbase = feat * PSQT_BUCKETS;
                for b in 0..PSQT_BUCKETS {
                    child_psqt[b] -= ft_psqt[pbase + b];
                }
            }
        }
        for &(sq, piece) in added {
            if !kings_are_features && piece.kind == PieceKind::King {
                continue;
            }
            let feat = make_index(self.arch, color, sq, piece.sf_index(), king_square);
            let base = feat * self.l1;
            crate::simd::add_i16(child, &self.ft_weight[base..base + self.l1]);
            if has_psqt {
                let pbase = feat * PSQT_BUCKETS;
                for b in 0..PSQT_BUCKETS {
                    child_psqt[b] += ft_psqt[pbase + b];
                }
            }
        }
    }

    /// Advance `parent` into `child` for the move that turns `parent_board` into
    /// `child_board`, writing the result into `child`.
    ///
    /// The changed pieces are derived by diffing the two boards, so every move
    /// type (captures, en passant, promotions, castling) is handled. When a king
    /// moves, that perspective is recomputed from scratch automatically.
    pub fn update(
        &self,
        parent_board: &impl Board,
        child_board: &impl Board,
        parent: &Accumulator,
        child: &mut Accumulator,
    ) {
        if self.arch == Arch::Sfnnv10 {
            self.update_v10(parent_board, child_board, parent, child);
            return;
        }
        let dummy = (0u8, Piece::new(Color::White, PieceKind::Pawn));
        let mut removed = [dummy; MAX_CHANGED];
        let mut added = [dummy; MAX_CHANGED];
        let (nr, na) = diff_boards(parent_board, child_board, &mut removed, &mut added);

        let delta = BoardDelta {
            removed,
            added,
            removed_len: nr as u8,
            added_len: na as u8,
            parent_kings: [
                parent_board.king_square(Color::White),
                parent_board.king_square(Color::Black),
            ],
        };
        self.update_delta(child_board, &delta, parent, child);
    }

    /// Advance `parent` into `child` using an exact compact move delta.
    ///
    /// This is the preferred engine-integration API: callers already know the
    /// moved, captured, promoted, and castling pieces, so a full two-board diff
    /// would only repeat work from move making.
    pub fn update_delta(
        &self,
        child_board: &impl Board,
        delta: &BoardDelta,
        parent: &Accumulator,
        child: &mut Accumulator,
    ) {
        if self.arch == Arch::Sfnnv10 {
            self.update_v10_delta(child_board, delta, parent, child);
            return;
        }

        let removed = delta.removed();
        let added = delta.added();

        let white_king_moved = delta.parent_kings[0] != child_board.king_square(Color::White);
        let black_king_moved = delta.parent_kings[1] != child_board.king_square(Color::Black);

        if white_king_moved {
            self.refresh_side(
                child_board,
                Color::White,
                &mut child.white,
                &mut child.psqt_white,
            );
        } else {
            let wk = child_board.king_square(Color::White);
            self.apply_side(
                &parent.white,
                &parent.psqt_white,
                &mut child.white,
                &mut child.psqt_white,
                wk,
                Color::White,
                removed,
                added,
            );
        }
        if black_king_moved {
            self.refresh_side(
                child_board,
                Color::Black,
                &mut child.black,
                &mut child.psqt_black,
            );
        } else {
            let bk = child_board.king_square(Color::Black);
            self.apply_side(
                &parent.black,
                &parent.psqt_black,
                &mut child.black,
                &mut child.psqt_black,
                bk,
                Color::Black,
                removed,
                added,
            );
        }

        child.piece_count = parent.piece_count + added.len() - removed.len();
    }

    fn update_v10(
        &self,
        parent_board: &impl Board,
        child_board: &impl Board,
        parent: &Accumulator,
        child: &mut Accumulator,
    ) {
        let dummy = (0u8, Piece::new(Color::White, PieceKind::Pawn));
        let mut removed = [dummy; MAX_CHANGED];
        let mut added = [dummy; MAX_CHANGED];
        let (nr, na) = diff_boards(parent_board, child_board, &mut removed, &mut added);

        let delta = BoardDelta {
            removed,
            added,
            removed_len: nr as u8,
            added_len: na as u8,
            parent_kings: [
                parent_board.king_square(Color::White),
                parent_board.king_square(Color::Black),
            ],
        };
        self.update_v10_delta(child_board, &delta, parent, child);
    }

    fn update_v10_delta(
        &self,
        child_board: &impl Board,
        delta: &BoardDelta,
        parent: &Accumulator,
        child: &mut Accumulator,
    ) {
        let pos = threats::PosInfo::from_board(child_board);
        threats::threat_pairs(&pos, &mut child.threat_pairs);
        let removed = delta.removed();
        let added = delta.added();

        let Accumulator {
            white,
            black,
            psqt_white,
            psqt_black,
            piece_count,
            threat_pairs,
        } = child;

        let wk = pos.kings[0];
        if delta.parent_kings[0] != wk {
            self.refresh_side_v10(&pos, Color::White, threat_pairs, white, psqt_white);
        } else {
            self.apply_side(
                &parent.white,
                &parent.psqt_white,
                white,
                psqt_white,
                wk,
                Color::White,
                removed,
                added,
            );
            self.apply_threat_diff(
                Color::White,
                wk,
                &parent.threat_pairs,
                threat_pairs,
                white,
                psqt_white,
            );
        }

        let bk = pos.kings[1];
        if delta.parent_kings[1] != bk {
            self.refresh_side_v10(&pos, Color::Black, threat_pairs, black, psqt_black);
        } else {
            self.apply_side(
                &parent.black,
                &parent.psqt_black,
                black,
                psqt_black,
                bk,
                Color::Black,
                removed,
                added,
            );
            self.apply_threat_diff(
                Color::Black,
                bk,
                &parent.threat_pairs,
                threat_pairs,
                black,
                psqt_black,
            );
        }

        *piece_count = pos.occ.count_ones() as usize;
    }

    /// Evaluate a ready accumulator for the given side to move.
    ///
    /// Returns an internal score in roughly centipawn-scaled units from `stm`'s
    /// perspective. `stm` is passed separately so the same accumulator can be
    /// reused across a null move (which flips the side to move without changing
    /// any features).
    pub fn evaluate_accumulator(&self, acc: &Accumulator, stm: Color) -> i32 {
        self.evaluate_accumulator_components(acc, stm).total()
    }

    /// Evaluate a ready accumulator while retaining the PSQT and positional
    /// components required by modern Stockfish-compatible score scaling.
    pub fn evaluate_accumulator_components(&self, acc: &Accumulator, stm: Color) -> Evaluation {
        match &self.layers {
            Layers::Sfnnv10 { buckets, .. } => self.evaluate_hm_style(acc, stm, 255, buckets),
            Layers::HalfKAv2Hm { buckets, .. } => self.evaluate_hm_style(acc, stm, 254, buckets),
            Layers::HalfKAv2 { buckets, .. } => self.evaluate_halfka_v2(acc, stm, buckets),
            Layers::HalfKP(bucket) => self.evaluate_halfkp(acc, stm, bucket),
        }
    }

    fn evaluate_hm_style(
        &self,
        acc: &Accumulator,
        stm: Color,
        hi: i16,
        buckets: &[Bucket],
    ) -> Evaluation {
        SCRATCH.with(|cell| {
            let mut s = cell.borrow_mut();
            if s.input.len() != self.l1 {
                s.input = vec![0u8; self.l1];
            }
            let input = &mut s.input;

            let (acc_stm, acc_opp, psqt_stm, psqt_opp) = match stm {
                Color::White => (&acc.white, &acc.black, &acc.psqt_white, &acc.psqt_black),
                Color::Black => (&acc.black, &acc.white, &acc.psqt_black, &acc.psqt_white),
            };

            let half = self.l1 / 2;
            let (in0, in1) = input.split_at_mut(half);
            crate::simd::pairwise_clip_mul(&acc_stm[..half], &acc_stm[half..], in0, hi, 9);
            crate::simd::pairwise_clip_mul(
                &acc_opp[..half],
                &acc_opp[half..],
                &mut in1[..half],
                hi,
                9,
            );

            let bucket = (acc.piece_count - 1) / 4;
            let psqt = (psqt_stm[bucket] - psqt_opp[bucket]) / 2;
            let positional = self.propagate_hm_stack(input, &buckets[bucket]);

            Evaluation {
                psqt: psqt / OUTPUT_SCALE,
                positional: positional / OUTPUT_SCALE,
            }
        })
    }

    fn evaluate_halfka_v2(&self, acc: &Accumulator, stm: Color, buckets: &[Bucket]) -> Evaluation {
        SCRATCH.with(|cell| {
            let mut s = cell.borrow_mut();
            let want = 2 * self.l1;
            if s.input.len() != want {
                s.input = vec![0u8; want];
            }
            let input = &mut s.input;

            let (acc_stm, acc_opp, psqt_stm, psqt_opp) = match stm {
                Color::White => (&acc.white, &acc.black, &acc.psqt_white, &acc.psqt_black),
                Color::Black => (&acc.black, &acc.white, &acc.psqt_black, &acc.psqt_white),
            };

            let (in0, in1) = input.split_at_mut(self.l1);
            crate::simd::clip_u8(acc_stm, in0);
            crate::simd::clip_u8(acc_opp, &mut in1[..self.l1]);

            let bucket = (acc.piece_count - 1) / 4;
            let psqt = (psqt_stm[bucket] - psqt_opp[bucket]) / 2;
            let positional = self.propagate_halfka_v2(input, &buckets[bucket]);

            Evaluation {
                psqt: psqt / OUTPUT_SCALE,
                positional: positional / OUTPUT_SCALE,
            }
        })
    }

    fn evaluate_halfkp(&self, acc: &Accumulator, stm: Color, bucket: &Bucket) -> Evaluation {
        SCRATCH.with(|cell| {
            let mut s = cell.borrow_mut();
            let want = 2 * self.l1;
            if s.input.len() != want {
                s.input = vec![0u8; want];
            }
            let input = &mut s.input;

            let (acc_stm, acc_opp) = match stm {
                Color::White => (&acc.white, &acc.black),
                Color::Black => (&acc.black, &acc.white),
            };

            let (in0, in1) = input.split_at_mut(self.l1);
            crate::simd::clip_u8(acc_stm, in0);
            crate::simd::clip_u8(acc_opp, &mut in1[..self.l1]);

            Evaluation {
                psqt: 0,
                positional: self.propagate_halfkp(input, bucket) / OUTPUT_SCALE,
            }
        })
    }

    /// Evaluate a position directly, computing a fresh accumulator.
    ///
    /// This is the simple, stateless entry point used by both the FEN and trait
    /// integration styles. For a search, prefer the incremental
    /// [`Network::accumulator`] + [`Network::update`] path.
    pub fn evaluate(&self, board: &impl Board) -> i32 {
        let acc = self.accumulator(board);
        self.evaluate_accumulator(&acc, board.side_to_move())
    }

    fn propagate_hm_stack(&self, input: &[u8], b: &Bucket) -> i32 {
        let mut fc0_out = [0i32; HM_FC0_OUT];
        let inp = &input[..self.l1];
        for (o, out) in fc0_out.iter_mut().enumerate() {
            let wbase = o * self.l1;
            *out =
                b.fc0_bias[o] + crate::simd::dot_u8_i8(inp, &b.fc0_weight[wbase..wbase + self.l1]);
        }

        let mut concat = [0u8; HM_FC1_IN];
        for i in 0..HM_L2 {
            let x = fc0_out[i] as i64;
            concat[i] = (x * x >> 19).min(127) as u8;
            concat[HM_L2 + i] = (fc0_out[i] >> WEIGHT_SCALE_BITS).clamp(0, 127) as u8;
        }

        let mut fc1_out = [0i32; HM_L3];
        for (o, out) in fc1_out.iter_mut().enumerate() {
            let mut sum = b.fc1_bias[o];
            let wbase = o * FC1_PAD;
            for i in 0..HM_FC1_IN {
                sum += b.fc1_weight[wbase + i] as i32 * concat[i] as i32;
            }
            *out = sum;
        }

        let mut ac1 = [0u8; HM_L3];
        for i in 0..HM_L3 {
            ac1[i] = (fc1_out[i] >> WEIGHT_SCALE_BITS).clamp(0, 127) as u8;
        }

        let mut fc2 = b.fc2_bias;
        for i in 0..HM_L3 {
            fc2 += b.fc2_weight[i] as i32 * ac1[i] as i32;
        }

        let fwd_out = fc0_out[HM_L2] * (600 * OUTPUT_SCALE) / (127 * (1 << WEIGHT_SCALE_BITS));
        fc2 + fwd_out
    }

    fn propagate_halfka_v2(&self, input: &[u8], b: &Bucket) -> i32 {
        let inp = &input[..2 * self.l1];

        let mut fc0 = [0u8; FC1_PAD];
        for o in 0..V2_FC0_OUT {
            let wbase = o * inp.len();
            let sum = b.fc0_bias[o]
                + crate::simd::dot_u8_i8(inp, &b.fc0_weight[wbase..wbase + inp.len()]);
            fc0[o] = (sum >> WEIGHT_SCALE_BITS).clamp(0, 127) as u8;
        }

        let mut fc1 = [0u8; V2_FC1_OUT];
        for o in 0..V2_FC1_OUT {
            let wbase = o * FC1_PAD;
            let sum =
                b.fc1_bias[o] + crate::simd::dot_u8_i8(&fc0, &b.fc1_weight[wbase..wbase + FC1_PAD]);
            fc1[o] = (sum >> WEIGHT_SCALE_BITS).clamp(0, 127) as u8;
        }

        b.fc2_bias + crate::simd::dot_u8_i8(&fc1, &b.fc2_weight)
    }

    fn propagate_halfkp(&self, input: &[u8], b: &Bucket) -> i32 {
        let inp = &input[..2 * self.l1];

        let mut fc0 = [0u8; HALFKP_FC0_OUT];
        for o in 0..HALFKP_FC0_OUT {
            let wbase = o * inp.len();
            let sum = b.fc0_bias[o]
                + crate::simd::dot_u8_i8(inp, &b.fc0_weight[wbase..wbase + inp.len()]);
            fc0[o] = (sum >> WEIGHT_SCALE_BITS).clamp(0, 127) as u8;
        }

        let mut fc1 = [0u8; HALFKP_FC1_OUT];
        for o in 0..HALFKP_FC1_OUT {
            let wbase = o * HALFKP_FC0_OUT;
            let sum = b.fc1_bias[o]
                + crate::simd::dot_u8_i8(&fc0, &b.fc1_weight[wbase..wbase + HALFKP_FC0_OUT]);
            fc1[o] = (sum >> WEIGHT_SCALE_BITS).clamp(0, 127) as u8;
        }

        b.fc2_bias + crate::simd::dot_u8_i8(&fc1, &b.fc2_weight)
    }
}

#[cfg(test)]
mod loader_tests {
    use super::*;

    struct HalfKaFixture {
        bytes: Vec<u8>,
        ft_hash_offset: usize,
        bucket_hash_offsets: Vec<usize>,
    }

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn append_zeroes(bytes: &mut Vec<u8>, count: usize) {
        bytes.resize(bytes.len() + count, 0);
    }

    // HalfKAv2 is raw rather than compressed and permits a 16-wide feature
    // transformer, so it gives the production loader a compact, deterministic
    // complete network fixture without checking a multi-megabyte binary into
    // the repository.
    fn halfka_fixture() -> HalfKaFixture {
        const L1: usize = 16;
        const DESCRIPTION: &[u8] = b"Volkrix loader validation fixture";
        const NETWORK_HASH: u32 = 0x6333_712a;

        let ft_hash = FEATURE_HASH_V2 ^ (2 * L1 as u32);
        let architecture_hash = ft_hash ^ NETWORK_HASH;
        let mut bytes = Vec::new();
        push_u32(&mut bytes, VERSION_HALFKA);
        push_u32(&mut bytes, architecture_hash);
        push_u32(&mut bytes, DESCRIPTION.len() as u32);
        bytes.extend_from_slice(DESCRIPTION);

        let ft_hash_offset = bytes.len();
        push_u32(&mut bytes, ft_hash);
        append_zeroes(&mut bytes, L1 * size_of::<i16>());
        append_zeroes(
            &mut bytes,
            L1 * Arch::HalfKAv2.input_dimensions() * size_of::<i16>(),
        );
        append_zeroes(
            &mut bytes,
            PSQT_BUCKETS * Arch::HalfKAv2.input_dimensions() * size_of::<i32>(),
        );

        let mut bucket_hash_offsets = Vec::with_capacity(LAYER_STACKS);
        for _ in 0..LAYER_STACKS {
            bucket_hash_offsets.push(bytes.len());
            push_u32(&mut bytes, NETWORK_HASH);
            append_zeroes(&mut bytes, V2_FC0_OUT * size_of::<i32>());
            append_zeroes(&mut bytes, V2_FC0_OUT * (2 * L1) * size_of::<i8>());
            append_zeroes(&mut bytes, V2_FC1_OUT * size_of::<i32>());
            append_zeroes(&mut bytes, V2_FC1_OUT * FC1_PAD * size_of::<i8>());
            append_zeroes(&mut bytes, size_of::<i32>());
            append_zeroes(&mut bytes, V2_FC1_OUT * size_of::<i8>());
        }

        HalfKaFixture {
            bytes,
            ft_hash_offset,
            bucket_hash_offsets,
        }
    }

    fn halfkp_fixture() -> (Vec<u8>, usize) {
        let mut bytes = Vec::new();
        push_u32(&mut bytes, VERSION_HALFKP);
        push_u32(&mut bytes, FEATURE_HASH_HALFKP ^ NETWORK_HASH_HALFKP);
        push_u32(&mut bytes, 0);
        push_u32(&mut bytes, FEATURE_HASH_HALFKP);
        append_zeroes(&mut bytes, HALFKP_HALF_DIMENSIONS * size_of::<i16>());
        append_zeroes(
            &mut bytes,
            HALFKP_HALF_DIMENSIONS * Arch::HalfKP.input_dimensions() * size_of::<i16>(),
        );
        let network_hash_offset = bytes.len();
        push_u32(&mut bytes, NETWORK_HASH_HALFKP);
        append_zeroes(&mut bytes, HALFKP_FC0_OUT * size_of::<i32>());
        append_zeroes(
            &mut bytes,
            HALFKP_FC0_OUT * (2 * HALFKP_HALF_DIMENSIONS) * size_of::<i8>(),
        );
        append_zeroes(&mut bytes, HALFKP_FC1_OUT * size_of::<i32>());
        append_zeroes(&mut bytes, HALFKP_FC1_OUT * FC1_PAD * size_of::<i8>());
        append_zeroes(&mut bytes, size_of::<i32>());
        append_zeroes(&mut bytes, HALFKP_FC1_OUT * size_of::<i8>());
        (bytes, network_hash_offset)
    }

    fn assert_invalid_data(bytes: &[u8], expected: &'static str) {
        let error = match Network::from_bytes(bytes) {
            Ok(_) => panic!("malformed network was accepted"),
            Err(error) => error,
        };
        assert!(matches!(error, Error::InvalidData(message) if message == expected));
    }

    fn mutate_u32(bytes: &mut [u8], offset: usize) {
        bytes[offset] ^= 1;
    }

    #[test]
    fn rejects_oversized_description_before_allocating_it() {
        let mut bytes = Vec::new();
        push_u32(&mut bytes, VERSION_HALFKA);
        push_u32(&mut bytes, 0);
        push_u32(&mut bytes, (MAX_DESCRIPTION_LEN as u32) + 1);

        assert_invalid_data(&bytes, "network description is too large");
    }

    #[test]
    fn validates_architecture_transformer_and_every_bucket_hash() {
        let mut fixture = halfka_fixture();
        assert!(Network::from_bytes(&fixture.bytes).is_ok());

        mutate_u32(&mut fixture.bytes, 4);
        assert_invalid_data(
            &fixture.bytes,
            "architecture and network bucket hashes do not match",
        );
        mutate_u32(&mut fixture.bytes, 4);

        mutate_u32(&mut fixture.bytes, fixture.ft_hash_offset);
        assert_invalid_data(&fixture.bytes, "bad feature-transformer width");
        mutate_u32(&mut fixture.bytes, fixture.ft_hash_offset);

        for &offset in &fixture.bucket_hash_offsets {
            mutate_u32(&mut fixture.bytes, offset);
            assert_invalid_data(
                &fixture.bytes,
                "architecture and network bucket hashes do not match",
            );
            mutate_u32(&mut fixture.bytes, offset);
        }
    }

    #[test]
    fn validates_fixed_halfkp_architecture_transformer_and_network_hashes() {
        let mut short_header = Vec::new();
        push_u32(&mut short_header, VERSION_HALFKP);
        push_u32(&mut short_header, 0);
        push_u32(&mut short_header, 0);
        assert_invalid_data(&short_header, "incompatible HalfKP architecture hash");

        short_header[4..8]
            .copy_from_slice(&(FEATURE_HASH_HALFKP ^ NETWORK_HASH_HALFKP).to_le_bytes());
        push_u32(&mut short_header, FEATURE_HASH_HALFKP ^ 1);
        assert_invalid_data(
            &short_header,
            "incompatible HalfKP feature-transformer hash",
        );

        let (mut bytes, network_hash_offset) = halfkp_fixture();
        assert!(Network::from_bytes(&bytes).is_ok());
        mutate_u32(&mut bytes, network_hash_offset);
        assert_invalid_data(&bytes, "incompatible HalfKP network hash");
    }

    #[test]
    fn rejects_trailing_data_after_a_complete_network() {
        let mut fixture = halfka_fixture();
        fixture.bytes.push(0);
        assert_invalid_data(&fixture.bytes, "trailing data after network");
    }
}
