use std::{
    mem::size_of,
    sync::atomic::{AtomicU8, AtomicU64, Ordering},
};

use crate::core::Move;

pub const DEFAULT_HASH_MB: usize = 16;
const ENTRIES_PER_CLUSTER: usize = 4;
const AGE_DEPTH_PENALTY: i16 = 2;
const GENERATION_BITS: u32 = 5;
const GENERATION_MASK: u8 = (1 << GENERATION_BITS) - 1;

// Entry layout (64 bits): move 20, score 16, eval 13, depth 8, bound 2, generation 5.
const MOVE_BITS: u32 = 20;
const SCORE_SHIFT: u32 = MOVE_BITS;
const EVAL_SHIFT: u32 = SCORE_SHIFT + 16;
const DEPTH_SHIFT: u32 = EVAL_SHIFT + 13;
const BOUND_SHIFT: u32 = DEPTH_SHIFT + 8;
const GENERATION_SHIFT: u32 = BOUND_SHIFT + 2;
const MOVE_MASK: u64 = (1 << MOVE_BITS) - 1;
const EVAL_MASK: u64 = (1 << 13) - 1;
const TT_MOVE_NONE: u32 = MOVE_MASK as u32;
const MIN_PACKED_EVAL: i16 = -(1 << 12);
const MAX_PACKED_EVAL: i16 = (1 << 12) - 1;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Bound {
    #[default]
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TtHit {
    pub key_tag: u64,
    pub best_move: Move,
    pub score: i16,
    pub eval: i16,
    pub depth: u8,
    pub bound: Bound,
    pub generation: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TtStore {
    pub best_move: Move,
    pub score: i16,
    pub eval: i16,
    pub depth: u8,
    pub bound: Bound,
}

/// One independently replaceable 16-byte TT entry.
///
/// The first word is a 64-bit verification code (`position_key XOR payload`). A writer publishes
/// the payload before the verification word; a reader accepts a pair only when XOR reconstructs
/// its requested key. Concurrently mixed words therefore become a normal 64-bit TT collision
/// rather than a torn entry. This established checksum scheme keeps probes and stores lock-free,
/// bounded, and data-race-free under Lazy SMP without a version lock or retry loop.
#[repr(C)]
struct AtomicEntry {
    key_xor_data: AtomicU64,
    data: AtomicU64,
}

impl AtomicEntry {
    const fn new() -> Self {
        Self {
            key_xor_data: AtomicU64::new(0),
            data: AtomicU64::new(0),
        }
    }

    fn read(&self) -> Option<EntrySnapshot> {
        let key_xor_data = self.key_xor_data.load(Ordering::Acquire);
        let data = self.data.load(Ordering::Relaxed);
        if key_xor_data == 0 && data == 0 {
            return None;
        }
        Some(EntrySnapshot {
            key: key_xor_data ^ data,
            data,
        })
    }

    fn clear(&self) {
        self.data.store(0, Ordering::Relaxed);
        self.key_xor_data.store(0, Ordering::Release);
    }

    fn write(&self, key: u64, data: u64) {
        self.data.store(data, Ordering::Relaxed);
        self.key_xor_data.store(key ^ data, Ordering::Release);
    }
}

#[derive(Clone, Copy)]
struct EntrySnapshot {
    key: u64,
    data: u64,
}

#[repr(C, align(64))]
struct Cluster {
    entries: [AtomicEntry; ENTRIES_PER_CLUSTER],
}

impl Cluster {
    const fn new() -> Self {
        Self {
            entries: [const { AtomicEntry::new() }; ENTRIES_PER_CLUSTER],
        }
    }
}

pub struct TranspositionTable {
    clusters: Vec<Cluster>,
    generation: AtomicU8,
}

impl TranspositionTable {
    pub fn new_mb(hash_mb: usize) -> Self {
        Self::with_cluster_count(cluster_count_for_mb(hash_mb))
    }

    pub fn clear(&self) {
        for cluster in &self.clusters {
            for entry in &cluster.entries {
                entry.clear();
            }
        }
        self.generation.store(0, Ordering::Relaxed);
    }

    pub fn new_generation(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    pub fn probe(&self, key: u64) -> Option<TtHit> {
        let cluster = &self.clusters[self.cluster_index(key)];
        for entry in &cluster.entries {
            let Some(snapshot) = entry.read() else {
                continue;
            };
            if snapshot.key == key {
                return Some(unpack_hit(key, snapshot.data));
            }
        }
        None
    }

    pub fn store(&self, key: u64, store: TtStore) {
        let generation = self.generation.load(Ordering::Relaxed) & GENERATION_MASK;
        let data = pack_entry(store, generation);
        let cluster = &self.clusters[self.cluster_index(key)];
        let replacement_index = select_replacement_slot(cluster, key, generation);
        cluster.entries[replacement_index].write(key, data);
    }

    fn cluster_index(&self, key: u64) -> usize {
        ((key as u128).wrapping_mul(self.clusters.len() as u128) >> 64) as usize
    }

    fn with_cluster_count(cluster_count: usize) -> Self {
        let mut clusters = Vec::with_capacity(cluster_count.max(1));
        clusters.resize_with(cluster_count.max(1), Cluster::new);
        Self {
            clusters,
            generation: AtomicU8::new(0),
        }
    }

    #[cfg(test)]
    fn with_cluster_count_for_test(cluster_count: usize) -> Self {
        Self::with_cluster_count(cluster_count)
    }

    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    pub fn debug_entry_count(&self) -> usize {
        self.clusters
            .iter()
            .flat_map(|cluster| cluster.entries.iter())
            .filter(|entry| entry.read().is_some())
            .count()
    }
}

pub fn normalize_score_for_store(score: i32, ply: usize) -> i16 {
    if score >= super::root::MATE_SCORE - super::root::MAX_PLY as i32 {
        (score + ply as i32) as i16
    } else if score <= -(super::root::MATE_SCORE - super::root::MAX_PLY as i32) {
        (score - ply as i32) as i16
    } else {
        score as i16
    }
}

pub fn denormalize_score_from_tt(score: i16, ply: usize) -> i32 {
    let score = score as i32;
    if score >= super::root::MATE_SCORE - super::root::MAX_PLY as i32 {
        score - ply as i32
    } else if score <= -(super::root::MATE_SCORE - super::root::MAX_PLY as i32) {
        score + ply as i32
    } else {
        score
    }
}

fn cluster_count_for_mb(hash_mb: usize) -> usize {
    let bytes = hash_mb.max(1).saturating_mul(1024 * 1024);
    (bytes / size_of::<Cluster>()).max(1)
}

fn select_replacement_slot(cluster: &Cluster, key: u64, generation: u8) -> usize {
    let snapshots =
        std::array::from_fn::<_, ENTRIES_PER_CLUSTER, _>(|index| cluster.entries[index].read());

    for (index, snapshot) in snapshots.iter().enumerate() {
        if snapshot.is_some_and(|snapshot| snapshot.key == key) {
            return index;
        }
    }
    for (index, snapshot) in snapshots.iter().enumerate() {
        if snapshot.is_none() {
            return index;
        }
    }

    let mut best_index = 0;
    let mut worst_value = replacement_value(
        snapshots[0]
            .expect("full cluster must have a first entry")
            .data,
        generation,
    );
    for (index, snapshot) in snapshots.iter().enumerate().skip(1) {
        let snapshot = snapshot.expect("full cluster must contain only entries");
        let value = replacement_value(snapshot.data, generation);
        if value < worst_value {
            worst_value = value;
            best_index = index;
        }
    }
    best_index
}

fn replacement_value(data: u64, generation: u8) -> i16 {
    let depth = ((data >> DEPTH_SHIFT) & u8::MAX as u64) as i16;
    let entry_generation = ((data >> GENERATION_SHIFT) & GENERATION_MASK as u64) as u8;
    let age = generation.wrapping_sub(entry_generation) & GENERATION_MASK;
    depth - age as i16 * AGE_DEPTH_PENALTY
}

fn pack_entry(store: TtStore, generation: u8) -> u64 {
    let move_bits = if store.best_move.is_none() {
        TT_MOVE_NONE
    } else {
        debug_assert!(store.best_move.raw() <= MOVE_MASK as u32);
        store.best_move.raw() & MOVE_MASK as u32
    };
    let eval = store.eval.clamp(MIN_PACKED_EVAL, MAX_PACKED_EVAL);
    u64::from(move_bits)
        | (u64::from(store.score as u16) << SCORE_SHIFT)
        | (u64::from((eval as u16) & EVAL_MASK as u16) << EVAL_SHIFT)
        | (u64::from(store.depth) << DEPTH_SHIFT)
        | (u64::from(bound_code(store.bound)) << BOUND_SHIFT)
        | (u64::from(generation & GENERATION_MASK) << GENERATION_SHIFT)
}

fn unpack_hit(key: u64, data: u64) -> TtHit {
    let move_bits = (data & MOVE_MASK) as u32;
    let eval_bits = ((data >> EVAL_SHIFT) & EVAL_MASK) as u16;
    let eval = if eval_bits & (1 << 12) != 0 {
        (eval_bits | !EVAL_MASK as u16) as i16
    } else {
        eval_bits as i16
    };
    TtHit {
        key_tag: key,
        best_move: if move_bits == TT_MOVE_NONE {
            Move::NONE
        } else {
            Move::from_raw(move_bits)
        },
        score: ((data >> SCORE_SHIFT) & u16::MAX as u64) as u16 as i16,
        eval,
        depth: ((data >> DEPTH_SHIFT) & u8::MAX as u64) as u8,
        bound: decode_bound(((data >> BOUND_SHIFT) & 0b11) as u8),
        generation: ((data >> GENERATION_SHIFT) & GENERATION_MASK as u64) as u8,
    }
}

const fn bound_code(bound: Bound) -> u8 {
    match bound {
        Bound::Exact => 0,
        Bound::Lower => 1,
        Bound::Upper => 2,
    }
}

const fn decode_bound(code: u8) -> Bound {
    match code {
        1 => Bound::Lower,
        2 => Bound::Upper,
        _ => Bound::Exact,
    }
}

#[cfg(test)]
mod tests {
    use std::{
        mem::{align_of, size_of},
        sync::Arc,
        thread,
    };

    use super::{
        Bound, Cluster, DEFAULT_HASH_MB, ENTRIES_PER_CLUSTER, TranspositionTable, TtStore,
        denormalize_score_from_tt, normalize_score_for_store,
    };
    use crate::core::{Move, Square, chess_move::FLAG_CAPTURE};

    fn square(text: &str) -> Square {
        Square::from_coord_text(text).expect("test square must parse")
    }

    fn sample_move() -> Move {
        Move::new(square("e2"), square("e4"))
    }

    fn sample_store(score: i16, depth: u8) -> TtStore {
        TtStore {
            best_move: sample_move(),
            score,
            eval: 17,
            depth,
            bound: Bound::Exact,
        }
    }

    #[test]
    fn cluster_is_exactly_one_aligned_cache_line() {
        assert_eq!(size_of::<Cluster>(), 64);
        assert_eq!(align_of::<Cluster>(), 64);
    }

    #[test]
    fn probe_misses_on_empty_table() {
        let table = TranspositionTable::new_mb(DEFAULT_HASH_MB);
        assert!(table.probe(0x1234_5678).is_none());
        assert!(table.probe(0).is_none());
        assert!(table.probe(u64::MAX).is_none());
    }

    #[test]
    fn store_and_probe_round_trip_fields() {
        let table = TranspositionTable::with_cluster_count_for_test(1);
        table.new_generation();
        let capture = sample_move().with_flags(FLAG_CAPTURE);
        table.store(
            0xabc,
            TtStore {
                best_move: capture,
                score: -42,
                eval: -117,
                depth: 6,
                bound: Bound::Upper,
            },
        );

        let hit = table.probe(0xabc).expect("stored key must probe");
        assert_eq!(hit.key_tag, 0xabc);
        assert_eq!(hit.best_move, capture);
        assert_eq!(hit.score, -42);
        assert_eq!(hit.eval, -117);
        assert_eq!(hit.depth, 6);
        assert_eq!(hit.bound, Bound::Upper);
        assert_eq!(hit.generation, 1);
    }

    #[test]
    fn none_move_and_extreme_scores_round_trip() {
        let table = TranspositionTable::with_cluster_count_for_test(1);
        table.store(
            7,
            TtStore {
                best_move: Move::NONE,
                score: i16::MIN,
                eval: i16::MAX,
                depth: u8::MAX,
                bound: Bound::Lower,
            },
        );
        let hit = table.probe(7).expect("entry must probe");
        assert!(hit.best_move.is_none());
        assert_eq!(hit.score, i16::MIN);
        assert_eq!(hit.eval, super::MAX_PACKED_EVAL);
        assert_eq!(hit.depth, u8::MAX);
    }

    #[test]
    fn same_key_overwrites_before_replacement() {
        let table = TranspositionTable::with_cluster_count_for_test(1);
        table.store(0xabc, sample_store(1, 3));
        table.store(
            0xabc,
            TtStore {
                best_move: Move::new(square("d2"), square("d4")),
                score: 9,
                eval: 8,
                depth: 7,
                bound: Bound::Lower,
            },
        );

        let hit = table.probe(0xabc).expect("key must still exist");
        assert_eq!(hit.best_move, Move::new(square("d2"), square("d4")));
        assert_eq!(hit.score, 9);
        assert_eq!(hit.depth, 7);
        assert_eq!(hit.bound, Bound::Lower);
    }

    #[test]
    fn replacement_prefers_oldest_lowest_depth_entry() {
        let table = TranspositionTable::with_cluster_count_for_test(1);
        for index in 0..ENTRIES_PER_CLUSTER {
            table.store(
                index as u64 + 1,
                sample_store(index as i16, (index + 1) as u8),
            );
            table.new_generation();
        }
        table.store(99, sample_store(99, 9));

        assert!(
            table.probe(1).is_none(),
            "oldest shallowest entry should be replaced"
        );
        assert!(table.probe(99).is_some());
    }

    #[test]
    fn collision_does_not_return_wrong_key() {
        let table = TranspositionTable::with_cluster_count_for_test(1);
        table.store(0x111, sample_store(1, 1));
        table.store(0x222, sample_store(2, 1));

        let hit = table.probe(0x111).expect("first key should still exist");
        assert_eq!(hit.key_tag, 0x111);
        assert!(table.probe(0x333).is_none());
    }

    #[test]
    fn clear_removes_all_entries() {
        let table = TranspositionTable::with_cluster_count_for_test(2);
        for key in 1..=8 {
            table.store(key, sample_store(key as i16, 1));
        }
        assert!(table.debug_entry_count() > 0);
        table.clear();
        assert_eq!(table.debug_entry_count(), 0);
        for key in 1..=8 {
            assert!(table.probe(key).is_none());
        }
    }

    #[test]
    fn concurrent_probes_and_stores_never_decode_torn_payloads() {
        let table = Arc::new(TranspositionTable::with_cluster_count_for_test(1));
        let mut workers = Vec::new();
        for worker in 0..8u64 {
            let table = Arc::clone(&table);
            workers.push(thread::spawn(move || {
                let key = worker + 1;
                for iteration in 0..10_000u16 {
                    table.store(
                        key,
                        TtStore {
                            best_move: Move::new(square("a2"), square("a4")),
                            score: iteration as i16,
                            eval: -(iteration as i16),
                            depth: worker as u8,
                            bound: Bound::Lower,
                        },
                    );
                    if let Some(hit) = table.probe(key) {
                        assert_eq!(hit.key_tag, key);
                        assert_eq!(hit.best_move, Move::new(square("a2"), square("a4")));
                        assert_eq!(hit.depth, worker as u8);
                        assert_eq!(hit.bound, Bound::Lower);
                        assert_eq!(hit.eval, (-hit.score).clamp(-4096, 4095));
                    }
                }
            }));
        }
        for worker in workers {
            worker.join().expect("TT worker must not panic");
        }
    }

    #[test]
    fn mate_score_normalization_round_trip_is_ply_safe() {
        let stored = normalize_score_for_store(29_996, 4);
        assert_eq!(denormalize_score_from_tt(stored, 4), 29_996);
        assert_eq!(denormalize_score_from_tt(stored, 1), 29_999);

        let stored_loss = normalize_score_for_store(-29_995, 5);
        assert_eq!(denormalize_score_from_tt(stored_loss, 5), -29_995);
        assert_eq!(denormalize_score_from_tt(stored_loss, 2), -29_998);
    }
}
