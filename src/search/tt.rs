use std::{
    mem::size_of,
    sync::{
        RwLock,
        atomic::{AtomicU8, Ordering},
    },
};

use crate::core::Move;

pub const DEFAULT_HASH_MB: usize = 16;
const ENTRIES_PER_CLUSTER: usize = 4;
const AGE_DEPTH_PENALTY: i16 = 2;

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TtEntry {
    occupied: bool,
    key_tag: u64,
    best_move: Move,
    score: i16,
    eval: i16,
    depth: u8,
    bound: Bound,
    generation: u8,
}

impl Default for TtEntry {
    fn default() -> Self {
        Self {
            occupied: false,
            key_tag: 0,
            best_move: Move::NONE,
            score: 0,
            eval: 0,
            depth: 0,
            bound: Bound::Exact,
            generation: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Cluster {
    entries: [TtEntry; ENTRIES_PER_CLUSTER],
}

pub struct TranspositionTable {
    clusters: Vec<RwLock<Cluster>>,
    generation: AtomicU8,
}

impl TranspositionTable {
    pub fn new_mb(hash_mb: usize) -> Self {
        Self::with_cluster_count(cluster_count_for_mb(hash_mb))
    }

    pub fn clear(&self) {
        for cluster in &self.clusters {
            *cluster.write().expect("TT cluster lock poisoned") = Cluster::default();
        }
        self.generation.store(0, Ordering::Relaxed);
    }

    pub fn new_generation(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    pub fn probe(&self, key: u64) -> Option<TtHit> {
        let cluster = self.clusters[self.cluster_index(key)]
            .read()
            .expect("TT cluster lock poisoned");
        for entry in cluster.entries {
            if entry.occupied && entry.key_tag == key {
                return Some(TtHit {
                    key_tag: entry.key_tag,
                    best_move: entry.best_move,
                    score: entry.score,
                    eval: entry.eval,
                    depth: entry.depth,
                    bound: entry.bound,
                    generation: entry.generation,
                });
            }
        }
        None
    }

    pub fn store(&self, key: u64, store: TtStore) {
        let generation = self.generation.load(Ordering::Relaxed);
        let cluster_index = self.cluster_index(key);
        let mut cluster = self.clusters[cluster_index]
            .write()
            .expect("TT cluster lock poisoned");
        let replacement_index = select_replacement_slot(&cluster.entries, key, generation);

        cluster.entries[replacement_index] = TtEntry {
            occupied: true,
            key_tag: key,
            best_move: store.best_move,
            score: store.score,
            eval: store.eval,
            depth: store.depth,
            bound: store.bound,
            generation,
        };
    }

    fn cluster_index(&self, key: u64) -> usize {
        ((key as u128).wrapping_mul(self.clusters.len() as u128) >> 64) as usize
    }

    fn with_cluster_count(cluster_count: usize) -> Self {
        Self {
            clusters: (0..cluster_count.max(1))
                .map(|_| RwLock::new(Cluster::default()))
                .collect(),
            generation: AtomicU8::new(0),
        }
    }

    #[cfg(test)]
    fn with_cluster_count_for_test(cluster_count: usize) -> Self {
        Self::with_cluster_count(cluster_count)
    }

    #[cfg(any(test, debug_assertions))]
    pub fn debug_entry_count(&self) -> usize {
        self.clusters
            .iter()
            .map(|cluster| {
                cluster
                    .read()
                    .expect("TT cluster lock poisoned")
                    .entries
                    .iter()
                    .filter(|entry| entry.occupied)
                    .count()
            })
            .sum()
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
    let bytes = hash_mb.max(1) * 1024 * 1024;
    (bytes / size_of::<Cluster>()).max(1)
}

fn select_replacement_slot(
    entries: &[TtEntry; ENTRIES_PER_CLUSTER],
    key: u64,
    generation: u8,
) -> usize {
    for (index, entry) in entries.iter().enumerate() {
        if entry.occupied && entry.key_tag == key {
            return index;
        }
    }

    for (index, entry) in entries.iter().enumerate() {
        if !entry.occupied {
            return index;
        }
    }

    let mut best_index = 0usize;
    let mut worst_value = replacement_value(entries[0], generation);
    for (index, entry) in entries.iter().enumerate().skip(1) {
        let value = replacement_value(*entry, generation);
        if value < worst_value {
            worst_value = value;
            best_index = index;
        }
    }
    best_index
}

fn replacement_value(entry: TtEntry, generation: u8) -> i16 {
    let age = generation.wrapping_sub(entry.generation) as i16;
    entry.depth as i16 - age * AGE_DEPTH_PENALTY
}

#[cfg(test)]
mod tests {
    use super::{
        Bound, DEFAULT_HASH_MB, ENTRIES_PER_CLUSTER, TranspositionTable, TtStore,
        denormalize_score_from_tt, normalize_score_for_store,
    };
    use crate::core::{Move, Square};

    fn square(text: &str) -> Square {
        Square::from_coord_text(text).expect("test square must parse")
    }

    fn sample_move() -> Move {
        Move::new(square("e2"), square("e4"))
    }

    #[test]
    fn probe_misses_on_empty_table() {
        let table = TranspositionTable::new_mb(DEFAULT_HASH_MB);
        assert!(table.probe(0x1234_5678).is_none());
    }

    #[test]
    fn store_and_probe_round_trip_fields() {
        let table = TranspositionTable::with_cluster_count_for_test(1);
        table.new_generation();
        table.store(
            0xabc,
            TtStore {
                best_move: sample_move(),
                score: 42,
                eval: 17,
                depth: 6,
                bound: Bound::Exact,
            },
        );

        let hit = table.probe(0xabc).expect("stored key must probe");
        assert_eq!(hit.key_tag, 0xabc);
        assert_eq!(hit.best_move, sample_move());
        assert_eq!(hit.score, 42);
        assert_eq!(hit.eval, 17);
        assert_eq!(hit.depth, 6);
        assert_eq!(hit.bound, Bound::Exact);
    }

    #[test]
    fn same_key_overwrites_before_replacement() {
        let table = TranspositionTable::with_cluster_count_for_test(1);
        table.store(
            0xabc,
            TtStore {
                best_move: sample_move(),
                score: 1,
                eval: 2,
                depth: 3,
                bound: Bound::Upper,
            },
        );
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
                TtStore {
                    best_move: sample_move(),
                    score: index as i16,
                    eval: 0,
                    depth: (index + 1) as u8,
                    bound: Bound::Exact,
                },
            );
            table.new_generation();
        }

        table.store(
            99,
            TtStore {
                best_move: sample_move(),
                score: 99,
                eval: 0,
                depth: 9,
                bound: Bound::Exact,
            },
        );

        assert!(
            table.probe(1).is_none(),
            "oldest shallowest entry should be replaced"
        );
        assert!(table.probe(99).is_some());
    }

    #[test]
    fn collision_does_not_return_wrong_key() {
        let table = TranspositionTable::with_cluster_count_for_test(1);
        table.store(
            0x111,
            TtStore {
                best_move: sample_move(),
                score: 1,
                eval: 0,
                depth: 1,
                bound: Bound::Exact,
            },
        );
        table.store(
            0x222,
            TtStore {
                best_move: sample_move(),
                score: 2,
                eval: 0,
                depth: 1,
                bound: Bound::Exact,
            },
        );

        let hit = table.probe(0x111).expect("first key should still exist");
        assert_eq!(hit.key_tag, 0x111);
        assert!(table.probe(0x333).is_none());
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
