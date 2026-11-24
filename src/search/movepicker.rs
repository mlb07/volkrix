use crate::core::{Move, MoveList, Position, movelist::MAX_MOVES};

use super::root::{MoveOrderHints, SearchContext};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
enum MoveStage {
    Pv,
    Tt,
    GoodCaptures,
    Promotions,
    Killers,
    Quiets,
    #[default]
    BadCaptures,
}

impl MoveStage {
    fn next(self) -> Option<Self> {
        match self {
            Self::Pv => Some(Self::Tt),
            Self::Tt => Some(Self::GoodCaptures),
            Self::GoodCaptures => Some(Self::Promotions),
            Self::Promotions => Some(Self::Killers),
            Self::Killers => Some(Self::Quiets),
            Self::Quiets => Some(Self::BadCaptures),
            Self::BadCaptures => None,
        }
    }
}

pub(crate) struct SearchMoves {
    list: Option<MoveList>,
    noisy_only: bool,
}

impl SearchMoves {
    pub(crate) fn new(noisy_only: bool) -> Self {
        Self {
            list: None,
            noisy_only,
        }
    }

    pub(crate) fn get(&mut self, position: &mut Position) -> &MoveList {
        self.list.get_or_insert_with(|| {
            let mut moves = MoveList::new();
            if self.noisy_only {
                position.generate_legal_noisy_moves(&mut moves);
            } else {
                position.generate_legal_moves(&mut moves);
            }
            moves
        })
    }
}

/// Tries a validated move hint before paying to score the full list.
pub(crate) struct DeferredMovePicker {
    hints: MoveOrderHints,
    first: Option<Move>,
    remaining: Option<MovePicker>,
}

impl DeferredMovePicker {
    pub(crate) fn new(hints: MoveOrderHints) -> Self {
        Self {
            hints,
            first: hints.pv_move.or(hints.tt_move),
            remaining: None,
        }
    }

    pub(crate) fn next(
        &mut self,
        context: &SearchContext,
        position: &mut Position,
        legal_moves: &mut SearchMoves,
    ) -> Option<Move> {
        if let Some(mv) = self.first.take() {
            return Some(mv);
        }

        self.remaining
            .get_or_insert_with(|| {
                let legal_moves = legal_moves.get(position);
                let mut picker = MovePicker::new(context, position, legal_moves, self.hints);
                if let Some(first) = self.hints.pv_move.or(self.hints.tt_move) {
                    // Keep the normal picker's swap, which also determines the order of ties.
                    let picked = picker.next();
                    debug_assert_eq!(picked, Some(first));
                }
                picker
            })
            .next()
    }
}

pub(crate) struct MovePicker {
    moves: [Move; MAX_MOVES],
    scores: [i32; MAX_MOVES],
    stages: [MoveStage; MAX_MOVES],
    len: usize,
    next_index: usize,
    stage: MoveStage,
}

impl MovePicker {
    pub(crate) fn new(
        context: &SearchContext,
        position: &Position,
        legal_moves: &MoveList,
        hints: MoveOrderHints,
    ) -> Self {
        let mut picker = Self {
            moves: [Move::NONE; MAX_MOVES],
            scores: [0; MAX_MOVES],
            stages: [MoveStage::BadCaptures; MAX_MOVES],
            len: 0,
            next_index: 0,
            stage: MoveStage::Pv,
        };

        for mv in legal_moves.as_slice().iter().copied() {
            if hints.quiescence_only && !super::root::is_quiescence_move(mv, position) {
                continue;
            }

            let score = context.score_move(position, mv, hints);
            picker.moves[picker.len] = mv;
            picker.scores[picker.len] = score;
            picker.stages[picker.len] = classify_move(mv, score, hints);
            picker.len += 1;
        }

        picker
    }

    pub(crate) fn next(&mut self) -> Option<Move> {
        loop {
            let Some(best_index) = self.best_index_for_stage(self.stage) else {
                self.stage = self.stage.next()?;
                continue;
            };
            self.moves.swap(self.next_index, best_index);
            self.scores.swap(self.next_index, best_index);
            self.stages.swap(self.next_index, best_index);
            let mv = self.moves[self.next_index];
            self.next_index += 1;
            return Some(mv);
        }
    }

    pub(crate) fn ordered(mut self) -> Vec<Move> {
        let mut ordered = Vec::with_capacity(self.len);
        while let Some(mv) = self.next() {
            ordered.push(mv);
        }
        ordered
    }

    /// Returns a bounded tactical prefix for speculative ProbCut probes.
    ///
    /// Promotions remain on the normal full-search path. Capture stages retain TT/PV priority,
    /// then use SEE and capture-history scores, so the most credible fail-highs are paid for
    /// first without changing the main move picker's ordering contract.
    #[cfg(any(test, debug_assertions, feature = "internal-testing"))]
    pub(crate) fn ordered_probcut_candidates(
        mut self,
        position: &Position,
        limit: usize,
    ) -> Vec<Move> {
        let mut candidates = Vec::with_capacity(limit.min(self.len));
        while candidates.len() < limit {
            let Some(mv) = self.next() else {
                break;
            };
            if mv.is_capture() && !mv.is_promotion() && position.see(mv).0 >= 0 {
                candidates.push(mv);
            }
        }
        candidates
    }

    fn best_index_for_stage(&self, target: MoveStage) -> Option<usize> {
        let mut best = None;
        let mut best_score = i32::MIN;

        for index in self.next_index..self.len {
            if self.stages[index] != target {
                continue;
            }
            let score = self.scores[index];
            if score > best_score {
                best_score = score;
                best = Some(index);
            }
        }

        best
    }
}

fn classify_move(mv: Move, score: i32, hints: MoveOrderHints) -> MoveStage {
    if hints.pv_move == Some(mv) {
        return MoveStage::Pv;
    }
    if hints.tt_move == Some(mv) {
        return MoveStage::Tt;
    }
    if mv.is_capture() {
        return if score >= 260_000 {
            MoveStage::GoodCaptures
        } else {
            MoveStage::BadCaptures
        };
    }
    if mv.is_promotion() {
        return MoveStage::Promotions;
    }
    if score >= 130_000 {
        return MoveStage::Killers;
    }
    MoveStage::Quiets
}

#[cfg(test)]
mod tests {
    use super::{DeferredMovePicker, MovePicker, SearchMoves};
    use crate::{
        core::{ParsedMove, Position},
        search::{
            SearchLimits,
            root::{MoveOrderHints, SearchContext},
        },
    };

    #[test]
    fn staged_picker_prefers_pv_tt_then_good_captures_before_quiets() {
        let mut position = Position::from_fen("6k1/8/8/5r2/3N4/8/4p3/6K1 w - - 0 1")
            .expect("FEN parse must succeed");
        let mut legal_moves = crate::core::MoveList::new();
        position.generate_legal_moves(&mut legal_moves);
        let pv = legal_moves
            .iter()
            .copied()
            .find(|mv| mv.matches_parsed(ParsedMove::parse("d4f5").expect("parse must succeed")))
            .expect("pv move must exist");
        let tt = legal_moves
            .iter()
            .copied()
            .find(|mv| mv.matches_parsed(ParsedMove::parse("d4e2").expect("parse must succeed")))
            .expect("tt move must exist");
        let quiet = legal_moves
            .iter()
            .copied()
            .find(|mv| mv.matches_parsed(ParsedMove::parse("g1h1").expect("parse must succeed")))
            .expect("quiet move must exist");

        let context = SearchContext::new(SearchLimits::new(3));
        let mut picker = MovePicker::new(
            &context,
            &position,
            &legal_moves,
            MoveOrderHints {
                ply: 0,
                quiescence_only: false,
                pv_move: Some(pv),
                tt_move: Some(tt),
            },
        );

        assert_eq!(picker.next(), Some(pv));
        assert_eq!(picker.next(), Some(tt));
        let remaining = picker.ordered();
        assert!(remaining.contains(&quiet));
    }

    #[test]
    fn deferred_picker_returns_the_hint_before_generating_alternatives() {
        let mut position = Position::startpos();
        let context = SearchContext::new(SearchLimits::new(3));
        let mut eager_moves = crate::core::MoveList::new();
        position.generate_legal_moves(&mut eager_moves);
        let hints = MoveOrderHints {
            pv_move: Some(eager_moves.get(9)),
            tt_move: Some(eager_moves.get(5)),
            ..MoveOrderHints::default()
        };
        let expected = MovePicker::new(&context, &position, &eager_moves, hints).ordered();
        let mut deferred = DeferredMovePicker::new(hints);
        let mut legal = SearchMoves::new(false);
        assert_eq!(
            deferred.next(&context, &mut position, &mut legal),
            hints.pv_move
        );
        assert!(legal.list.is_none());
        assert!(deferred.remaining.is_none());
        for mv in expected.into_iter().skip(1) {
            assert_eq!(deferred.next(&context, &mut position, &mut legal), Some(mv));
        }
        assert_eq!(deferred.next(&context, &mut position, &mut legal), None);
    }

    #[test]
    fn probcut_prefix_filters_before_bounding_and_is_deterministic() {
        let mut position = Position::from_fen("6k1/8/8/5r2/3N4/8/4p3/6K1 w - - 0 1")
            .expect("FEN parse must succeed");
        let mut legal_moves = crate::core::MoveList::new();
        position.generate_legal_moves(&mut legal_moves);
        let tt = legal_moves
            .iter()
            .copied()
            .find(|mv| mv.matches_parsed(ParsedMove::parse("d4f5").expect("parse must succeed")))
            .expect("winning TT capture must exist");
        let hints = MoveOrderHints {
            ply: 0,
            quiescence_only: false,
            pv_move: None,
            tt_move: Some(tt),
        };
        let context = SearchContext::new(SearchLimits::new(3));

        let first = MovePicker::new(&context, &position, &legal_moves, hints)
            .ordered_probcut_candidates(&position, 1);
        let second = MovePicker::new(&context, &position, &legal_moves, hints)
            .ordered_probcut_candidates(&position, 1);
        assert_eq!(first, second);
        assert_eq!(first, vec![tt]);
        assert!(
            first
                .iter()
                .all(|mv| { mv.is_capture() && !mv.is_promotion() && position.see(*mv).0 >= 0 })
        );
    }
}
