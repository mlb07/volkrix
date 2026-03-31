use crate::core::{Move, MoveList, Position, movelist::MAX_MOVES, position::MoveGenStage};

use super::root::{MoveOrderHints, SearchContext};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PickerStage {
    Lead,
    GoodCaptures,
    Quiets,
    BadCaptures,
    Done,
}

#[derive(Clone)]
struct ScoredMoves {
    moves: [Move; MAX_MOVES],
    scores: [i32; MAX_MOVES],
    len: usize,
    index: usize,
}

impl Default for ScoredMoves {
    fn default() -> Self {
        Self {
            moves: [Move::NONE; MAX_MOVES],
            scores: [0; MAX_MOVES],
            len: 0,
            index: 0,
        }
    }
}

impl ScoredMoves {
    fn push_sorted(&mut self, mv: Move, score: i32) {
        assert!(self.len < MAX_MOVES, "staged move list overflow");

        let mut insert = self.len;
        while insert > 0 && score > self.scores[insert - 1] {
            self.moves[insert] = self.moves[insert - 1];
            self.scores[insert] = self.scores[insert - 1];
            insert -= 1;
        }
        self.moves[insert] = mv;
        self.scores[insert] = score;
        self.len += 1;
    }

    fn next(&mut self) -> Option<Move> {
        if self.index >= self.len {
            None
        } else {
            let mv = self.moves[self.index];
            self.index += 1;
            Some(mv)
        }
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }
}

pub(crate) struct MovePicker {
    pv_move: Option<Move>,
    tt_move: Option<Move>,
    lead_index: usize,
    good_captures: ScoredMoves,
    quiets: ScoredMoves,
    bad_captures: ScoredMoves,
    stage: PickerStage,
}

impl MovePicker {
    pub(crate) fn new(
        context: &SearchContext,
        position: &mut Position,
        hints: MoveOrderHints,
    ) -> Self {
        let mut picker = Self {
            pv_move: None,
            tt_move: None,
            lead_index: 0,
            good_captures: ScoredMoves::default(),
            quiets: ScoredMoves::default(),
            bad_captures: ScoredMoves::default(),
            stage: PickerStage::Lead,
        };

        let info = position.check_info();
        if info.checkers != 0 {
            let mut evasions = MoveList::new();
            position.generate_evasions(&info, &mut evasions);
            for mv in evasions.as_slice().iter().copied() {
                if position.is_legal_fast(mv, &info) {
                    picker.push_move(context, position, hints, mv);
                }
            }
            return picker;
        }

        let mut captures = MoveList::new();
        position.generate_pseudo_legal(MoveGenStage::Captures, &mut captures);
        for mv in captures.as_slice().iter().copied() {
            if position.is_legal_fast(mv, &info) {
                picker.push_move(context, position, hints, mv);
            }
        }

        let mut quiets = MoveList::new();
        position.generate_pseudo_legal(MoveGenStage::Quiets, &mut quiets);
        for mv in quiets.as_slice().iter().copied() {
            if hints.quiescence_only && !mv.is_promotion() {
                continue;
            }
            if position.is_legal_fast(mv, &info) {
                picker.push_move(context, position, hints, mv);
            }
        }

        picker
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.pv_move.is_none()
            && self.tt_move.is_none()
            && self.good_captures.is_empty()
            && self.quiets.is_empty()
            && self.bad_captures.is_empty()
    }

    pub(crate) fn next(&mut self) -> Option<Move> {
        loop {
            match self.stage {
                PickerStage::Lead => {
                    while self.lead_index < 2 {
                        let mv = match self.lead_index {
                            0 => self.pv_move,
                            1 => self.tt_move,
                            _ => None,
                        };
                        self.lead_index += 1;
                        if let Some(mv) = mv {
                            return Some(mv);
                        }
                    }
                    self.stage = PickerStage::GoodCaptures;
                }
                PickerStage::GoodCaptures => {
                    if let Some(mv) = self.good_captures.next() {
                        return Some(mv);
                    }
                    self.stage = PickerStage::Quiets;
                }
                PickerStage::Quiets => {
                    if let Some(mv) = self.quiets.next() {
                        return Some(mv);
                    }
                    self.stage = PickerStage::BadCaptures;
                }
                PickerStage::BadCaptures => {
                    if let Some(mv) = self.bad_captures.next() {
                        return Some(mv);
                    }
                    self.stage = PickerStage::Done;
                }
                PickerStage::Done => return None,
            }
        }
    }

    fn push_move(
        &mut self,
        context: &SearchContext,
        position: &Position,
        hints: MoveOrderHints,
        mv: Move,
    ) {
        if hints.pv_move == Some(mv) {
            self.pv_move = Some(mv);
            return;
        }
        if hints.tt_move == Some(mv) {
            if self.pv_move != Some(mv) {
                self.tt_move = Some(mv);
            }
            return;
        }

        if mv.is_capture() {
            let see_score = position.see(mv).0 as i32;
            let score = context.capture_order_score_with_see(position, mv, see_score);
            if see_score < 0 {
                self.bad_captures.push_sorted(mv, score);
            } else {
                self.good_captures.push_sorted(mv, score);
            }
            return;
        }

        let score = context.score_move(position, mv, hints);
        self.quiets.push_sorted(mv, score);
    }
}

#[cfg(test)]
mod tests {
    use super::MovePicker;
    use crate::core::{MoveList, ParsedMove, Position};
    use crate::search::{
        SearchLimits,
        root::{MoveOrderHints, SearchContext},
    };

    #[test]
    fn lead_moves_are_emitted_before_staged_lists() {
        let mut position = Position::startpos();
        let mut legal_moves = MoveList::new();
        position.generate_legal_moves(&mut legal_moves);
        let pv_move = legal_moves
            .iter()
            .copied()
            .find(|mv| mv.matches_parsed(ParsedMove::parse("e2e4").expect("parse must succeed")))
            .expect("pv move must exist");
        let tt_move = legal_moves
            .iter()
            .copied()
            .find(|mv| mv.matches_parsed(ParsedMove::parse("d2d4").expect("parse must succeed")))
            .expect("tt move must exist");

        let context = SearchContext::new(SearchLimits::new(3));
        let mut picker = MovePicker::new(
            &context,
            &mut position,
            MoveOrderHints {
                ply: 0,
                quiescence_only: false,
                pv_move: Some(pv_move),
                tt_move: Some(tt_move),
            },
        );

        assert_eq!(picker.next(), Some(pv_move));
        assert_eq!(picker.next(), Some(tt_move));
    }

    #[test]
    fn bad_captures_are_deferred_after_quiets() {
        let mut position = Position::from_fen("4k3/8/8/5r1q/3N4/8/4p3/4K3 w - - 0 1")
            .expect("FEN parse must succeed");
        let context = SearchContext::new(SearchLimits::new(3));
        let mut picker = MovePicker::new(
            &context,
            &mut position,
            MoveOrderHints {
                ply: 0,
                quiescence_only: false,
                pv_move: None,
                tt_move: None,
            },
        );

        let ordered: Vec<_> = std::iter::from_fn(|| picker.next()).collect();
        let winning_capture = ordered
            .iter()
            .position(|mv| {
                mv.matches_parsed(ParsedMove::parse("d4f5").expect("parse must succeed"))
            })
            .expect("winning capture must exist");
        let quiet = ordered
            .iter()
            .position(|mv| {
                mv.matches_parsed(ParsedMove::parse("d4f3").expect("parse must succeed"))
            })
            .expect("quiet must exist");
        let losing_capture = ordered
            .iter()
            .position(|mv| {
                mv.matches_parsed(ParsedMove::parse("d4e2").expect("parse must succeed"))
            })
            .expect("losing capture must exist");

        assert!(winning_capture < quiet);
        assert!(quiet < losing_capture);
    }
}
