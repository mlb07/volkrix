use crate::core::{Move, MoveList, Position};

pub fn perft(position: &mut Position, depth: u8) -> u64 {
    if depth == 0 {
        return 1;
    }

    let mut moves = MoveList::new();
    position.generate_legal_moves(&mut moves);

    if depth == 1 {
        return moves.len() as u64;
    }

    let mut nodes = 0u64;
    for mv in moves.as_slice().iter().copied() {
        let undo = position
            .make_move(mv)
            .expect("moves returned from generate_legal_moves must be legal");
        nodes += perft(position, depth - 1);
        position.unmake_move(mv, undo);
    }

    nodes
}

pub fn divide(position: &mut Position, depth: u8) -> Vec<(Move, u64)> {
    let mut moves = MoveList::new();
    position.generate_legal_moves(&mut moves);

    let mut results = Vec::with_capacity(moves.len());
    for mv in moves.as_slice().iter().copied() {
        let undo = position
            .make_move(mv)
            .expect("moves returned from generate_legal_moves must be legal");
        let nodes = if depth == 0 {
            1
        } else {
            perft(position, depth.saturating_sub(1))
        };
        position.unmake_move(mv, undo);
        results.push((mv, nodes));
    }

    results.sort_by_key(|(mv, _)| mv.to_string());
    results
}
