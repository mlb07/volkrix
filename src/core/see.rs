use super::{piece::PieceType, types::Value};

pub(crate) const SEE_VALUES: [i16; 6] = [100, 320, 330, 500, 900, 0];

pub(crate) fn piece_value(piece_type: PieceType) -> Value {
    Value(SEE_VALUES[piece_type.index()])
}

pub(crate) fn promotion_gain(piece_type: PieceType) -> Value {
    Value(piece_value(piece_type).0 - piece_value(PieceType::Pawn).0)
}
