from __future__ import annotations

from enum import IntEnum

SENTINEL = -1


class TOKEN_KIND(IntEnum):
    PAD = 0
    RANGE = 1
    TENSOR_START = 2
    SYMMETRY_START = 3
    SYMMETRY_PERM = 4
    SYMMETRY_END = 5
    TENSOR_END = 6
    DEF_START = 7
    EXT_INDEX = 8
    TERM_START = 9
    COEFF = 10
    SUM_INDEX = 11
    FACTOR_START = 12
    FACTOR_INDEX = 13
    FACTOR_END = 14
    TERM_END = 15
    DEF_END = 16
    ACTION_SPACE_START = 17
    CANDIDATE_START = 18
    SIDE_START = 19
    SIDE_END = 20
    CANDIDATE_END = 21
    ACTION_SPACE_END = 22


class SEGMENT(IntEnum):
    RANGES = 0
    TENSORS = 1
    DEFINITIONS = 2
    ACTION_SPACE = 3


class SIDE(IntEnum):
    LEFT = 0
    RIGHT = 1


class SYM_ACTION(IntEnum):
    IDENTITY = 0
    NEGATE = 1


TOKEN_FIELDS = (
    "token_kind",
    "segment",
    "position",
    "def_index",
    "term_index",
    "factor_index",
    "tensor_id",
    "range_id",
    "index_id",
    "candidate_index",
    "side",
    "coeff_num",
    "coeff_den",
    "symmetry_index",
    "symmetry_action",
    "perm_index",
    "perm_value",
)
