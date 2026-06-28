from __future__ import annotations

from enum import IntEnum

SENTINEL = -1


class TOKEN_KIND(IntEnum):
    PAD = 0
    RANGE = 1
    TENSOR = 2
    DEF_START = 3
    EXT_INDEX = 4
    TERM_START = 5
    COEFF = 6
    SUM_INDEX = 7
    FACTOR_START = 8
    FACTOR_INDEX = 9
    FACTOR_END = 10
    TERM_END = 11
    DEF_END = 12
    ACTION_SPACE_START = 13
    CANDIDATE_START = 14
    SIDE_START = 15
    SIDE_END = 16
    CANDIDATE_END = 17
    ACTION_SPACE_END = 18


class SEGMENT(IntEnum):
    RANGES = 0
    TENSORS = 1
    DEFINITIONS = 2
    ACTION_SPACE = 3


class SIDE(IntEnum):
    LEFT = 0
    RIGHT = 1
    REWRITTEN = 2


STATE_TOKEN_FIELDS = (
    "token_kind",
    "segment",
    "def_index",
    "term_index",
    "factor_index",
    "tensor_id",
    "range_id",
    "index_id",
    "coeff_num",
    "coeff_den",
    "position",
)

ACTION_TOKEN_FIELDS = STATE_TOKEN_FIELDS + ("candidate_index", "side")
