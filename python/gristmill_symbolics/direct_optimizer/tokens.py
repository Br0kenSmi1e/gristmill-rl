from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

TOKEN_FIELDS = ("kind", "keyword", "scalar_type", "scalar_value", "mask")
SENTINEL = -1

KIND = {"PAD": 0, "BOS": 1, "EOS": 2, "KEYWORD": 3, "SCALAR": 4}
KEYWORD = {
    "range": 0,
    "id": 1,
    "size": 2,
    "tensor": 3,
    "symmetry": 4,
    "action": 5,
    "perm": 6,
    "endtensor": 7,
    "endsymmetry": 8,
    "def": 9,
    "base": 10,
    "ext": 11,
    "term": 12,
    "coeff": 13,
    "numer": 14,
    "denom": 15,
    "sum": 16,
    "factor": 17,
    "index": 18,
    "endfactor": 19,
    "endterm": 20,
    "enddef": 21,
}
SCALAR_TYPE = {
    "range_id": 0,
    "tensor_id": 1,
    "index_id": 2,
    "dim_size": 3,
    "coeff_num": 4,
    "coeff_den": 5,
    "sym_action": 6,
    "axis": 7,
}
SYM_ACTION_VALUE = {"Identity": 0, "Negate": 1}
VALUE_SYM_ACTION = {value: key for key, value in SYM_ACTION_VALUE.items()}


@dataclass(frozen=True)
class LogicalToken:
    kind: str
    keyword: str | None = None
    scalar_type: str | None = None
    scalar_value: int | str | None = None


VALUE_KIND = {value: key for key, value in KIND.items()}
VALUE_KEYWORD = {value: key for key, value in KEYWORD.items()}
VALUE_SCALAR_TYPE = {value: key for key, value in SCALAR_TYPE.items()}

_RecordPattern = tuple[tuple[str, str], ...]
_RECORD_PATTERNS: Mapping[str, _RecordPattern] = {
    "range": (
        ("keyword", "range"),
        ("keyword", "id"),
        ("scalar", "range_id"),
        ("keyword", "size"),
        ("scalar", "dim_size"),
    ),
    "tensor": (
        ("keyword", "tensor"),
        ("keyword", "id"),
        ("scalar", "tensor_id"),
    ),
    "symmetry": (
        ("keyword", "symmetry"),
        ("keyword", "action"),
        ("scalar", "sym_action"),
    ),
    "perm": (("keyword", "perm"), ("scalar", "axis")),
    "endtensor": (("keyword", "endtensor"),),
    "endsymmetry": (("keyword", "endsymmetry"),),
    "def": (
        ("keyword", "def"),
        ("keyword", "base"),
        ("scalar", "tensor_id"),
    ),
    "ext": (
        ("keyword", "ext"),
        ("keyword", "id"),
        ("scalar", "index_id"),
        ("keyword", "range"),
        ("scalar", "range_id"),
    ),
    "term": (("keyword", "term"),),
    "coeff": (
        ("keyword", "coeff"),
        ("keyword", "numer"),
        ("scalar", "coeff_num"),
        ("keyword", "denom"),
        ("scalar", "coeff_den"),
    ),
    "sum": (
        ("keyword", "sum"),
        ("keyword", "id"),
        ("scalar", "index_id"),
        ("keyword", "range"),
        ("scalar", "range_id"),
    ),
    "factor": (
        ("keyword", "factor"),
        ("keyword", "tensor"),
        ("scalar", "tensor_id"),
    ),
    "index": (("keyword", "index"), ("scalar", "index_id")),
    "endfactor": (("keyword", "endfactor"),),
    "endterm": (("keyword", "endterm"),),
    "enddef": (("keyword", "enddef"),),
}


def text_to_logical_tokens(text: str) -> list[LogicalToken]:
    tokens: list[LogicalToken] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if line == "":
            raise ValueError(f"line {line_number}: empty record")
        tokens.extend(
            _text_part_to_logical_token(part, line_number)
            for part in line.split()
        )
    return tokens


def logical_tokens_to_text(tokens: list[LogicalToken]) -> str:
    record_tokens = _without_control_tokens(tokens)
    lines: list[str] = []
    position = 0
    while position < len(record_tokens):
        start = record_tokens[position]
        if start.kind != "KEYWORD" or start.keyword is None:
            raise ValueError(f"expected record keyword at token {position}")
        pattern = _RECORD_PATTERNS.get(start.keyword)
        if pattern is None:
            raise ValueError(
                f"unexpected keyword '{start.keyword}' at token {position}"
            )
        if position + len(pattern) > len(record_tokens):
            raise ValueError(f"incomplete record for {start.keyword}")
        parts = [
            _render_expected_token(record_tokens[position + offset], expected)
            for offset, expected in enumerate(pattern)
        ]
        lines.append(" ".join(parts))
        position += len(pattern)
    return "\n".join(lines)


def encode_text(text: str) -> dict[str, np.ndarray]:
    logical_tokens = [
        LogicalToken("BOS"),
        *text_to_logical_tokens(text),
        LogicalToken("EOS"),
    ]
    return _encode_logical_tokens(logical_tokens)


def decode_token_row_to_text(tokens: Mapping[str, Any]) -> str:
    arrays = _token_arrays_1d(tokens)
    logical_tokens: list[LogicalToken] = []
    for position in range(len(arrays["kind"])):
        if not bool(arrays["mask"][position]):
            continue
        kind_id = _int_array_value(arrays["kind"][position], "kind", position)
        kind = VALUE_KIND.get(kind_id)
        if kind is None:
            raise ValueError(f"token {position}: unknown kind {kind_id}")
        if kind == "PAD":
            continue
        if kind == "BOS":
            logical_tokens.append(LogicalToken("BOS"))
            continue
        if kind == "EOS":
            logical_tokens.append(LogicalToken("EOS"))
            break
        if kind == "KEYWORD":
            keyword_id = _int_array_value(
                arrays["keyword"][position],
                "keyword",
                position,
            )
            keyword = VALUE_KEYWORD.get(keyword_id)
            if keyword is None:
                raise ValueError(f"token {position}: unknown keyword {keyword_id}")
            logical_tokens.append(LogicalToken("KEYWORD", keyword=keyword))
            continue
        if kind == "SCALAR":
            scalar_type_id = _int_array_value(
                arrays["scalar_type"][position],
                "scalar_type",
                position,
            )
            scalar_type = VALUE_SCALAR_TYPE.get(scalar_type_id)
            if scalar_type is None:
                raise ValueError(
                    f"token {position}: unknown scalar_type {scalar_type_id}"
                )
            scalar_value = _int_array_value(
                arrays["scalar_value"][position],
                "scalar_value",
                position,
            )
            logical_tokens.append(
                LogicalToken(
                    "SCALAR",
                    scalar_type=scalar_type,
                    scalar_value=scalar_value,
                )
            )
            continue
        raise ValueError(f"token {position}: unsupported kind {kind}")
    return logical_tokens_to_text(logical_tokens)


def pad_tokens(tokens: Mapping[str, Any], *, length: int) -> dict[str, np.ndarray]:
    arrays = _token_arrays_1d(tokens)
    if length < 0:
        raise ValueError("length must be nonnegative")
    current_length = len(arrays["kind"])
    if current_length > length:
        raise ValueError(
            "cannot pad token row of length "
            f"{current_length} to shorter length {length}"
        )
    padded = {
        "kind": np.full(length, KIND["PAD"], dtype=np.int64),
        "keyword": np.full(length, SENTINEL, dtype=np.int64),
        "scalar_type": np.full(length, SENTINEL, dtype=np.int64),
        "scalar_value": np.full(length, SENTINEL, dtype=np.int64),
        "mask": np.zeros(length, dtype=bool),
    }
    for field in TOKEN_FIELDS:
        padded[field][:current_length] = arrays[field]
    return padded


def repeat_token_row(
    tokens: Mapping[str, Any],
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    if batch_size < 0:
        raise ValueError("batch_size must be nonnegative")
    arrays = _token_arrays_1d(tokens)
    return {
        field: np.repeat(arrays[field][np.newaxis, :], batch_size, axis=0)
        for field in TOKEN_FIELDS
    }


def validate_scalar_bounds(
    tokens: Mapping[str, Any],
    *,
    scalar_value_min: int,
    scalar_value_max: int,
) -> None:
    if scalar_value_min > scalar_value_max:
        raise ValueError("scalar_value_min must be <= scalar_value_max")
    arrays = _token_arrays_same_shape(tokens)
    scalar_positions = (
        np.asarray(arrays["mask"], dtype=bool)
        & (arrays["kind"] == KIND["SCALAR"])
    )
    if not np.any(scalar_positions):
        return
    scalar_values = arrays["scalar_value"][scalar_positions]
    out_of_bounds = (scalar_values < scalar_value_min) | (
        scalar_values > scalar_value_max
    )
    if np.any(out_of_bounds):
        bad_value = int(scalar_values[np.flatnonzero(out_of_bounds)[0]])
        raise ValueError(
            "scalar_value out of bounds: "
            f"{bad_value} not in [{scalar_value_min}, {scalar_value_max}]"
        )


def _text_part_to_logical_token(part: str, line_number: int) -> LogicalToken:
    if part in KEYWORD:
        return LogicalToken("KEYWORD", keyword=part)
    scalar_type, separator, raw_value = part.partition(":")
    if separator == "" or scalar_type == "" or raw_value == "":
        raise ValueError(f"line {line_number}: malformed scalar token '{part}'")
    if scalar_type not in SCALAR_TYPE:
        raise ValueError(f"line {line_number}: unknown scalar type '{scalar_type}'")
    if scalar_type == "sym_action":
        if raw_value not in SYM_ACTION_VALUE:
            raise ValueError(f"line {line_number}: unknown sym_action '{raw_value}'")
        return LogicalToken(
            "SCALAR",
            scalar_type=scalar_type,
            scalar_value=raw_value,
        )
    try:
        scalar_value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"line {line_number}: malformed {scalar_type}: {raw_value}"
        ) from exc
    return LogicalToken("SCALAR", scalar_type=scalar_type, scalar_value=scalar_value)


def _without_control_tokens(tokens: list[LogicalToken]) -> list[LogicalToken]:
    record_tokens: list[LogicalToken] = []
    for token in tokens:
        kind = token.kind.upper()
        if kind in {"PAD", "BOS"}:
            continue
        if kind == "EOS":
            break
        if kind == "KEYWORD":
            if token.keyword not in KEYWORD:
                raise ValueError(f"unknown keyword: {token.keyword}")
            record_tokens.append(LogicalToken("KEYWORD", keyword=token.keyword))
            continue
        if kind == "SCALAR":
            if token.scalar_type not in SCALAR_TYPE:
                raise ValueError(f"unknown scalar type: {token.scalar_type}")
            _scalar_text(token.scalar_type, token.scalar_value)
            record_tokens.append(
                LogicalToken(
                    "SCALAR",
                    scalar_type=token.scalar_type,
                    scalar_value=token.scalar_value,
                )
            )
            continue
        raise ValueError(f"unknown token kind: {token.kind}")
    return record_tokens


def _render_expected_token(token: LogicalToken, expected: tuple[str, str]) -> str:
    expected_kind, expected_value = expected
    if expected_kind == "keyword":
        if token.kind != "KEYWORD" or token.keyword != expected_value:
            raise ValueError(f"expected keyword {expected_value}")
        return expected_value
    if token.kind != "SCALAR" or token.scalar_type != expected_value:
        raise ValueError(f"expected scalar {expected_value}")
    return _scalar_text(expected_value, token.scalar_value)


def _scalar_text(scalar_type: str, scalar_value: int | str | None) -> str:
    if scalar_value is None:
        raise ValueError(f"missing scalar value for {scalar_type}")
    if scalar_type == "sym_action":
        if isinstance(scalar_value, str):
            if scalar_value not in SYM_ACTION_VALUE:
                raise ValueError(f"unknown sym_action: {scalar_value}")
            text_value = scalar_value
        else:
            value = _int_value(scalar_value, scalar_type)
            text_value = VALUE_SYM_ACTION.get(value)
            if text_value is None:
                raise ValueError(f"unknown sym_action value: {value}")
        return f"{scalar_type}:{text_value}"
    value = _int_value(scalar_value, scalar_type)
    return f"{scalar_type}:{value}"


def _encode_logical_tokens(tokens: list[LogicalToken]) -> dict[str, np.ndarray]:
    length = len(tokens)
    encoded = {
        "kind": np.full(length, SENTINEL, dtype=np.int64),
        "keyword": np.full(length, SENTINEL, dtype=np.int64),
        "scalar_type": np.full(length, SENTINEL, dtype=np.int64),
        "scalar_value": np.full(length, SENTINEL, dtype=np.int64),
        "mask": np.ones(length, dtype=bool),
    }
    for position, token in enumerate(tokens):
        kind = token.kind.upper()
        if kind not in KIND:
            raise ValueError(f"unknown token kind: {token.kind}")
        encoded["kind"][position] = KIND[kind]
        if kind == "KEYWORD":
            if token.keyword not in KEYWORD:
                raise ValueError(f"unknown keyword: {token.keyword}")
            encoded["keyword"][position] = KEYWORD[token.keyword]
        elif kind == "SCALAR":
            if token.scalar_type not in SCALAR_TYPE:
                raise ValueError(f"unknown scalar type: {token.scalar_type}")
            encoded["scalar_type"][position] = SCALAR_TYPE[token.scalar_type]
            encoded["scalar_value"][position] = _scalar_int_value(
                token.scalar_type,
                token.scalar_value,
            )
    return encoded


def _scalar_int_value(scalar_type: str, scalar_value: int | str | None) -> int:
    if scalar_value is None:
        raise ValueError(f"missing scalar value for {scalar_type}")
    if scalar_type == "sym_action":
        if isinstance(scalar_value, str):
            value = SYM_ACTION_VALUE.get(scalar_value)
            if value is None:
                raise ValueError(f"unknown sym_action: {scalar_value}")
            return value
        value = _int_value(scalar_value, scalar_type)
        if value not in VALUE_SYM_ACTION:
            raise ValueError(f"unknown sym_action value: {value}")
        return value
    return _int_value(scalar_value, scalar_type)


def _int_value(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"expected integer {context}, got {value!r}")
    return int(value)


def _token_arrays_1d(tokens: Mapping[str, Any]) -> dict[str, np.ndarray]:
    arrays = _token_arrays_same_shape(tokens)
    for field, array in arrays.items():
        if array.ndim != 1:
            raise ValueError(f"expected {field} to be a 1D token row")
    return arrays


def _token_arrays_same_shape(tokens: Mapping[str, Any]) -> dict[str, np.ndarray]:
    missing = [field for field in TOKEN_FIELDS if field not in tokens]
    if missing:
        raise ValueError(f"missing token fields: {missing}")
    arrays = {field: np.asarray(tokens[field]) for field in TOKEN_FIELDS}
    shape = arrays["kind"].shape
    for field, array in arrays.items():
        if array.shape != shape:
            raise ValueError(
                f"expected {field} shape {array.shape} to match kind shape {shape}"
            )
    return arrays


def _int_array_value(value: Any, field: str, position: int) -> int:
    try:
        return _int_value(value, field)
    except ValueError as exc:
        raise ValueError(f"token {position}: malformed {field}") from exc
