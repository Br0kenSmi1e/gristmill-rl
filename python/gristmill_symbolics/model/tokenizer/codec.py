from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax

from .vocabulary import (
    SENTINEL,
    SEGMENT,
    SIDE,
    SYM_ACTION,
    TOKEN_FIELDS,
    TOKEN_KIND,
)
from .token_arrays import TokenArrays, make_token_arrays, validate_token_arrays


_COMMON_FIELDS = {"token_kind", "segment", "position"}

_KIND_SEGMENTS = {
    TOKEN_KIND.RANGE: (SEGMENT.RANGES,),
    TOKEN_KIND.TENSOR_START: (SEGMENT.TENSORS,),
    TOKEN_KIND.SYMMETRY_START: (SEGMENT.TENSORS,),
    TOKEN_KIND.SYMMETRY_PERM: (SEGMENT.TENSORS,),
    TOKEN_KIND.SYMMETRY_END: (SEGMENT.TENSORS,),
    TOKEN_KIND.TENSOR_END: (SEGMENT.TENSORS,),
    TOKEN_KIND.DEF_START: (SEGMENT.DEFINITIONS, SEGMENT.ACTION_SPACE),
    TOKEN_KIND.EXT_INDEX: (SEGMENT.DEFINITIONS, SEGMENT.ACTION_SPACE),
    TOKEN_KIND.TERM_START: (SEGMENT.DEFINITIONS, SEGMENT.ACTION_SPACE),
    TOKEN_KIND.COEFF: (SEGMENT.DEFINITIONS, SEGMENT.ACTION_SPACE),
    TOKEN_KIND.SUM_INDEX: (SEGMENT.DEFINITIONS, SEGMENT.ACTION_SPACE),
    TOKEN_KIND.FACTOR_START: (SEGMENT.DEFINITIONS, SEGMENT.ACTION_SPACE),
    TOKEN_KIND.FACTOR_INDEX: (SEGMENT.DEFINITIONS, SEGMENT.ACTION_SPACE),
    TOKEN_KIND.FACTOR_END: (SEGMENT.DEFINITIONS, SEGMENT.ACTION_SPACE),
    TOKEN_KIND.TERM_END: (SEGMENT.DEFINITIONS, SEGMENT.ACTION_SPACE),
    TOKEN_KIND.DEF_END: (SEGMENT.DEFINITIONS, SEGMENT.ACTION_SPACE),
    TOKEN_KIND.ACTION_SPACE_START: (SEGMENT.ACTION_SPACE,),
    TOKEN_KIND.CANDIDATE_START: (SEGMENT.ACTION_SPACE,),
    TOKEN_KIND.SIDE_START: (SEGMENT.ACTION_SPACE,),
    TOKEN_KIND.SIDE_END: (SEGMENT.ACTION_SPACE,),
    TOKEN_KIND.CANDIDATE_END: (SEGMENT.ACTION_SPACE,),
    TOKEN_KIND.ACTION_SPACE_END: (SEGMENT.ACTION_SPACE,),
}

_KIND_FIELDS = {
    TOKEN_KIND.RANGE: {"range_id", "coeff_num", "coeff_den"},
    TOKEN_KIND.TENSOR_START: {"tensor_id"},
    TOKEN_KIND.SYMMETRY_START: {
        "tensor_id",
        "symmetry_index",
        "symmetry_action",
    },
    TOKEN_KIND.SYMMETRY_PERM: {
        "tensor_id",
        "symmetry_index",
        "perm_index",
        "perm_value",
    },
    TOKEN_KIND.SYMMETRY_END: {"tensor_id", "symmetry_index"},
    TOKEN_KIND.TENSOR_END: {"tensor_id"},
    TOKEN_KIND.DEF_START: {"def_index", "tensor_id"},
    TOKEN_KIND.EXT_INDEX: {"def_index", "index_id", "range_id"},
    TOKEN_KIND.TERM_START: {"def_index", "term_index"},
    TOKEN_KIND.COEFF: {
        "def_index",
        "term_index",
        "coeff_num",
        "coeff_den",
    },
    TOKEN_KIND.SUM_INDEX: {
        "def_index",
        "term_index",
        "index_id",
        "range_id",
    },
    TOKEN_KIND.FACTOR_START: {
        "def_index",
        "term_index",
        "factor_index",
        "tensor_id",
    },
    TOKEN_KIND.FACTOR_INDEX: {
        "def_index",
        "term_index",
        "factor_index",
        "index_id",
        "range_id",
    },
    TOKEN_KIND.FACTOR_END: {"def_index", "term_index", "factor_index"},
    TOKEN_KIND.TERM_END: {"def_index", "term_index"},
    TOKEN_KIND.DEF_END: {"def_index"},
    TOKEN_KIND.ACTION_SPACE_START: {"def_index"},
    TOKEN_KIND.CANDIDATE_START: {"def_index", "candidate_index"},
    TOKEN_KIND.SIDE_START: {"def_index", "candidate_index", "side"},
    TOKEN_KIND.SIDE_END: {"def_index", "candidate_index", "side"},
    TOKEN_KIND.CANDIDATE_END: {"def_index", "candidate_index"},
    TOKEN_KIND.ACTION_SPACE_END: {"def_index"},
}

_DEFINITION_KINDS = {
    TOKEN_KIND.DEF_START,
    TOKEN_KIND.EXT_INDEX,
    TOKEN_KIND.TERM_START,
    TOKEN_KIND.COEFF,
    TOKEN_KIND.SUM_INDEX,
    TOKEN_KIND.FACTOR_START,
    TOKEN_KIND.FACTOR_INDEX,
    TOKEN_KIND.FACTOR_END,
    TOKEN_KIND.TERM_END,
    TOKEN_KIND.DEF_END,
}


@dataclass
class _Cursor:
    rows: list[dict[str, int]]
    position: int = 0

    def done(self) -> bool:
        return self.position >= len(self.rows)

    def peek(self, kind: TOKEN_KIND) -> bool:
        return not self.done() and self.kind() == kind

    def kind(self) -> TOKEN_KIND:
        return TOKEN_KIND(self.rows[self.position]["token_kind"])

    def take(self, expected: TOKEN_KIND) -> dict[str, int]:
        if self.done():
            raise ValueError(f"expected {expected.name}, got end of tokens")
        row = self.rows[self.position]
        actual = TOKEN_KIND(row["token_kind"])
        if actual != expected:
            raise ValueError(f"expected {expected.name}, got {actual.name}")
        self.position += 1
        return row


def tokenize_computation_snapshot(
    snapshot: dict[str, Any],
) -> tuple[TokenArrays, jax.Array]:
    rows: list[dict[str, int]] = []
    for range_info in snapshot.get("ranges", []):
        _append_range(rows, range_info)
    for tensor in snapshot.get("tensors", []):
        _serialize_tensor(rows, tensor)
    for def_index, definition in enumerate(snapshot.get("definitions", [])):
        _serialize_definition(
            rows,
            definition,
            def_index=def_index,
            segment=SEGMENT.DEFINITIONS,
        )
    return make_token_arrays(rows)


def decode_computation_snapshot(
    tokens: TokenArrays,
    mask: jax.Array,
) -> dict[str, Any]:
    cursor = _Cursor(_token_rows(tokens, mask))
    ranges = _decode_many(cursor, TOKEN_KIND.RANGE, _decode_range)
    tensors = _decode_many(cursor, TOKEN_KIND.TENSOR_START, _decode_tensor)
    definitions = []
    while cursor.peek(TOKEN_KIND.DEF_START):
        scope = {"def_index": len(definitions)}
        definitions.append(_decode_definition(cursor, scope))
    if not cursor.done():
        raise ValueError(f"unexpected {cursor.kind().name}")
    return {
        "ranges": ranges,
        "tensors": tensors,
        "definitions": definitions,
    }


def tokenize_action_space_snapshot(
    snapshot: dict[str, Any],
) -> tuple[TokenArrays, jax.Array]:
    rows: list[dict[str, int]] = []
    def_index = int(snapshot["def_index"])
    _append(rows, TOKEN_KIND.ACTION_SPACE_START, def_index=def_index)
    for candidate_index, candidate in enumerate(
        snapshot.get("candidate_templates", [])
    ):
        _serialize_candidate(rows, candidate, def_index, candidate_index)
    _append(rows, TOKEN_KIND.ACTION_SPACE_END, def_index=def_index)
    return make_token_arrays(rows)


def decode_action_space_snapshot(
    tokens: TokenArrays,
    mask: jax.Array,
) -> dict[str, Any]:
    cursor = _Cursor(_token_rows(tokens, mask))
    start = cursor.take(TOKEN_KIND.ACTION_SPACE_START)
    def_index = _need(start, "def_index")
    candidates = []
    while not cursor.peek(TOKEN_KIND.ACTION_SPACE_END):
        scope = {
            "def_index": def_index,
            "candidate_index": len(candidates),
        }
        candidates.append(_decode_candidate(cursor, scope))
    end = cursor.take(TOKEN_KIND.ACTION_SPACE_END)
    _check_scope(end, {"def_index": def_index})
    if not cursor.done():
        raise ValueError(f"unexpected {cursor.kind().name}")
    return {"def_index": def_index, "candidate_templates": candidates}


def _append(
    rows: list[dict[str, int]],
    token_kind: TOKEN_KIND,
    *,
    segment: SEGMENT = SEGMENT.ACTION_SPACE,
    **fields: int,
) -> None:
    row = {
        "token_kind": int(token_kind),
        "segment": int(segment),
        "position": len(rows),
    }
    row.update({key: int(value) for key, value in fields.items()})
    rows.append(row)


def _append_range(
    rows: list[dict[str, int]],
    range_info: dict[str, Any],
) -> None:
    _append(
        rows,
        TOKEN_KIND.RANGE,
        segment=SEGMENT.RANGES,
        range_id=int(range_info["id"]),
        coeff_num=int(range_info["size"]),
        coeff_den=1,
    )


def _serialize_tensor(
    rows: list[dict[str, int]],
    tensor: dict[str, Any],
) -> None:
    tensor_id = int(tensor["id"])
    _append(
        rows,
        TOKEN_KIND.TENSOR_START,
        segment=SEGMENT.TENSORS,
        tensor_id=tensor_id,
    )
    for symmetry_index, symmetry in enumerate(tensor.get("symmetry", [])):
        _serialize_symmetry(rows, tensor_id, symmetry_index, symmetry)
    _append(
        rows,
        TOKEN_KIND.TENSOR_END,
        segment=SEGMENT.TENSORS,
        tensor_id=tensor_id,
    )


def _serialize_symmetry(
    rows: list[dict[str, int]],
    tensor_id: int,
    symmetry_index: int,
    symmetry: dict[str, Any],
) -> None:
    action = _sym_action_value(str(symmetry["action"]))
    fields = {"tensor_id": tensor_id, "symmetry_index": symmetry_index}
    _append(
        rows,
        TOKEN_KIND.SYMMETRY_START,
        segment=SEGMENT.TENSORS,
        symmetry_action=action,
        **fields,
    )
    for perm_index, perm_value in enumerate(symmetry.get("perm", [])):
        _append_symmetry_perm(rows, fields, perm_index, int(perm_value))
    _append(rows, TOKEN_KIND.SYMMETRY_END, segment=SEGMENT.TENSORS, **fields)


def _append_symmetry_perm(
    rows: list[dict[str, int]],
    fields: dict[str, int],
    perm_index: int,
    perm_value: int,
) -> None:
    _append(
        rows,
        TOKEN_KIND.SYMMETRY_PERM,
        segment=SEGMENT.TENSORS,
        perm_index=perm_index,
        perm_value=perm_value,
        **fields,
    )


def _serialize_definition(
    rows: list[dict[str, int]],
    definition: dict[str, Any],
    *,
    def_index: int,
    segment: SEGMENT,
    candidate_index: int = SENTINEL,
    side: int = SENTINEL,
) -> None:
    scope = _scope(def_index, candidate_index, side)
    _append(
        rows,
        TOKEN_KIND.DEF_START,
        segment=segment,
        tensor_id=int(definition["base"]),
        **scope,
    )
    for ext in definition.get("ext_indices", []):
        _append_index(rows, TOKEN_KIND.EXT_INDEX, segment, ext, scope)
    ranges = _definition_index_ranges(definition)
    for term_index, term in enumerate(definition.get("terms", [])):
        _serialize_term(rows, term, term_index, segment, scope, ranges)
    _append(rows, TOKEN_KIND.DEF_END, segment=segment, **scope)


def _serialize_term(
    rows: list[dict[str, int]],
    term: dict[str, Any],
    term_index: int,
    segment: SEGMENT,
    scope: dict[str, int],
    ranges: dict[int, int],
) -> None:
    term_scope = dict(scope, term_index=term_index)
    _append(rows, TOKEN_KIND.TERM_START, segment=segment, **term_scope)
    numer, denom = _coeff_parts(term["coeff"])
    _append(
        rows,
        TOKEN_KIND.COEFF,
        segment=segment,
        coeff_num=numer,
        coeff_den=denom,
        **term_scope,
    )
    for index in term.get("sum_indices", []):
        _append_index(rows, TOKEN_KIND.SUM_INDEX, segment, index, term_scope)
    for factor_index, factor in enumerate(term.get("factors", [])):
        _serialize_factor(
            rows,
            factor,
            factor_index,
            segment,
            term_scope,
            ranges,
        )
    _append(rows, TOKEN_KIND.TERM_END, segment=segment, **term_scope)


def _serialize_factor(
    rows: list[dict[str, int]],
    factor: dict[str, Any],
    factor_index: int,
    segment: SEGMENT,
    term_scope: dict[str, int],
    ranges: dict[int, int],
) -> None:
    scope = dict(term_scope, factor_index=factor_index)
    _append(
        rows,
        TOKEN_KIND.FACTOR_START,
        segment=segment,
        tensor_id=int(factor["tensor"]),
        **scope,
    )
    for index_id in factor.get("indices", []):
        index_id = int(index_id)
        _append_factor_index(rows, segment, scope, index_id, ranges)
    _append(rows, TOKEN_KIND.FACTOR_END, segment=segment, **scope)


def _append_index(
    rows: list[dict[str, int]],
    kind: TOKEN_KIND,
    segment: SEGMENT,
    index: dict[str, Any],
    scope: dict[str, int],
) -> None:
    _append(
        rows,
        kind,
        segment=segment,
        index_id=int(index["id"]),
        range_id=int(index["range"]),
        **scope,
    )


def _append_factor_index(
    rows: list[dict[str, int]],
    segment: SEGMENT,
    scope: dict[str, int],
    index_id: int,
    ranges: dict[int, int],
) -> None:
    _append(
        rows,
        TOKEN_KIND.FACTOR_INDEX,
        segment=segment,
        index_id=index_id,
        range_id=ranges.get(index_id, SENTINEL),
        **scope,
    )


def _serialize_candidate(
    rows: list[dict[str, int]],
    candidate: dict[str, Any],
    def_index: int,
    candidate_index: int,
) -> None:
    scope = {"def_index": def_index, "candidate_index": candidate_index}
    _append(rows, TOKEN_KIND.CANDIDATE_START, **scope)
    _serialize_side(rows, candidate, "left_definition", SIDE.LEFT, scope)
    _serialize_side(rows, candidate, "right_definition", SIDE.RIGHT, scope)
    _append(rows, TOKEN_KIND.CANDIDATE_END, **scope)


def _serialize_side(
    rows: list[dict[str, int]],
    candidate: dict[str, Any],
    key: str,
    side: SIDE,
    scope: dict[str, int],
) -> None:
    side_scope = dict(scope, side=int(side))
    _append(rows, TOKEN_KIND.SIDE_START, **side_scope)
    _serialize_definition(
        rows,
        candidate[key],
        def_index=scope["def_index"],
        segment=SEGMENT.ACTION_SPACE,
        candidate_index=scope["candidate_index"],
        side=int(side),
    )
    _append(rows, TOKEN_KIND.SIDE_END, **side_scope)


def _decode_many(
    cursor: _Cursor,
    kind: TOKEN_KIND,
    decoder,
) -> list[Any]:
    decoded = []
    while cursor.peek(kind):
        decoded.append(decoder(cursor))
    return decoded


def _decode_range(cursor: _Cursor) -> dict[str, int]:
    row = cursor.take(TOKEN_KIND.RANGE)
    return {"id": _need(row, "range_id"), "size": _need(row, "coeff_num")}


def _decode_tensor(cursor: _Cursor) -> dict[str, Any]:
    start = cursor.take(TOKEN_KIND.TENSOR_START)
    tensor_id = _need(start, "tensor_id")
    scope = {"tensor_id": tensor_id}
    symmetry = []
    while not cursor.peek(TOKEN_KIND.TENSOR_END):
        symmetry.append(_decode_symmetry(cursor, scope, len(symmetry)))
    end = cursor.take(TOKEN_KIND.TENSOR_END)
    _check_scope(end, scope)
    return {"id": tensor_id, "symmetry": symmetry}


def _decode_symmetry(
    cursor: _Cursor,
    tensor_scope: dict[str, int],
    expected_index: int,
) -> dict[str, Any]:
    start = cursor.take(TOKEN_KIND.SYMMETRY_START)
    _check_scope(start, tensor_scope)
    _check_exact(start, "symmetry_index", expected_index)
    action = _sym_action_name(_need(start, "symmetry_action"))
    scope = dict(tensor_scope, symmetry_index=_need(start, "symmetry_index"))
    perm = []
    while not cursor.peek(TOKEN_KIND.SYMMETRY_END):
        row = cursor.take(TOKEN_KIND.SYMMETRY_PERM)
        _check_scope(row, scope)
        _check_perm_index(row, len(perm))
        perm.append(_need(row, "perm_value"))
    end = cursor.take(TOKEN_KIND.SYMMETRY_END)
    _check_scope(end, scope)
    return {"perm": perm, "action": action}


def _decode_definition(
    cursor: _Cursor,
    parent_scope: dict[str, int] | None = None,
) -> dict[str, Any]:
    start = cursor.take(TOKEN_KIND.DEF_START)
    if parent_scope is not None:
        _check_scope(start, parent_scope)
    scope = _row_scope(start, ("def_index", "candidate_index", "side"))
    definition = {
        "base": _need(start, "tensor_id"),
        "ext_indices": [],
        "terms": [],
    }
    while cursor.peek(TOKEN_KIND.EXT_INDEX):
        definition["ext_indices"].append(
            _decode_index(cursor, TOKEN_KIND.EXT_INDEX, scope)
        )
    while not cursor.peek(TOKEN_KIND.DEF_END):
        term_index = len(definition["terms"])
        definition["terms"].append(_decode_term(cursor, scope, term_index))
    end = cursor.take(TOKEN_KIND.DEF_END)
    _check_scope(end, scope)
    return definition


def _decode_term(
    cursor: _Cursor,
    definition_scope: dict[str, int],
    expected_index: int,
) -> dict[str, Any]:
    start = cursor.take(TOKEN_KIND.TERM_START)
    _check_scope(start, definition_scope)
    _check_exact(start, "term_index", expected_index)
    scope = dict(definition_scope, term_index=_need(start, "term_index"))
    coeff = _decode_coeff(cursor, scope)
    sum_indices = []
    factors = []
    while not cursor.peek(TOKEN_KIND.TERM_END):
        if cursor.peek(TOKEN_KIND.SUM_INDEX):
            sum_indices.append(
                _decode_index(cursor, TOKEN_KIND.SUM_INDEX, scope)
            )
        elif cursor.peek(TOKEN_KIND.FACTOR_START):
            factors.append(_decode_factor(cursor, scope, len(factors)))
        else:
            raise ValueError(
                f"expected SUM_INDEX or FACTOR_START, got {cursor.kind()}"
            )
    end = cursor.take(TOKEN_KIND.TERM_END)
    _check_scope(end, scope)
    return {"coeff": coeff, "sum_indices": sum_indices, "factors": factors}


def _decode_coeff(
    cursor: _Cursor,
    term_scope: dict[str, int],
) -> dict[str, int]:
    row = cursor.take(TOKEN_KIND.COEFF)
    _check_scope(row, term_scope)
    return {"numer": _need(row, "coeff_num"), "denom": _need(row, "coeff_den")}


def _decode_factor(
    cursor: _Cursor,
    term_scope: dict[str, int],
    expected_index: int,
) -> dict[str, Any]:
    start = cursor.take(TOKEN_KIND.FACTOR_START)
    _check_scope(start, term_scope)
    _check_exact(start, "factor_index", expected_index)
    scope = dict(term_scope, factor_index=_need(start, "factor_index"))
    indices = []
    while not cursor.peek(TOKEN_KIND.FACTOR_END):
        row = cursor.take(TOKEN_KIND.FACTOR_INDEX)
        _check_scope(row, scope)
        indices.append(_need(row, "index_id"))
    end = cursor.take(TOKEN_KIND.FACTOR_END)
    _check_scope(end, scope)
    return {"tensor": _need(start, "tensor_id"), "indices": indices}


def _decode_index(
    cursor: _Cursor,
    kind: TOKEN_KIND,
    scope: dict[str, int],
) -> dict[str, int]:
    row = cursor.take(kind)
    _check_scope(row, scope)
    return {"id": _need(row, "index_id"), "range": _need(row, "range_id")}


def _decode_candidate(
    cursor: _Cursor,
    candidate_scope: dict[str, int],
) -> dict[str, Any]:
    start = cursor.take(TOKEN_KIND.CANDIDATE_START)
    _check_scope(start, candidate_scope)
    left = _decode_side(cursor, SIDE.LEFT, candidate_scope)
    right = _decode_side(cursor, SIDE.RIGHT, candidate_scope)
    end = cursor.take(TOKEN_KIND.CANDIDATE_END)
    _check_scope(end, candidate_scope)
    return {"left_definition": left, "right_definition": right}


def _decode_side(
    cursor: _Cursor,
    side: SIDE,
    candidate_scope: dict[str, int],
) -> dict[str, Any]:
    start = cursor.take(TOKEN_KIND.SIDE_START)
    _check_scope(start, candidate_scope)
    _check_side(start, side)
    scope = dict(candidate_scope, side=int(side))
    definition = _decode_definition(cursor, scope)
    end = cursor.take(TOKEN_KIND.SIDE_END)
    _check_scope(end, scope)
    return definition


def _token_rows(tokens: TokenArrays, mask: jax.Array) -> list[dict[str, int]]:
    _check_field_set(tokens)
    length = validate_token_arrays(tokens, mask)
    columns = {field: tokens[field].tolist() for field in TOKEN_FIELDS}
    mask_values = mask.tolist()
    rows = []
    for position in range(length):
        row = {field: int(columns[field][position]) for field in TOKEN_FIELDS}
        _validate_row_mask(row, bool(mask_values[position]), position)
        if mask_values[position]:
            rows.append(row)
    return rows


def _validate_row_mask(
    row: dict[str, int],
    valid: bool,
    position: int,
) -> None:
    if valid:
        _validate_valid_row(row, position)
    else:
        _check_pad_row(row)


def _validate_valid_row(row: dict[str, int], position: int) -> None:
    kind = _checked_token_kind(row["token_kind"])
    segment = _checked_segment(row["segment"], kind)
    _check_exact(row, "position", position)
    expected = _expected_fields(kind, segment)
    _check_row_fields(row, expected)


def _check_pad_row(row: dict[str, int]) -> None:
    if row["token_kind"] != int(TOKEN_KIND.PAD):
        raise ValueError("masked out rows must use PAD token_kind")
    for field in TOKEN_FIELDS:
        if field != "token_kind" and row[field] != SENTINEL:
            raise ValueError("masked out PAD rows must be empty")


def _checked_token_kind(value: int) -> TOKEN_KIND:
    try:
        token_kind = TOKEN_KIND(value)
    except ValueError as error:
        raise ValueError(f"unknown token kind {value}") from error
    if token_kind == TOKEN_KIND.PAD:
        raise ValueError("PAD rows cannot be valid tokens")
    return token_kind


def _checked_segment(value: int, kind: TOKEN_KIND) -> SEGMENT:
    try:
        segment = SEGMENT(value)
    except ValueError as error:
        raise ValueError(f"unknown segment {value}") from error
    allowed = _KIND_SEGMENTS[kind]
    if segment not in allowed:
        names = ", ".join(item.name for item in allowed)
        raise ValueError(f"expected segment {names}, got {segment.name}")
    return segment


def _check_field_set(tokens: TokenArrays) -> None:
    fields = set(tokens)
    expected = set(TOKEN_FIELDS)
    if fields != expected:
        raise ValueError(
            f"token arrays field set mismatch: {fields} != {expected}"
        )


def _expected_fields(kind: TOKEN_KIND, segment: SEGMENT) -> set[str]:
    fields = set(_COMMON_FIELDS)
    fields.update(_KIND_FIELDS[kind])
    if kind in _DEFINITION_KINDS and segment == SEGMENT.ACTION_SPACE:
        fields.update(("candidate_index", "side"))
    return fields


def _check_row_fields(row: dict[str, int], expected: set[str]) -> None:
    kind = TOKEN_KIND(row["token_kind"]).name
    for field in TOKEN_FIELDS:
        value = int(row[field])
        if field in expected and value == SENTINEL:
            raise ValueError(f"{field} is required for {kind}")
        if field not in expected and value != SENTINEL:
            raise ValueError(f"{field} must be SENTINEL for {kind}")


def _need(row: dict[str, int], field: str) -> int:
    value = int(row[field])
    if value == SENTINEL:
        token_kind = TOKEN_KIND(row["token_kind"]).name
        raise ValueError(f"{field} is required for {token_kind}")
    return value


def _row_scope(
    row: dict[str, int],
    fields: tuple[str, ...],
) -> dict[str, int]:
    return {field: int(row[field]) for field in fields}


def _check_scope(row: dict[str, int], expected: dict[str, int]) -> None:
    for field, value in expected.items():
        _check_exact(row, field, value)


def _check_exact(row: dict[str, int], field: str, expected: int) -> None:
    actual = int(row[field])
    if actual != expected:
        raise ValueError(f"expected {field} {expected}, got {actual}")


def _scope(
    def_index: int,
    candidate_index: int = SENTINEL,
    side: int = SENTINEL,
) -> dict[str, int]:
    scope = {"def_index": def_index}
    if candidate_index != SENTINEL:
        scope["candidate_index"] = candidate_index
    if side != SENTINEL:
        scope["side"] = side
    return scope


def _coeff_parts(coeff: Any) -> tuple[int, int]:
    if isinstance(coeff, dict):
        return int(coeff["numer"]), int(coeff["denom"])
    return int(coeff[0]), int(coeff[1])


def _definition_index_ranges(definition: dict[str, Any]) -> dict[int, int]:
    ranges: dict[int, int] = {}
    for index in definition.get("ext_indices", []):
        ranges[int(index["id"])] = int(index["range"])
    for term in definition.get("terms", []):
        for index in term.get("sum_indices", []):
            ranges[int(index["id"])] = int(index["range"])
    return ranges


def _sym_action_value(action: str) -> int:
    if action == "Identity":
        return int(SYM_ACTION.IDENTITY)
    if action == "Negate":
        return int(SYM_ACTION.NEGATE)
    raise ValueError(f"unknown symmetry action {action}")


def _sym_action_name(action: int) -> str:
    if action == int(SYM_ACTION.IDENTITY):
        return "Identity"
    if action == int(SYM_ACTION.NEGATE):
        return "Negate"
    raise ValueError(f"unknown symmetry action {action}")


def _check_perm_index(row: dict[str, int], expected: int) -> None:
    actual = _need(row, "perm_index")
    if actual != expected:
        raise ValueError(f"expected perm_index {expected}, got {actual}")


def _check_side(row: dict[str, int], expected: SIDE) -> None:
    actual = _need(row, "side")
    if actual != int(expected):
        raise ValueError(f"expected side {int(expected)}, got {actual}")
