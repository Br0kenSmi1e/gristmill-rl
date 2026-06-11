from __future__ import annotations

from typing import Any

import jax

from .constants import SENTINEL, SEGMENT, STATE_TOKEN_FIELDS, TOKEN_KIND
from .tree import make_token_tree
from .types import TokenTree


def _coeff_parts(coeff: Any) -> tuple[int, int]:
    if isinstance(coeff, dict):
        return int(coeff["numer"]), int(coeff["denom"])
    return int(coeff[0]), int(coeff[1])


def _index_ranges(snapshot: dict[str, Any]) -> dict[int, int]:
    ranges: dict[int, int] = {}
    for definition in snapshot.get("definitions", []):
        for index in definition.get("ext_indices", []):
            ranges[int(index["id"])] = int(index["range"])
        for term in definition.get("terms", []):
            for index in term.get("sum_indices", []):
                ranges[int(index["id"])] = int(index["range"])
    return ranges


def _append(
    rows: list[dict[str, int]],
    token_kind: TOKEN_KIND,
    *,
    segment: SEGMENT,
    **fields: int,
) -> None:
    row = {
        "token_kind": int(token_kind),
        "segment": int(segment),
        "position": len(rows),
    }
    row.update({key: int(value) for key, value in fields.items()})
    rows.append(row)


def _serialize_definition(
    rows: list[dict[str, int]],
    definition: dict[str, Any],
    *,
    def_index: int,
    segment: SEGMENT,
    index_ranges: dict[int, int],
    candidate_index: int = SENTINEL,
    side: int = SENTINEL,
) -> None:
    common = {"def_index": def_index}
    if candidate_index != SENTINEL:
        common["candidate_index"] = candidate_index
    if side != SENTINEL:
        common["side"] = side

    _append(
        rows,
        TOKEN_KIND.DEF_START,
        segment=segment,
        tensor_id=int(definition["base"]),
        **common,
    )
    for ext in definition.get("ext_indices", []):
        _append(
            rows,
            TOKEN_KIND.EXT_INDEX,
            segment=segment,
            index_id=int(ext["id"]),
            range_id=int(ext["range"]),
            **common,
    )
    for term_index, term in enumerate(definition.get("terms", [])):
        _append(
            rows,
            TOKEN_KIND.TERM_START,
            segment=segment,
            term_index=term_index,
            **common,
        )
        numer, denom = _coeff_parts(term["coeff"])
        _append(
            rows,
            TOKEN_KIND.COEFF,
            segment=segment,
            term_index=term_index,
            coeff_num=numer,
            coeff_den=denom,
            **common,
        )
        for index in term.get("sum_indices", []):
            _append(
                rows,
                TOKEN_KIND.SUM_INDEX,
                segment=segment,
                term_index=term_index,
                index_id=int(index["id"]),
                range_id=int(index["range"]),
                **common,
            )
        for factor_index, factor in enumerate(term.get("factors", [])):
            _append(
                rows,
                TOKEN_KIND.FACTOR_START,
                segment=segment,
                term_index=term_index,
                factor_index=factor_index,
                tensor_id=int(factor["tensor"]),
                **common,
            )
            for index_id in factor.get("indices", []):
                index_id = int(index_id)
                _append(
                    rows,
                    TOKEN_KIND.FACTOR_INDEX,
                    segment=segment,
                    term_index=term_index,
                    factor_index=factor_index,
                    index_id=index_id,
                    range_id=index_ranges.get(index_id, SENTINEL),
                    **common,
                )
            _append(
                rows,
                TOKEN_KIND.FACTOR_END,
                segment=segment,
                term_index=term_index,
                factor_index=factor_index,
                **common,
            )
        _append(
            rows,
            TOKEN_KIND.TERM_END,
            segment=segment,
            term_index=term_index,
            **common,
        )
    _append(rows, TOKEN_KIND.DEF_END, segment=segment, **common)


def tokenize_state_snapshot(snapshot: dict[str, Any]) -> tuple[TokenTree, jax.Array]:
    rows: list[dict[str, int]] = []
    index_ranges = _index_ranges(snapshot)
    for range_info in snapshot.get("ranges", []):
        _append(
            rows,
            TOKEN_KIND.RANGE,
            segment=SEGMENT.RANGES,
            range_id=int(range_info["id"]),
            coeff_num=int(range_info["size"]),
            coeff_den=1,
        )
    for tensor in snapshot.get("tensors", []):
        _append(
            rows,
            TOKEN_KIND.TENSOR,
            segment=SEGMENT.TENSORS,
            tensor_id=int(tensor["id"]),
        )
    for def_index, definition in enumerate(snapshot.get("definitions", [])):
        _serialize_definition(
            rows,
            definition,
            def_index=def_index,
            segment=SEGMENT.DEFINITIONS,
            index_ranges=index_ranges,
        )
    return make_token_tree(rows, STATE_TOKEN_FIELDS)
