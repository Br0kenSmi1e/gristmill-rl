from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from gristmill_symbolics import TensorComputation

CONVERTER_SCHEMA_VERSION = 1

SOURCE_RECORD_STARTS = {
    "range",
    "tensor",
    "symmetry",
    "perm",
    "endsymmetry",
    "endtensor",
    "def",
    "ext",
    "term",
    "coeff",
    "sum",
    "factor",
    "index",
    "endfactor",
    "endterm",
    "enddef",
}
TARGET_RECORD_STARTS = {
    "def",
    "ext",
    "term",
    "coeff",
    "sum",
    "factor",
    "index",
    "endfactor",
    "endterm",
    "enddef",
}
VALID_SYM_ACTIONS = {"Identity", "Negate"}


def computation_to_source_text(comp: TensorComputation) -> str:
    snapshot = comp.snapshot()
    return _source_snapshot_to_text(snapshot)


def computation_to_target_text(comp: TensorComputation) -> str:
    return _definitions_to_text(comp.snapshot()["definitions"])


def source_text_to_snapshot(text: str) -> dict[str, Any]:
    parser = _Parser(text, allow_source_records=True)
    return parser.parse_source()


def target_text_to_definitions(text: str) -> list[dict[str, Any]]:
    parser = _Parser(text, allow_source_records=False)
    return parser.parse_definitions_until_end()


def _source_snapshot_to_text(snapshot: dict[str, Any]) -> str:
    _expect_mapping_keys(snapshot, {"ranges", "tensors", "definitions"}, "snapshot")
    lines: list[str] = []
    for range_info in _sequence(snapshot["ranges"], "ranges"):
        lines.extend(_range_to_lines(range_info))
    for tensor in _sequence(snapshot["tensors"], "tensors"):
        lines.extend(_tensor_to_lines(tensor))
    for definition in _sequence(snapshot["definitions"], "definitions"):
        lines.extend(_definition_to_lines(definition))
    return "\n".join(lines)


def _definitions_to_text(definitions: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = []
    for definition in _sequence(definitions, "definitions"):
        lines.extend(_definition_to_lines(definition))
    return "\n".join(lines)


def _range_to_lines(range_info: dict[str, Any]) -> list[str]:
    _expect_mapping_keys(range_info, {"id", "size"}, "range")
    range_id = _nonnegative_int_value(range_info["id"], "range.id")
    size = _nonnegative_int_value(range_info["size"], "range.size")
    return [f"range id range_id:{range_id} size dim_size:{size}"]


def _tensor_to_lines(tensor: dict[str, Any]) -> list[str]:
    _expect_mapping_keys(tensor, {"id", "symmetry"}, "tensor")
    tensor_id = _nonnegative_int_value(tensor["id"], "tensor.id")
    lines = [f"tensor id tensor_id:{tensor_id}"]
    for symmetry in _sequence(tensor["symmetry"], "tensor.symmetry"):
        lines.extend(_symmetry_to_lines(symmetry))
    lines.append("endtensor")
    return lines


def _definition_to_lines(definition: dict[str, Any]) -> list[str]:
    _expect_mapping_keys(definition, {"base", "ext_indices", "terms"}, "definition")
    base = _nonnegative_int_value(definition["base"], "definition.base")
    lines = [f"def base tensor_id:{base}"]
    for ext_index in _sequence(definition["ext_indices"], "definition.ext_indices"):
        _expect_mapping_keys(ext_index, {"id", "range"}, "ext index")
        index_id = _nonnegative_int_value(ext_index["id"], "ext index.id")
        range_id = _nonnegative_int_value(ext_index["range"], "ext index.range")
        lines.append(f"ext id index_id:{index_id} range range_id:{range_id}")
    for term in _sequence(definition["terms"], "definition.terms"):
        lines.extend(_term_to_lines(term))
    lines.append("enddef")
    return lines


def _term_to_lines(term: dict[str, Any]) -> list[str]:
    _expect_mapping_keys(term, {"coeff", "sum_indices", "factors"}, "term")
    coeff = _coeff_value(term["coeff"])
    lines = [
        "term",
        f"coeff numer coeff_num:{coeff['numer']} denom coeff_den:{coeff['denom']}",
    ]
    for sum_index in _sequence(term["sum_indices"], "term.sum_indices"):
        _expect_mapping_keys(sum_index, {"id", "range"}, "sum index")
        index_id = _nonnegative_int_value(sum_index["id"], "sum index.id")
        range_id = _nonnegative_int_value(sum_index["range"], "sum index.range")
        lines.append(f"sum id index_id:{index_id} range range_id:{range_id}")
    for factor in _sequence(term["factors"], "term.factors"):
        lines.extend(_factor_to_lines(factor))
    lines.append("endterm")
    return lines


def _factor_to_lines(factor: dict[str, Any]) -> list[str]:
    _expect_mapping_keys(factor, {"tensor", "indices"}, "factor")
    tensor_id = _nonnegative_int_value(factor["tensor"], "factor.tensor")
    lines = [f"factor tensor tensor_id:{tensor_id}"]
    for index_id in _sequence(factor["indices"], "factor.indices"):
        lines.append(
            f"index index_id:{_nonnegative_int_value(index_id, 'factor.index')}"
        )
    lines.append("endfactor")
    return lines


def _symmetry_to_lines(symmetry: dict[str, Any]) -> list[str]:
    _expect_mapping_keys(symmetry, {"perm", "action"}, "symmetry")
    action = _string_value(symmetry["action"], "symmetry.action")
    if action not in VALID_SYM_ACTIONS:
        raise ValueError(
            f"expected symmetry.action to be one of {sorted(VALID_SYM_ACTIONS)}"
        )
    lines = [f"symmetry action sym_action:{action}"]
    for axis in _sequence(symmetry["perm"], "symmetry.perm"):
        lines.append(f"perm axis:{_nonnegative_int_value(axis, 'symmetry.perm')}")
    lines.append("endsymmetry")
    return lines


def _coeff_value(coeff: Any) -> dict[str, int]:
    if isinstance(coeff, dict):
        _expect_mapping_keys(coeff, {"numer", "denom"}, "coeff")
        return {
            "numer": _int_value(coeff["numer"], "coeff.numer"),
            "denom": _positive_int_value(coeff["denom"], "coeff.denom"),
        }
    if _is_sequence(coeff) and len(coeff) == 2:
        return {
            "numer": _int_value(coeff[0], "coeff.numer"),
            "denom": _positive_int_value(coeff[1], "coeff.denom"),
        }
    raise ValueError(
        f"expected coeff object or pair, got {json.dumps(coeff, default=str)}"
    )


def _expect_mapping_keys(
    value: Any,
    expected: set[str],
    context: str,
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"expected {context} to be an object")
    actual = set(value)
    missing = expected - actual
    if missing:
        raise ValueError(f"missing fields for {context}: {sorted(missing)}")
    extra = actual - expected
    if extra:
        raise ValueError(f"extra fields for {context}: {sorted(extra)}")


def _sequence(value: Any, context: str) -> Sequence[Any]:
    if not _is_sequence(value):
        raise ValueError(f"expected {context} to be a sequence")
    return value


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    )


def _int_value(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"expected {context} to be an integer")
    return value


def _nonnegative_int_value(value: Any, context: str) -> int:
    parsed = _int_value(value, context)
    if parsed < 0:
        raise ValueError(f"expected {context} to be nonnegative")
    return parsed


def _positive_int_value(value: Any, context: str) -> int:
    parsed = _int_value(value, context)
    if parsed <= 0:
        raise ValueError(f"expected {context} to be positive")
    return parsed


def _string_value(value: Any, context: str) -> str:
    if not isinstance(value, str) or value == "" or any(c.isspace() for c in value):
        raise ValueError(f"expected {context} to be a non-empty scalar string")
    return value


class _Record:
    def __init__(self, line_number: int, line: str) -> None:
        self.line_number = line_number
        self.line = line
        self.parts = line.split()
        self.keyword = self.parts[0]


class _Parser:
    def __init__(self, text: str, *, allow_source_records: bool) -> None:
        allowed = SOURCE_RECORD_STARTS if allow_source_records else TARGET_RECORD_STARTS
        self.records: list[_Record] = []
        self.position = 0
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if line == "":
                raise ValueError(f"line {line_number}: empty record")
            record = _Record(line_number, line)
            if record.keyword not in allowed:
                raise ValueError(
                    f"line {line_number}: unknown keyword '{record.keyword}'"
                )
            self.records.append(record)

    def parse_source(self) -> dict[str, Any]:
        ranges = []
        while self._peek("range"):
            ranges.append(self._parse_range())
        tensors = []
        while self._peek("tensor"):
            tensors.append(self._parse_tensor())
        definitions = self.parse_definitions_until_end()
        return {
            "ranges": ranges,
            "tensors": tensors,
            "definitions": definitions,
        }

    def parse_definitions_until_end(self) -> list[dict[str, Any]]:
        definitions = []
        while not self._done():
            if not self._peek("def"):
                self._unexpected_current()
            definitions.append(self._parse_definition())
        return definitions

    def _parse_range(self) -> dict[str, Any]:
        record = self._take("range")
        self._expect_count(record, 5, "range")
        self._expect_literal(record, 1, "id")
        range_id = self._typed_nonnegative_int(record.parts[2], "range_id", record)
        self._expect_literal(record, 3, "size")
        size = self._typed_nonnegative_int(record.parts[4], "dim_size", record)
        return {"id": range_id, "size": size}

    def _parse_tensor(self) -> dict[str, Any]:
        record = self._take("tensor")
        self._expect_count(record, 3, "tensor")
        self._expect_literal(record, 1, "id")
        tensor_id = self._typed_nonnegative_int(record.parts[2], "tensor_id", record)
        symmetry = []
        while not self._done() and not self._peek("endtensor"):
            if self._peek("symmetry"):
                symmetry.append(self._parse_symmetry())
            elif self._peek_any({"range", "tensor", "def"}):
                raise ValueError(
                    f"line {self._current().line_number}: unclosed tensor"
                )
            else:
                self._unexpected_current()
        if self._done():
            raise ValueError("unclosed tensor")
        end = self._take("endtensor")
        self._expect_count(end, 1, "endtensor")
        return {"id": tensor_id, "symmetry": symmetry}

    def _parse_symmetry(self) -> dict[str, Any]:
        record = self._take("symmetry")
        self._expect_count(record, 3, "symmetry")
        self._expect_literal(record, 1, "action")
        action = self._typed_string(record.parts[2], "sym_action", record)
        if action not in VALID_SYM_ACTIONS:
            raise ValueError(
                f"line {record.line_number}: expected sym_action to be one of "
                f"{sorted(VALID_SYM_ACTIONS)}, got {action}"
            )
        perm = []
        while not self._done() and not self._peek("endsymmetry"):
            if self._peek("perm"):
                perm.append(self._parse_perm())
            elif self._peek_any({"endtensor", "range", "tensor", "def"}):
                raise ValueError(
                    f"line {self._current().line_number}: unclosed symmetry"
                )
            else:
                self._unexpected_current()
        if self._done():
            raise ValueError("unclosed symmetry")
        end = self._take("endsymmetry")
        self._expect_count(end, 1, "endsymmetry")
        return {"perm": perm, "action": action}

    def _parse_perm(self) -> int:
        record = self._take("perm")
        self._expect_count(record, 2, "perm")
        return self._typed_nonnegative_int(record.parts[1], "axis", record)

    def _parse_definition(self) -> dict[str, Any]:
        record = self._take("def")
        self._expect_count(record, 3, "def")
        self._expect_literal(record, 1, "base")
        base = self._typed_nonnegative_int(record.parts[2], "tensor_id", record)
        ext_indices = []
        terms = []
        seen_term = False
        while not self._done() and not self._peek("enddef"):
            if self._peek("ext"):
                if seen_term:
                    raise ValueError(
                        f"line {self._current().line_number}: unexpected ext"
                    )
                ext_indices.append(self._parse_index_range("ext"))
            elif self._peek("term"):
                seen_term = True
                terms.append(self._parse_term())
            elif self._peek_any({"range", "tensor", "def"}):
                raise ValueError(f"line {self._current().line_number}: unclosed def")
            else:
                self._unexpected_current()
        if self._done():
            raise ValueError("unclosed def")
        end = self._take("enddef")
        self._expect_count(end, 1, "enddef")
        return {"base": base, "ext_indices": ext_indices, "terms": terms}

    def _parse_term(self) -> dict[str, Any]:
        record = self._take("term")
        self._expect_count(record, 1, "term")
        coeff: dict[str, int] | None = None
        sum_indices = []
        factors = []
        seen_factor = False
        while not self._done() and not self._peek("endterm"):
            if self._peek("coeff"):
                if coeff is not None:
                    raise ValueError(
                        f"line {self._current().line_number}: unexpected coeff"
                    )
                coeff = self._parse_coeff()
            elif self._peek("sum"):
                if seen_factor:
                    raise ValueError(
                        f"line {self._current().line_number}: unexpected sum"
                    )
                sum_indices.append(self._parse_index_range("sum"))
            elif self._peek("factor"):
                seen_factor = True
                factors.append(self._parse_factor())
            elif self._peek_any({"enddef", "def", "ext", "range", "tensor"}):
                raise ValueError(f"line {self._current().line_number}: unclosed term")
            else:
                self._unexpected_current()
        if self._done():
            raise ValueError("unclosed term")
        end = self._take("endterm")
        self._expect_count(end, 1, "endterm")
        if coeff is None:
            raise ValueError("missing fields for term: coeff")
        return {
            "coeff": coeff,
            "sum_indices": sum_indices,
            "factors": factors,
        }

    def _parse_coeff(self) -> dict[str, int]:
        record = self._take("coeff")
        self._expect_count(record, 5, "coeff")
        self._expect_literal(record, 1, "numer")
        numer = self._typed_int(record.parts[2], "coeff_num", record)
        self._expect_literal(record, 3, "denom")
        denom = self._typed_positive_int(record.parts[4], "coeff_den", record)
        return {"numer": numer, "denom": denom}

    def _parse_factor(self) -> dict[str, Any]:
        record = self._take("factor")
        self._expect_count(record, 3, "factor")
        self._expect_literal(record, 1, "tensor")
        tensor_id = self._typed_nonnegative_int(record.parts[2], "tensor_id", record)
        indices = []
        while not self._done() and not self._peek("endfactor"):
            if self._peek("index"):
                indices.append(self._parse_index())
            elif self._peek_any(
                {"endterm", "enddef", "factor", "term", "def", "range", "tensor"}
            ):
                raise ValueError(f"line {self._current().line_number}: unclosed factor")
            else:
                self._unexpected_current()
        if self._done():
            raise ValueError("unclosed factor")
        end = self._take("endfactor")
        self._expect_count(end, 1, "endfactor")
        return {"tensor": tensor_id, "indices": indices}

    def _parse_index(self) -> int:
        record = self._take("index")
        self._expect_count(record, 2, "index")
        return self._typed_nonnegative_int(record.parts[1], "index_id", record)

    def _parse_index_range(self, keyword: str) -> dict[str, int]:
        record = self._take(keyword)
        self._expect_count(record, 5, keyword)
        self._expect_literal(record, 1, "id")
        index_id = self._typed_nonnegative_int(record.parts[2], "index_id", record)
        self._expect_literal(record, 3, "range")
        range_id = self._typed_nonnegative_int(record.parts[4], "range_id", record)
        return {"id": index_id, "range": range_id}

    def _done(self) -> bool:
        return self.position >= len(self.records)

    def _current(self) -> _Record:
        if self._done():
            raise ValueError("unexpected end of input")
        return self.records[self.position]

    def _peek(self, keyword: str) -> bool:
        return not self._done() and self._current().keyword == keyword

    def _peek_any(self, keywords: set[str]) -> bool:
        return not self._done() and self._current().keyword in keywords

    def _take(self, keyword: str) -> _Record:
        if self._done():
            raise ValueError(f"expected {keyword}, got end of input")
        record = self._current()
        if record.keyword != keyword:
            raise ValueError(
                f"line {record.line_number}: expected {keyword}, got {record.keyword}"
            )
        self.position += 1
        return record

    def _unexpected_current(self) -> None:
        record = self._current()
        raise ValueError(f"line {record.line_number}: unexpected {record.keyword}")

    def _expect_count(self, record: _Record, count: int, context: str) -> None:
        if len(record.parts) < count:
            raise ValueError(
                f"line {record.line_number}: missing fields for {context}"
            )
        if len(record.parts) > count:
            raise ValueError(f"line {record.line_number}: extra fields for {context}")

    def _expect_literal(self, record: _Record, index: int, expected: str) -> None:
        actual = record.parts[index]
        if actual != expected:
            raise ValueError(
                f"line {record.line_number}: expected {expected}, got {actual}"
            )

    def _typed_int(self, token: str, expected: str, record: _Record) -> int:
        prefix = expected + ":"
        if not token.startswith(prefix):
            raise ValueError(
                f"line {record.line_number}: expected {expected}, got {token}"
            )
        raw_value = token[len(prefix) :]
        try:
            return int(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"line {record.line_number}: malformed {expected}: {raw_value}"
            ) from exc

    def _typed_nonnegative_int(
        self,
        token: str,
        expected: str,
        record: _Record,
    ) -> int:
        value = self._typed_int(token, expected, record)
        if value < 0:
            raise ValueError(
                f"line {record.line_number}: expected nonnegative {expected}, "
                f"got {value}"
            )
        return value

    def _typed_positive_int(
        self,
        token: str,
        expected: str,
        record: _Record,
    ) -> int:
        value = self._typed_int(token, expected, record)
        if value <= 0:
            raise ValueError(
                f"line {record.line_number}: expected positive {expected}, "
                f"got {value}"
            )
        return value

    def _typed_string(self, token: str, expected: str, record: _Record) -> str:
        prefix = expected + ":"
        if not token.startswith(prefix):
            raise ValueError(
                f"line {record.line_number}: expected {expected}, got {token}"
            )
        value = token[len(prefix) :]
        if value == "":
            raise ValueError(f"line {record.line_number}: malformed {expected}")
        return value
