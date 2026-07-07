from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from operator import index as _index


__all__ = ("FlatDefinitionTokenizer", "TokenizerError")


class TokenizerError(ValueError):
    """Raised when a definition or token stream cannot be tokenized."""


@dataclass(frozen=True)
class _TokenSpec:
    name: str
    kind: str
    value: int | None = None


class FlatDefinitionTokenizer:
    def __init__(
        self,
        *,
        max_range_id: int,
        max_tensor_id: int,
        max_index_id: int,
        coeff_nums: Sequence[int],
        coeff_dens: Sequence[int],
    ):
        self.max_range_id = max_range_id
        self.max_tensor_id = max_tensor_id
        self.max_index_id = max_index_id
        self.coeff_nums = coeff_nums
        self.coeff_dens = coeff_dens

        self._token_specs: list[_TokenSpec] = []
        self._token_ids: dict[str, int] = {}
        self._add_token("pad", "pad")
        self._add_token("def_start", "def_start")
        self._add_token("def_end", "def_end")
        for value in range(self.max_range_id + 1):
            self._add_token(f"rangeid{value}", "rangeid", value)
        for value in range(self.max_tensor_id + 1):
            self._add_token(f"tensorid{value}", "tensorid", value)
        for value in range(self.max_index_id + 1):
            self._add_token(f"indexid{value}", "indexid", value)
        for value in self.coeff_nums:
            self._add_token(f"coeff_num{value}", "coeff_num", value)
        for value in self.coeff_dens:
            self._add_token(f"coeff_den{value}", "coeff_den", value)

    @property
    def pad_token_id(self) -> int:
        return self._token_ids["pad"]

    @property
    def vocab_size(self) -> int:
        return len(self._token_specs)

    def token_name(self, token_id: int) -> str:
        return self._spec_for_token_id("token_id", token_id).name

    def encode_definition(self, definition: Mapping[str, object]) -> list[int]:
        ids = [
            self._token_ids["def_start"],
            self._tensor_token_id("base", definition["base"]),
        ]
        for index in definition["ext_indices"]:
            ids.extend(self._encode_index("external index", index))
        for term in definition["terms"]:
            ids.extend(self._encode_term(term))
        ids.append(self._token_ids["def_end"])
        return ids

    def encode_definitions(
        self,
        definitions: Sequence[Mapping[str, object]],
    ) -> list[int]:
        ids: list[int] = []
        for definition in definitions:
            ids.extend(self.encode_definition(definition))
        return ids

    def decode_definition(self, ids: Sequence[int]) -> dict[str, object]:
        specs = self._token_specs_for_stream(ids, allow_pad=False)
        pos = 0

        def peek() -> _TokenSpec | None:
            if pos >= len(specs):
                return None
            return specs[pos]

        def consume(kind: str, description: str) -> _TokenSpec:
            nonlocal pos
            if pos >= len(specs):
                raise TokenizerError(
                    f"expected {description}, reached end of token stream"
                )
            spec = specs[pos]
            if spec.kind != kind:
                raise TokenizerError(f"expected {description}, got {spec.name}")
            pos += 1
            return spec

        consume("def_start", "def_start")
        base = consume("tensorid", "base tensorid").value
        ext_indices: list[dict[str, int]] = []
        while (spec := peek()) is not None and spec.kind == "indexid":
            index_id = consume("indexid", "external indexid").value
            range_id = consume("rangeid", "external index rangeid").value
            ext_indices.append({"id": index_id, "range": range_id})

        terms: list[dict[str, object]] = []
        while (spec := peek()) is not None and spec.kind == "coeff_num":
            numer = consume("coeff_num", "coeff_num").value
            denom = consume("coeff_den", "coeff_den").value
            sum_indices: list[dict[str, int]] = []
            while (spec := peek()) is not None and spec.kind == "indexid":
                next_pos = pos + 1
                if next_pos >= len(specs) or specs[next_pos].kind != "rangeid":
                    raise TokenizerError(
                        "sum index must be encoded as indexid/rangeid before factors"
                    )
                index_id = consume("indexid", "sum indexid").value
                range_id = consume("rangeid", "sum index rangeid").value
                sum_indices.append({"id": index_id, "range": range_id})

            factors: list[dict[str, object]] = []
            while (spec := peek()) is not None and spec.kind == "tensorid":
                tensor_id = consume("tensorid", "factor tensorid").value
                factor_indices: list[int] = []
                while (spec := peek()) is not None and spec.kind == "indexid":
                    factor_indices.append(consume("indexid", "factor indexid").value)
                factors.append({"tensor": tensor_id, "indices": factor_indices})

            terms.append(
                {
                    "coeff": {"numer": numer, "denom": denom},
                    "sum_indices": sum_indices,
                    "factors": factors,
                }
            )

        if (spec := peek()) is not None and spec.kind != "def_end":
            raise TokenizerError(f"expected coeff_num or def_end, got {spec.name}")
        consume("def_end", "def_end")
        if (spec := peek()) is not None:
            raise TokenizerError(f"unexpected token after def_end: {spec.name}")

        return {
            "base": base,
            "ext_indices": ext_indices,
            "terms": terms,
        }

    def decode_definitions(self, ids: Sequence[int]) -> list[dict[str, object]]:
        if isinstance(ids, (str, bytes, Mapping)) or not isinstance(ids, Sequence):
            raise TokenizerError("token stream must be a sequence of integer IDs")
        if not ids:
            return []

        specs = self._token_specs_for_stream(ids, allow_pad=False)
        raw_ids = list(ids)
        definitions: list[dict[str, object]] = []
        start: int | None = None
        for pos, spec in enumerate(specs):
            if start is None:
                if spec.kind != "def_start":
                    raise TokenizerError(f"expected def_start, got {spec.name}")
                start = pos
            elif spec.kind == "def_end":
                definitions.append(self.decode_definition(raw_ids[start : pos + 1]))
                start = None

        if start is not None:
            raise TokenizerError("expected def_end, reached end of token stream")
        return definitions

    def _add_token(
        self,
        name: str,
        kind: str,
        value: int | None = None,
    ) -> None:
        if name in self._token_ids:
            raise TokenizerError(f"duplicate token name {name!r}")
        self._token_ids[name] = len(self._token_specs)
        self._token_specs.append(_TokenSpec(name=name, kind=kind, value=value))

    def _token_id(self, name: str) -> int:
        try:
            return self._token_ids[name]
        except KeyError as exc:
            raise TokenizerError(f"unsupported token {name}") from exc

    def _spec_for_token_id(self, name: str, token_id: int) -> _TokenSpec:
        token_id = _integer(name, token_id)
        if token_id < 0 or token_id >= len(self._token_specs):
            raise TokenizerError(f"unknown token id {token_id}")
        return self._token_specs[token_id]

    def _token_specs_for_stream(
        self,
        ids: Sequence[int],
        *,
        allow_pad: bool,
    ) -> list[_TokenSpec]:
        if isinstance(ids, (str, bytes, Mapping)) or not isinstance(ids, Sequence):
            raise TokenizerError("token stream must be a sequence of integer IDs")
        raw_ids = list(ids)
        if not raw_ids:
            raise TokenizerError("token stream must not be empty")

        specs: list[_TokenSpec] = []
        for index, token_id in enumerate(raw_ids):
            spec = self._spec_for_token_id(f"token[{index}]", token_id)
            if not allow_pad and spec.kind == "pad":
                raise TokenizerError("raw token stream cannot contain pad")
            specs.append(spec)
        return specs

    def _encode_term(self, term: object) -> list[int]:
        coeff = term["coeff"]
        ids = [
            self._coeff_num_token_id(coeff["numer"]),
            self._coeff_den_token_id(coeff["denom"]),
        ]
        for index in term["sum_indices"]:
            ids.extend(self._encode_index("sum index", index))
        for factor in term["factors"]:
            ids.extend(self._encode_factor(factor))
        return ids

    def _encode_index(self, name: str, index: object) -> list[int]:
        return [
            self._index_token_id(f"{name}.id", index["id"]),
            self._range_token_id(f"{name}.range", index["range"]),
        ]

    def _encode_factor(self, factor: object) -> list[int]:
        ids = [self._tensor_token_id("factor.tensor", factor["tensor"])]
        for index in factor["indices"]:
            ids.append(self._index_token_id("factor index", index))
        return ids

    def _range_token_id(self, name: str, value: object) -> int:
        value = _bounded_int(name, value, upper=self.max_range_id)
        return self._token_id(f"rangeid{value}")

    def _tensor_token_id(self, name: str, value: object) -> int:
        value = _bounded_int(name, value, upper=self.max_tensor_id)
        return self._token_id(f"tensorid{value}")

    def _index_token_id(self, name: str, value: object) -> int:
        value = _bounded_int(name, value, upper=self.max_index_id)
        return self._token_id(f"indexid{value}")

    def _coeff_num_token_id(self, value: object) -> int:
        value = _integer("coeff.numer", value)
        if value not in self.coeff_nums:
            raise TokenizerError(f"unsupported coefficient numerator coeff_num{value}")
        return self._token_id(f"coeff_num{value}")

    def _coeff_den_token_id(self, value: object) -> int:
        value = _integer("coeff.denom", value)
        if value not in self.coeff_dens:
            raise TokenizerError(
                f"unsupported coefficient denominator coeff_den{value}"
            )
        return self._token_id(f"coeff_den{value}")


def _integer(name: str, value: object) -> int:
    try:
        return _index(value)
    except TypeError as exc:
        raise TokenizerError(f"{name} must be an integer") from exc


def _bounded_int(name: str, value: object, *, upper: int) -> int:
    value = _integer(name, value)
    if value < 0 or value > upper:
        raise TokenizerError(
            f"{name} value {value} is outside supported range 0..{upper}"
        )
    return value
