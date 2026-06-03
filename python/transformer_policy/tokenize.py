from __future__ import annotations

from typing import Any

from transformer_policy.types import T, Token


def _coeff_pair(coeff: Any) -> tuple[int, int]:
    if isinstance(coeff, dict) and "numer" in coeff and "denom" in coeff:
        return int(coeff["numer"]), int(coeff["denom"])
    if isinstance(coeff, list | tuple) and len(coeff) == 2:
        return int(coeff[0]), int(coeff[1])
    raise TypeError(f"unsupported coeff shape: {coeff!r}")


def tokenize_tensor_def(definition: dict[str, Any]) -> tuple[Token, ...]:
    tokens: list[Token] = [
        T("DEF_START"),
        T("BASE", tensor=int(definition["base"])),
    ]
    for position, index in enumerate(definition["ext_indices"]):
        tokens.append(
            T(
                "EXT_INDEX",
                position=position,
                id=int(index["id"]),
                range=int(index["range"]),
            )
        )
    for term_position, term in enumerate(definition["terms"]):
        numer, denom = _coeff_pair(term["coeff"])
        tokens.append(T("TERM_START", position=term_position))
        tokens.append(T("COEFF_NUM", value=numer))
        tokens.append(T("COEFF_DEN", value=denom))
        for sum_position, index in enumerate(term["sum_indices"]):
            tokens.append(
                T(
                    "SUM_INDEX",
                    position=sum_position,
                    id=int(index["id"]),
                    range=int(index["range"]),
                )
            )
        for factor_position, factor in enumerate(term["factors"]):
            indices = factor["indices"]
            tokens.append(
                T(
                    "FACTOR",
                    position=factor_position,
                    tensor=int(factor["tensor"]),
                    arity=len(indices),
                )
            )
            for index_position, index_id in enumerate(indices):
                tokens.append(T("INDEX", position=index_position, id=int(index_id)))
        tokens.append(T("TERM_END", position=term_position))
    tokens.append(T("DEF_END"))
    return tuple(tokens)


def build_state_context(comp_snapshot: dict[str, Any]) -> tuple[Token, ...]:
    tokens: list[Token] = [T("STATE_START")]
    for def_index, definition in enumerate(comp_snapshot["definitions"]):
        tokens.append(T("STATE_DEF", def_index=def_index))
        tokens.extend(tokenize_tensor_def(definition))
    tokens.append(T("STATE_END"))
    return tuple(tokens)


def build_action_space_context(
    action_space_snapshot: dict[str, Any],
) -> tuple[Token, ...]:
    def_index = int(action_space_snapshot["def_index"])
    tokens: list[Token] = [T("ACTION_SPACE_START", def_index=def_index)]
    for candidate_index, candidate in enumerate(
        action_space_snapshot["candidate_templates"]
    ):
        tokens.append(T("CAND_START", candidate_index=candidate_index))
        tokens.append(T("LEFT_DEF_START", candidate_index=candidate_index))
        tokens.extend(tokenize_tensor_def(candidate["left_definition"]))
        tokens.append(T("LEFT_DEF_END", candidate_index=candidate_index))
        tokens.append(T("RIGHT_DEF_START", candidate_index=candidate_index))
        tokens.extend(tokenize_tensor_def(candidate["right_definition"]))
        tokens.append(T("RIGHT_DEF_END", candidate_index=candidate_index))
        tokens.append(T("REWRITTEN_DEF_START", candidate_index=candidate_index))
        tokens.extend(tokenize_tensor_def(candidate["rewritten_definition"]))
        tokens.append(T("REWRITTEN_DEF_END", candidate_index=candidate_index))
        tokens.append(T("CAND_END", candidate_index=candidate_index))
    tokens.append(T("ACTION_SPACE_END"))
    return tuple(tokens)
