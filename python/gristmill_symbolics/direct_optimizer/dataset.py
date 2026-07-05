from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gristmill_symbolics import (
    GristmillSymbolicsError,
    TensorComputation,
    equivalent_computations,
)

from .converter import computation_to_source_text, computation_to_target_text


@dataclass(frozen=True)
class GenerationConfig:
    seed: int
    trajectories_per_input: int
    max_steps: int
    random_subsets: bool = False
    collect_intermediates: bool = True


@dataclass(frozen=True)
class BuildConfig:
    beta: float = 1.0
    verify: bool = False


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.8
    valid_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 0


def write_raw_candidates_jsonl(records: Sequence[dict[str, Any]], path: Path) -> None:
    _write_jsonl(records, path)


def read_raw_candidates_jsonl(path: Path) -> list[dict[str, Any]]:
    return _read_jsonl(path)


def write_processed_jsonl(rows: Sequence[dict[str, Any]], path: Path) -> None:
    _write_jsonl(rows, path)


def read_processed_jsonl(path: Path) -> list[dict[str, Any]]:
    return _read_jsonl(path)


def split_raw_candidates(
    raw_records: Sequence[dict[str, Any]],
    config: SplitConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    fractions = (
        float(config.train_fraction),
        float(config.valid_fraction),
        float(config.test_fraction),
    )
    if any(not math.isfinite(fraction) or fraction < 0.0 for fraction in fractions):
        raise ValueError("split fractions must be finite and non-negative")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("split fractions must sum to 1.0")

    records = list(raw_records)
    random.Random(config.seed).shuffle(records)

    count = len(records)
    train_count = int(count * config.train_fraction)
    valid_count = int(count * config.valid_fraction)
    test_count = count - train_count - valid_count

    train = records[:train_count]
    valid = records[train_count : train_count + valid_count]
    test = records[train_count + valid_count : train_count + valid_count + test_count]

    if not train:
        raise ValueError("train split must not be empty")
    if config.valid_fraction > 0.0 and not valid:
        raise ValueError("valid split must not be empty when valid_fraction > 0")
    if config.test_fraction > 0.0 and not test:
        raise ValueError("test split must not be empty when test_fraction > 0")

    return train, valid, test


def build_processed_dataset(
    raw_records: Sequence[dict[str, Any]],
    config: BuildConfig,
) -> list[dict[str, Any]]:
    beta = float(config.beta)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("BuildConfig.beta must be finite and positive")

    groups: dict[str, dict[str, dict[str, Any]]] = {}
    group_order: list[str] = []
    candidate_order: dict[str, list[str]] = {}

    for raw_record in raw_records:
        try:
            row = _processed_row_without_weight(raw_record, verify=config.verify)
        except (GristmillSymbolicsError, KeyError, TypeError, ValueError):
            continue

        input_key = row["input_key"]
        candidate_key = row["candidate_key"]
        if input_key not in groups:
            groups[input_key] = {}
            group_order.append(input_key)
            candidate_order[input_key] = []
        if candidate_key not in groups[input_key]:
            groups[input_key][candidate_key] = row
            candidate_order[input_key].append(candidate_key)
            continue
        if row["candidate_log_flops"] < groups[input_key][candidate_key][
            "candidate_log_flops"
        ]:
            groups[input_key][candidate_key] = row

    processed: list[dict[str, Any]] = []
    for input_key in group_order:
        candidates = [
            groups[input_key][candidate_key]
            for candidate_key in candidate_order[input_key]
            if candidate_key in groups[input_key]
        ]
        if not candidates:
            continue
        min_cost = min(row["candidate_log_flops"] for row in candidates)
        raw_weights = [
            math.exp(-beta * (row["candidate_log_flops"] - min_cost))
            for row in candidates
        ]
        weight_sum = sum(raw_weights)
        for row, raw_weight in zip(candidates, raw_weights, strict=True):
            weighted = dict(row)
            weighted["weight"] = raw_weight / weight_sum
            processed.append(weighted)
    return processed


def _write_jsonl(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            file.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if stripped == "":
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number}: expected JSON object")
            rows.append(value)
    return rows


def _processed_row_without_weight(
    raw_record: dict[str, Any],
    *,
    verify: bool,
) -> dict[str, Any]:
    input_comp = _parse_comp(raw_record["input_computation"])
    candidate_comp = _parse_comp(raw_record["candidate_computation"])
    outputs = _validate_outputs(raw_record["outputs"])
    if "initial_log_flops" in raw_record:
        initial_log_flops_value = raw_record["initial_log_flops"]
    else:
        initial_log_flops_value = input_comp.log_total_flops()
    initial_log_flops = _finite_cost(initial_log_flops_value)

    if "candidate_log_flops" in raw_record:
        candidate_log_flops_value = raw_record["candidate_log_flops"]
    else:
        candidate_log_flops_value = candidate_comp.log_total_flops()
    candidate_log_flops = _finite_cost(candidate_log_flops_value)

    if verify:
        try:
            equivalent = equivalent_computations(input_comp, candidate_comp, outputs)
        except Exception:
            equivalent = False
        if not equivalent:
            raise ValueError("computations are not equivalent for outputs")

    source_text = computation_to_source_text(input_comp)
    target_text = computation_to_target_text(candidate_comp)
    input_key = _stable_hash({"source_text": source_text, "outputs": outputs})
    candidate_key = _stable_hash({"target_text": target_text})
    return {
        "input_key": input_key,
        "candidate_key": candidate_key,
        "outputs": outputs,
        "source_text": source_text,
        "target_text": target_text,
        "initial_log_flops": initial_log_flops,
        "candidate_log_flops": candidate_log_flops,
    }


def _stable_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _parse_comp(value: Any) -> TensorComputation:
    text = value if isinstance(value, str) else json.dumps(value)
    return TensorComputation.from_json_string(text)


def _validate_outputs(value: Any) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError("outputs must be a non-empty list")
    outputs: list[int] = []
    seen: set[int] = set()
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError("outputs must contain integer tensor ids")
        if item in seen:
            raise ValueError("outputs must not contain duplicates")
        seen.add(item)
        outputs.append(item)
    return outputs


def _finite_cost(value: Any) -> float:
    cost = float(value)
    if not math.isfinite(cost):
        raise ValueError("cost must be finite")
    return cost
