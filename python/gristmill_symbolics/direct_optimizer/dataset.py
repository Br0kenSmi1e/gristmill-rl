from __future__ import annotations

import argparse
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
    action_space_for_def,
    apply_decision,
    equivalent_computations,
    validate_decision,
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


def generate_raw_candidates(
    inputs: Sequence[tuple[TensorComputation, Sequence[int]]],
    config: GenerationConfig,
) -> list[dict[str, Any]]:
    if config.trajectories_per_input <= 0:
        raise ValueError("GenerationConfig.trajectories_per_input must be positive")
    if config.max_steps <= 0:
        raise ValueError("GenerationConfig.max_steps must be positive")

    rng = random.Random(config.seed)
    records: list[dict[str, Any]] = []

    for input_comp, output_value in inputs:
        outputs = _validate_outputs(list(output_value))
        initial_json = input_comp.to_json_string()
        initial_log_flops = _finite_cost(input_comp.log_total_flops())
        for _trajectory in range(config.trajectories_per_input):
            comp = _clone_comp(input_comp)
            trajectory_records: list[dict[str, Any]] = []
            for _step in range(config.max_steps):
                spaces = _actionable_spaces(comp)
                if not spaces:
                    break
                _def_index, space = rng.choice(spaces)
                try:
                    decision = _sample_decision(space, rng, config.random_subsets)
                    validate_decision(space, decision)
                    apply_decision(comp, space, decision)
                    candidate_json = comp.to_json_string()
                    if candidate_json == initial_json:
                        continue
                    trajectory_records.append({
                        "input_computation": initial_json,
                        "candidate_computation": candidate_json,
                        "outputs": outputs,
                        "initial_log_flops": initial_log_flops,
                        "candidate_log_flops": _finite_cost(comp.log_total_flops()),
                    })
                except Exception:
                    break
            if config.collect_intermediates:
                records.extend(trajectory_records)
            elif trajectory_records:
                records.append(trajectory_records[-1])

    return records


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gristmill_symbolics.direct_optimizer.dataset",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--input", required=True, type=Path)
    generate_parser.add_argument("--outputs", required=True, nargs="+")
    generate_parser.add_argument("--raw-output", required=True, type=Path)
    generate_parser.add_argument("--seed", required=True, type=int)
    generate_parser.add_argument("--trajectories", required=True, type=int)
    generate_parser.add_argument("--max-steps", required=True, type=int)
    generate_parser.add_argument("--random-subsets", action="store_true")
    generate_parser.add_argument("--collect-intermediates", action="store_true")

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--raw-input", required=True, type=Path)
    build_parser.add_argument("--output", required=True, type=Path)
    build_parser.add_argument("--beta", default=1.0, type=float)
    build_parser.add_argument("--verify", action="store_true")

    split_parser = subparsers.add_parser("build-splits")
    split_parser.add_argument("--raw-input", required=True, type=Path)
    split_parser.add_argument("--train-output", required=True, type=Path)
    split_parser.add_argument("--valid-output", required=True, type=Path)
    split_parser.add_argument("--test-output", required=True, type=Path)
    split_parser.add_argument("--train-fraction", default=0.8, type=float)
    split_parser.add_argument("--valid-fraction", default=0.1, type=float)
    split_parser.add_argument("--test-fraction", default=0.1, type=float)
    split_parser.add_argument("--split-seed", default=0, type=int)
    split_parser.add_argument("--beta", default=1.0, type=float)
    split_parser.add_argument("--verify", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "generate":
        input_comp = TensorComputation.from_json_string(
            args.input.read_text(encoding="utf-8")
        )
        outputs = _parse_outputs(" ".join(args.outputs))
        records = generate_raw_candidates(
            [(input_comp, outputs)],
            GenerationConfig(
                seed=args.seed,
                trajectories_per_input=args.trajectories,
                max_steps=args.max_steps,
                random_subsets=args.random_subsets,
                collect_intermediates=args.collect_intermediates,
            ),
        )
        write_raw_candidates_jsonl(records, args.raw_output)
        return 0

    if args.command == "build":
        raw_records = read_raw_candidates_jsonl(args.raw_input)
        rows = build_processed_dataset(
            raw_records,
            BuildConfig(beta=args.beta, verify=args.verify),
        )
        write_processed_jsonl(rows, args.output)
        return 0

    if args.command == "build-splits":
        raw_records = read_raw_candidates_jsonl(args.raw_input)
        train_raw, valid_raw, test_raw = split_raw_candidates(
            raw_records,
            SplitConfig(
                train_fraction=args.train_fraction,
                valid_fraction=args.valid_fraction,
                test_fraction=args.test_fraction,
                seed=args.split_seed,
            ),
        )
        build_config = BuildConfig(beta=args.beta, verify=args.verify)
        write_processed_jsonl(
            build_processed_dataset(train_raw, build_config),
            args.train_output,
        )
        write_processed_jsonl(
            build_processed_dataset(valid_raw, build_config),
            args.valid_output,
        )
        write_processed_jsonl(
            build_processed_dataset(test_raw, build_config),
            args.test_output,
        )
        return 0

    raise AssertionError(f"unknown command: {args.command}")


def _clone_comp(comp: TensorComputation) -> TensorComputation:
    return TensorComputation.from_json_string(comp.to_json_string())


def _actionable_spaces(comp: TensorComputation) -> list[tuple[int, Any]]:
    snapshot = comp.snapshot()
    definitions = snapshot.get("definitions", [])
    spaces: list[tuple[int, Any]] = []
    for def_index in range(len(definitions)):
        try:
            space = action_space_for_def(comp, def_index)
        except Exception:
            continue
        if space is not None:
            spaces.append((def_index, space))
    return spaces


def _sample_decision(space: Any, rng: random.Random, random_subsets: bool) -> dict[str, Any]:
    templates = space.snapshot()["candidate_templates"]
    candidate_index = rng.randrange(len(templates))
    template = templates[candidate_index]
    return {
        "candidate_index": candidate_index,
        "left_mask": _sample_mask(
            len(template["left_definition"]["terms"]),
            rng,
            random_subsets,
        ),
        "right_mask": _sample_mask(
            len(template["right_definition"]["terms"]),
            rng,
            random_subsets,
        ),
    }


def _sample_mask(count: int, rng: random.Random, random_subset: bool) -> list[bool]:
    if not random_subset:
        return [True] * count
    if count == 0:
        return []
    mask = [bool(rng.getrandbits(1)) for _ in range(count)]
    if not any(mask):
        mask[rng.randrange(count)] = True
    return mask


def _parse_outputs(value: str) -> list[int]:
    if "," in value:
        parts = value.split(",")
    else:
        parts = value.split()
    outputs = [int(part) for part in parts if part != ""]
    return _validate_outputs(outputs)


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


if __name__ == "__main__":
    raise SystemExit(main())
