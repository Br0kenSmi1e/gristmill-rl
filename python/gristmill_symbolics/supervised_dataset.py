from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from operator import index as _index
from pathlib import Path
from typing import Any

import numpy as np

from . import TensorComputation

__all__ = (
    "SupervisedDatasetError",
    "iter_supervised_batches",
    "load_preprocessed_supervised_dataset",
    "preprocess_supervised_jsonl",
    "preprocess_supervised_rows",
    "save_preprocessed_supervised_dataset",
)


class SupervisedDatasetError(ValueError):
    """Raised when supervised dataset rows cannot be preprocessed."""


def preprocess_supervised_rows(
    rows: Iterable[Mapping[str, object]],
    tokenizer: Any,
    *,
    source_len: int,
    target_len: int,
) -> dict[str, object]:
    source_len = _positive_length("source_len", source_len)
    target_len = _positive_length("target_len", target_len)

    source_rows: list[list[int]] = []
    decoder_rows: list[list[int]] = []
    target_rows: list[list[int]] = []
    mask_rows: list[list[bool]] = []
    weights: list[float] = []

    for row_number, row in enumerate(rows, start=1):
        try:
            source_ids, decoder_ids, target_ids, target_mask, weight = _preprocess_row(
                row,
                tokenizer,
                source_len=source_len,
                target_len=target_len,
            )
        except SupervisedDatasetError:
            raise
        except Exception as exc:
            raise SupervisedDatasetError(f"row {row_number}: {exc}") from exc

        source_rows.append(source_ids)
        decoder_rows.append(decoder_ids)
        target_rows.append(target_ids)
        mask_rows.append(target_mask)
        weights.append(weight)

    num_examples = len(weights)
    return {
        "source_ids": np.asarray(source_rows, dtype=np.int32).reshape(
            num_examples, source_len
        ),
        "decoder_input_ids": np.asarray(decoder_rows, dtype=np.int32).reshape(
            num_examples, target_len
        ),
        "target_ids": np.asarray(target_rows, dtype=np.int32).reshape(
            num_examples, target_len
        ),
        "target_mask": np.asarray(mask_rows, dtype=np.bool_).reshape(
            num_examples, target_len
        ),
        "example_weight": np.asarray(weights, dtype=np.float32),
        "metadata": _metadata(
            tokenizer,
            source_len=source_len,
            target_len=target_len,
            num_examples=num_examples,
        ),
    }


def preprocess_supervised_jsonl(
    input_path: str | Path,
    arrays_path: str | Path,
    metadata_path: str | Path,
    tokenizer: Any,
    *,
    source_len: int,
    target_len: int,
) -> dict[str, object]:
    dataset = preprocess_supervised_rows(
        _read_jsonl_rows(input_path),
        tokenizer,
        source_len=source_len,
        target_len=target_len,
    )
    save_preprocessed_supervised_dataset(dataset, arrays_path, metadata_path)
    return dataset


def save_preprocessed_supervised_dataset(
    dataset: Mapping[str, object],
    arrays_path: str | Path,
    metadata_path: str | Path,
) -> None:
    np.savez_compressed(
        arrays_path,
        source_ids=dataset["source_ids"],
        decoder_input_ids=dataset["decoder_input_ids"],
        target_ids=dataset["target_ids"],
        target_mask=dataset["target_mask"],
        example_weight=dataset["example_weight"],
    )
    Path(metadata_path).write_text(
        f"{json.dumps(dataset['metadata'], indent=2, sort_keys=True)}\n"
    )


def load_preprocessed_supervised_dataset(
    arrays_path: str | Path,
    metadata_path: str | Path,
) -> dict[str, object]:
    metadata = json.loads(Path(metadata_path).read_text())
    with np.load(arrays_path) as arrays:
        return {
            "source_ids": np.asarray(arrays["source_ids"], dtype=np.int32),
            "decoder_input_ids": np.asarray(
                arrays["decoder_input_ids"],
                dtype=np.int32,
            ),
            "target_ids": np.asarray(arrays["target_ids"], dtype=np.int32),
            "target_mask": np.asarray(arrays["target_mask"], dtype=np.bool_),
            "example_weight": np.asarray(arrays["example_weight"], dtype=np.float32),
            "metadata": metadata,
        }


def iter_supervised_batches(
    dataset: Mapping[str, object],
    *,
    batch_size: int,
    shuffle: bool = False,
    rng: np.random.Generator | None = None,
):
    batch_size = _positive_batch_size(batch_size)
    num_examples = int(dataset["example_weight"].shape[0])
    order = np.arange(num_examples)
    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        rng.shuffle(order)

    def batches():
        for start in range(0, num_examples - batch_size + 1, batch_size):
            batch_order = order[start : start + batch_size]
            yield {
                "source_ids": dataset["source_ids"][batch_order],
                "decoder_input_ids": dataset["decoder_input_ids"][batch_order],
                "target_ids": dataset["target_ids"][batch_order],
                "target_mask": dataset["target_mask"][batch_order],
                "example_weight": dataset["example_weight"][batch_order],
            }

    return batches()


def _preprocess_row(
    row: Mapping[str, object],
    tokenizer: Any,
    *,
    source_len: int,
    target_len: int,
) -> tuple[list[int], list[int], list[int], list[bool], float]:
    if not isinstance(row, Mapping):
        raise TypeError("row must be a Mapping")
    for key in ("source", "target", "weight"):
        if key not in row:
            raise ValueError(f"missing {key!r}")

    source = _computation_from_json_value(row["source"], "source")
    target = _computation_from_json_value(row["target"], "target")
    weight = _weight(row["weight"])

    source_definitions = source.snapshot()["definitions"]
    target_definitions = target.snapshot()["definitions"]
    source_ids = tokenizer.encode_definitions_padded(
        source_definitions,
        length=source_len,
    )
    content_ids = tokenizer.encode_definitions(target_definitions)
    if len(content_ids) + 1 > target_len:
        raise ValueError(
            f"target content length {len(content_ids)} plus eos exceeds target_len {target_len}"
        )

    decoder_pad_count = target_len - len(content_ids) - 1
    target_pad_count = target_len - len(content_ids) - 1
    decoder_ids = [
        tokenizer.bos_token_id,
        *content_ids,
        *([tokenizer.pad_token_id] * decoder_pad_count),
    ]
    target_ids = [
        *content_ids,
        tokenizer.eos_token_id,
        *([tokenizer.pad_token_id] * target_pad_count),
    ]
    target_mask = [True] * (len(content_ids) + 1) + [False] * target_pad_count
    return source_ids, decoder_ids, target_ids, target_mask, weight


def _computation_from_json_value(value: object, name: str):
    try:
        text = json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON serializable") from exc
    return TensorComputation.from_json_string(text)


def _weight(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("weight must be a non-bool int or float")
    weight = float(value)
    if not math.isfinite(weight):
        raise ValueError("weight must be finite")
    if weight < 0.0:
        raise ValueError("weight must be non-negative")
    return weight


def _positive_length(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    length = _index(value)
    if length <= 0:
        raise ValueError(f"{name} must be positive")
    return length


def _positive_batch_size(value: object) -> int:
    if isinstance(value, bool):
        raise SupervisedDatasetError("batch_size must be positive")
    try:
        batch_size = _index(value)
    except TypeError as exc:
        raise SupervisedDatasetError("batch_size must be positive") from exc
    if batch_size <= 0:
        raise SupervisedDatasetError("batch_size must be positive")
    return batch_size


def _metadata(
    tokenizer: Any,
    *,
    source_len: int,
    target_len: int,
    num_examples: int,
) -> dict[str, object]:
    return {
        "source_len": source_len,
        "target_len": target_len,
        "vocab_size": int(tokenizer.vocab_size),
        "num_examples": num_examples,
        "tokenizer": {
            "max_range_id": tokenizer.max_range_id,
            "max_tensor_id": tokenizer.max_tensor_id,
            "max_index_id": tokenizer.max_index_id,
            "coeff_nums": list(tokenizer.coeff_nums),
            "coeff_dens": list(tokenizer.coeff_dens),
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
    }


def _read_jsonl_rows(input_path: str | Path) -> Iterable[object]:
    for row_number, line in enumerate(
        Path(input_path).read_text().splitlines(),
        start=1,
    ):
        if not line.strip():
            raise SupervisedDatasetError(f"row {row_number}: empty JSONL row")
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            raise SupervisedDatasetError(f"row {row_number}: invalid JSON") from exc
