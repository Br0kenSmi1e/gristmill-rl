from __future__ import annotations

import argparse
import sys

from .supervised_dataset import preprocess_supervised_jsonl
from .tokenizer import FlatDefinitionTokenizer


def _parse_int_csv(value: str) -> tuple[int, ...]:
    try:
        items = [item.strip() for item in value.split(",")]
        if not items or any(item == "" for item in items):
            raise ValueError
        return tuple(int(item) for item in items)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integers"
        ) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess one supervised JSONL file for flat seq2seq training."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--arrays-out", required=True)
    parser.add_argument("--metadata-out", required=True)
    parser.add_argument("--source-len", type=int, required=True)
    parser.add_argument("--target-len", type=int, required=True)
    parser.add_argument("--max-range-id", type=int, required=True)
    parser.add_argument("--max-tensor-id", type=int, required=True)
    parser.add_argument("--max-index-id", type=int, required=True)
    parser.add_argument("--coeff-nums", type=_parse_int_csv, required=True)
    parser.add_argument("--coeff-dens", type=_parse_int_csv, required=True)
    return parser


def _normalize_coeff_args(argv: list[str]) -> list[str]:
    normalized = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg in ("--coeff-nums", "--coeff-dens") and index + 1 < len(argv):
            value = argv[index + 1]
            if not value.startswith("--"):
                normalized.append(f"{arg}={value}")
                index += 2
                continue
        normalized.append(arg)
        index += 1
    return normalized


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    args = _parser().parse_args(_normalize_coeff_args(list(raw_argv)))
    tokenizer = FlatDefinitionTokenizer(
        max_range_id=args.max_range_id,
        max_tensor_id=args.max_tensor_id,
        max_index_id=args.max_index_id,
        coeff_nums=args.coeff_nums,
        coeff_dens=args.coeff_dens,
    )
    preprocess_supervised_jsonl(
        args.input,
        args.arrays_out,
        args.metadata_out,
        tokenizer,
        source_len=args.source_len,
        target_len=args.target_len,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
