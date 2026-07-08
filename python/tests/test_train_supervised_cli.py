import pytest

import gristmill_symbolics.train_supervised as train_supervised
from gristmill_symbolics.tokenizer import FlatDefinitionTokenizer


def _metadata(*, source_len: int, target_len: int) -> dict[str, object]:
    tokenizer = FlatDefinitionTokenizer(
        max_range_id=1,
        max_tensor_id=3,
        max_index_id=2,
        coeff_nums=(1, 2),
        coeff_dens=(1,),
    )
    return {
        "source_len": source_len,
        "target_len": target_len,
        "vocab_size": tokenizer.vocab_size,
        "num_examples": 2,
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


def test_tokenizer_from_metadata_rebuilds_flat_definition_tokenizer():
    metadata = _metadata(source_len=12, target_len=14)

    tokenizer = train_supervised._tokenizer_from_metadata(metadata)

    assert tokenizer.max_range_id == 1
    assert tokenizer.max_tensor_id == 3
    assert tokenizer.max_index_id == 2
    assert tokenizer.coeff_nums == (1, 2)
    assert tokenizer.coeff_dens == (1,)
    assert tokenizer.pad_token_id == metadata["tokenizer"]["pad_token_id"]
    assert tokenizer.bos_token_id == metadata["tokenizer"]["bos_token_id"]
    assert tokenizer.eos_token_id == metadata["tokenizer"]["eos_token_id"]


def test_validate_metadata_accepts_matching_train_and_valid_metadata():
    train = _metadata(source_len=12, target_len=14)
    valid = _metadata(source_len=12, target_len=14)

    result = train_supervised._validate_compatible_metadata(train, valid)

    assert result is None


def test_validate_metadata_rejects_shape_or_tokenizer_mismatch():
    train = _metadata(source_len=12, target_len=14)
    valid = _metadata(source_len=12, target_len=15)

    with pytest.raises(ValueError, match="target_len"):
        train_supervised._validate_compatible_metadata(train, valid)
