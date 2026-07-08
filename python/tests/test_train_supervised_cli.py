import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

import gristmill_symbolics.train_supervised as train_supervised
from gristmill_symbolics.grammar import FlatDefinitionGrammar
from gristmill_symbolics.tokenizer import FlatDefinitionTokenizer


class _ConstantLogitModel(nnx.Module):
    def __init__(self, *, batch_size: int, target_len: int, vocab_size: int):
        self.batch_size = batch_size
        self.target_len = target_len
        self.vocab_size = vocab_size

    def __call__(
        self,
        source_ids: jax.Array,
        decoder_input_ids: jax.Array,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        del source_ids, decoder_input_ids, deterministic
        return jnp.zeros(
            (self.batch_size, self.target_len, self.vocab_size),
            dtype=jnp.float32,
        )


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


def _tokenizer() -> FlatDefinitionTokenizer:
    return FlatDefinitionTokenizer(
        max_range_id=1,
        max_tensor_id=3,
        max_index_id=2,
        coeff_nums=(1, 2),
        coeff_dens=(1,),
    )


def _array_dataset(
    *,
    num_examples: int,
    source_len: int,
    target_len: int,
) -> dict[str, object]:
    metadata = _metadata(source_len=source_len, target_len=target_len)
    metadata["num_examples"] = num_examples
    return {
        "source_ids": np.arange(
            num_examples * source_len,
            dtype=np.int32,
        ).reshape(num_examples, source_len),
        "decoder_input_ids": np.zeros((num_examples, target_len), dtype=np.int32),
        "target_ids": np.zeros((num_examples, target_len), dtype=np.int32),
        "target_mask": np.ones((num_examples, target_len), dtype=np.bool_),
        "example_weight": np.ones(num_examples, dtype=np.float32),
        "metadata": metadata,
    }


def _valid_flat_dataset(
    tokenizer: FlatDefinitionTokenizer,
    *,
    num_examples: int,
) -> dict[str, object]:
    source_len = 3
    target_len = 4
    def_start = tokenizer.token_ids_for_kind("def_start")[0]
    tensor0 = tokenizer.token_ids_for_kind("tensorid")[0]
    def_end = tokenizer.token_ids_for_kind("def_end")[0]
    metadata = _metadata(source_len=source_len, target_len=target_len)
    metadata["num_examples"] = num_examples
    metadata["vocab_size"] = tokenizer.vocab_size
    return {
        "source_ids": np.zeros((num_examples, source_len), dtype=np.int32),
        "decoder_input_ids": np.asarray(
            [[tokenizer.bos_token_id, def_start, tensor0, def_end]]
            * num_examples,
            dtype=np.int32,
        ),
        "target_ids": np.asarray(
            [[def_start, tensor0, def_end, tokenizer.eos_token_id]]
            * num_examples,
            dtype=np.int32,
        ),
        "target_mask": np.ones((num_examples, target_len), dtype=np.bool_),
        "example_weight": np.arange(1, num_examples + 1, dtype=np.float32),
        "metadata": metadata,
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


def test_tokenizer_from_metadata_rejects_vocab_size_mismatch():
    metadata = _metadata(source_len=12, target_len=14)
    metadata["vocab_size"] += 1

    with pytest.raises(ValueError, match="vocab_size"):
        train_supervised._tokenizer_from_metadata(metadata)


def test_validate_metadata_accepts_matching_train_and_valid_metadata():
    train = _metadata(source_len=12, target_len=14)
    valid = _metadata(source_len=12, target_len=14)

    result = train_supervised._validate_compatible_metadata(train, valid)

    assert result is None


def test_validate_metadata_rejects_target_len_mismatch():
    train = _metadata(source_len=12, target_len=14)
    valid = _metadata(source_len=12, target_len=15)

    with pytest.raises(ValueError, match="target_len"):
        train_supervised._validate_compatible_metadata(train, valid)


def test_validate_metadata_rejects_tokenizer_mismatch():
    train = _metadata(source_len=12, target_len=14)
    valid = _metadata(source_len=12, target_len=14)
    valid["tokenizer"]["max_tensor_id"] += 1

    with pytest.raises(ValueError, match="tokenizer"):
        train_supervised._validate_compatible_metadata(train, valid)


def test_attention_from_name_accepts_only_supported_attention_names():
    assert train_supervised._attention_from_name("default") is None
    assert train_supervised._attention_from_name("xla") == "xla"
    assert train_supervised._attention_from_name("cudnn") == "cudnn"

    with pytest.raises(ValueError, match="unsupported attention"):
        train_supervised._attention_from_name("flash")


def test_iter_update_groups_drops_partial_microbatch():
    dataset = _array_dataset(num_examples=5, source_len=3, target_len=4)

    groups = list(
        train_supervised._iter_update_groups(
            dataset,
            batch_size=2,
            accumulate_steps=2,
            rng=np.random.default_rng(0),
        )
    )

    assert len(groups) == 1
    assert len(groups[0]) == 2
    assert groups[0][0]["source_ids"].shape == (2, 3)
    assert groups[0][1]["target_ids"].shape == (2, 4)


def test_iter_update_groups_drops_trailing_full_microbatch():
    dataset = _array_dataset(num_examples=6, source_len=3, target_len=4)

    groups = list(
        train_supervised._iter_update_groups(
            dataset,
            batch_size=2,
            accumulate_steps=2,
            rng=np.random.default_rng(0),
        )
    )

    assert len(groups) == 1
    assert len(groups[0]) == 2
    assert groups[0][0]["source_ids"].shape == (2, 3)
    assert groups[0][1]["target_ids"].shape == (2, 4)


def test_evaluate_dataset_returns_weighted_totals_for_full_batches_only():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    dataset = _valid_flat_dataset(tokenizer, num_examples=3)
    model = _ConstantLogitModel(
        batch_size=2,
        target_len=dataset["metadata"]["target_len"],
        vocab_size=tokenizer.vocab_size,
    )

    metrics = train_supervised._evaluate_dataset(
        model,
        dataset,
        grammar,
        batch_size=2,
    )

    assert metrics["num_batches"] == 1
    assert metrics["weight_sum"] > 0.0
    assert metrics["weighted_nll_sum"] > 0.0
    assert metrics["mean_nll"] == pytest.approx(
        metrics["weighted_nll_sum"] / metrics["weight_sum"]
    )
