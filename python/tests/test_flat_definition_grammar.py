import jax
import jax.numpy as jnp

from gristmill_symbolics.grammar import FlatDefinitionGrammar, apply_grammar_mask
from gristmill_symbolics.tokenizer import FlatDefinitionTokenizer


def _tokenizer() -> FlatDefinitionTokenizer:
    return FlatDefinitionTokenizer(
        max_range_id=2,
        max_tensor_id=3,
        max_index_id=4,
        coeff_nums=(-1, 1, 2),
        coeff_dens=(1, 2),
    )


def _id(tokenizer: FlatDefinitionTokenizer, kind: str, offset: int = 0) -> int:
    return tokenizer.token_ids_for_kind(kind)[offset]


def _allowed_kinds(
    tokenizer: FlatDefinitionTokenizer,
    mask: jax.Array,
) -> set[str]:
    return {
        tokenizer.token_kind(token_id)
        for token_id, allowed in enumerate(list(mask))
        if bool(allowed)
    }


def _allows_all_kind_ids(
    tokenizer: FlatDefinitionTokenizer,
    mask: jax.Array,
    kind: str,
) -> bool:
    return all(bool(mask[token_id]) for token_id in tokenizer.token_ids_for_kind(kind))


def test_prefix_masks_allow_expected_token_families():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    def_start = _id(tokenizer, "def_start")
    def_end = _id(tokenizer, "def_end")
    tensor0 = _id(tokenizer, "tensorid")
    index0 = _id(tokenizer, "indexid")
    range0 = _id(tokenizer, "rangeid")
    coeff_num = _id(tokenizer, "coeff_num", 1)
    coeff_den = _id(tokenizer, "coeff_den")

    cases = [
        ([bos], {"def_start", "eos"}),
        ([bos, def_start], {"tensorid"}),
        ([bos, def_start, tensor0], {"indexid", "coeff_num", "def_end"}),
        ([bos, def_start, tensor0, index0], {"rangeid"}),
        (
            [bos, def_start, tensor0, index0, range0],
            {"indexid", "coeff_num", "def_end"},
        ),
        ([bos, def_start, tensor0, coeff_num], {"coeff_den"}),
        (
            [bos, def_start, tensor0, coeff_num, coeff_den],
            {"indexid", "tensorid", "coeff_num", "def_end"},
        ),
        (
            [bos, def_start, tensor0, coeff_num, coeff_den, tensor0],
            {"indexid", "tensorid", "coeff_num", "def_end"},
        ),
        ([bos, def_start, tensor0, def_end], {"def_start", "eos"}),
        ([bos, eos], {"pad"}),
    ]

    for prefix, expected_kinds in cases:
        mask = grammar.valid_next_mask_from_prefix(jnp.asarray([prefix], dtype=jnp.int32))
        assert _allowed_kinds(tokenizer, mask[0]) == expected_kinds


def test_prefix_mask_allows_every_id_in_allowed_family():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    prefix = jnp.asarray(
        [[tokenizer.bos_token_id, _id(tokenizer, "def_start")]],
        dtype=jnp.int32,
    )

    mask = grammar.valid_next_mask_from_prefix(prefix)

    assert _allows_all_kind_ids(tokenizer, mask[0], "tensorid")
    assert not bool(mask[0, _id(tokenizer, "indexid")])


def test_decoder_input_masks_align_with_target_ids():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    content = [
        _id(tokenizer, "def_start"),
        _id(tokenizer, "tensorid"),
        _id(tokenizer, "indexid"),
        _id(tokenizer, "rangeid"),
        _id(tokenizer, "coeff_num", 1),
        _id(tokenizer, "coeff_den"),
        _id(tokenizer, "tensorid", 1),
        _id(tokenizer, "indexid"),
        _id(tokenizer, "def_end"),
    ]
    decoder_input_ids = jnp.asarray(
        [[tokenizer.bos_token_id, *content, tokenizer.pad_token_id]],
        dtype=jnp.int32,
    )
    target_ids = [
        *content,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
    ]

    masks = grammar.valid_next_masks_for_decoder_input(decoder_input_ids)

    assert masks.shape == (
        1,
        decoder_input_ids.shape[1],
        tokenizer.vocab_size,
    )
    for position, target_id in enumerate(target_ids[:-1]):
        assert bool(masks[0, position, target_id])
    assert not bool(masks[0, len(target_ids) - 1, tokenizer.pad_token_id])


def test_invalid_prefix_has_no_valid_next_tokens():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    invalid_prefix = jnp.asarray(
        [
            [
                tokenizer.bos_token_id,
                _id(tokenizer, "def_start"),
                _id(tokenizer, "indexid"),
            ]
        ],
        dtype=jnp.int32,
    )

    mask = grammar.valid_next_mask_from_prefix(invalid_prefix)

    assert not bool(jnp.any(mask))


def test_advance_state_supports_scalar_and_batched_arrays():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)

    scalar_state = grammar.initial_state(())
    scalar_state = grammar.advance_state(
        scalar_state,
        jnp.asarray(tokenizer.bos_token_id, dtype=jnp.int32),
    )
    scalar_mask = grammar.allowed_by_state[scalar_state]

    assert _allowed_kinds(tokenizer, scalar_mask) == {"def_start", "eos"}

    batched_state = grammar.initial_state((2,))
    batched_state = grammar.advance_state(
        batched_state,
        jnp.asarray(
            [tokenizer.bos_token_id, tokenizer.bos_token_id],
            dtype=jnp.int32,
        ),
    )
    batched_state = grammar.advance_state(
        batched_state,
        jnp.asarray(
            [tokenizer.eos_token_id, _id(tokenizer, "def_start")],
            dtype=jnp.int32,
        ),
    )
    batched_masks = jnp.take(grammar.allowed_by_state, batched_state, axis=0)

    assert _allowed_kinds(tokenizer, batched_masks[0]) == {"pad"}
    assert _allowed_kinds(tokenizer, batched_masks[1]) == {"tensorid"}


def test_grammar_methods_are_jittable_for_fixed_shapes():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    decoder_input_ids = jnp.asarray(
        [
            [
                tokenizer.bos_token_id,
                _id(tokenizer, "def_start"),
                _id(tokenizer, "tensorid"),
                _id(tokenizer, "def_end"),
            ]
        ],
        dtype=jnp.int32,
    )

    masks = jax.jit(grammar.valid_next_masks_for_decoder_input)(decoder_input_ids)

    assert masks.shape == (1, 4, tokenizer.vocab_size)
    assert bool(masks[0, 3, tokenizer.eos_token_id])


def test_apply_grammar_mask_preserves_valid_logits_and_masks_invalid_logits():
    logits = jnp.asarray([[1.0, 2.0, 3.0]], dtype=jnp.float32)
    valid_next = jnp.asarray([[True, False, True]])

    masked = apply_grammar_mask(logits, valid_next)

    assert float(masked[0, 0]) == 1.0
    assert float(masked[0, 2]) == 3.0
    assert float(masked[0, 1]) < -1.0e20
