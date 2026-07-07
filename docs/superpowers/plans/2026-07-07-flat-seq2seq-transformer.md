# Flat Seq2Seq Transformer Wrapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a token-level NNX seq2seq wrapper that maps padded source and decoder token IDs to vocabulary logits.

**Architecture:** Create `gristmill_symbolics.nn.flat_seq2seq` as a thin module around the existing stable vector Transformer core. The wrapper owns token/position embeddings, derives source and target padding masks from `pad_token_id`, zeroes embedded pad positions, calls `TransformerEncoder` and `TransformerDecoder`, and projects decoder states to vocabulary logits.

**Tech Stack:** Python 3.11, JAX, Flax NNX, pytest, uv.

---

## File Structure

- Create `python/gristmill_symbolics/nn/flat_seq2seq.py`: defines `FlatDefinitionSeq2SeqTransformer`.
- Modify `python/gristmill_symbolics/nn/__init__.py`: exports `FlatDefinitionSeq2SeqTransformer`.
- Create `python/tests/test_flat_seq2seq_transformer.py`: focused wrapper tests.
- Do not modify `python/gristmill_symbolics/nn/transformer.py`.

## Task 1: Wrapper Shape And Export Tests

- [ ] Add tests importing `FlatDefinitionSeq2SeqTransformer` from `gristmill_symbolics.nn`.
- [ ] Add a test constructing the model with `nnx.Rngs` and asserting logits have shape `[B, target_len, vocab_size]`.
- [ ] Run `uv run pytest tests/test_flat_seq2seq_transformer.py -q` from `python/` and verify it fails before implementation because the wrapper is missing.

## Task 2: Padding And Determinism Tests

- [ ] Add a test showing changing source pad token IDs does not affect logits when those positions are masked.
- [ ] Add a test showing decoder pad positions produce zero logits after masking and output projection.
- [ ] Add a test showing deterministic calls repeat with dropout disabled.
- [ ] Add a test showing the wrapper module can be imported without importing tokenizer or grammar modules.
- [ ] Run `uv run pytest tests/test_flat_seq2seq_transformer.py -q` and verify the tests fail before implementation.

## Task 3: Wrapper Implementation

- [ ] Create `python/gristmill_symbolics/nn/flat_seq2seq.py`.
- [ ] Implement constructor fields for `source_len`, `target_len`, `vocab_size`, `pad_token_id`, `d_model`, `num_layers`, `num_heads`, `mlp_hidden_dim`, `dropout`, `attention_implementation`, `dtype`, `param_dtype`, and `rngs`.
- [ ] Add shared token embedding, source positional embedding, target positional embedding, embedding dropout, encoder, decoder, and output head.
- [ ] Implement `_embed(ids, position_embed, deterministic)` using token plus position embeddings, zeroing positions where `ids == pad_token_id`.
- [ ] Implement `__call__(source_ids, decoder_input_ids, deterministic=True)` to derive masks, embed IDs, call encoder/decoder, zero padded decoded states, and return vocabulary logits.
- [ ] Export `FlatDefinitionSeq2SeqTransformer` from `python/gristmill_symbolics/nn/__init__.py`.
- [ ] Re-run `uv run pytest tests/test_flat_seq2seq_transformer.py -q`.

## Task 4: Verification

- [ ] Run `uv run pytest tests/test_flat_seq2seq_transformer.py tests/test_nnx_transformer_core.py -q`.
- [ ] Run `uv run pytest -q`.
- [ ] Run `cargo test`.
- [ ] Run `git diff --check`.
- [ ] Review `git diff --stat` and `git diff` for scope control and confirm `nn/transformer.py` is unchanged.
