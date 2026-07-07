# NNX Transformer Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable Flax NNX Transformer encoder-decoder core that operates on dense vectors and masks.

**Architecture:** Add a focused `gristmill_symbolics.nn` package containing the reusable vector Transformer core. The core is independent from tokenizers and training code, uses pre-norm attention/MLP blocks, and calls `jax.nn.dot_product_attention` with a configurable `attention_implementation`.

**Tech Stack:** Python 3.11, JAX, Flax NNX, pytest, uv.

---

## File Structure

- Create `python/gristmill_symbolics/nn/__init__.py`: exports the reusable NNX Transformer core symbols.
- Create `python/gristmill_symbolics/nn/transformer.py`: `TransformerEncoder`, `TransformerDecoder`, `EncoderBlock`, `DecoderBlock`, and the private multi-head attention/MLP helpers.
- Create `python/tests/test_nnx_transformer_core.py`: focused tests for shape contracts, causal masking, source masking, deterministic behavior, tokenizer independence, and attention backend pass-through.

## Task 1: Public Shape Contract Tests

**Files:**
- Create: `python/tests/test_nnx_transformer_core.py`

- [ ] **Step 1: Write failing shape and import tests**

Create `python/tests/test_nnx_transformer_core.py` with:

```python
import jax
import jax.numpy as jnp
from flax import nnx

from gristmill_symbolics.nn import (
    DecoderBlock,
    EncoderBlock,
    TransformerDecoder,
    TransformerEncoder,
)


def _source_vectors(batch: int = 2, length: int = 5, d_model: int = 8):
    return jnp.arange(batch * length * d_model, dtype=jnp.float32).reshape(
        batch,
        length,
        d_model,
    ) / 100.0


def _target_vectors(batch: int = 2, length: int = 4, d_model: int = 8):
    return jnp.arange(batch * length * d_model, dtype=jnp.float32).reshape(
        batch,
        length,
        d_model,
    ) / 50.0


def test_encoder_and_decoder_return_vector_shapes():
    source = _source_vectors()
    target = _target_vectors()
    source_mask = jnp.array(
        [
            [True, True, True, False, False],
            [True, True, False, False, False],
        ]
    )
    target_mask = jnp.array(
        [
            [True, True, True, False],
            [True, True, False, False],
        ]
    )
    encoder = TransformerEncoder(
        d_model=8,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        rngs=nnx.Rngs(0),
    )
    decoder = TransformerDecoder(
        d_model=8,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        rngs=nnx.Rngs(1),
    )

    memory = encoder(source, source_mask, deterministic=True)
    decoded = decoder(
        target,
        memory,
        target_mask=target_mask,
        source_mask=source_mask,
        deterministic=True,
    )

    assert memory.shape == source.shape
    assert decoded.shape == target.shape


def test_blocks_return_vector_shapes():
    source = _source_vectors()
    target = _target_vectors()
    source_mask = jnp.ones(source.shape[:2], dtype=bool)
    target_mask = jnp.ones(target.shape[:2], dtype=bool)
    encoder_block = EncoderBlock(
        d_model=8,
        num_heads=2,
        dropout=0.0,
        rngs=nnx.Rngs(2),
    )
    decoder_block = DecoderBlock(
        d_model=8,
        num_heads=2,
        dropout=0.0,
        rngs=nnx.Rngs(3),
    )

    memory = encoder_block(source, source_mask, deterministic=True)
    decoded = decoder_block(
        target,
        memory,
        target_mask=target_mask,
        source_mask=source_mask,
        deterministic=True,
    )

    assert memory.shape == source.shape
    assert decoded.shape == target.shape
```

- [ ] **Step 2: Run the tests and verify RED**

Run from `python/`:

```bash
uv run pytest tests/test_nnx_transformer_core.py -q
```

Expected before implementation: FAIL during import with
`ModuleNotFoundError: No module named 'gristmill_symbolics.nn'`.

## Task 2: Minimal NNX Transformer Core

**Files:**
- Create: `python/gristmill_symbolics/nn/__init__.py`
- Create: `python/gristmill_symbolics/nn/transformer.py`
- Test: `python/tests/test_nnx_transformer_core.py`

- [ ] **Step 1: Add the package exports**

Create `python/gristmill_symbolics/nn/__init__.py`:

```python
from __future__ import annotations

from .transformer import (
    DecoderBlock,
    EncoderBlock,
    TransformerDecoder,
    TransformerEncoder,
)

__all__ = (
    "DecoderBlock",
    "EncoderBlock",
    "TransformerDecoder",
    "TransformerEncoder",
)
```

- [ ] **Step 2: Add the minimal Transformer implementation**

Create `python/gristmill_symbolics/nn/transformer.py`:

```python
from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx


AttentionImplementation = Literal["xla", "cudnn"] | None

__all__ = (
    "DecoderBlock",
    "EncoderBlock",
    "TransformerDecoder",
    "TransformerEncoder",
)


class TransformerEncoder(nnx.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_layers: int,
        num_heads: int,
        mlp_hidden_dim: int | None = None,
        dropout: float = 0.0,
        attention_implementation: AttentionImplementation = None,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_hidden_dim = mlp_hidden_dim or 4 * d_model
        self.dropout = dropout
        self.attention_implementation = attention_implementation
        self.layers = [
            EncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                mlp_hidden_dim=self.mlp_hidden_dim,
                dropout=dropout,
                attention_implementation=attention_implementation,
                rngs=rngs,
            )
            for _ in range(num_layers)
        ]
        self.final_norm = nnx.LayerNorm(d_model, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        source_mask: jax.Array | None = None,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        for layer in self.layers:
            x = layer(x, source_mask, deterministic=deterministic)
        return self.final_norm(x)


class TransformerDecoder(nnx.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_layers: int,
        num_heads: int,
        mlp_hidden_dim: int | None = None,
        dropout: float = 0.0,
        attention_implementation: AttentionImplementation = None,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_hidden_dim = mlp_hidden_dim or 4 * d_model
        self.dropout = dropout
        self.attention_implementation = attention_implementation
        self.layers = [
            DecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                mlp_hidden_dim=self.mlp_hidden_dim,
                dropout=dropout,
                attention_implementation=attention_implementation,
                rngs=rngs,
            )
            for _ in range(num_layers)
        ]
        self.final_norm = nnx.LayerNorm(d_model, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        memory: jax.Array,
        *,
        target_mask: jax.Array | None = None,
        source_mask: jax.Array | None = None,
        deterministic: bool = True,
    ) -> jax.Array:
        for layer in self.layers:
            x = layer(
                x,
                memory,
                target_mask=target_mask,
                source_mask=source_mask,
                deterministic=deterministic,
            )
        return self.final_norm(x)


class EncoderBlock(nnx.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        mlp_hidden_dim: int | None = None,
        dropout: float = 0.0,
        attention_implementation: AttentionImplementation = None,
        rngs: nnx.Rngs,
    ):
        self.self_attention = _MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            attention_implementation=attention_implementation,
            rngs=rngs,
        )
        self.mlp = _FeedForward(
            d_model=d_model,
            hidden_dim=mlp_hidden_dim or 4 * d_model,
            dropout=dropout,
            rngs=rngs,
        )
        self.attention_norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.mlp_norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.residual_dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        source_mask: jax.Array | None = None,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        attention_input = self.attention_norm(x)
        attention_output = self.self_attention(
            attention_input,
            attention_input,
            attention_input,
            key_mask=source_mask,
            deterministic=deterministic,
        )
        x = x + self.residual_dropout(attention_output, deterministic=deterministic)
        mlp_output = self.mlp(self.mlp_norm(x), deterministic=deterministic)
        x = x + self.residual_dropout(mlp_output, deterministic=deterministic)
        return _apply_query_mask(x, source_mask)


class DecoderBlock(nnx.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        mlp_hidden_dim: int | None = None,
        dropout: float = 0.0,
        attention_implementation: AttentionImplementation = None,
        rngs: nnx.Rngs,
    ):
        self.self_attention = _MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            attention_implementation=attention_implementation,
            rngs=rngs,
        )
        self.cross_attention = _MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            attention_implementation=attention_implementation,
            rngs=rngs,
        )
        self.mlp = _FeedForward(
            d_model=d_model,
            hidden_dim=mlp_hidden_dim or 4 * d_model,
            dropout=dropout,
            rngs=rngs,
        )
        self.self_attention_norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.cross_attention_norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.mlp_norm = nnx.LayerNorm(d_model, rngs=rngs)
        self.residual_dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        memory: jax.Array,
        *,
        target_mask: jax.Array | None = None,
        source_mask: jax.Array | None = None,
        deterministic: bool = True,
    ) -> jax.Array:
        self_attention_input = self.self_attention_norm(x)
        self_attention_output = self.self_attention(
            self_attention_input,
            self_attention_input,
            self_attention_input,
            key_mask=target_mask,
            is_causal=True,
            deterministic=deterministic,
        )
        x = x + self.residual_dropout(
            self_attention_output,
            deterministic=deterministic,
        )

        cross_attention_output = self.cross_attention(
            self.cross_attention_norm(x),
            memory,
            memory,
            key_mask=source_mask,
            deterministic=deterministic,
        )
        x = x + self.residual_dropout(
            cross_attention_output,
            deterministic=deterministic,
        )
        mlp_output = self.mlp(self.mlp_norm(x), deterministic=deterministic)
        x = x + self.residual_dropout(mlp_output, deterministic=deterministic)
        return _apply_query_mask(x, target_mask)


class _MultiHeadAttention(nnx.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        dropout: float,
        attention_implementation: AttentionImplementation,
        rngs: nnx.Rngs,
    ):
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attention_implementation = attention_implementation
        self.query = nnx.Linear(d_model, d_model, rngs=rngs)
        self.key = nnx.Linear(d_model, d_model, rngs=rngs)
        self.value = nnx.Linear(d_model, d_model, rngs=rngs)
        self.output = nnx.Linear(d_model, d_model, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(
        self,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        *,
        key_mask: jax.Array | None = None,
        is_causal: bool = False,
        deterministic: bool = True,
    ) -> jax.Array:
        query_heads = self._split_heads(self.query(query))
        key_heads = self._split_heads(self.key(key))
        value_heads = self._split_heads(self.value(value))
        attention_mask = _key_attention_mask(key_mask)
        attended = jax.nn.dot_product_attention(
            query_heads,
            key_heads,
            value_heads,
            mask=attention_mask,
            is_causal=is_causal,
            implementation=self.attention_implementation,
        )
        attended = self._merge_heads(attended)
        attended = self.output(attended)
        return self.dropout(attended, deterministic=deterministic)

    def _split_heads(self, x: jax.Array) -> jax.Array:
        batch, length, _ = x.shape
        return x.reshape(batch, length, self.num_heads, self.head_dim)

    def _merge_heads(self, x: jax.Array) -> jax.Array:
        batch, length, _, _ = x.shape
        return x.reshape(batch, length, self.d_model)


class _FeedForward(nnx.Module):
    def __init__(
        self,
        *,
        d_model: int,
        hidden_dim: int,
        dropout: float,
        rngs: nnx.Rngs,
    ):
        self.input = nnx.Linear(d_model, hidden_dim, rngs=rngs)
        self.output = nnx.Linear(hidden_dim, d_model, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        x = self.input(x)
        x = jax.nn.gelu(x)
        x = self.dropout(x, deterministic=deterministic)
        return self.output(x)


def _key_attention_mask(mask: jax.Array | None) -> jax.Array | None:
    if mask is None:
        return None
    return mask[:, None, None, :]


def _apply_query_mask(x: jax.Array, mask: jax.Array | None) -> jax.Array:
    if mask is None:
        return x
    return jnp.where(mask[..., None], x, 0.0)
```

- [ ] **Step 3: Run shape tests and verify GREEN**

Run from `python/`:

```bash
uv run pytest tests/test_nnx_transformer_core.py -q
```

Expected: PASS for the two shape tests.

## Task 3: Masking Behavior Tests

**Files:**
- Modify: `python/tests/test_nnx_transformer_core.py`
- Verify: `python/gristmill_symbolics/nn/transformer.py`

- [ ] **Step 1: Add causal and source-mask tests**

Append these tests to `python/tests/test_nnx_transformer_core.py`:

```python
def test_decoder_causal_mask_hides_later_target_positions():
    source = _source_vectors(batch=1, length=3, d_model=8)
    target = _target_vectors(batch=1, length=5, d_model=8)
    changed_future = target.at[:, 4, :].add(1000.0)
    source_mask = jnp.ones(source.shape[:2], dtype=bool)
    target_mask = jnp.ones(target.shape[:2], dtype=bool)
    encoder = TransformerEncoder(
        d_model=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        rngs=nnx.Rngs(4),
    )
    decoder = TransformerDecoder(
        d_model=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        rngs=nnx.Rngs(5),
    )
    memory = encoder(source, source_mask, deterministic=True)

    decoded = decoder(
        target,
        memory,
        target_mask=target_mask,
        source_mask=source_mask,
        deterministic=True,
    )
    changed_decoded = decoder(
        changed_future,
        memory,
        target_mask=target_mask,
        source_mask=source_mask,
        deterministic=True,
    )

    assert jnp.allclose(decoded[:, :4, :], changed_decoded[:, :4, :], atol=1e-5)


def test_masked_source_positions_do_not_affect_decoder_outputs():
    source = _source_vectors(batch=1, length=4, d_model=8)
    changed_masked_source = source.at[:, 3, :].add(5000.0)
    target = _target_vectors(batch=1, length=3, d_model=8)
    source_mask = jnp.array([[True, True, True, False]])
    target_mask = jnp.ones(target.shape[:2], dtype=bool)
    encoder = TransformerEncoder(
        d_model=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        rngs=nnx.Rngs(6),
    )
    decoder = TransformerDecoder(
        d_model=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        rngs=nnx.Rngs(7),
    )

    memory = encoder(source, source_mask, deterministic=True)
    changed_memory = encoder(changed_masked_source, source_mask, deterministic=True)
    decoded = decoder(
        target,
        memory,
        target_mask=target_mask,
        source_mask=source_mask,
        deterministic=True,
    )
    changed_decoded = decoder(
        target,
        changed_memory,
        target_mask=target_mask,
        source_mask=source_mask,
        deterministic=True,
    )

    assert jnp.allclose(decoded, changed_decoded, atol=1e-5)
```

- [ ] **Step 2: Run behavior tests and verify result**

Run from `python/`:

```bash
uv run pytest tests/test_nnx_transformer_core.py -q
```

Expected: PASS. If either test fails, fix only mask construction or masked
output handling in `python/gristmill_symbolics/nn/transformer.py`.

## Task 4: Determinism, Backend Pass-Through, and Scope Tests

**Files:**
- Modify: `python/tests/test_nnx_transformer_core.py`
- Verify: `python/gristmill_symbolics/nn/transformer.py`

- [ ] **Step 1: Add deterministic and backend tests**

Append these tests to `python/tests/test_nnx_transformer_core.py`:

```python
def test_deterministic_calls_are_repeatable_with_dropout_configured():
    source = _source_vectors(batch=1, length=3, d_model=8)
    source_mask = jnp.ones(source.shape[:2], dtype=bool)
    encoder = TransformerEncoder(
        d_model=8,
        num_layers=1,
        num_heads=2,
        dropout=0.25,
        rngs=nnx.Rngs(8),
    )

    first = encoder(source, source_mask, deterministic=True)
    second = encoder(source, source_mask, deterministic=True)

    assert jnp.allclose(first, second)


def test_attention_implementation_is_passed_to_jax_attention(monkeypatch):
    source = _source_vectors(batch=1, length=3, d_model=8)
    source_mask = jnp.ones(source.shape[:2], dtype=bool)
    seen: list[str | None] = []
    original_attention = jax.nn.dot_product_attention

    def spy_attention(*args, **kwargs):
        seen.append(kwargs.get("implementation"))
        return original_attention(*args, **kwargs)

    monkeypatch.setattr(jax.nn, "dot_product_attention", spy_attention)
    encoder = TransformerEncoder(
        d_model=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        attention_implementation="xla",
        rngs=nnx.Rngs(9),
    )

    encoder(source, source_mask, deterministic=True)

    assert seen == ["xla"]


def test_transformer_core_does_not_import_tokenizer():
    import gristmill_symbolics.nn.transformer as transformer

    assert not hasattr(transformer, "FlatDefinitionTokenizer")
```

- [ ] **Step 2: Run transformer tests**

Run from `python/`:

```bash
uv run pytest tests/test_nnx_transformer_core.py -q
```

Expected: PASS.

## Task 5: Final Verification and Commit

**Files:**
- Verify: `python/gristmill_symbolics/nn/__init__.py`
- Verify: `python/gristmill_symbolics/nn/transformer.py`
- Verify: `python/tests/test_nnx_transformer_core.py`
- Verify: `docs/superpowers/specs/2026-07-07-nnx-transformer-core-story.md`
- Verify: `docs/superpowers/plans/2026-07-07-nnx-transformer-core.md`

- [ ] **Step 1: Run focused transformer tests**

Run from `python/`:

```bash
uv run pytest tests/test_nnx_transformer_core.py -q
```

Expected: PASS.

- [ ] **Step 2: Run all Python tests**

Run from `python/`:

```bash
uv run pytest -q
```

Expected: PASS.

- [ ] **Step 3: Run Rust tests**

Run from the worktree root:

```bash
cargo test
```

Expected: PASS.

- [ ] **Step 4: Check whitespace**

Run from the worktree root:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 5: Commit**

Run from the worktree root:

```bash
git add docs/superpowers/specs/2026-07-07-nnx-transformer-core-story.md docs/superpowers/plans/2026-07-07-nnx-transformer-core.md python/gristmill_symbolics/nn/__init__.py python/gristmill_symbolics/nn/transformer.py python/tests/test_nnx_transformer_core.py
git commit -m "feat: add reusable nnx transformer core"
```
