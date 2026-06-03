# Transformer Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `python/transformer_policy` package that tokenizes symbolic rewrite states, scores grammar-constrained next-token choices with a causal Transformer policy, and exposes `sample_step` / `score_step` without depending on the deprecated `gristmill_rl` package.

**Architecture:** `transformer_policy` is split into tokenizer/context builders, token records, token embedder, replaceable neural sequence scorer, constrained decoder, and a high-level policy wrapper. It uses the merged PyO3 API: `RewriteState.from_computation`, `definition_mask`, `action_space_for_def`, `step_with_space`, and `ActionSpace.snapshot`.

**Tech Stack:** Python 3.11, NumPy, JAX, Flax NNX, PyO3 extension module `gristmill_symbolics`, pytest, `uv`.

---

## File Structure

- Create `python/transformer_policy/__init__.py`: public package exports.
- Create `python/transformer_policy/types.py`: immutable token records, stage-1 attempt records, policy sample records.
- Create `python/transformer_policy/tokenize.py`: faithful `TensorDef` tokenizer and state/action-space context builders.
- Create `python/transformer_policy/embed.py`: token feature matrix and default NNX token embedder.
- Create `python/transformer_policy/sequence_model.py`: causal Transformer next-token scorer.
- Create `python/transformer_policy/decoder.py`: dynamic grammar masks, sampling, scoring, and decision construction.
- Create `python/transformer_policy/policy.py`: high-level `TransformerPolicy` wrapper.
- Modify `python/pyproject.toml`: add `transformer_policy` to `python-packages`.
- Create `python/tests/transformer_policy_fixtures.py`: fixtures using the merged `RewriteState` API.
- Create `python/tests/test_transformer_policy_package.py`.
- Create `python/tests/test_transformer_policy_types.py`.
- Create `python/tests/test_transformer_policy_tokenize.py`.
- Create `python/tests/test_transformer_policy_embed.py`.
- Create `python/tests/test_transformer_policy_sequence_model.py`.
- Create `python/tests/test_transformer_policy_decoder.py`.
- Create `python/tests/test_transformer_policy_policy.py`.

Before running Python tests, refresh the local extension from `python/`:

```bash
uv run maturin develop
```

Expected: the `gristmill_symbolics` extension is built and importable in the `uv` environment.

---

### Task 1: Package Scaffold And Metadata

**Files:**
- Create: `python/transformer_policy/__init__.py`
- Modify: `python/pyproject.toml`
- Test: `python/tests/test_transformer_policy_package.py`

- [ ] **Step 1: Write the failing package test**

Create `python/tests/test_transformer_policy_package.py`:

```python
import importlib
import sys


def test_transformer_policy_imports_without_legacy_rl():
    sys.modules.pop("gristmill_rl", None)

    module = importlib.import_module("transformer_policy")

    assert module.__all__ == ()
    assert "gristmill_rl" not in sys.modules
```

- [ ] **Step 2: Run the test to verify it fails**

From `python/`, run:

```bash
uv run pytest tests/test_transformer_policy_package.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'transformer_policy'`.

- [ ] **Step 3: Create package exports and package metadata**

Create `python/transformer_policy/__init__.py`:

```python
"""Transformer policy for symbolic tensor rewrite decisions."""

__all__ = ()
```

Modify `python/pyproject.toml` so the maturin package list is:

```toml
python-packages = ["gristmill_rl", "transformer_policy"]
```

- [ ] **Step 4: Run the package test**

From `python/`, run:

```bash
uv run pytest tests/test_transformer_policy_package.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the scaffold**

```bash
git add python/pyproject.toml python/transformer_policy/__init__.py python/tests/test_transformer_policy_package.py
git commit -m "feat: scaffold transformer policy package"
```

---

### Task 2: Token And Policy Sample Types

**Files:**
- Create: `python/transformer_policy/types.py`
- Modify: `python/transformer_policy/__init__.py`
- Modify: `python/tests/test_transformer_policy_package.py`
- Test: `python/tests/test_transformer_policy_types.py`
- Re-test: `python/tests/test_transformer_policy_package.py`

- [ ] **Step 1: Write failing type tests**

Create `python/tests/test_transformer_policy_types.py`:

```python
import pytest

from transformer_policy.types import PolicySample, Stage1Attempt, T, Token


def test_token_factory_sorts_payload_for_stable_equality():
    left = T("FACTOR", tensor=3, position=1)
    right = Token(kind="FACTOR", payload=(("position", 1), ("tensor", 3)))

    assert left == right
    assert left.payload_dict() == {"position": 1, "tensor": 3}


def test_token_rejects_empty_kind():
    with pytest.raises(ValueError, match="token kind must not be empty"):
        T("")


def test_token_rejects_unsupported_payload_value():
    with pytest.raises(TypeError, match="unsupported token payload"):
        T("FACTOR", tensor=None)


def test_policy_sample_requires_consistent_terminal_shape():
    stopped = PolicySample(stopped=True, log_prob=0.25)

    assert stopped.def_index is None
    assert stopped.action_space is None
    assert stopped.decision is None
    assert stopped.def_attempts == ()
    assert stopped.decision_tokens == ()


def test_policy_sample_rejects_stopped_with_decision():
    with pytest.raises(ValueError, match="stopped sample must not contain a decision"):
        PolicySample(
            stopped=True,
            def_index=0,
            action_space=object(),
            decision={"candidate_index": 0, "left_mask": [], "right_mask": []},
            log_prob=0.0,
        )


def test_policy_sample_rejects_rewrite_without_decision():
    with pytest.raises(ValueError, match="rewrite sample requires def_index"):
        PolicySample(stopped=False, log_prob=0.0)


def test_stage1_attempt_records_acceptance_and_log_prob():
    attempt = Stage1Attempt(def_index=2, log_prob=-0.5, accepted=False)

    assert attempt.def_index == 2
    assert attempt.log_prob == -0.5
    assert not attempt.accepted


def test_package_exports_policy_types():
    import transformer_policy

    assert transformer_policy.__all__ == (
        "Token",
        "T",
        "Stage1Attempt",
        "PolicySample",
    )
```

- [ ] **Step 2: Run the type tests to verify they fail**

From `python/`, run:

```bash
uv run pytest tests/test_transformer_policy_types.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'transformer_policy.types'`.

- [ ] **Step 3: Implement immutable records**

Create `python/transformer_policy/types.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


PayloadValue = int | float | str | bool
Payload = tuple[tuple[str, PayloadValue], ...]


def _normalize_payload(payload: dict[str, PayloadValue]) -> Payload:
    normalized: list[tuple[str, PayloadValue]] = []
    for key, value in payload.items():
        if not isinstance(key, str) or not key:
            raise ValueError("token payload keys must be nonempty strings")
        if not isinstance(value, (int, float, str, bool)):
            raise TypeError(f"unsupported token payload for key '{key}'")
        normalized.append((key, value))
    return tuple(sorted(normalized, key=lambda item: item[0]))


@dataclass(frozen=True)
class Token:
    kind: str
    payload: Payload = ()

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("token kind must not be empty")
        for key, value in self.payload:
            if not isinstance(key, str) or not key:
                raise ValueError("token payload keys must be nonempty strings")
            if not isinstance(value, (int, float, str, bool)):
                raise TypeError(f"unsupported token payload for key '{key}'")

    @staticmethod
    def make(kind: str, **payload: PayloadValue) -> Token:
        return Token(kind=kind, payload=_normalize_payload(payload))

    def payload_dict(self) -> dict[str, PayloadValue]:
        return dict(self.payload)


T = Token.make


@dataclass(frozen=True)
class Stage1Attempt:
    def_index: int
    log_prob: float
    accepted: bool


@dataclass(frozen=True)
class PolicySample:
    stopped: bool
    log_prob: float
    def_index: int | None = None
    action_space: Any | None = None
    decision: dict[str, Any] | None = None
    def_attempts: tuple[Stage1Attempt, ...] = ()
    decision_tokens: tuple[Token, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.stopped:
            if self.def_index is not None or self.action_space is not None:
                raise ValueError("stopped sample must not contain a decision")
            if self.decision is not None:
                raise ValueError("stopped sample must not contain a decision")
            return
        if self.def_index is None:
            raise ValueError("rewrite sample requires def_index")
        if self.action_space is None:
            raise ValueError("rewrite sample requires action_space")
        if self.decision is None:
            raise ValueError("rewrite sample requires decision")
```

Modify `python/transformer_policy/__init__.py`:

```python
"""Transformer policy for symbolic tensor rewrite decisions."""

from transformer_policy.types import PolicySample, Stage1Attempt, T, Token

__all__ = (
    "Token",
    "T",
    "Stage1Attempt",
    "PolicySample",
)
```

Modify `python/tests/test_transformer_policy_package.py` so the export assertion is:

```python
    assert module.__all__ == (
        "Token",
        "T",
        "Stage1Attempt",
        "PolicySample",
    )
```

- [ ] **Step 4: Run package and type tests**

From `python/`, run:

```bash
uv run pytest tests/test_transformer_policy_package.py tests/test_transformer_policy_types.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit token and sample types**

```bash
git add python/transformer_policy/__init__.py python/transformer_policy/types.py python/tests/test_transformer_policy_package.py python/tests/test_transformer_policy_types.py
git commit -m "feat: add transformer policy types"
```

---

### Task 3: Faithful Tokenizer And Context Builders

**Files:**
- Create: `python/transformer_policy/tokenize.py`
- Create: `python/tests/transformer_policy_fixtures.py`
- Test: `python/tests/test_transformer_policy_tokenize.py`

- [ ] **Step 1: Write failing tokenizer tests**

Create `python/tests/transformer_policy_fixtures.py`:

```python
import json

from gristmill_symbolics import RewriteState, TensorComputation


def actionable_json() -> str:
    return json.dumps(
        {
            "ranges": [{"id": 0, "size": 8}],
            "tensors": [
                {"id": 0, "symmetry": []},
                {"id": 1, "symmetry": []},
                {"id": 2, "symmetry": []},
                {"id": 3, "symmetry": []},
            ],
            "definitions": [
                {
                    "base": 3,
                    "ext_indices": [
                        {"id": 0, "range": 0},
                        {"id": 1, "range": 0},
                    ],
                    "terms": [
                        {
                            "coeff": [1, 1],
                            "sum_indices": [{"id": 2, "range": 0}],
                            "factors": [
                                {"tensor": 0, "indices": [0, 2]},
                                {"tensor": 1, "indices": [2, 1]},
                            ],
                        },
                        {
                            "coeff": [1, 1],
                            "sum_indices": [{"id": 3, "range": 0}],
                            "factors": [
                                {"tensor": 0, "indices": [0, 3]},
                                {"tensor": 2, "indices": [3, 1]},
                            ],
                        },
                    ],
                }
            ],
        }
    )


def actionable_state():
    comp = TensorComputation.from_json_string(actionable_json())
    return RewriteState.from_computation(comp)


def actionable_space():
    state = actionable_state()
    space = state.action_space_for_def(0)
    assert space is not None
    return state, space
```

Create `python/tests/test_transformer_policy_tokenize.py`:

```python
from transformer_policy.tokenize import (
    build_action_space_context,
    build_state_context,
    tokenize_tensor_def,
)
from transformer_policy.types import T

from .transformer_policy_fixtures import actionable_space, actionable_state


def tensor_definition_snapshot():
    state = actionable_state()
    return state.snapshot()["definitions"][0]


def test_tokenize_tensor_def_preserves_field_order_and_raw_ids():
    tokens = tokenize_tensor_def(tensor_definition_snapshot())

    assert tokens[:15] == (
        T("DEF_START"),
        T("BASE", tensor=3),
        T("EXT_INDEX", position=0, id=0, range=0),
        T("EXT_INDEX", position=1, id=1, range=0),
        T("TERM_START", position=0),
        T("COEFF_NUM", value=1),
        T("COEFF_DEN", value=1),
        T("SUM_INDEX", position=0, id=2, range=0),
        T("FACTOR", position=0, tensor=0, arity=2),
        T("INDEX", position=0, id=0),
        T("INDEX", position=1, id=2),
        T("FACTOR", position=1, tensor=1, arity=2),
        T("INDEX", position=0, id=2),
        T("INDEX", position=1, id=1),
        T("TERM_END", position=0),
    )
    assert tokens[-1] == T("DEF_END")


def test_build_state_context_wraps_definitions():
    state = actionable_state()

    tokens = build_state_context(state.snapshot())

    assert tokens[0] == T("STATE_START")
    assert tokens[1] == T("STATE_DEF", def_index=0)
    assert T("BASE", tensor=3) in tokens
    assert tokens[-1] == T("STATE_END")


def test_build_action_space_context_wraps_candidates_and_nested_defs():
    _, space = actionable_space()

    tokens = build_action_space_context(space.snapshot())

    assert tokens[0] == T("ACTION_SPACE_START", def_index=0)
    assert T("CAND_START", candidate_index=0) in tokens
    assert T("LEFT_DEF_START", candidate_index=0) in tokens
    assert T("RIGHT_DEF_START", candidate_index=0) in tokens
    assert T("REWRITTEN_DEF_START", candidate_index=0) in tokens
    assert tokens[-1] == T("ACTION_SPACE_END")
```

- [ ] **Step 2: Run tokenizer tests to verify they fail**

From `python/`, run:

```bash
uv run pytest tests/test_transformer_policy_tokenize.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'transformer_policy.tokenize'`.

- [ ] **Step 3: Implement tokenizer and context builders**

Create `python/transformer_policy/tokenize.py`:

```python
from __future__ import annotations

from typing import Any

from transformer_policy.types import T, Token


def _coeff_pair(coeff: Any) -> tuple[int, int]:
    if isinstance(coeff, dict):
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
                tokens.append(
                    T("INDEX", position=index_position, id=int(index_id))
                )
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


def build_action_space_context(action_space_snapshot: dict[str, Any]) -> tuple[Token, ...]:
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
```

- [ ] **Step 4: Run tokenizer tests**

From `python/`, run:

```bash
uv run pytest tests/test_transformer_policy_tokenize.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit tokenizer**

```bash
git add python/transformer_policy/tokenize.py python/tests/transformer_policy_fixtures.py python/tests/test_transformer_policy_tokenize.py
git commit -m "feat: tokenize symbolic tensor contexts"
```

---

### Task 4: Token Embedder Interface

**Files:**
- Create: `python/transformer_policy/embed.py`
- Test: `python/tests/test_transformer_policy_embed.py`

- [ ] **Step 1: Write failing embedder tests**

Create `python/tests/test_transformer_policy_embed.py`:

```python
import numpy as np
from flax import nnx

from transformer_policy.embed import (
    PAYLOAD_KEYS,
    TOKEN_FEATURE_DIM,
    TokenEmbedder,
    token_features,
)
from transformer_policy.types import T


def test_token_features_are_deterministic_float32_matrix():
    tokens = (
        T("DEF_START"),
        T("FACTOR", tensor=3, position=1, arity=2),
    )

    features = token_features(tokens)

    assert features.shape == (2, TOKEN_FEATURE_DIM)
    assert features.dtype == np.float32
    assert TOKEN_FEATURE_DIM == 2 + len(PAYLOAD_KEYS)
    np.testing.assert_array_equal(features, token_features(tokens))


def test_token_features_reject_unknown_kind():
    tokens = (T("UNKNOWN_KIND"),)

    try:
        token_features(tokens)
    except ValueError as error:
        assert "unknown token kind" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_token_embedder_projects_tokens_to_hidden_vectors():
    tokens = (
        T("STATE_START"),
        T("STATE_END"),
    )
    embedder = TokenEmbedder(hidden_dim=8, rngs=nnx.Rngs(0))

    values = embedder(tokens)

    assert values.shape == (2, 8)
    assert np.isfinite(np.asarray(values)).all()
```

- [ ] **Step 2: Run embedder tests to verify they fail**

From `python/`, run:

```bash
uv run pytest tests/test_transformer_policy_embed.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'transformer_policy.embed'`.

- [ ] **Step 3: Implement token features and embedder**

Create `python/transformer_policy/embed.py`:

```python
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from transformer_policy.types import PayloadValue, Token


TOKEN_KINDS = (
    "ACTION_SPACE_END",
    "ACTION_SPACE_START",
    "BASE",
    "CAND",
    "CAND_END",
    "CAND_START",
    "COEFF_DEN",
    "COEFF_NUM",
    "DEF",
    "DEF_END",
    "DEF_START",
    "END",
    "EXT_INDEX",
    "FACTOR",
    "INDEX",
    "LEFT_DEF_END",
    "LEFT_DEF_START",
    "LEFT_DROP",
    "LEFT_KEEP",
    "REWRITTEN_DEF_END",
    "REWRITTEN_DEF_START",
    "RIGHT_DEF_END",
    "RIGHT_DEF_START",
    "RIGHT_DROP",
    "RIGHT_KEEP",
    "STATE_DEF",
    "STATE_END",
    "STATE_START",
    "STOP",
    "SUM_INDEX",
    "TERM_END",
    "TERM_START",
)
TOKEN_KIND_TO_ID = {kind: index for index, kind in enumerate(TOKEN_KINDS)}

PAYLOAD_KEYS = (
    "accepted",
    "arity",
    "candidate_index",
    "def_index",
    "id",
    "position",
    "range",
    "tensor",
    "value",
)
PAYLOAD_KEY_TO_COLUMN = {key: index for index, key in enumerate(PAYLOAD_KEYS)}
TOKEN_FEATURE_DIM = 2 + len(PAYLOAD_KEYS)


def _payload_value(value: PayloadValue) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return float(jax.random.bits(jax.random.key(abs(hash(value)) % (2**31)), ())) / float(2**32)


def token_features(tokens: tuple[Token, ...]) -> np.ndarray:
    features = np.zeros((len(tokens), TOKEN_FEATURE_DIM), dtype=np.float32)
    denominator = max(len(TOKEN_KINDS) - 1, 1)
    for row, token in enumerate(tokens):
        if token.kind not in TOKEN_KIND_TO_ID:
            raise ValueError(f"unknown token kind '{token.kind}'")
        features[row, 0] = float(TOKEN_KIND_TO_ID[token.kind]) / float(denominator)
        features[row, 1] = float(row)
        for key, value in token.payload:
            if key not in PAYLOAD_KEY_TO_COLUMN:
                continue
            features[row, 2 + PAYLOAD_KEY_TO_COLUMN[key]] = _payload_value(value)
    return features


class TokenEmbedder(nnx.Module):
    def __init__(self, *, hidden_dim: int, rngs: nnx.Rngs):
        self.proj = nnx.Linear(TOKEN_FEATURE_DIM, hidden_dim, rngs=rngs)

    def __call__(self, tokens: tuple[Token, ...]) -> jax.Array:
        features = jnp.asarray(token_features(tokens), dtype=jnp.float32)
        return nnx.relu(self.proj(features))
```

- [ ] **Step 4: Run embedder tests**

From `python/`, run:

```bash
uv run pytest tests/test_transformer_policy_embed.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit embedder**

```bash
git add python/transformer_policy/embed.py python/tests/test_transformer_policy_embed.py
git commit -m "feat: add transformer policy token embedder"
```

---

### Task 5: Causal Transformer Next-Token Scorer

**Files:**
- Create: `python/transformer_policy/sequence_model.py`
- Test: `python/tests/test_transformer_policy_sequence_model.py`

- [ ] **Step 1: Write failing sequence model tests**

Create `python/tests/test_transformer_policy_sequence_model.py`:

```python
import numpy as np
import pytest
from flax import nnx

from transformer_policy.sequence_model import CausalTransformerScorer
from transformer_policy.types import T


def test_sequence_model_scores_legal_next_tokens():
    scorer = CausalTransformerScorer(
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        mlp_dim=32,
        rngs=nnx.Rngs(0),
    )
    context = (T("STATE_START"), T("STATE_END"))
    legal = (T("STOP"), T("DEF", def_index=0))

    logits = scorer.score_next(context, (), legal)

    assert logits.shape == (2,)
    assert np.isfinite(np.asarray(logits)).all()


def test_sequence_model_uses_decision_prefix():
    scorer = CausalTransformerScorer(
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        mlp_dim=32,
        rngs=nnx.Rngs(1),
    )
    context = (T("STATE_START"), T("STATE_END"))
    legal = (T("LEFT_KEEP"), T("LEFT_DROP"))

    without_prefix = np.asarray(scorer.score_next(context, (), legal))
    with_prefix = np.asarray(scorer.score_next(context, (T("CAND", candidate_index=0),), legal))

    assert without_prefix.shape == with_prefix.shape
    assert not np.array_equal(without_prefix, with_prefix)


def test_sequence_model_rejects_empty_legal_set():
    scorer = CausalTransformerScorer(
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        mlp_dim=32,
        rngs=nnx.Rngs(2),
    )

    with pytest.raises(ValueError, match="legal_next_tokens must not be empty"):
        scorer.score_next((T("STATE_START"),), (), ())
```

- [ ] **Step 2: Run sequence model tests to verify they fail**

From `python/`, run:

```bash
uv run pytest tests/test_transformer_policy_sequence_model.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'transformer_policy.sequence_model'`.

- [ ] **Step 3: Implement the causal Transformer scorer**

Create `python/transformer_policy/sequence_model.py`:

```python
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from transformer_policy.embed import TokenEmbedder
from transformer_policy.types import Token


class TransformerBlock(nnx.Module):
    def __init__(self, *, hidden_dim: int, num_heads: int, mlp_dim: int, rngs: nnx.Rngs):
        self.ln_1 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_dim,
            qkv_features=hidden_dim,
            rngs=rngs,
        )
        self.ln_2 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.mlp_1 = nnx.Linear(hidden_dim, mlp_dim, rngs=rngs)
        self.mlp_2 = nnx.Linear(mlp_dim, hidden_dim, rngs=rngs)

    def __call__(self, values: jax.Array, mask: jax.Array) -> jax.Array:
        attended = self.attn(self.ln_1(values), mask=mask, deterministic=True)
        values = values + attended
        mlp = self.mlp_2(nnx.gelu(self.mlp_1(self.ln_2(values))))
        return values + mlp


class CausalTransformerScorer(nnx.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 1,
        mlp_dim: int = 64,
        rngs: nnx.Rngs,
    ):
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.embedder = TokenEmbedder(hidden_dim=hidden_dim, rngs=rngs)
        self.blocks = [
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                rngs=rngs,
            )
            for _ in range(num_layers)
        ]
        self.final_ln = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.query = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)

    def _encode(self, tokens: tuple[Token, ...]) -> jax.Array:
        if not tokens:
            raise ValueError("context plus prefix must not be empty")
        values = self.embedder(tokens)[None, :, :]
        token_mask = jnp.ones((1, len(tokens)), dtype=bool)
        causal_mask = nnx.make_causal_mask(token_mask)
        for block in self.blocks:
            values = block(values, causal_mask)
        return self.final_ln(values[0])

    def score_next(
        self,
        context_tokens: tuple[Token, ...],
        decision_prefix: tuple[Token, ...],
        legal_next_tokens: tuple[Token, ...],
    ) -> jax.Array:
        if not legal_next_tokens:
            raise ValueError("legal_next_tokens must not be empty")
        hidden = self._encode((*context_tokens, *decision_prefix))[-1]
        query = self.query(hidden)
        legal_embeddings = self.embedder(legal_next_tokens)
        return jnp.matmul(legal_embeddings, query)
```

- [ ] **Step 4: Run sequence model tests**

From `python/`, run:

```bash
uv run pytest tests/test_transformer_policy_sequence_model.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit sequence model**

```bash
git add python/transformer_policy/sequence_model.py python/tests/test_transformer_policy_sequence_model.py
git commit -m "feat: add transformer policy sequence scorer"
```

---

### Task 6: Constrained Decoder Sampling

**Files:**
- Create: `python/transformer_policy/decoder.py`
- Test: `python/tests/test_transformer_policy_decoder.py`

- [ ] **Step 1: Write failing decoder tests**

Create `python/tests/test_transformer_policy_decoder.py`:

```python
import numpy as np
import pytest

from gristmill_symbolics import TensorComputation, RewriteState
from transformer_policy.decoder import sample_step, score_step
from transformer_policy.types import T

from .transformer_policy_fixtures import actionable_state


class PreferenceScorer:
    def score_next(self, context_tokens, decision_prefix, legal_next_tokens):
        scores = []
        for token in legal_next_tokens:
            payload = token.payload_dict()
            if token.kind == "DEF":
                scores.append(1.0e6)
            elif token.kind == "STOP":
                scores.append(-1.0e6)
            elif token.kind == "CAND" and payload.get("candidate_index") == 0:
                scores.append(1.0e6)
            elif token.kind.endswith("KEEP"):
                scores.append(1.0e6)
            elif token.kind == "END":
                scores.append(1.0e6)
            else:
                scores.append(-1.0e6)
        return np.asarray(scores, dtype=np.float32)


class StopScorer:
    def score_next(self, context_tokens, decision_prefix, legal_next_tokens):
        return np.asarray(
            [
                1.0e6 if token.kind == "STOP" else -1.0e6
                for token in legal_next_tokens
            ],
            dtype=np.float32,
        )


def empty_state():
    text = """
    {
      "ranges": [{"id": 0, "size": 8}],
      "tensors": [{"id": 0, "symmetry": []}],
      "definitions": [
        {
          "base": 0,
          "ext_indices": [{"id": 0, "range": 0}],
          "terms": [
            {
              "coeff": [1, 1],
              "sum_indices": [],
              "factors": [{"tensor": 0, "indices": [0]}]
            }
          ]
        }
      ]
    }
    """
    return RewriteState.from_computation(TensorComputation.from_json_string(text))


def test_sample_step_can_stop_when_mask_is_empty():
    sample = sample_step(empty_state(), StopScorer(), np.random.default_rng(0))

    assert sample.stopped
    assert sample.decision is None
    assert sample.def_attempts == ()
    assert sample.decision_tokens == (T("STOP"),)


def test_sample_step_returns_rewrite_decision_that_rust_can_apply():
    state = actionable_state()

    sample = sample_step(state, PreferenceScorer(), np.random.default_rng(0))

    assert not sample.stopped
    assert sample.def_index == 0
    assert sample.decision == {
        "candidate_index": 0,
        "left_mask": [True],
        "right_mask": [True, True],
    }
    assert sample.action_space is not None
    state.step_with_space(sample.action_space, sample.decision)


def test_score_step_replays_sample_log_probability():
    state = actionable_state()
    sample = sample_step(state, PreferenceScorer(), np.random.default_rng(0))

    rescored = score_step(actionable_state(), PreferenceScorer(), sample)

    assert rescored == pytest.approx(sample.log_prob)


def test_score_step_rejects_invalid_mask_length():
    sample = sample_step(actionable_state(), PreferenceScorer(), np.random.default_rng(0))
    invalid = sample.__class__(
        stopped=False,
        def_index=sample.def_index,
        action_space=sample.action_space,
        decision={
            "candidate_index": sample.decision["candidate_index"],
            "left_mask": [True, False],
            "right_mask": sample.decision["right_mask"],
        },
        log_prob=sample.log_prob,
        def_attempts=sample.def_attempts,
        decision_tokens=sample.decision_tokens,
    )

    with pytest.raises(ValueError, match="invalid left_mask length"):
        score_step(actionable_state(), PreferenceScorer(), invalid)
```

- [ ] **Step 2: Run decoder tests to verify they fail**

From `python/`, run:

```bash
uv run pytest tests/test_transformer_policy_decoder.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'transformer_policy.decoder'`.

- [ ] **Step 3: Implement constrained sampling and scoring**

Create `python/transformer_policy/decoder.py`:

```python
from __future__ import annotations

from typing import Protocol

import numpy as np

from transformer_policy.tokenize import (
    build_action_space_context,
    build_state_context,
)
from transformer_policy.types import PolicySample, Stage1Attempt, T, Token


class NextTokenScorer(Protocol):
    def score_next(
        self,
        context_tokens: tuple[Token, ...],
        decision_prefix: tuple[Token, ...],
        legal_next_tokens: tuple[Token, ...],
    ):
        raise NotImplementedError


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return shifted - np.log(exp.sum())


def _sample_token(
    scorer: NextTokenScorer,
    context: tuple[Token, ...],
    prefix: tuple[Token, ...],
    legal: tuple[Token, ...],
    rng: np.random.Generator,
) -> tuple[Token, float]:
    if not legal:
        raise ValueError("legal token set must not be empty")
    logits = np.asarray(scorer.score_next(context, prefix, legal), dtype=np.float64)
    log_probs = _log_softmax(logits)
    probs = np.exp(log_probs)
    index = int(rng.choice(len(legal), p=probs))
    return legal[index], float(log_probs[index])


def _score_token(
    scorer: NextTokenScorer,
    context: tuple[Token, ...],
    prefix: tuple[Token, ...],
    legal: tuple[Token, ...],
    chosen: Token,
) -> float:
    if chosen not in legal:
        raise ValueError(f"illegal token {chosen.kind}")
    logits = np.asarray(scorer.score_next(context, prefix, legal), dtype=np.float64)
    log_probs = _log_softmax(logits)
    return float(log_probs[legal.index(chosen)])


def _stage1_legal(state) -> tuple[Token, ...]:
    tokens = [T("STOP")]
    for def_index, allowed in enumerate(state.definition_mask()):
        if allowed:
            tokens.append(T("DEF", def_index=def_index))
    return tuple(tokens)


def _candidate_legal(space_snapshot: dict) -> tuple[Token, ...]:
    return tuple(
        T("CAND", candidate_index=index)
        for index, _candidate in enumerate(space_snapshot["candidate_templates"])
    )


def _bit_legal(kind_prefix: str, is_final: bool, kept_any: bool) -> tuple[Token, ...]:
    keep = T(f"{kind_prefix}_KEEP")
    drop = T(f"{kind_prefix}_DROP")
    if is_final and not kept_any:
        return (keep,)
    return (keep, drop)


def _sample_bits(
    *,
    scorer: NextTokenScorer,
    context: tuple[Token, ...],
    prefix: list[Token],
    kind_prefix: str,
    term_count: int,
    rng: np.random.Generator,
) -> tuple[list[bool], float]:
    bits: list[bool] = []
    log_prob = 0.0
    kept_any = False
    for term_index in range(term_count):
        legal = _bit_legal(kind_prefix, term_index == term_count - 1, kept_any)
        token, token_log_prob = _sample_token(scorer, context, tuple(prefix), legal, rng)
        prefix.append(token)
        keep = token.kind.endswith("KEEP")
        bits.append(keep)
        kept_any = kept_any or keep
        log_prob += token_log_prob
    return bits, log_prob


def _score_bits(
    *,
    scorer: NextTokenScorer,
    context: tuple[Token, ...],
    prefix: list[Token],
    kind_prefix: str,
    bits: list[bool],
) -> float:
    log_prob = 0.0
    kept_any = False
    for index, keep in enumerate(bits):
        legal = _bit_legal(kind_prefix, index == len(bits) - 1, kept_any)
        chosen = T(f"{kind_prefix}_{'KEEP' if keep else 'DROP'}")
        log_prob += _score_token(scorer, context, tuple(prefix), legal, chosen)
        prefix.append(chosen)
        kept_any = kept_any or keep
    return log_prob


def sample_step(state, scorer: NextTokenScorer, rng: np.random.Generator) -> PolicySample:
    attempts: list[Stage1Attempt] = []
    total_log_prob = 0.0
    while True:
        state_context = build_state_context(state.snapshot())
        stage1_token, stage1_log_prob = _sample_token(
            scorer, state_context, (), _stage1_legal(state), rng
        )
        total_log_prob += stage1_log_prob
        if stage1_token.kind == "STOP":
            return PolicySample(
                stopped=True,
                log_prob=total_log_prob,
                def_attempts=tuple(attempts),
                decision_tokens=(T("STOP"),),
            )
        def_index = int(stage1_token.payload_dict()["def_index"])
        space = state.action_space_for_def(def_index)
        accepted = space is not None
        attempts.append(
            Stage1Attempt(
                def_index=def_index,
                log_prob=stage1_log_prob,
                accepted=accepted,
            )
        )
        if accepted:
            break

    assert space is not None
    space_snapshot = space.snapshot()
    context = (*build_state_context(state.snapshot()), *build_action_space_context(space_snapshot))
    prefix: list[Token] = []
    candidate_token, candidate_log_prob = _sample_token(
        scorer, context, tuple(prefix), _candidate_legal(space_snapshot), rng
    )
    prefix.append(candidate_token)
    total_log_prob += candidate_log_prob
    candidate_index = int(candidate_token.payload_dict()["candidate_index"])
    candidate = space_snapshot["candidate_templates"][candidate_index]

    left_bits, left_log_prob = _sample_bits(
        scorer=scorer,
        context=context,
        prefix=prefix,
        kind_prefix="LEFT",
        term_count=len(candidate["left_definition"]["terms"]),
        rng=rng,
    )
    right_bits, right_log_prob = _sample_bits(
        scorer=scorer,
        context=context,
        prefix=prefix,
        kind_prefix="RIGHT",
        term_count=len(candidate["right_definition"]["terms"]),
        rng=rng,
    )
    prefix.append(T("END"))
    total_log_prob += left_log_prob + right_log_prob
    return PolicySample(
        stopped=False,
        def_index=def_index,
        action_space=space,
        decision={
            "candidate_index": candidate_index,
            "left_mask": left_bits,
            "right_mask": right_bits,
        },
        log_prob=total_log_prob,
        def_attempts=tuple(attempts),
        decision_tokens=tuple(prefix),
    )


def score_step(state, scorer: NextTokenScorer, sample: PolicySample) -> float:
    total_log_prob = 0.0
    accepted_space = None
    for attempt in sample.def_attempts:
        context = build_state_context(state.snapshot())
        chosen = T("DEF", def_index=attempt.def_index)
        total_log_prob += _score_token(scorer, context, (), _stage1_legal(state), chosen)
        space = state.action_space_for_def(attempt.def_index)
        if attempt.accepted:
            if space is None:
                raise ValueError("invalid def_index accepted by sample trace")
            accepted_space = space
        else:
            if space is not None:
                raise ValueError("sample trace rejects an available def_index")
    if sample.stopped:
        context = build_state_context(state.snapshot())
        total_log_prob += _score_token(scorer, context, (), _stage1_legal(state), T("STOP"))
        return total_log_prob
    if accepted_space is None:
        raise ValueError("rewrite sample requires an accepted def_index")
    if sample.decision is None:
        raise ValueError("rewrite sample requires decision")
    space_snapshot = accepted_space.snapshot()
    context = (*build_state_context(state.snapshot()), *build_action_space_context(space_snapshot))
    prefix: list[Token] = []
    candidate_index = int(sample.decision["candidate_index"])
    candidate_token = T("CAND", candidate_index=candidate_index)
    total_log_prob += _score_token(
        scorer, context, tuple(prefix), _candidate_legal(space_snapshot), candidate_token
    )
    prefix.append(candidate_token)
    candidate = space_snapshot["candidate_templates"][candidate_index]
    left_mask = [bool(value) for value in sample.decision["left_mask"]]
    right_mask = [bool(value) for value in sample.decision["right_mask"]]
    if len(left_mask) != len(candidate["left_definition"]["terms"]):
        raise ValueError("invalid left_mask length")
    if len(right_mask) != len(candidate["right_definition"]["terms"]):
        raise ValueError("invalid right_mask length")
    total_log_prob += _score_bits(
        scorer=scorer,
        context=context,
        prefix=prefix,
        kind_prefix="LEFT",
        bits=left_mask,
    )
    total_log_prob += _score_bits(
        scorer=scorer,
        context=context,
        prefix=prefix,
        kind_prefix="RIGHT",
        bits=right_mask,
    )
    return total_log_prob
```

- [ ] **Step 4: Run decoder tests**

From `python/`, run:

```bash
uv run pytest tests/test_transformer_policy_decoder.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit decoder**

```bash
git add python/transformer_policy/decoder.py python/tests/test_transformer_policy_decoder.py
git commit -m "feat: add constrained transformer policy decoder"
```

---

### Task 7: High-Level Policy Wrapper

**Files:**
- Create: `python/transformer_policy/policy.py`
- Modify: `python/transformer_policy/__init__.py`
- Test: `python/tests/test_transformer_policy_policy.py`

- [ ] **Step 1: Write failing policy wrapper tests**

Create `python/tests/test_transformer_policy_policy.py`:

```python
import numpy as np
from flax import nnx

from transformer_policy import TransformerPolicy
from transformer_policy.sequence_model import CausalTransformerScorer

from .transformer_policy_fixtures import actionable_state


def test_transformer_policy_wires_sequence_model_to_sample_and_score():
    scorer = CausalTransformerScorer(
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        mlp_dim=32,
        rngs=nnx.Rngs(0),
    )
    policy = TransformerPolicy(scorer=scorer)
    sample = policy.sample_step(actionable_state(), np.random.default_rng(0))

    assert np.isfinite(sample.log_prob)
    rescored = policy.score_step(actionable_state(), sample)
    assert np.isfinite(rescored)
```

- [ ] **Step 2: Run policy wrapper test to verify it fails**

From `python/`, run:

```bash
uv run pytest tests/test_transformer_policy_policy.py -q
```

Expected: FAIL with `ImportError` for `TransformerPolicy`.

- [ ] **Step 3: Implement wrapper and public exports**

Create `python/transformer_policy/policy.py`:

```python
from __future__ import annotations

import numpy as np

from transformer_policy.decoder import NextTokenScorer, sample_step, score_step
from transformer_policy.types import PolicySample


class TransformerPolicy:
    def __init__(self, *, scorer: NextTokenScorer):
        self.scorer = scorer

    def sample_step(self, state, rng: np.random.Generator) -> PolicySample:
        return sample_step(state, self.scorer, rng)

    def score_step(self, state, sample: PolicySample) -> float:
        return score_step(state, self.scorer, sample)
```

Modify `python/transformer_policy/__init__.py`:

```python
"""Transformer policy for symbolic tensor rewrite decisions."""

from transformer_policy.policy import TransformerPolicy
from transformer_policy.types import PolicySample, Stage1Attempt, T, Token

__all__ = (
    "Token",
    "T",
    "Stage1Attempt",
    "PolicySample",
    "TransformerPolicy",
)
```

Update `python/tests/test_transformer_policy_package.py` expected `__all__`:

```python
    assert module.__all__ == (
        "Token",
        "T",
        "Stage1Attempt",
        "PolicySample",
        "TransformerPolicy",
    )
```

- [ ] **Step 4: Run package and policy tests**

From `python/`, run:

```bash
uv run pytest tests/test_transformer_policy_package.py tests/test_transformer_policy_policy.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit policy wrapper**

```bash
git add python/transformer_policy/__init__.py python/transformer_policy/policy.py python/tests/test_transformer_policy_package.py python/tests/test_transformer_policy_policy.py
git commit -m "feat: add transformer policy wrapper"
```

---

### Task 8: Full Verification

**Files:**
- Read: `python/transformer_policy/*.py`
- Read: `python/tests/test_transformer_policy_*.py`
- Read: `python/pyproject.toml`

- [ ] **Step 1: Verify no dependency on deprecated RL package**

From repository root, run:

```bash
rg -n "gristmill_rl" python/transformer_policy python/tests/test_transformer_policy_*.py
```

Expected: no matches in `python/transformer_policy`. Test files may mention `gristmill_rl` only in the package import guard test.

- [ ] **Step 2: Run all transformer policy tests**

From `python/`, run:

```bash
uv run pytest tests/test_transformer_policy_package.py tests/test_transformer_policy_types.py tests/test_transformer_policy_tokenize.py tests/test_transformer_policy_embed.py tests/test_transformer_policy_sequence_model.py tests/test_transformer_policy_decoder.py tests/test_transformer_policy_policy.py -q
```

Expected: PASS.

- [ ] **Step 3: Run PyO3 binding tests around `RewriteState`**

From `python/`, run:

```bash
uv run pytest tests/test_bindings.py -q
```

Expected: PASS.

- [ ] **Step 4: Run Rust rewrite tests**

From repository root, run:

```bash
cargo test rewrite --test rewrite
```

Expected: PASS.

- [ ] **Step 5: Commit final verification notes if files changed during fixes**

If Task 8 required code fixes, commit those files:

```bash
git add python/transformer_policy python/tests python/pyproject.toml
git commit -m "test: verify transformer policy package"
```

Expected: commit is created only if verification fixes changed files.

---

## Self-Review

- Spec coverage: The plan creates `transformer_policy`, keeps it independent of `gristmill_rl`, implements faithful tokenization, defines token embedding, adds a replaceable causal Transformer scorer, implements grammar-constrained sampling/scoring, supports `STOP`, uses lazy stage-1 exact action-space queries, and reserves `reinforce_training` for later.
- Scope check: This plan does not implement REINFORCE, MCTS, replay, training loops, objectives, metrics, checkpoints, graph encoding, DeepSets, or ID canonicalization.
- Type consistency: `Token`, `T`, `Stage1Attempt`, `PolicySample`, `TokenEmbedder`, `CausalTransformerScorer`, `sample_step`, `score_step`, and `TransformerPolicy` are introduced before use in later tasks.
- Verification: The final task verifies new policy tests, PyO3 `RewriteState` bindings, Rust rewrite tests, and the no-import boundary against `gristmill_rl`.
