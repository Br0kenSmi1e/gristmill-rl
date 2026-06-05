# Policy And REINFORCE API Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the token-decoder-centered policy/training prototype with a semantic two-stage rewrite policy API and a REINFORCE trainer that trains from policy log-probability terms.

**Architecture:** `transformer_policy` exposes `TargetSelection`, `ActionConstruction`, `RewriteDecision`, policy-owned decision traces, and a v1 attention-based `RewriteAttentionPolicy`. `reinforce_training` executes semantic decisions, computes rewards and advantages, scores policy traces in bounded chunks, and applies one optimizer update per logical batch. The old token-choice decoder, trace, batch, and sequence-scorer training surfaces are removed from the supported public surface.

**Tech Stack:** Python 3.11, NumPy, JAX, Flax NNX, Optax, Orbax, PyO3 `gristmill_symbolics`, pytest, uv.

---

## File Structure

- Create `python/transformer_policy/api.py`: public semantic policy records and policy score records.
- Create `python/transformer_policy/attention_policy.py`: v1 attention policy with direct target logits and autoregressive action construction.
- Replace `python/transformer_policy/policy.py`: public `TransformerPolicy` compatibility name should alias or wrap `RewriteAttentionPolicy` only during migration; no old decoder dependency remains.
- Modify `python/transformer_policy/__init__.py`: export only the semantic policy surface plus token utilities still needed by the model.
- Delete `python/transformer_policy/decoder.py`: old next-token decoder surface.
- Delete `python/transformer_policy/trace.py`: old token event trace surface.
- Delete `python/transformer_policy/batch.py`: old padded token event batch surface.
- Delete `python/transformer_policy/sequence_model.py`: old causal next-token scorer surface.
- Keep `python/transformer_policy/types.py`, `tokenize.py`, and `embed.py`: token records and token features remain internal representation utilities.
- Replace `python/reinforce_training/trace.py`: episode traces store semantic decisions and opaque policy traces, not token events.
- Replace `python/reinforce_training/rollout.py`: rollout samples through `RewritePolicy.sample_decision`.
- Replace `python/reinforce_training/objective.py`: objective consumes policy traces and returned `PolicyLogpTerm` values.
- Modify `python/reinforce_training/train.py`: add `--score-chunk-size`, remove token-event batching, report target/action diagnostics.
- Modify `python/reinforce_training/checkpoint.py`: checkpoint the new policy config and model class.
- Modify `python/reinforce_training/__init__.py`: export semantic episode trace records only.
- Replace token-decoder tests with semantic policy tests:
  - Create `python/tests/test_transformer_policy_api.py`
  - Create `python/tests/test_transformer_policy_attention_policy.py`
  - Replace `python/tests/test_transformer_policy_policy.py`
  - Delete `python/tests/test_transformer_policy_decoder.py`
  - Delete `python/tests/test_transformer_policy_trace.py`
  - Delete `python/tests/test_transformer_policy_batch.py`
  - Delete `python/tests/test_transformer_policy_sequence_model.py`
- Replace REINFORCE tests:
  - Replace `python/tests/test_reinforce_trace.py`
  - Replace `python/tests/test_reinforce_rollout.py`
  - Replace `python/tests/test_reinforce_objective.py`
  - Modify `python/tests/test_reinforce_checkpoint.py`
  - Modify `python/tests/test_reinforce_train.py`

Before running Python tests after Rust-facing changes, refresh the extension from `python/`:

```bash
uv run maturin develop
```

Expected: the `gristmill_symbolics` extension is built and importable.

---

### Task 1: Add Semantic Policy API Records

**Files:**
- Create: `python/transformer_policy/api.py`
- Modify: `python/transformer_policy/__init__.py`
- Create: `python/tests/test_transformer_policy_api.py`
- Modify: `python/tests/test_transformer_policy_package.py`

- [ ] **Step 1: Write failing semantic API tests**

Create `python/tests/test_transformer_policy_api.py`:

```python
import jax.numpy as jnp
import pytest

from transformer_policy.api import (
    ActionConstruction,
    PolicyDecisionTrace,
    PolicyLogpTerm,
    PolicySample,
    PolicyScoreBatch,
    RewriteDecision,
    TargetSelection,
)


def test_target_selection_stop_and_definition_records():
    stop = TargetSelection.stop()
    definition = TargetSelection.definition(3)

    assert stop.kind == "stop"
    assert stop.def_index is None
    assert definition.kind == "definition"
    assert definition.def_index == 3


def test_target_selection_rejects_invalid_definition_index():
    with pytest.raises(ValueError, match="def_index must be non-negative"):
        TargetSelection.definition(-1)

    with pytest.raises(ValueError, match="stop target must not carry def_index"):
        TargetSelection(kind="stop", def_index=0)


def test_action_construction_normalizes_masks_and_rust_decision():
    action = ActionConstruction(
        candidate_index=2,
        left_mask=[True, False],
        right_mask=(False, True),
    )

    assert action.left_mask == (True, False)
    assert action.right_mask == (False, True)
    assert action.to_rust_decision() == {
        "candidate_index": 2,
        "left_mask": [True, False],
        "right_mask": [False, True],
    }


def test_rewrite_decision_validates_stop_and_action_shape():
    stop = RewriteDecision.stop()
    rewrite = RewriteDecision.rewrite(
        def_index=1,
        action_construction=ActionConstruction(
            candidate_index=0,
            left_mask=(True,),
            right_mask=(True, True),
        ),
    )

    assert stop.stopped
    assert stop.def_index is None
    assert rewrite.def_index == 1
    assert rewrite.action_construction is not None

    with pytest.raises(ValueError, match="stop decision must not contain action"):
        RewriteDecision(
            target_selection=TargetSelection.stop(),
            action_construction=ActionConstruction(
                candidate_index=0,
                left_mask=(True,),
                right_mask=(True,),
            ),
        )


def test_policy_trace_exposes_metadata_not_model_internals():
    trace = PolicyDecisionTrace(
        trace_id=7,
        rollout_step=2,
        decision=RewriteDecision.stop(),
        target_selection_present=True,
        action_construction_present=False,
        trainable_choice_count=1,
        payload={"private": "policy-owned"},
    )

    assert trace.trace_id == 7
    assert trace.rollout_step == 2
    assert trace.metadata == {
        "trace_id": 7,
        "rollout_step": 2,
        "target_selection_present": True,
        "action_construction_present": False,
        "trainable_choice_count": 1,
    }


def test_policy_score_batch_groups_logp_terms():
    term = PolicyLogpTerm(
        trace_id=1,
        rollout_step=0,
        kind="target_selection",
        logp=jnp.asarray(-0.5),
        role="target",
    )
    batch = PolicyScoreBatch(terms=(term,), diagnostics={"mean_entropy": 0.0})

    assert batch.terms == (term,)
    assert batch.diagnostics == {"mean_entropy": 0.0}


def test_policy_sample_requires_trace_decision_match():
    decision = RewriteDecision.stop()

    with pytest.raises(ValueError, match="sample decision must match trace decision"):
        PolicySample(
            decision=decision,
            trace=PolicyDecisionTrace(
                trace_id=0,
                rollout_step=0,
                decision=RewriteDecision.rewrite(
                    def_index=0,
                    action_construction=ActionConstruction(
                        candidate_index=0,
                        left_mask=(True,),
                        right_mask=(True,),
                    ),
                ),
                target_selection_present=True,
                action_construction_present=True,
                trainable_choice_count=2,
                payload={},
            ),
            metrics={},
        )
```

Modify `python/tests/test_transformer_policy_package.py` so it expects the new public names:

```python
import importlib
import sys


def test_transformer_policy_imports_without_legacy_rl():
    sys.modules.pop("gristmill_rl", None)

    module = importlib.import_module("transformer_policy")

    assert "gristmill_rl" not in sys.modules
    assert "RewriteDecision" in module.__all__
    assert "RewriteAttentionPolicy" in module.__all__
```

- [ ] **Step 2: Run API tests to verify they fail**

From `python/`, run:

```bash
uv run pytest tests/test_transformer_policy_api.py tests/test_transformer_policy_package.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'transformer_policy.api'` or missing exports.

- [ ] **Step 3: Implement semantic records**

Create `python/transformer_policy/api.py`:

```python
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Sequence

import numpy as np


TargetKind = Literal["stop", "definition"]
LogpKind = Literal["target_selection", "action_construction"]
PolicyMode = Literal["reinforce", "inference"]


@dataclass(frozen=True)
class TargetSelection:
    kind: TargetKind
    def_index: int | None = None

    def __post_init__(self) -> None:
        if self.kind == "stop":
            if self.def_index is not None:
                raise ValueError("stop target must not carry def_index")
            return
        if self.kind != "definition":
            raise ValueError("target kind must be 'stop' or 'definition'")
        if type(self.def_index) is not int or self.def_index < 0:
            raise ValueError("def_index must be non-negative")

    @staticmethod
    def stop() -> "TargetSelection":
        return TargetSelection(kind="stop")

    @staticmethod
    def definition(def_index: int) -> "TargetSelection":
        return TargetSelection(kind="definition", def_index=def_index)


@dataclass(frozen=True)
class ActionConstruction:
    candidate_index: int
    left_mask: Sequence[bool]
    right_mask: Sequence[bool]

    def __post_init__(self) -> None:
        if type(self.candidate_index) is not int or self.candidate_index < 0:
            raise ValueError("candidate_index must be non-negative")
        left = tuple(self.left_mask)
        right = tuple(self.right_mask)
        if not left:
            raise ValueError("left_mask must not be empty")
        if not right:
            raise ValueError("right_mask must not be empty")
        if any(type(value) is not bool for value in (*left, *right)):
            raise ValueError("mask entries must be bool")
        object.__setattr__(self, "left_mask", left)
        object.__setattr__(self, "right_mask", right)

    def to_rust_decision(self) -> dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "left_mask": list(self.left_mask),
            "right_mask": list(self.right_mask),
        }


@dataclass(frozen=True)
class RewriteDecision:
    target_selection: TargetSelection
    action_construction: ActionConstruction | None = None

    def __post_init__(self) -> None:
        if self.target_selection.kind == "stop":
            if self.action_construction is not None:
                raise ValueError("stop decision must not contain action")
            return
        if self.action_construction is None:
            raise ValueError("rewrite decision requires action construction")

    @staticmethod
    def stop() -> "RewriteDecision":
        return RewriteDecision(target_selection=TargetSelection.stop())

    @staticmethod
    def rewrite(
        *,
        def_index: int,
        action_construction: ActionConstruction,
    ) -> "RewriteDecision":
        return RewriteDecision(
            target_selection=TargetSelection.definition(def_index),
            action_construction=action_construction,
        )

    @property
    def stopped(self) -> bool:
        return self.target_selection.kind == "stop"

    @property
    def def_index(self) -> int | None:
        return self.target_selection.def_index

    def to_rust_decision(self) -> dict[str, Any] | None:
        if self.action_construction is None:
            return None
        return self.action_construction.to_rust_decision()


@dataclass(frozen=True)
class PolicyDecisionTrace:
    trace_id: int
    rollout_step: int
    decision: RewriteDecision
    target_selection_present: bool
    action_construction_present: bool
    trainable_choice_count: int
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        if type(self.trace_id) is not int or self.trace_id < 0:
            raise ValueError("trace_id must be non-negative")
        if type(self.rollout_step) is not int or self.rollout_step < 0:
            raise ValueError("rollout_step must be non-negative")
        if self.trainable_choice_count <= 0:
            raise ValueError("trainable_choice_count must be positive")
        object.__setattr__(self, "payload", deepcopy(self.payload))

    @property
    def metadata(self) -> dict[str, int | bool]:
        return {
            "trace_id": self.trace_id,
            "rollout_step": self.rollout_step,
            "target_selection_present": self.target_selection_present,
            "action_construction_present": self.action_construction_present,
            "trainable_choice_count": self.trainable_choice_count,
        }


@dataclass(frozen=True)
class PolicySample:
    decision: RewriteDecision
    trace: PolicyDecisionTrace
    metrics: dict[str, float | int | bool]

    def __post_init__(self) -> None:
        if self.decision != self.trace.decision:
            raise ValueError("sample decision must match trace decision")
        metrics = dict(self.metrics)
        for key, value in metrics.items():
            if not isinstance(key, str):
                raise ValueError("metric keys must be strings")
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError("metric values must be finite")
        object.__setattr__(self, "metrics", metrics)


@dataclass(frozen=True)
class PolicyLogpTerm:
    trace_id: int
    rollout_step: int
    kind: LogpKind
    logp: Any
    role: str

    def __post_init__(self) -> None:
        if type(self.trace_id) is not int or self.trace_id < 0:
            raise ValueError("trace_id must be non-negative")
        if type(self.rollout_step) is not int or self.rollout_step < 0:
            raise ValueError("rollout_step must be non-negative")
        if self.kind not in {"target_selection", "action_construction"}:
            raise ValueError("logp term kind is invalid")
        if not self.role:
            raise ValueError("role must not be empty")


@dataclass(frozen=True)
class PolicyScoreBatch:
    terms: tuple[PolicyLogpTerm, ...]
    diagnostics: dict[str, float | int]

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("score batch must contain at least one logp term")
        object.__setattr__(self, "terms", tuple(self.terms))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


class RewritePolicy(Protocol):
    def sample_decision(
        self,
        state,
        rng,
        *,
        trace_id: int,
        rollout_step: int,
        mode: PolicyMode,
    ) -> PolicySample:
        raise NotImplementedError

    def score_traces(
        self,
        traces: tuple[PolicyDecisionTrace, ...],
    ) -> PolicyScoreBatch:
        raise NotImplementedError
```

Modify `python/transformer_policy/__init__.py`:

```python
"""Transformer policy for semantic symbolic tensor rewrite decisions."""

from transformer_policy.api import (
    ActionConstruction,
    PolicyDecisionTrace,
    PolicyLogpTerm,
    PolicySample,
    PolicyScoreBatch,
    RewriteDecision,
    RewritePolicy,
    TargetSelection,
)
from transformer_policy.attention_policy import AttentionPolicyConfig, RewriteAttentionPolicy
from transformer_policy.types import T, Token

TransformerPolicy = RewriteAttentionPolicy

__all__ = (
    "Token",
    "T",
    "TargetSelection",
    "ActionConstruction",
    "RewriteDecision",
    "PolicyDecisionTrace",
    "PolicyLogpTerm",
    "PolicySample",
    "PolicyScoreBatch",
    "RewritePolicy",
    "AttentionPolicyConfig",
    "RewriteAttentionPolicy",
    "TransformerPolicy",
)
```

Temporarily create `python/transformer_policy/attention_policy.py` with a stub so imports succeed until Task 2:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionPolicyConfig:
    hidden_dim: int = 32
    num_heads: int = 4
    num_layers: int = 1
    mlp_dim: int = 64


class RewriteAttentionPolicy:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("RewriteAttentionPolicy is implemented in Task 2")
```

- [ ] **Step 4: Run API tests**

From `python/`, run:

```bash
uv run pytest tests/test_transformer_policy_api.py tests/test_transformer_policy_package.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit semantic API records**

```bash
git add python/transformer_policy/api.py python/transformer_policy/__init__.py python/transformer_policy/attention_policy.py python/tests/test_transformer_policy_api.py python/tests/test_transformer_policy_package.py
git commit -m "feat: add semantic rewrite policy API"
```

---

### Task 2: Add V1 Attention Rewrite Policy

**Files:**
- Replace: `python/transformer_policy/attention_policy.py`
- Replace: `python/transformer_policy/policy.py`
- Create: `python/tests/test_transformer_policy_attention_policy.py`
- Modify: `python/tests/test_transformer_policy_policy.py`

- [ ] **Step 1: Write failing attention policy tests**

Create `python/tests/test_transformer_policy_attention_policy.py`:

```python
import numpy as np
import pytest
from flax import nnx

from transformer_policy.api import RewriteDecision
from transformer_policy.attention_policy import AttentionPolicyConfig, RewriteAttentionPolicy

from .transformer_policy_fixtures import actionable_state


def _policy(seed: int = 0) -> RewriteAttentionPolicy:
    return RewriteAttentionPolicy(
        config=AttentionPolicyConfig(hidden_dim=16, num_heads=4, num_layers=1, mlp_dim=32),
        rngs=nnx.Rngs(seed),
    )


def test_attention_policy_samples_semantic_decision_and_trace():
    sample = _policy().sample_decision(
        actionable_state(),
        np.random.default_rng(0),
        trace_id=0,
        rollout_step=3,
        mode="reinforce",
    )

    assert isinstance(sample.decision, RewriteDecision)
    assert sample.trace.rollout_step == 3
    assert sample.trace.target_selection_present
    assert sample.trace.trainable_choice_count >= 1
    assert "token_events" not in sample.trace.payload


def test_attention_policy_score_traces_returns_stage_logp_terms():
    policy = _policy()
    sample = policy.sample_decision(
        actionable_state(),
        np.random.default_rng(0),
        trace_id=1,
        rollout_step=0,
        mode="reinforce",
    )

    scores = policy.score_traces((sample.trace,))

    assert {term.kind for term in scores.terms}.issubset(
        {"target_selection", "action_construction"}
    )
    assert any(term.kind == "target_selection" for term in scores.terms)
    assert all(term.trace_id == sample.trace.trace_id for term in scores.terms)
    assert all(np.isfinite(float(term.logp)) for term in scores.terms)


def test_attention_policy_can_score_rewrite_action_terms_when_sample_rewrites():
    for seed in range(20):
        policy = _policy(seed)
        sample = policy.sample_decision(
            actionable_state(),
            np.random.default_rng(seed),
            trace_id=seed,
            rollout_step=0,
            mode="reinforce",
        )
        if not sample.decision.stopped:
            scores = policy.score_traces((sample.trace,))
            assert any(term.kind == "action_construction" for term in scores.terms)
            roles = {term.role for term in scores.terms}
            assert {"candidate", "left_mask", "right_mask"}.issubset(roles)
            return

    pytest.skip("random seeds produced only STOP decisions")


def test_attention_policy_rejects_invalid_config():
    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        AttentionPolicyConfig(hidden_dim=0)

    with pytest.raises(ValueError, match="hidden_dim must be divisible by num_heads"):
        AttentionPolicyConfig(hidden_dim=18, num_heads=4)
```

Replace `python/tests/test_transformer_policy_policy.py`:

```python
import numpy as np
from flax import nnx

from transformer_policy import AttentionPolicyConfig, TransformerPolicy

from .transformer_policy_fixtures import actionable_state


def test_transformer_policy_alias_uses_semantic_attention_policy():
    policy = TransformerPolicy(
        config=AttentionPolicyConfig(hidden_dim=16, num_heads=4, num_layers=1, mlp_dim=32),
        rngs=nnx.Rngs(0),
    )

    sample = policy.sample_decision(
        actionable_state(),
        np.random.default_rng(0),
        trace_id=0,
        rollout_step=0,
        mode="reinforce",
    )
    scores = policy.score_traces((sample.trace,))

    assert sample.trace.decision == sample.decision
    assert scores.terms
```

- [ ] **Step 2: Run attention policy tests to verify they fail**

From `python/`, run:

```bash
uv run pytest tests/test_transformer_policy_attention_policy.py tests/test_transformer_policy_policy.py -q
```

Expected: FAIL with `NotImplementedError: RewriteAttentionPolicy is implemented in Task 2`.

- [ ] **Step 3: Implement attention policy**

Replace `python/transformer_policy/attention_policy.py` with a real implementation. Use these signatures and invariants:

```python
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from transformer_policy.api import (
    ActionConstruction,
    PolicyDecisionTrace,
    PolicyLogpTerm,
    PolicyMode,
    PolicySample,
    PolicyScoreBatch,
    RewriteDecision,
    TargetSelection,
)
from transformer_policy.embed import TokenEmbedder, token_features
from transformer_policy.tokenize import build_action_space_context, build_state_context
from transformer_policy.types import T, Token


def _validate_positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class AttentionPolicyConfig:
    hidden_dim: int = 32
    num_heads: int = 4
    num_layers: int = 1
    mlp_dim: int = 64

    def __post_init__(self) -> None:
        _validate_positive("hidden_dim", self.hidden_dim)
        _validate_positive("num_heads", self.num_heads)
        _validate_positive("num_layers", self.num_layers)
        _validate_positive("mlp_dim", self.mlp_dim)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")


class AttentionBlock(nnx.Module):
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
        attended = self.attn(self.ln_1(values), mask=mask, deterministic=True, decode=False)
        values = values + attended
        return values + self.mlp_2(nnx.gelu(self.mlp_1(self.ln_2(values))))


class RewriteAttentionPolicy(nnx.Module):
    def __init__(self, *, config: AttentionPolicyConfig, rngs: nnx.Rngs):
        self.config = config
        self.embedder = TokenEmbedder(hidden_dim=config.hidden_dim, rngs=rngs)
        self.blocks = nnx.List(
            AttentionBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                mlp_dim=config.mlp_dim,
                rngs=rngs,
            )
            for _ in range(config.num_layers)
        )
        self.final_ln = nnx.LayerNorm(config.hidden_dim, rngs=rngs)
        self.stop_head = nnx.Linear(config.hidden_dim, 1, rngs=rngs)
        self.target_head = nnx.Linear(config.hidden_dim * 2, 1, rngs=rngs)
        self.candidate_head = nnx.Linear(config.hidden_dim * 2, 1, rngs=rngs)
        self.mask_head = nnx.Linear(config.hidden_dim * 3 + 1, 2, rngs=rngs)

    def _encode_tokens(self, tokens: tuple[Token, ...]) -> jax.Array:
        if not tokens:
            raise ValueError("tokens must not be empty")
        features = jnp.asarray(token_features(tokens), dtype=jnp.float32)
        values = self.embedder.proj(features)[None, :, :]
        length = values.shape[1]
        mask = jnp.ones((1, 1, length, length), dtype=bool)
        for block in self.blocks:
            values = block(values, mask)
        return self.final_ln(values[0])

    def _state_vectors(self, state_snapshot: dict[str, Any]) -> tuple[jax.Array, dict[int, jax.Array]]:
        tokens = build_state_context(state_snapshot)
        encoded = self._encode_tokens(tokens)
        state_vector = jnp.mean(encoded, axis=0)
        def_vectors: dict[int, jax.Array] = {}
        for index, token in enumerate(tokens):
            if token.kind == "STATE_DEF":
                def_vectors[int(token.payload_dict()["def_index"])] = encoded[index]
        return state_vector, def_vectors

    def _target_logits(
        self,
        state_snapshot: dict[str, Any],
        legal_def_indices: tuple[int, ...],
    ) -> tuple[jax.Array, tuple[TargetSelection, ...]]:
        state_vector, def_vectors = self._state_vectors(state_snapshot)
        labels = [TargetSelection.stop()]
        logits = [self.stop_head(state_vector)[0]]
        for def_index in legal_def_indices:
            labels.append(TargetSelection.definition(def_index))
            pair = jnp.concatenate([state_vector, def_vectors[def_index]], axis=0)
            logits.append(self.target_head(pair)[0])
        return jnp.stack(logits), tuple(labels)

    def _action_tokens_and_vectors(
        self,
        state_snapshot: dict[str, Any],
        action_space_snapshot: dict[str, Any],
    ) -> tuple[jax.Array, dict[int, jax.Array]]:
        tokens = (*build_state_context(state_snapshot), *build_action_space_context(action_space_snapshot))
        encoded = self._encode_tokens(tokens)
        action_vector = jnp.mean(encoded, axis=0)
        candidate_vectors: dict[int, jax.Array] = {}
        for index, token in enumerate(tokens):
            if token.kind == "CAND_START":
                candidate_vectors[int(token.payload_dict()["candidate_index"])] = encoded[index]
        return action_vector, candidate_vectors

    def _candidate_logits(
        self,
        state_snapshot: dict[str, Any],
        action_space_snapshot: dict[str, Any],
    ) -> jax.Array:
        action_vector, candidate_vectors = self._action_tokens_and_vectors(
            state_snapshot,
            action_space_snapshot,
        )
        logits = []
        for candidate_index, _candidate in enumerate(action_space_snapshot["candidate_templates"]):
            pair = jnp.concatenate([action_vector, candidate_vectors[candidate_index]], axis=0)
            logits.append(self.candidate_head(pair)[0])
        return jnp.stack(logits)

    def _mask_logits(
        self,
        *,
        state_snapshot: dict[str, Any],
        action_space_snapshot: dict[str, Any],
        candidate_index: int,
        side: str,
        previous_mask: tuple[bool, ...],
        term_count: int,
    ) -> jax.Array:
        action_vector, candidate_vectors = self._action_tokens_and_vectors(
            state_snapshot,
            action_space_snapshot,
        )
        candidate_vector = candidate_vectors[candidate_index]
        logits = []
        previous_keep_count = sum(previous_mask)
        for term_index in range(term_count):
            progress = jnp.asarray(
                [term_index / max(term_count - 1, 1)],
                dtype=jnp.float32,
            )
            side_value = 0.0 if side == "left" else 1.0
            side_feature = jnp.asarray([side_value + previous_keep_count], dtype=jnp.float32)
            features = jnp.concatenate(
                [action_vector, candidate_vector, action_vector * candidate_vector, progress + side_feature],
                axis=0,
            )
            row = self.mask_head(features)
            is_final = term_index == term_count - 1
            if is_final and previous_keep_count == 0:
                row = jnp.asarray([-jnp.inf, row[1]], dtype=row.dtype)
            logits.append(row)
        return jnp.stack(logits)

    def _sample_index(self, logits: jax.Array, rng: np.random.Generator) -> tuple[int, float]:
        values = np.asarray(logits, dtype=np.float64)
        shifted = values - np.max(values)
        log_probs = shifted - np.log(np.exp(shifted).sum())
        probs = np.exp(log_probs)
        index = int(rng.choice(len(values), p=probs))
        return index, float(log_probs[index])

    def sample_decision(
        self,
        state,
        rng: np.random.Generator,
        *,
        trace_id: int,
        rollout_step: int,
        mode: PolicyMode,
    ) -> PolicySample:
        if type(trace_id) is not int or trace_id < 0:
            raise ValueError("trace_id must be non-negative")
        state_snapshot = state.snapshot()
        legal_def_indices = tuple(
            index for index, allowed in enumerate(state.definition_mask()) if allowed
        )
        target_logits, target_labels = self._target_logits(state_snapshot, legal_def_indices)
        target_index, target_logp = self._sample_index(target_logits, rng)
        target = target_labels[target_index]
        payload: dict[str, Any] = {
            "state_snapshot": deepcopy(state_snapshot),
            "legal_def_indices": legal_def_indices,
            "target_index": target_index,
        }
        trainable_count = 1
        action_present = False
        logp = target_logp
        if target.kind == "stop":
            decision = RewriteDecision.stop()
        else:
            assert target.def_index is not None
            action_space = state.action_space_for_def(target.def_index)
            if action_space is None:
                raise ValueError("selected definition has no action space")
            action_space_snapshot = action_space.snapshot()
            candidate_logits = self._candidate_logits(state_snapshot, action_space_snapshot)
            candidate_index, candidate_logp = self._sample_index(candidate_logits, rng)
            candidate = action_space_snapshot["candidate_templates"][candidate_index]
            left_count = len(candidate["left_definition"]["terms"])
            right_count = len(candidate["right_definition"]["terms"])
            left_mask = self._sample_mask(
                state_snapshot=state_snapshot,
                action_space_snapshot=action_space_snapshot,
                candidate_index=candidate_index,
                side="left",
                term_count=left_count,
                rng=rng,
            )
            right_mask = self._sample_mask(
                state_snapshot=state_snapshot,
                action_space_snapshot=action_space_snapshot,
                candidate_index=candidate_index,
                side="right",
                term_count=right_count,
                rng=rng,
            )
            action = ActionConstruction(
                candidate_index=candidate_index,
                left_mask=left_mask[0],
                right_mask=right_mask[0],
            )
            decision = RewriteDecision.rewrite(
                def_index=target.def_index,
                action_construction=action,
            )
            logp += candidate_logp + left_mask[1] + right_mask[1]
            trainable_count += 3
            action_present = True
            payload.update(
                {
                    "action_space_snapshot": deepcopy(action_space_snapshot),
                    "candidate_index": candidate_index,
                    "left_mask": tuple(left_mask[0]),
                    "right_mask": tuple(right_mask[0]),
                }
            )
        trace = PolicyDecisionTrace(
            trace_id=trace_id,
            rollout_step=rollout_step,
            decision=decision,
            target_selection_present=True,
            action_construction_present=action_present,
            trainable_choice_count=trainable_count,
            payload=payload,
        )
        return PolicySample(
            decision=decision,
            trace=trace,
            metrics={"sample_logp": logp},
        )

    def _sample_mask(
        self,
        *,
        state_snapshot: dict[str, Any],
        action_space_snapshot: dict[str, Any],
        candidate_index: int,
        side: str,
        term_count: int,
        rng: np.random.Generator,
    ) -> tuple[tuple[bool, ...], float]:
        mask: list[bool] = []
        total_logp = 0.0
        for _term_index in range(term_count):
            logits = self._mask_logits(
                state_snapshot=state_snapshot,
                action_space_snapshot=action_space_snapshot,
                candidate_index=candidate_index,
                side=side,
                previous_mask=tuple(mask),
                term_count=term_count,
            )[len(mask)]
            choice, logp = self._sample_index(logits, rng)
            keep = choice == 1
            mask.append(keep)
            total_logp += logp
        return tuple(mask), total_logp

    def _chosen_logp(self, logits: jax.Array, chosen_index: int) -> jax.Array:
        log_probs = jax.nn.log_softmax(logits)
        return log_probs[chosen_index]

    def score_traces(
        self,
        traces: tuple[PolicyDecisionTrace, ...],
    ) -> PolicyScoreBatch:
        terms: list[PolicyLogpTerm] = []
        for trace in traces:
            payload = trace.payload
            state_snapshot = payload["state_snapshot"]
            legal_def_indices = tuple(payload["legal_def_indices"])
            target_logits, _labels = self._target_logits(state_snapshot, legal_def_indices)
            terms.append(
                PolicyLogpTerm(
                    trace_id=trace.trace_id,
                    rollout_step=trace.rollout_step,
                    kind="target_selection",
                    logp=self._chosen_logp(target_logits, int(payload["target_index"])),
                    role="target",
                )
            )
            if trace.decision.stopped:
                continue
            action_space_snapshot = payload["action_space_snapshot"]
            candidate_index = int(payload["candidate_index"])
            candidate_logits = self._candidate_logits(state_snapshot, action_space_snapshot)
            terms.append(
                PolicyLogpTerm(
                    trace_id=trace.trace_id,
                    rollout_step=trace.rollout_step,
                    kind="action_construction",
                    logp=self._chosen_logp(candidate_logits, candidate_index),
                    role="candidate",
                )
            )
            for side, mask in (
                ("left", tuple(payload["left_mask"])),
                ("right", tuple(payload["right_mask"])),
            ):
                term_count = len(mask)
                side_logp = jnp.asarray(0.0, dtype=jnp.float32)
                previous: list[bool] = []
                for keep in mask:
                    logits = self._mask_logits(
                        state_snapshot=state_snapshot,
                        action_space_snapshot=action_space_snapshot,
                        candidate_index=candidate_index,
                        side=side,
                        previous_mask=tuple(previous),
                        term_count=term_count,
                    )[len(previous)]
                    side_logp = side_logp + self._chosen_logp(logits, 1 if keep else 0)
                    previous.append(bool(keep))
                terms.append(
                    PolicyLogpTerm(
                        trace_id=trace.trace_id,
                        rollout_step=trace.rollout_step,
                        kind="action_construction",
                        logp=side_logp,
                        role=f"{side}_mask",
                    )
                )
        return PolicyScoreBatch(
            terms=tuple(terms),
            diagnostics={"term_count": len(terms)},
        )
```

Replace `python/transformer_policy/policy.py`:

```python
from __future__ import annotations

from transformer_policy.attention_policy import AttentionPolicyConfig, RewriteAttentionPolicy

TransformerPolicy = RewriteAttentionPolicy

__all__ = ("AttentionPolicyConfig", "RewriteAttentionPolicy", "TransformerPolicy")
```

- [ ] **Step 4: Run attention policy tests**

From `python/`, run:

```bash
uv run pytest tests/test_transformer_policy_attention_policy.py tests/test_transformer_policy_policy.py tests/test_transformer_policy_api.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit attention policy**

```bash
git add python/transformer_policy/attention_policy.py python/transformer_policy/policy.py python/tests/test_transformer_policy_attention_policy.py python/tests/test_transformer_policy_policy.py
git commit -m "feat: add attention rewrite policy"
```

---

### Task 3: Replace REINFORCE Episode Traces With Semantic Traces

**Files:**
- Replace: `python/reinforce_training/trace.py`
- Replace: `python/tests/test_reinforce_trace.py`
- Modify: `python/reinforce_training/__init__.py`

- [ ] **Step 1: Write failing semantic trace tests**

Replace `python/tests/test_reinforce_trace.py`:

```python
import pytest

from reinforce_training.trace import EpisodeTrace, StepTrace
from transformer_policy.api import (
    ActionConstruction,
    PolicyDecisionTrace,
    RewriteDecision,
)


def _stop_trace() -> PolicyDecisionTrace:
    decision = RewriteDecision.stop()
    return PolicyDecisionTrace(
        trace_id=0,
        rollout_step=0,
        decision=decision,
        target_selection_present=True,
        action_construction_present=False,
        trainable_choice_count=1,
        payload={"state_snapshot": {"definitions": []}, "legal_def_indices": (), "target_index": 0},
    )


def _rewrite_decision() -> RewriteDecision:
    return RewriteDecision.rewrite(
        def_index=0,
        action_construction=ActionConstruction(
            candidate_index=0,
            left_mask=(True,),
            right_mask=(True, True),
        ),
    )


def _rewrite_trace() -> PolicyDecisionTrace:
    decision = _rewrite_decision()
    return PolicyDecisionTrace(
        trace_id=1,
        rollout_step=0,
        decision=decision,
        target_selection_present=True,
        action_construction_present=True,
        trainable_choice_count=4,
        payload={"private": "policy-owned"},
    )


def test_step_trace_stores_semantic_decision_and_policy_trace():
    step = StepTrace(
        step_index=0,
        state_snapshot={"definitions": []},
        decision=RewriteDecision.stop(),
        policy_trace=_stop_trace(),
        sample_metrics={"sample_logp": -0.5},
    )

    assert step.stopped
    assert step.policy_trace.trace_id == 0
    assert not hasattr(step, "token_events")
    assert not hasattr(step, "action_space")


def test_step_trace_requires_matching_policy_trace_decision():
    with pytest.raises(ValueError, match="policy_trace decision must match step decision"):
        StepTrace(
            step_index=0,
            state_snapshot={},
            decision=RewriteDecision.stop(),
            policy_trace=_rewrite_trace(),
            sample_metrics={},
        )


def test_step_trace_copies_mutable_inputs():
    state_snapshot = {"definitions": [{"terms": [0]}]}
    step = StepTrace(
        step_index=0,
        state_snapshot=state_snapshot,
        decision=_rewrite_decision(),
        policy_trace=_rewrite_trace(),
        sample_metrics={"sample_logp": -1.0},
    )
    state_snapshot["definitions"][0]["terms"].append(1)

    assert step.state_snapshot["definitions"][0]["terms"] == [0]


def test_episode_trace_validates_terminal_reason():
    episode = EpisodeTrace(
        episode_index=0,
        episode_seed=10,
        steps=(
            StepTrace(
                step_index=0,
                state_snapshot={},
                decision=RewriteDecision.stop(),
                policy_trace=_stop_trace(),
                sample_metrics={},
            ),
        ),
        final_snapshot={},
        final_log_flops=7.0,
        reward=-7.0,
        terminal_reason="stop",
    )

    assert episode.reward == -episode.final_log_flops

    with pytest.raises(ValueError, match="stop episode requires a final stopped step"):
        EpisodeTrace(
            episode_index=0,
            episode_seed=10,
            steps=(
                StepTrace(
                    step_index=0,
                    state_snapshot={},
                    decision=_rewrite_decision(),
                    policy_trace=_rewrite_trace(),
                    sample_metrics={},
                ),
            ),
            final_snapshot={},
            final_log_flops=1.0,
            reward=-1.0,
            terminal_reason="stop",
        )
```

- [ ] **Step 2: Run trace tests to verify they fail**

From `python/`, run:

```bash
uv run pytest tests/test_reinforce_trace.py -q
```

Expected: FAIL because `StepTrace` still requires token-event fields.

- [ ] **Step 3: Implement semantic trace records**

Replace `python/reinforce_training/trace.py`:

```python
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from transformer_policy.api import PolicyDecisionTrace, RewriteDecision


TerminalReason = Literal["stop", "max_steps"]


@dataclass(frozen=True)
class StepTrace:
    step_index: int
    state_snapshot: dict[str, Any]
    decision: RewriteDecision
    policy_trace: PolicyDecisionTrace
    sample_metrics: dict[str, float | int | bool]

    def __post_init__(self) -> None:
        if type(self.step_index) is not int or self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        if self.policy_trace.rollout_step != self.step_index:
            raise ValueError("policy_trace rollout_step must match step_index")
        if self.policy_trace.decision != self.decision:
            raise ValueError("policy_trace decision must match step decision")
        metrics = dict(self.sample_metrics)
        for key, value in metrics.items():
            if not isinstance(key, str):
                raise ValueError("sample metric keys must be strings")
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError("sample metric values must be finite")
        object.__setattr__(self, "state_snapshot", deepcopy(self.state_snapshot))
        object.__setattr__(self, "sample_metrics", metrics)

    @property
    def stopped(self) -> bool:
        return self.decision.stopped


@dataclass(frozen=True)
class EpisodeTrace:
    episode_index: int
    episode_seed: int
    steps: tuple[StepTrace, ...]
    final_snapshot: dict[str, Any]
    final_log_flops: float
    reward: float
    terminal_reason: TerminalReason

    def __post_init__(self) -> None:
        if type(self.episode_index) is not int or self.episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        if type(self.episode_seed) is not int:
            raise ValueError("episode_seed must be an integer")
        if self.terminal_reason not in {"stop", "max_steps"}:
            raise ValueError("terminal_reason must be 'stop' or 'max_steps'")
        steps = tuple(self.steps)
        if not steps:
            raise ValueError("episode trace must contain at least one step")
        final_log_flops = float(self.final_log_flops)
        reward = float(self.reward)
        if not np.isfinite(final_log_flops):
            raise ValueError("final_log_flops must be finite")
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        if self.terminal_reason == "stop" and not steps[-1].stopped:
            raise ValueError("stop episode requires a final stopped step")
        if self.terminal_reason == "max_steps" and steps[-1].stopped:
            raise ValueError("max_steps episode must not end with stop")
        object.__setattr__(self, "steps", deepcopy(steps))
        object.__setattr__(self, "final_snapshot", deepcopy(self.final_snapshot))
        object.__setattr__(self, "final_log_flops", final_log_flops)
        object.__setattr__(self, "reward", reward)
```

Modify `python/reinforce_training/__init__.py`:

```python
"""REINFORCE training for semantic rewrite policies."""

from reinforce_training.trace import EpisodeTrace, StepTrace

__all__ = (
    "EpisodeTrace",
    "StepTrace",
)
```

- [ ] **Step 4: Run trace tests**

From `python/`, run:

```bash
uv run pytest tests/test_reinforce_trace.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit semantic trace records**

```bash
git add python/reinforce_training/trace.py python/reinforce_training/__init__.py python/tests/test_reinforce_trace.py
git commit -m "refactor: use semantic reinforce traces"
```

---

### Task 4: Replace Rollout With Policy API Sampling

**Files:**
- Replace: `python/reinforce_training/rollout.py`
- Replace: `python/tests/test_reinforce_rollout.py`

- [ ] **Step 1: Write failing semantic rollout tests**

Replace `python/tests/test_reinforce_rollout.py`:

```python
import pytest

from reinforce_training.rollout import (
    PolicyConfig,
    RolloutConfig,
    collect_episode_batch,
    sample_episode,
)

from .transformer_policy_fixtures import actionable_json


def test_sample_episode_returns_semantic_trace():
    policy = PolicyConfig(hidden_dim=16, num_heads=4, num_layers=1, mlp_dim=32).create_policy(seed=0)

    episode = sample_episode(
        input_json=actionable_json(),
        policy=policy,
        config=RolloutConfig(max_steps=1),
        episode_index=0,
        episode_seed=0,
    )

    assert episode.episode_index == 0
    assert episode.episode_seed == 0
    assert episode.reward == -episode.final_log_flops
    assert episode.terminal_reason in {"stop", "max_steps"}
    assert len(episode.steps) >= 1
    assert episode.steps[0].policy_trace is not None
    assert not hasattr(episode.steps[0], "token_events")


def test_sample_episode_applies_rewrite_decision_when_policy_does_not_stop():
    policy = PolicyConfig(hidden_dim=16, num_heads=4, num_layers=1, mlp_dim=32).create_policy(seed=3)

    episode = sample_episode(
        input_json=actionable_json(),
        policy=policy,
        config=RolloutConfig(max_steps=1),
        episode_index=0,
        episode_seed=3,
    )

    step = episode.steps[0]
    if not step.decision.stopped:
        assert step.decision.to_rust_decision() is not None
        assert episode.terminal_reason == "max_steps"


def test_collect_episode_batch_returns_sorted_full_batch_with_workers():
    config = PolicyConfig(hidden_dim=16, num_heads=4, num_layers=1, mlp_dim=32)
    policy = config.create_policy(seed=0)

    episodes = collect_episode_batch(
        input_json=actionable_json(),
        policy=policy,
        policy_config=config,
        rollout_config=RolloutConfig(max_steps=1),
        update_index=0,
        batch_size=2,
        num_workers=2,
        seed=10,
    )

    assert [episode.episode_index for episode in episodes] == [0, 1]
    assert [episode.episode_seed for episode in episodes] == [10, 11]
    assert all(not hasattr(step, "action_space") for episode in episodes for step in episode.steps)


def test_collect_episode_batch_uses_deterministic_update_offset_seeds():
    config = PolicyConfig(hidden_dim=16, num_heads=4, num_layers=1, mlp_dim=32)
    policy = config.create_policy(seed=0)

    episodes = collect_episode_batch(
        input_json=actionable_json(),
        policy=policy,
        policy_config=config,
        rollout_config=RolloutConfig(max_steps=1),
        update_index=3,
        batch_size=2,
        num_workers=1,
        seed=10,
    )

    assert [episode.episode_seed for episode in episodes] == [16, 17]


def test_rollout_validates_arguments():
    with pytest.raises(ValueError, match="hidden_dim must be positive integer"):
        PolicyConfig(hidden_dim=0)

    with pytest.raises(ValueError, match="max_steps must be positive"):
        RolloutConfig(max_steps=0)
```

- [ ] **Step 2: Run rollout tests to verify they fail**

From `python/`, run:

```bash
uv run pytest tests/test_reinforce_rollout.py -q
```

Expected: FAIL because `PolicyConfig.create_policy` and semantic rollout are not implemented.

- [ ] **Step 3: Implement semantic rollout**

Replace `python/reinforce_training/rollout.py`:

```python
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
from flax import nnx

from gristmill_symbolics import RewriteState, TensorComputation
from reinforce_training.trace import EpisodeTrace, StepTrace
from transformer_policy.attention_policy import AttentionPolicyConfig, RewriteAttentionPolicy


def _validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be positive integer")


@dataclass(frozen=True)
class PolicyConfig:
    hidden_dim: int = 32
    num_heads: int = 4
    num_layers: int = 1
    mlp_dim: int = 64

    def __post_init__(self) -> None:
        _validate_positive_integer("hidden_dim", self.hidden_dim)
        _validate_positive_integer("num_heads", self.num_heads)
        _validate_positive_integer("num_layers", self.num_layers)
        _validate_positive_integer("mlp_dim", self.mlp_dim)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

    def attention_config(self) -> AttentionPolicyConfig:
        return AttentionPolicyConfig(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            mlp_dim=self.mlp_dim,
        )

    def create_policy(self, *, seed: int) -> RewriteAttentionPolicy:
        return RewriteAttentionPolicy(config=self.attention_config(), rngs=nnx.Rngs(seed))


@dataclass(frozen=True)
class RolloutConfig:
    max_steps: int = 4

    def __post_init__(self) -> None:
        _validate_positive_integer("max_steps", self.max_steps)


def _model_state(policy) -> Any:
    return nnx.state(policy, nnx.Param)


def _restore_policy(policy_config: PolicyConfig, state: Any) -> RewriteAttentionPolicy:
    policy = policy_config.create_policy(seed=0)
    nnx.update(policy, state)
    return policy


def sample_episode(
    *,
    input_json: str,
    policy: RewriteAttentionPolicy,
    config: RolloutConfig,
    episode_index: int,
    episode_seed: int,
) -> EpisodeTrace:
    comp = TensorComputation.from_json_string(input_json)
    state = RewriteState.from_computation(comp)
    rng = np.random.default_rng(episode_seed)
    steps: list[StepTrace] = []
    terminal_reason = "max_steps"

    for step_index in range(config.max_steps):
        state_snapshot = state.snapshot()
        sample = policy.sample_decision(
            state,
            rng,
            trace_id=episode_index * config.max_steps + step_index,
            rollout_step=step_index,
            mode="reinforce",
        )
        steps.append(
            StepTrace(
                step_index=step_index,
                state_snapshot=state_snapshot,
                decision=sample.decision,
                policy_trace=sample.trace,
                sample_metrics=sample.metrics,
            )
        )
        if sample.decision.stopped:
            terminal_reason = "stop"
            break
        def_index = sample.decision.def_index
        rust_decision = sample.decision.to_rust_decision()
        if def_index is None or rust_decision is None:
            raise ValueError("rewrite decision must contain def_index and rust decision")
        action_space = state.action_space_for_def(def_index)
        if action_space is None:
            raise ValueError("rewrite decision selected unavailable definition")
        state.step_with_space(action_space, rust_decision)

    final_log_flops = float(state.log_total_flops())
    return EpisodeTrace(
        episode_index=episode_index,
        episode_seed=episode_seed,
        steps=tuple(steps),
        final_snapshot=state.snapshot(),
        final_log_flops=final_log_flops,
        reward=-final_log_flops,
        terminal_reason=terminal_reason,
    )


def _episode_job(
    *,
    input_json: str,
    policy_config: PolicyConfig,
    model_state: Any,
    rollout_config: RolloutConfig,
    episode_index: int,
    episode_seed: int,
) -> EpisodeTrace:
    try:
        policy = _restore_policy(policy_config, model_state)
        return sample_episode(
            input_json=input_json,
            policy=policy,
            config=rollout_config,
            episode_index=episode_index,
            episode_seed=episode_seed,
        )
    except Exception as exc:
        raise RuntimeError(
            "rollout episode failed "
            f"episode_index={episode_index} episode_seed={episode_seed}"
        ) from exc


def _validate_nonnegative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be non-negative")


def collect_episode_batch(
    *,
    input_json: str,
    policy: RewriteAttentionPolicy,
    policy_config: PolicyConfig,
    rollout_config: RolloutConfig,
    update_index: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[EpisodeTrace, ...]:
    _validate_positive_integer("batch_size", batch_size)
    _validate_positive_integer("num_workers", num_workers)
    _validate_nonnegative_integer("update_index", update_index)
    model_state = _model_state(policy)

    def episode_seed(index: int) -> int:
        return int(seed + update_index * batch_size + index)

    if num_workers == 1:
        episodes = [
            _episode_job(
                input_json=input_json,
                policy_config=policy_config,
                model_state=model_state,
                rollout_config=rollout_config,
                episode_index=index,
                episode_seed=episode_seed(index),
            )
            for index in range(batch_size)
        ]
    else:
        episodes = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _episode_job,
                    input_json=input_json,
                    policy_config=policy_config,
                    model_state=model_state,
                    rollout_config=rollout_config,
                    episode_index=index,
                    episode_seed=episode_seed(index),
                )
                for index in range(batch_size)
            ]
            for future in as_completed(futures):
                episodes.append(future.result())

    return tuple(sorted(episodes, key=lambda episode: episode.episode_index))
```

- [ ] **Step 4: Run rollout tests**

From `python/`, run:

```bash
uv run pytest tests/test_reinforce_rollout.py tests/test_reinforce_trace.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit semantic rollout**

```bash
git add python/reinforce_training/rollout.py python/tests/test_reinforce_rollout.py
git commit -m "refactor: sample reinforce rollouts through policy API"
```

---

### Task 5: Replace REINFORCE Objective With Policy Logp Terms

**Files:**
- Replace: `python/reinforce_training/objective.py`
- Replace: `python/tests/test_reinforce_objective.py`

- [ ] **Step 1: Write failing objective tests**

Replace `python/tests/test_reinforce_objective.py`:

```python
import numpy as np
import pytest
from flax import nnx

from reinforce_training.objective import (
    TrainConfig,
    create_optimizer,
    rewards_and_advantages,
    train_step,
)
from reinforce_training.rollout import PolicyConfig, RolloutConfig, collect_episode_batch

from .transformer_policy_fixtures import actionable_json


def _policy():
    return PolicyConfig(hidden_dim=16, num_heads=4, num_layers=1, mlp_dim=32).create_policy(seed=0)


def _episodes(policy):
    return collect_episode_batch(
        input_json=actionable_json(),
        policy=policy,
        policy_config=PolicyConfig(hidden_dim=16, num_heads=4, num_layers=1, mlp_dim=32),
        rollout_config=RolloutConfig(max_steps=1),
        update_index=0,
        batch_size=2,
        num_workers=1,
        seed=0,
    )


def test_rewards_and_advantages_use_float64_negative_final_log_flops():
    rewards, advantages = rewards_and_advantages(np.asarray([2.0, 4.0], dtype=np.float64))

    assert rewards.dtype == np.float64
    assert advantages.dtype == np.float64
    np.testing.assert_allclose(rewards, [-2.0, -4.0])
    np.testing.assert_allclose(advantages, [1.0, -1.0])


def test_train_step_scores_policy_traces_in_chunks_and_updates_params():
    policy = _policy()
    optimizer = create_optimizer(policy, TrainConfig(learning_rate=1e-2))
    episodes = _episodes(policy)
    final_log_flops = np.asarray([episode.final_log_flops for episode in episodes], dtype=np.float64)
    _rewards, advantages = rewards_and_advantages(final_log_flops)

    metrics = train_step(
        policy,
        optimizer=optimizer,
        episodes=episodes,
        advantages=advantages,
        score_chunk_size=1,
    )

    assert np.isfinite(metrics["loss"])
    assert metrics["params_changed"] in {True, False}
    assert metrics["score_chunk_count"] >= 2
    assert metrics["target_selection_logp_count"] >= 1
    assert "mean_action_construction_logp" in metrics


def test_train_step_rejects_invalid_score_chunk_size():
    policy = _policy()
    optimizer = create_optimizer(policy, TrainConfig(learning_rate=1e-2))
    episodes = _episodes(policy)
    _rewards, advantages = rewards_and_advantages(
        np.asarray([episode.final_log_flops for episode in episodes], dtype=np.float64)
    )

    with pytest.raises(ValueError, match="score_chunk_size must be positive"):
        train_step(
            policy,
            optimizer=optimizer,
            episodes=episodes,
            advantages=advantages,
            score_chunk_size=0,
        )


def test_train_step_rejects_advantage_length_mismatch():
    policy = _policy()
    optimizer = create_optimizer(policy, TrainConfig(learning_rate=1e-2))
    episodes = _episodes(policy)

    with pytest.raises(ValueError, match="advantages length must match episodes"):
        train_step(
            policy,
            optimizer=optimizer,
            episodes=episodes,
            advantages=np.asarray([1.0], dtype=np.float64),
            score_chunk_size=1,
        )
```

- [ ] **Step 2: Run objective tests to verify they fail**

From `python/`, run:

```bash
uv run pytest tests/test_reinforce_objective.py -q
```

Expected: FAIL because `train_step` still expects a padded token batch.

- [ ] **Step 3: Implement logp-term objective**

Replace `python/reinforce_training/objective.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from reinforce_training.trace import EpisodeTrace
from transformer_policy.api import PolicyDecisionTrace, PolicyLogpTerm


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 1e-3


def create_optimizer(policy, config: TrainConfig) -> nnx.Optimizer:
    if config.learning_rate <= 0.0 or not np.isfinite(config.learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    return nnx.Optimizer(policy, optax.adam(config.learning_rate), wrt=nnx.Param)


def rewards_and_advantages(final_log_flops: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(final_log_flops, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("final_log_flops must be a nonempty 1-D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("final_log_flops must be finite")
    rewards = -values
    advantages = rewards - np.mean(rewards, dtype=np.float64)
    return rewards.astype(np.float64), advantages.astype(np.float64)


def _episode_traces(episodes: tuple[EpisodeTrace, ...]) -> tuple[PolicyDecisionTrace, ...]:
    traces = []
    for episode in episodes:
        for step in episode.steps:
            traces.append(step.policy_trace)
    if not traces:
        raise ValueError("episodes must contain at least one policy trace")
    return tuple(traces)


def _trace_episode_advantages(
    episodes: tuple[EpisodeTrace, ...],
    advantages: np.ndarray,
) -> dict[int, float]:
    values = np.asarray(advantages, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(episodes):
        raise ValueError("advantages length must match episodes")
    if not np.all(np.isfinite(values)):
        raise ValueError("advantages must be finite")
    mapping: dict[int, float] = {}
    for episode, advantage in zip(episodes, values, strict=True):
        for step in episode.steps:
            mapping[step.policy_trace.trace_id] = float(advantage)
    return mapping


def _chunks(values: tuple[PolicyDecisionTrace, ...], size: int):
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("score_chunk_size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _flat_param_values(module) -> list[np.ndarray]:
    state = nnx.state(module, nnx.Param)
    values = []
    for leaf in jax.tree_util.tree_leaves(state):
        value = getattr(leaf, "value", leaf)
        values.append(np.asarray(value).copy())
    return values


def _tree_all_finite(tree) -> bool:
    for leaf in jax.tree_util.tree_leaves(tree):
        value = getattr(leaf, "value", leaf)
        if not np.all(np.isfinite(np.asarray(value))):
            return False
    return True


def _loss_from_terms(
    terms: tuple[PolicyLogpTerm, ...],
    *,
    advantage_by_trace_id: dict[int, float],
) -> tuple[jax.Array, dict[str, jax.Array]]:
    weighted = []
    target_terms = []
    action_terms = []
    for term in terms:
        advantage = jax.lax.stop_gradient(
            jnp.asarray(advantage_by_trace_id[term.trace_id], dtype=jnp.float32)
        )
        value = advantage * term.logp
        weighted.append(value)
        if term.kind == "target_selection":
            target_terms.append(term.logp)
        else:
            action_terms.append(term.logp)
    if not weighted:
        raise ValueError("logp terms must not be empty")
    loss = -jnp.mean(jnp.stack(weighted))
    return loss, {
        "mean_target_selection_logp": (
            jnp.mean(jnp.stack(target_terms)) if target_terms else jnp.asarray(0.0)
        ),
        "mean_action_construction_logp": (
            jnp.mean(jnp.stack(action_terms)) if action_terms else jnp.asarray(0.0)
        ),
        "target_selection_logp_count": jnp.asarray(len(target_terms)),
        "action_construction_logp_count": jnp.asarray(len(action_terms)),
    }


def _reinforce_loss_for_grad(
    policy,
    traces: tuple[PolicyDecisionTrace, ...],
    advantage_by_trace_id: dict[int, float],
    score_chunk_size: int,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    losses = []
    target_counts = []
    action_counts = []
    mean_targets = []
    mean_actions = []
    for chunk in _chunks(traces, score_chunk_size):
        scored = policy.score_traces(tuple(chunk))
        loss, aux = _loss_from_terms(
            scored.terms,
            advantage_by_trace_id=advantage_by_trace_id,
        )
        losses.append(loss)
        target_counts.append(aux["target_selection_logp_count"])
        action_counts.append(aux["action_construction_logp_count"])
        mean_targets.append(aux["mean_target_selection_logp"])
        mean_actions.append(aux["mean_action_construction_logp"])
    total_loss = jnp.mean(jnp.stack(losses))
    return total_loss, {
        "loss": total_loss,
        "mean_target_selection_logp": jnp.mean(jnp.stack(mean_targets)),
        "mean_action_construction_logp": jnp.mean(jnp.stack(mean_actions)),
        "target_selection_logp_count": jnp.sum(jnp.stack(target_counts)),
        "action_construction_logp_count": jnp.sum(jnp.stack(action_counts)),
        "score_chunk_count": jnp.asarray(len(losses)),
    }


def train_step(
    policy,
    *,
    optimizer: nnx.Optimizer,
    episodes: tuple[EpisodeTrace, ...],
    advantages: np.ndarray,
    score_chunk_size: int,
) -> dict[str, float | int | bool]:
    traces = _episode_traces(tuple(episodes))
    advantage_by_trace_id = _trace_episode_advantages(tuple(episodes), advantages)
    before = _flat_param_values(policy)
    grad_fn = nnx.value_and_grad(_reinforce_loss_for_grad, has_aux=True)
    (loss, aux), grads = grad_fn(
        policy,
        traces,
        advantage_by_trace_id,
        score_chunk_size,
    )
    if not np.isfinite(float(loss)):
        raise ValueError("loss must be finite before optimizer update")
    if not _tree_all_finite(grads):
        raise ValueError("gradients must be finite before optimizer update")
    optimizer.update(policy, grads)
    after = _flat_param_values(policy)
    params_changed = any(
        not np.array_equal(left, right)
        for left, right in zip(before, after, strict=True)
    )
    return {
        "loss": float(loss),
        "mean_target_selection_logp": float(aux["mean_target_selection_logp"]),
        "mean_action_construction_logp": float(aux["mean_action_construction_logp"]),
        "target_selection_logp_count": int(aux["target_selection_logp_count"]),
        "action_construction_logp_count": int(aux["action_construction_logp_count"]),
        "score_chunk_count": int(aux["score_chunk_count"]),
        "params_changed": params_changed,
    }
```

- [ ] **Step 4: Run objective tests**

From `python/`, run:

```bash
uv run pytest tests/test_reinforce_objective.py tests/test_reinforce_rollout.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit logp-term objective**

```bash
git add python/reinforce_training/objective.py python/tests/test_reinforce_objective.py
git commit -m "refactor: train reinforce from policy logp terms"
```

---

### Task 6: Update CLI For Chunked Semantic Training

**Files:**
- Modify: `python/reinforce_training/train.py`
- Modify: `python/tests/test_reinforce_train.py`

- [ ] **Step 1: Write failing CLI tests**

Modify `python/tests/test_reinforce_train.py`:

```python
import json
import math
import subprocess
import sys

from .transformer_policy_fixtures import actionable_json


def _tiny_train_command(input_path, *extra_args):
    return [
        sys.executable,
        "-m",
        "reinforce_training.train",
        "--input",
        str(input_path),
        "--updates",
        "1",
        "--batch-size",
        "2",
        "--max-steps",
        "1",
        "--num-workers",
        "1",
        "--score-chunk-size",
        "1",
        "--hidden-dim",
        "16",
        "--num-heads",
        "4",
        "--num-layers",
        "1",
        "--mlp-dim",
        "32",
        "--seed",
        "0",
        *extra_args,
    ]


def _json_stdout_lines(result):
    return [
        json.loads(line)
        for line in result.stdout.strip().splitlines()
        if line.strip().startswith("{")
    ]


def test_reinforce_train_cli_completes_tiny_run(tmp_path):
    input_path = tmp_path / "input.json"
    input_path.write_text(actionable_json())

    result = subprocess.run(
        _tiny_train_command(input_path),
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(result.stdout.strip().splitlines()[-1])
    assert metrics["updates"] == 1
    assert metrics["batch_size"] == 2
    assert metrics["num_workers"] == 1
    assert metrics["score_chunk_size"] == 1
    assert metrics["score_chunk_count"] >= 2
    assert isinstance(metrics["params_changed"], bool)
    assert math.isfinite(metrics["loss"])
    assert math.isfinite(metrics["mean_reward"])
    assert math.isfinite(metrics["mean_final_log_flops"])
    assert math.isfinite(metrics["mean_target_selection_logp"])
    assert "mean_sample_log_prob" not in metrics
    assert "mean_trajectory_log_prob" not in metrics
    assert metrics["checkpoint_out"] is None


def test_reinforce_train_cli_writes_checkpoint(tmp_path):
    input_path = tmp_path / "input.json"
    checkpoint_path = tmp_path / "checkpoint"
    input_path.write_text(actionable_json())

    result = subprocess.run(
        _tiny_train_command(input_path, "--checkpoint-out", str(checkpoint_path)),
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(result.stdout.strip().splitlines()[-1])
    assert metrics["checkpoint_out"] == str(checkpoint_path)
    assert (checkpoint_path / "metadata.json").exists()
    assert (checkpoint_path / "state").exists()


def test_reinforce_train_cli_existing_checkpoint_fails_without_misleading_metrics(tmp_path):
    input_path = tmp_path / "input.json"
    checkpoint_path = tmp_path / "checkpoint"
    marker_path = checkpoint_path / "marker.txt"
    input_path.write_text(actionable_json())
    checkpoint_path.mkdir()
    marker_path.write_text("unchanged")

    result = subprocess.run(
        _tiny_train_command(input_path, "--checkpoint-out", str(checkpoint_path)),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "checkpoint path already exists" in result.stderr
    assert all(
        metrics.get("checkpoint_out") != str(checkpoint_path)
        for metrics in _json_stdout_lines(result)
    )
    assert marker_path.read_text() == "unchanged"
```

- [ ] **Step 2: Run CLI tests to verify they fail**

From `python/`, run:

```bash
uv run pytest tests/test_reinforce_train.py -q
```

Expected: FAIL because `--score-chunk-size` is not parsed and train still pads token events.

- [ ] **Step 3: Update CLI training flow**

Modify `python/reinforce_training/train.py`:

- Remove import of `pad_token_choice_events`.
- Add parser argument:

```python
parser.add_argument("--score-chunk-size", type=_positive_int, default=32)
```

- Replace `_episode_events` with:

```python
def _mean_sample_logp(episodes) -> float:
    values = [
        float(step.sample_metrics["sample_logp"])
        for episode in episodes
        for step in episode.steps
        if "sample_logp" in step.sample_metrics
    ]
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))
```

- In `_run_configs`, rename local `scorer` to `policy` and use `policy_config.create_policy(seed=seed)`.
- In `run`, call `collect_episode_batch(..., policy=policy, ...)`.
- Remove `events, episode_ids = _episode_events(episodes)` and `pad_token_choice_events`.
- Call:

```python
train_metrics = train_step(
    policy,
    optimizer=optimizer,
    episodes=episodes,
    advantages=advantages,
    score_chunk_size=args.score_chunk_size,
)
```

- In `_update_metrics`, add:

```python
"score_chunk_size": score_chunk_size,
"score_chunk_count": int(train_metrics["score_chunk_count"]),
"mean_target_selection_logp": float(train_metrics["mean_target_selection_logp"]),
"mean_action_construction_logp": float(train_metrics["mean_action_construction_logp"]),
"target_selection_logp_count": int(train_metrics["target_selection_logp_count"]),
"action_construction_logp_count": int(train_metrics["action_construction_logp_count"]),
"mean_sample_logp": _mean_sample_logp(episodes),
```

- Remove old `"mean_sample_log_prob"` and `"mean_trajectory_log_prob"` metrics.
- Pass `score_chunk_size=args.score_chunk_size` into `_update_metrics`.

- [ ] **Step 4: Run CLI tests**

From `python/`, run:

```bash
uv run pytest tests/test_reinforce_train.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit CLI update**

```bash
git add python/reinforce_training/train.py python/tests/test_reinforce_train.py
git commit -m "refactor: run reinforce CLI on semantic policy traces"
```

---

### Task 7: Update Checkpointing For New Policy Class

**Files:**
- Modify: `python/reinforce_training/checkpoint.py`
- Modify: `python/tests/test_reinforce_checkpoint.py`

- [ ] **Step 1: Update checkpoint tests**

In `python/tests/test_reinforce_checkpoint.py`, replace references to:

```python
loaded.scorer
scorer=
_MODEL_CLASS = "CausalTransformerScorer"
```

with:

```python
loaded.policy
policy=
model_class == "RewriteAttentionPolicy"
```

Update helper setup so it creates policies with:

```python
policy_config = PolicyConfig(hidden_dim=16, num_heads=4, num_layers=1, mlp_dim=32)
policy = policy_config.create_policy(seed=123)
optimizer = create_optimizer(policy, train_config)
```

- [ ] **Step 2: Run checkpoint tests to verify they fail**

From `python/`, run:

```bash
uv run pytest tests/test_reinforce_checkpoint.py -q
```

Expected: FAIL because checkpoint load/save still uses `scorer`.

- [ ] **Step 3: Update checkpoint implementation**

Modify `python/reinforce_training/checkpoint.py`:

- Set:

```python
_MODEL_CLASS = "RewriteAttentionPolicy"
```

- Change `LoadedCheckpoint.scorer` to:

```python
policy: Any
```

- Rename function parameters and metadata helpers from `scorer` to `policy`.
- In `_validate_checkpoint_state_shapes`, create expected policy:

```python
expected_policy = policy_config.create_policy(seed=0)
```

- Validate `nnx.state(policy, nnx.Param)` against `expected_policy`.
- In `_state_payload`, store:

```python
{"policy": nnx.state(policy), "optimizer": nnx.state(optimizer)}
```

- In `save_checkpoint`, require `policy=` and write policy state.
- In `load_checkpoint`, restore with:

```python
policy = policy_config.create_policy(seed=0)
nnx.update(policy, restored["policy"])
optimizer = create_optimizer(policy, train_config)
nnx.update(optimizer, restored["optimizer"])
```

- Return `LoadedCheckpoint(policy=policy, ...)`.

- [ ] **Step 4: Run checkpoint and CLI tests**

From `python/`, run:

```bash
uv run pytest tests/test_reinforce_checkpoint.py tests/test_reinforce_train.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit checkpoint migration**

```bash
git add python/reinforce_training/checkpoint.py python/tests/test_reinforce_checkpoint.py
git commit -m "refactor: checkpoint attention rewrite policies"
```

---

### Task 8: Remove Deprecated Token-Decoder Training Surfaces

**Files:**
- Delete: `python/transformer_policy/decoder.py`
- Delete: `python/transformer_policy/trace.py`
- Delete: `python/transformer_policy/batch.py`
- Delete: `python/transformer_policy/sequence_model.py`
- Delete: `python/tests/test_transformer_policy_decoder.py`
- Delete: `python/tests/test_transformer_policy_trace.py`
- Delete: `python/tests/test_transformer_policy_batch.py`
- Delete: `python/tests/test_transformer_policy_sequence_model.py`
- Modify: any imports found by `rg`

- [ ] **Step 1: Remove deprecated files**

From repo root, run:

```bash
git rm python/transformer_policy/decoder.py \
  python/transformer_policy/trace.py \
  python/transformer_policy/batch.py \
  python/transformer_policy/sequence_model.py \
  python/tests/test_transformer_policy_decoder.py \
  python/tests/test_transformer_policy_trace.py \
  python/tests/test_transformer_policy_batch.py \
  python/tests/test_transformer_policy_sequence_model.py
```

Expected: files are staged as deleted.

- [ ] **Step 2: Search for stale imports**

From repo root, run:

```bash
rg -n "transformer_policy\\.(decoder|trace|batch|sequence_model)|TokenChoiceEvent|PaddedTokenChoiceBatch|sample_step_with_events|sample_step\\(|score_step\\(" python docs
```

Expected: only historical docs may match. No matches under `python/` except deleted-file paths in git status.

- [ ] **Step 3: Fix stale imports if the search finds Python matches**

If the search finds Python imports, replace them with the semantic API:

```python
from transformer_policy.api import (
    ActionConstruction,
    PolicyDecisionTrace,
    PolicyLogpTerm,
    PolicySample,
    PolicyScoreBatch,
    RewriteDecision,
    TargetSelection,
)
from transformer_policy.attention_policy import AttentionPolicyConfig, RewriteAttentionPolicy
```

Do not keep adapters that recreate token-event behavior.

- [ ] **Step 4: Run package import tests**

From `python/`, run:

```bash
uv run pytest tests/test_transformer_policy_package.py tests/test_reinforce_package.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit deprecated surface removal**

```bash
git add python
git commit -m "refactor: remove token decoder training surfaces"
```

---

### Task 9: Full Verification And Documentation Check

**Files:**
- Modify only if verification reveals stale docs or imports.

- [ ] **Step 1: Run focused Python test suite**

From `python/`, run:

```bash
uv run pytest tests/test_transformer_policy_api.py tests/test_transformer_policy_attention_policy.py tests/test_transformer_policy_policy.py tests/test_reinforce_trace.py tests/test_reinforce_rollout.py tests/test_reinforce_objective.py tests/test_reinforce_checkpoint.py tests/test_reinforce_train.py -q
```

Expected: PASS.

- [ ] **Step 2: Run all Python tests**

From `python/`, run:

```bash
uv run pytest -q
```

Expected: PASS.

- [ ] **Step 3: Run Rust tests**

From repo root, run:

```bash
cargo test
```

Expected: PASS.

- [ ] **Step 4: Verify no deprecated Python imports remain**

From repo root, run:

```bash
rg -n "transformer_policy\\.(decoder|trace|batch|sequence_model)|TokenChoiceEvent|PaddedTokenChoiceBatch|sample_step_with_events|mean_trajectory_log_prob" python
```

Expected: no output.

- [ ] **Step 5: Verify spec acceptance criteria manually**

Check these against code and tests:

```text
reinforce_training samples and executes RewriteDecision values
reinforce_training does not import decoder, trace, batch, or sequence_model
transformer_policy exposes sample_decision and score_traces
REINFORCE loss uses PolicyLogpTerm values weighted by advantages
CLI reports target-selection and action-construction diagnostics
num_workers, batch_size, and score_chunk_size are independent
warm start is not implemented
```

Expected: every line is true.

- [ ] **Step 6: Commit verification fixes if any were needed**

If Step 1-5 required file changes:

```bash
git add python docs
git commit -m "test: verify policy reinforce api refactor"
```

If no file changes were needed, do not create an empty commit.
