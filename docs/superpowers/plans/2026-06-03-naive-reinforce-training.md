# Naive REINFORCE Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone `reinforce_training` package and CLI that trains the merged `transformer_policy` with process-parallel episode sampling and a padded JAX REINFORCE objective.

**Architecture:** First extend `transformer_policy` with public traced-sampling and padded event-scoring hooks, because the current `score_step` returns a detached Python float. Then add `reinforce_training` modules for serializable traces, on-policy objectives, process-parallel rollout, checkpoints, and the end-to-end CLI. Training uses token-choice events from rollout workers and rescoring in the main process; it never imports `gristmill_rl`.

**Tech Stack:** Python 3.11, PyO3 `gristmill_symbolics`, NumPy, JAX, Flax NNX, Optax, Orbax, pytest, `uv`.

---

## File Structure

- Create `python/transformer_policy/trace.py`: public token-choice event records, traced sample records, and token/event wire serialization helpers.
- Modify `python/transformer_policy/decoder.py`: add `sample_step_with_events` while preserving existing `sample_step` behavior.
- Create `python/transformer_policy/batch.py`: pad token-choice events and compute chosen log-probs/trajectory log-probs from batched logits.
- Modify `python/transformer_policy/sequence_model.py`: add a batch-friendly feature scoring method on `CausalTransformerScorer`.
- Modify `python/transformer_policy/__init__.py`: export public trace records.
- Create `python/tests/test_transformer_policy_trace.py`.
- Create `python/tests/test_transformer_policy_batch.py`.
- Create `python/reinforce_training/__init__.py`: package exports.
- Create `python/reinforce_training/trace.py`: serializable episode and step trace records.
- Create `python/reinforce_training/objective.py`: rewards, advantages, REINFORCE loss, and one update step.
- Create `python/reinforce_training/rollout.py`: single-episode rollout and process-parallel batch collection.
- Create `python/reinforce_training/checkpoint.py`: Orbax save/load for scorer, optimizer state, config, and counters.
- Create `python/reinforce_training/train.py`: CLI and training loop.
- Modify `python/pyproject.toml`: add `reinforce_training` to `python-packages` and exclude pycache.
- Create `python/tests/test_reinforce_package.py`.
- Create `python/tests/test_reinforce_trace.py`.
- Create `python/tests/test_reinforce_objective.py`.
- Create `python/tests/test_reinforce_rollout.py`.
- Create `python/tests/test_reinforce_checkpoint.py`.
- Create `python/tests/test_reinforce_train.py`.

Before Python tests that import `gristmill_symbolics`, refresh the local extension from `python/`:

```bash
uv run maturin develop
```

Expected: the PyO3 extension builds and `import gristmill_symbolics` works in the `uv` environment.

---

### Task 1: Transformer Policy Trace Events

**Files:**
- Create: `python/transformer_policy/trace.py`
- Modify: `python/transformer_policy/decoder.py`
- Modify: `python/transformer_policy/__init__.py`
- Test: `python/tests/test_transformer_policy_trace.py`
- Re-test: `python/tests/test_transformer_policy_decoder.py`

- [ ] **Step 1: Write failing trace tests**

Create `python/tests/test_transformer_policy_trace.py`:

```python
import numpy as np
import pytest

from transformer_policy.decoder import sample_step, sample_step_with_events
from transformer_policy.trace import (
    TokenChoiceEvent,
    TracedPolicySample,
    event_from_wire,
    event_to_wire,
    token_from_wire,
    token_to_wire,
)
from transformer_policy.types import T

from .test_transformer_policy_decoder import PreferenceScorer, StopScorer, empty_state
from .transformer_policy_fixtures import actionable_state


def test_token_wire_round_trip_preserves_payload_order():
    token = T("FACTOR", position=1, tensor=3, arity=2)

    wire = token_to_wire(token)
    restored = token_from_wire(wire)

    assert wire == {
        "kind": "FACTOR",
        "payload": (("arity", 2), ("position", 1), ("tensor", 3)),
    }
    assert restored == token


def test_token_wire_rejects_invalid_shape():
    with pytest.raises(ValueError, match="token wire kind must be a string"):
        token_from_wire({"kind": 1, "payload": ()})

    with pytest.raises(ValueError, match="token wire payload must be a tuple"):
        token_from_wire({"kind": "STOP", "payload": []})


def test_event_wire_round_trip():
    event = TokenChoiceEvent(
        sequence_tokens=(T("STATE_START"), T("STATE_END")),
        legal_next_tokens=(T("STOP"), T("DEF", def_index=0)),
        chosen_index=1,
        phase="def",
        step_index=2,
    )

    restored = event_from_wire(event_to_wire(event))

    assert restored == event


def test_traced_sampling_matches_plain_sampling_for_same_seed():
    state = actionable_state()
    traced = sample_step_with_events(
        state,
        PreferenceScorer(),
        np.random.default_rng(0),
    )
    plain = sample_step(
        actionable_state(),
        PreferenceScorer(),
        np.random.default_rng(0),
    )

    assert isinstance(traced, TracedPolicySample)
    assert traced.sample.stopped == plain.stopped
    assert traced.sample.def_index == plain.def_index
    assert traced.sample.decision == plain.decision
    assert traced.sample.def_attempts == plain.def_attempts
    assert traced.sample.decision_tokens == plain.decision_tokens
    assert traced.sample.log_prob == pytest.approx(plain.log_prob)


def test_traced_sampling_records_one_event_per_sampled_decision():
    traced = sample_step_with_events(
        actionable_state(),
        PreferenceScorer(),
        np.random.default_rng(0),
    )

    phases = [event.phase for event in traced.events]

    assert phases == ["def", "candidate", "left_bit", "right_bit", "right_bit"]
    assert [event.step_index for event in traced.events] == [0, 0, 0, 0, 0]
    assert traced.events[0].legal_next_tokens[0] == T("STOP")
    assert traced.events[0].legal_next_tokens[1] == T("DEF", def_index=0)
    assert traced.events[0].chosen_index == 1
    assert traced.events[1].legal_next_tokens[traced.events[1].chosen_index] == T(
        "CAND",
        candidate_index=0,
    )


def test_stop_only_state_records_zero_log_prob_stop_event():
    traced = sample_step_with_events(empty_state(), StopScorer(), np.random.default_rng(0))

    assert traced.sample.stopped
    assert traced.sample.log_prob == pytest.approx(0.0)
    assert len(traced.events) == 1
    event = traced.events[0]
    assert event.phase == "def"
    assert event.legal_next_tokens == (T("STOP"),)
    assert event.chosen_index == 0
```

- [ ] **Step 2: Run trace tests to verify they fail**

From `python/`, run:

```bash
uv run pytest tests/test_transformer_policy_trace.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'transformer_policy.trace'` or `ImportError` for `sample_step_with_events`.

- [ ] **Step 3: Create trace records and wire helpers**

Create `python/transformer_policy/trace.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

from transformer_policy.types import Payload, PayloadValue, PolicySample, T, Token


TracePhase = Literal["def", "candidate", "left_bit", "right_bit"]


class TokenWire(TypedDict):
    kind: str
    payload: tuple[tuple[str, PayloadValue], ...]


class TokenChoiceEventWire(TypedDict):
    sequence_tokens: tuple[TokenWire, ...]
    legal_next_tokens: tuple[TokenWire, ...]
    chosen_index: int
    phase: TracePhase
    step_index: int


@dataclass(frozen=True)
class TokenChoiceEvent:
    sequence_tokens: tuple[Token, ...]
    legal_next_tokens: tuple[Token, ...]
    chosen_index: int
    phase: TracePhase
    step_index: int

    def __post_init__(self) -> None:
        if not self.sequence_tokens:
            raise ValueError("sequence_tokens must not be empty")
        if not self.legal_next_tokens:
            raise ValueError("legal_next_tokens must not be empty")
        if self.chosen_index < 0 or self.chosen_index >= len(self.legal_next_tokens):
            raise ValueError("chosen_index is outside legal_next_tokens")
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")


@dataclass(frozen=True)
class TracedPolicySample:
    sample: PolicySample
    events: tuple[TokenChoiceEvent, ...]

    def __post_init__(self) -> None:
        if not self.events:
            raise ValueError("traced sample must contain at least one token event")


def token_to_wire(token: Token) -> TokenWire:
    return {"kind": token.kind, "payload": token.payload}


def token_from_wire(wire: TokenWire) -> Token:
    kind = wire.get("kind")
    payload = wire.get("payload")
    if not isinstance(kind, str):
        raise ValueError("token wire kind must be a string")
    if not isinstance(payload, tuple):
        raise ValueError("token wire payload must be a tuple")
    normalized: list[tuple[str, PayloadValue]] = []
    for item in payload:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("token wire payload entries must be pairs")
        key, value = item
        if not isinstance(key, str):
            raise ValueError("token wire payload keys must be strings")
        if not isinstance(value, (int, float, str, bool)):
            raise ValueError("token wire payload values must be scalar")
        normalized.append((key, value))
    return Token(kind=kind, payload=tuple(normalized))


def event_to_wire(event: TokenChoiceEvent) -> TokenChoiceEventWire:
    return {
        "sequence_tokens": tuple(token_to_wire(token) for token in event.sequence_tokens),
        "legal_next_tokens": tuple(token_to_wire(token) for token in event.legal_next_tokens),
        "chosen_index": event.chosen_index,
        "phase": event.phase,
        "step_index": event.step_index,
    }


def event_from_wire(wire: TokenChoiceEventWire) -> TokenChoiceEvent:
    phase = wire.get("phase")
    if phase not in {"def", "candidate", "left_bit", "right_bit"}:
        raise ValueError("event wire phase is invalid")
    chosen_index = wire.get("chosen_index")
    step_index = wire.get("step_index")
    if type(chosen_index) is not int:
        raise ValueError("event wire chosen_index must be an int")
    if type(step_index) is not int:
        raise ValueError("event wire step_index must be an int")
    return TokenChoiceEvent(
        sequence_tokens=tuple(token_from_wire(token) for token in wire["sequence_tokens"]),
        legal_next_tokens=tuple(token_from_wire(token) for token in wire["legal_next_tokens"]),
        chosen_index=chosen_index,
        phase=phase,
        step_index=step_index,
    )
```

- [ ] **Step 4: Add traced decoder sampling**

Modify `python/transformer_policy/decoder.py`:

```python
from transformer_policy.trace import TokenChoiceEvent, TracedPolicySample
```

Add these helpers near `_sample_token`:

```python
def _sample_token_index(
    scorer: NextTokenScorer,
    context: tuple[Token, ...],
    prefix: tuple[Token, ...],
    legal: tuple[Token, ...],
    rng: np.random.Generator,
) -> tuple[int, float]:
    if not legal:
        raise ValueError("legal token set must not be empty")
    logits = _validated_logits(scorer, context, prefix, legal)
    log_probs = _log_softmax(logits)
    probs = np.exp(log_probs)
    index = int(rng.choice(len(legal), p=probs))
    return index, float(log_probs[index])


def _sample_token(
    scorer: NextTokenScorer,
    context: tuple[Token, ...],
    prefix: tuple[Token, ...],
    legal: tuple[Token, ...],
    rng: np.random.Generator,
) -> tuple[Token, float]:
    index, log_prob = _sample_token_index(scorer, context, prefix, legal, rng)
    return legal[index], log_prob
```

Replace the existing `_sample_token` definition with the version above. Then add:

```python
def _sample_token_with_event(
    *,
    scorer: NextTokenScorer,
    context: tuple[Token, ...],
    prefix: tuple[Token, ...],
    legal: tuple[Token, ...],
    rng: np.random.Generator,
    phase: str,
    step_index: int,
) -> tuple[Token, float, TokenChoiceEvent]:
    chosen_index, log_prob = _sample_token_index(scorer, context, prefix, legal, rng)
    event = TokenChoiceEvent(
        sequence_tokens=(*context, *prefix),
        legal_next_tokens=legal,
        chosen_index=chosen_index,
        phase=phase,
        step_index=step_index,
    )
    return legal[chosen_index], log_prob, event
```

Add traced bit sampling:

```python
def _sample_bits_with_events(
    *,
    scorer: NextTokenScorer,
    context: tuple[Token, ...],
    prefix: list[Token],
    kind_prefix: str,
    term_count: int,
    rng: np.random.Generator,
    step_index: int,
) -> tuple[list[bool], float, list[TokenChoiceEvent]]:
    bits: list[bool] = []
    log_prob = 0.0
    events: list[TokenChoiceEvent] = []
    kept_any = False
    phase = "left_bit" if kind_prefix == "LEFT" else "right_bit"
    for term_index in range(term_count):
        legal = _bit_legal(kind_prefix, term_index == term_count - 1, kept_any)
        token, token_log_prob, event = _sample_token_with_event(
            scorer=scorer,
            context=context,
            prefix=tuple(prefix),
            legal=legal,
            rng=rng,
            phase=phase,
            step_index=step_index,
        )
        prefix.append(token)
        keep = token.kind.endswith("KEEP")
        bits.append(keep)
        kept_any = kept_any or keep
        log_prob += token_log_prob
        events.append(event)
    return bits, log_prob, events
```

Add the public traced function and make existing `sample_step` delegate:

```python
def sample_step(state, scorer: NextTokenScorer, rng: np.random.Generator) -> PolicySample:
    return sample_step_with_events(state, scorer, rng).sample


def sample_step_with_events(
    state,
    scorer: NextTokenScorer,
    rng: np.random.Generator,
    *,
    step_index: int = 0,
) -> TracedPolicySample:
    sample_state = _fresh_replay_state(state)
    attempts: list[Stage1Attempt] = []
    events: list[TokenChoiceEvent] = []
    total_log_prob = 0.0
    while True:
        state_context = build_state_context(sample_state.snapshot())
        legal = _stage1_legal(sample_state)
        stage1_token, stage1_log_prob, event = _sample_token_with_event(
            scorer=scorer,
            context=state_context,
            prefix=(),
            legal=legal,
            rng=rng,
            phase="def",
            step_index=step_index,
        )
        events.append(event)
        total_log_prob += stage1_log_prob
        if stage1_token.kind == "STOP":
            return TracedPolicySample(
                sample=PolicySample(
                    stopped=True,
                    log_prob=total_log_prob,
                    def_attempts=tuple(attempts),
                    decision_tokens=(T("STOP"),),
                ),
                events=tuple(events),
            )
        def_index = int(stage1_token.payload_dict()["def_index"])
        space = sample_state.action_space_for_def(def_index)
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
    context = (
        *build_state_context(sample_state.snapshot()),
        *build_action_space_context(space_snapshot),
    )
    prefix: list[Token] = []
    candidate_token, candidate_log_prob, candidate_event = _sample_token_with_event(
        scorer=scorer,
        context=context,
        prefix=tuple(prefix),
        legal=_candidate_legal(space_snapshot),
        rng=rng,
        phase="candidate",
        step_index=step_index,
    )
    prefix.append(candidate_token)
    events.append(candidate_event)
    total_log_prob += candidate_log_prob
    candidate_index = int(candidate_token.payload_dict()["candidate_index"])
    candidate = space_snapshot["candidate_templates"][candidate_index]

    left_bits, left_log_prob, left_events = _sample_bits_with_events(
        scorer=scorer,
        context=context,
        prefix=prefix,
        kind_prefix="LEFT",
        term_count=len(candidate["left_definition"]["terms"]),
        rng=rng,
        step_index=step_index,
    )
    right_bits, right_log_prob, right_events = _sample_bits_with_events(
        scorer=scorer,
        context=context,
        prefix=prefix,
        kind_prefix="RIGHT",
        term_count=len(candidate["right_definition"]["terms"]),
        rng=rng,
        step_index=step_index,
    )
    prefix.append(T("END"))
    total_log_prob += left_log_prob + right_log_prob
    events.extend(left_events)
    events.extend(right_events)
    return TracedPolicySample(
        sample=PolicySample(
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
        ),
        events=tuple(events),
    )
```

- [ ] **Step 5: Export public trace types**

Modify `python/transformer_policy/__init__.py`:

```python
"""Transformer policy for symbolic tensor rewrite decisions."""

from transformer_policy.policy import TransformerPolicy
from transformer_policy.trace import TokenChoiceEvent, TracedPolicySample
from transformer_policy.types import PolicySample, Stage1Attempt, T, Token

__all__ = (
    "Token",
    "T",
    "Stage1Attempt",
    "PolicySample",
    "TransformerPolicy",
    "TokenChoiceEvent",
    "TracedPolicySample",
)
```

- [ ] **Step 6: Run trace and existing decoder tests**

From `python/`, run:

```bash
uv run pytest tests/test_transformer_policy_trace.py tests/test_transformer_policy_decoder.py tests/test_transformer_policy_package.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit traced sampling**

```bash
git add python/transformer_policy/trace.py python/transformer_policy/decoder.py python/transformer_policy/__init__.py python/tests/test_transformer_policy_trace.py
git commit -m "feat: add transformer policy trace events"
```

---

### Task 2: Padded Transformer Event Scoring

**Files:**
- Create: `python/transformer_policy/batch.py`
- Modify: `python/transformer_policy/sequence_model.py`
- Test: `python/tests/test_transformer_policy_batch.py`
- Re-test: `python/tests/test_transformer_policy_sequence_model.py`

- [ ] **Step 1: Write failing batch tests**

Create `python/tests/test_transformer_policy_batch.py`:

```python
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from transformer_policy.batch import (
    chosen_event_log_probs,
    pad_token_choice_events,
    score_event_batch,
    trajectory_log_probs,
)
from transformer_policy.decoder import sample_step_with_events
from transformer_policy.sequence_model import CausalTransformerScorer
from transformer_policy.trace import TokenChoiceEvent
from transformer_policy.types import T

from .test_transformer_policy_decoder import PreferenceScorer
from .transformer_policy_fixtures import actionable_state


def test_pad_token_choice_events_shapes_and_masks():
    events = (
        TokenChoiceEvent(
            sequence_tokens=(T("STATE_START"), T("STATE_END")),
            legal_next_tokens=(T("STOP"), T("DEF", def_index=0)),
            chosen_index=1,
            phase="def",
            step_index=0,
        ),
        TokenChoiceEvent(
            sequence_tokens=(T("STATE_START"), T("STATE_END"), T("DEF", def_index=0)),
            legal_next_tokens=(T("CAND", candidate_index=0),),
            chosen_index=0,
            phase="candidate",
            step_index=0,
        ),
    )

    batch = pad_token_choice_events(events, episode_ids=np.asarray([0, 0]))

    assert batch.sequence_features.shape[0] == 2
    assert batch.sequence_mask.tolist() == [[True, True, False], [True, True, True]]
    assert batch.legal_mask.tolist() == [[True, True], [True, False]]
    assert batch.chosen_index.tolist() == [1, 0]
    assert batch.episode_id.tolist() == [0, 0]
    assert batch.next_position.tolist() == [2, 3]


def test_pad_token_choice_events_rejects_empty_events():
    with pytest.raises(ValueError, match="events must not be empty"):
        pad_token_choice_events((), episode_ids=np.asarray([], dtype=np.int32))


def test_score_event_batch_matches_score_next_for_each_event():
    scorer = CausalTransformerScorer(
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        mlp_dim=32,
        rngs=nnx.Rngs(0),
    )
    events = sample_step_with_events(
        actionable_state(),
        PreferenceScorer(),
        np.random.default_rng(0),
    ).events
    batch = pad_token_choice_events(events, episode_ids=np.zeros(len(events), dtype=np.int32))

    batched_logits = np.asarray(score_event_batch(scorer, batch))

    for index, event in enumerate(events):
        context_prefix = event.sequence_tokens
        expected = np.asarray(scorer.score_next(context_prefix, (), event.legal_next_tokens))
        actual = batched_logits[index, : len(event.legal_next_tokens)]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_chosen_event_and_trajectory_log_probs():
    logits = jnp.asarray([[0.0, 1.0, -1.0], [2.0, -2.0, -1.0]], dtype=jnp.float32)
    legal_mask = jnp.asarray([[True, True, False], [True, True, True]])
    chosen = jnp.asarray([1, 0], dtype=jnp.int32)
    episode_id = jnp.asarray([0, 1], dtype=jnp.int32)

    chosen_logp = chosen_event_log_probs(logits, legal_mask, chosen)
    per_episode = trajectory_log_probs(chosen_logp, episode_id, episode_count=2)

    assert chosen_logp.shape == (2,)
    assert per_episode.shape == (2,)
    assert np.isfinite(np.asarray(chosen_logp)).all()
    assert np.isfinite(np.asarray(per_episode)).all()
```

- [ ] **Step 2: Run batch tests to verify they fail**

From `python/`, run:

```bash
uv run pytest tests/test_transformer_policy_batch.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'transformer_policy.batch'`.

- [ ] **Step 3: Add feature scoring to the sequence model**

Modify `python/transformer_policy/sequence_model.py` by adding these methods to `CausalTransformerScorer`:

```python
    def _encode_features(
        self,
        sequence_features: jax.Array,
        sequence_mask: jax.Array,
    ) -> jax.Array:
        if sequence_features.ndim != 2:
            raise ValueError("sequence_features must be a 2-D matrix")
        if sequence_mask.ndim != 1 or sequence_mask.shape[0] != sequence_features.shape[0]:
            raise ValueError("sequence_mask must match sequence_features length")
        values = self.embedder.proj(sequence_features)[None, :, :]
        token_mask = sequence_mask[None, :]
        causal_mask = nnx.make_causal_mask(token_mask)
        for block in self.blocks:
            values = block(values, causal_mask)
        encoded = self.final_ln(values[0])
        final_index = jnp.maximum(jnp.sum(sequence_mask.astype(jnp.int32)) - 1, 0)
        return encoded[final_index]

    def score_next_features(
        self,
        sequence_features: jax.Array,
        sequence_mask: jax.Array,
        legal_features: jax.Array,
        legal_mask: jax.Array,
    ) -> jax.Array:
        hidden = self._encode_features(sequence_features, sequence_mask)
        query = self.query(hidden)
        legal_embeddings = self.embedder.proj(legal_features)
        logits = jnp.matmul(legal_embeddings, query)
        return jnp.where(legal_mask, logits, -jnp.inf)
```

- [ ] **Step 4: Implement event padding and scoring**

Create `python/transformer_policy/batch.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from transformer_policy.embed import TOKEN_FEATURE_DIM, token_features
from transformer_policy.trace import TokenChoiceEvent


@dataclass(frozen=True)
class PaddedTokenChoiceBatch:
    sequence_features: np.ndarray
    sequence_mask: np.ndarray
    legal_features: np.ndarray
    legal_mask: np.ndarray
    next_position: np.ndarray
    chosen_index: np.ndarray
    episode_id: np.ndarray
    event_mask: np.ndarray


def _validate_episode_ids(events: tuple[TokenChoiceEvent, ...], episode_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(episode_ids, dtype=np.int32)
    if ids.ndim != 1 or len(ids) != len(events):
        raise ValueError("episode_ids must be a 1-D array matching events")
    if np.any(ids < 0):
        raise ValueError("episode_ids must be non-negative")
    return ids


def pad_token_choice_events(
    events: tuple[TokenChoiceEvent, ...],
    *,
    episode_ids: np.ndarray,
) -> PaddedTokenChoiceBatch:
    if not events:
        raise ValueError("events must not be empty")
    ids = _validate_episode_ids(events, episode_ids)
    max_sequence_len = max(len(event.sequence_tokens) for event in events)
    max_legal = max(len(event.legal_next_tokens) for event in events)
    event_count = len(events)
    sequence_features = np.zeros(
        (event_count, max_sequence_len, TOKEN_FEATURE_DIM),
        dtype=np.float32,
    )
    sequence_mask = np.zeros((event_count, max_sequence_len), dtype=bool)
    legal_features = np.zeros(
        (event_count, max_legal, TOKEN_FEATURE_DIM),
        dtype=np.float32,
    )
    legal_mask = np.zeros((event_count, max_legal), dtype=bool)
    next_position = np.zeros(event_count, dtype=np.int32)
    chosen_index = np.zeros(event_count, dtype=np.int32)
    for row, event in enumerate(events):
        sequence = token_features(event.sequence_tokens)
        legal = token_features(event.legal_next_tokens)
        next_pos = len(event.sequence_tokens)
        legal[:, 1] = float(next_pos)
        sequence_features[row, : len(event.sequence_tokens), :] = sequence
        sequence_mask[row, : len(event.sequence_tokens)] = True
        legal_features[row, : len(event.legal_next_tokens), :] = legal
        legal_mask[row, : len(event.legal_next_tokens)] = True
        next_position[row] = next_pos
        chosen_index[row] = event.chosen_index
    return PaddedTokenChoiceBatch(
        sequence_features=sequence_features,
        sequence_mask=sequence_mask,
        legal_features=legal_features,
        legal_mask=legal_mask,
        next_position=next_position,
        chosen_index=chosen_index,
        episode_id=ids,
        event_mask=np.ones(event_count, dtype=bool),
    )


def score_event_batch(scorer, batch: PaddedTokenChoiceBatch) -> jax.Array:
    return jax.vmap(scorer.score_next_features)(
        jnp.asarray(batch.sequence_features, dtype=jnp.float32),
        jnp.asarray(batch.sequence_mask, dtype=bool),
        jnp.asarray(batch.legal_features, dtype=jnp.float32),
        jnp.asarray(batch.legal_mask, dtype=bool),
    )


def chosen_event_log_probs(
    logits: jax.Array,
    legal_mask: jax.Array,
    chosen_index: jax.Array,
) -> jax.Array:
    masked_logits = jnp.where(legal_mask, logits, -jnp.inf)
    log_probs = jax.nn.log_softmax(masked_logits, axis=-1)
    return jnp.take_along_axis(log_probs, chosen_index[:, None], axis=-1).squeeze(-1)


def trajectory_log_probs(
    chosen_log_probs: jax.Array,
    episode_id: jax.Array,
    *,
    episode_count: int,
) -> jax.Array:
    return jnp.zeros((episode_count,), dtype=chosen_log_probs.dtype).at[episode_id].add(
        chosen_log_probs
    )
```

- [ ] **Step 5: Run batch tests**

From `python/`, run:

```bash
uv run pytest tests/test_transformer_policy_batch.py tests/test_transformer_policy_sequence_model.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit padded event scoring**

```bash
git add python/transformer_policy/batch.py python/transformer_policy/sequence_model.py python/tests/test_transformer_policy_batch.py
git commit -m "feat: add transformer event batch scoring"
```

---

### Task 3: Reinforce Package Scaffold And Trace Records

**Files:**
- Create: `python/reinforce_training/__init__.py`
- Create: `python/reinforce_training/trace.py`
- Modify: `python/pyproject.toml`
- Test: `python/tests/test_reinforce_package.py`
- Test: `python/tests/test_reinforce_trace.py`

- [ ] **Step 1: Write failing package and trace tests**

Create `python/tests/test_reinforce_package.py`:

```python
import importlib
import sys


def test_reinforce_training_imports_without_legacy_rl():
    sys.modules.pop("gristmill_rl", None)

    module = importlib.import_module("reinforce_training")

    assert module.__all__ == (
        "EpisodeTrace",
        "Stage1AttemptTrace",
        "StepTrace",
    )
    assert "gristmill_rl" not in sys.modules
```

Create `python/tests/test_reinforce_trace.py`:

```python
import pytest

from transformer_policy.trace import TokenChoiceEvent
from transformer_policy.types import Stage1Attempt, T
from reinforce_training.trace import (
    EpisodeTrace,
    Stage1AttemptTrace,
    StepTrace,
    step_trace_from_traced_sample,
)
from transformer_policy.decoder import sample_step_with_events

from .test_transformer_policy_decoder import PreferenceScorer
from .transformer_policy_fixtures import actionable_state


def test_stage1_attempt_trace_from_policy_attempt():
    trace = Stage1AttemptTrace.from_policy_attempt(
        Stage1Attempt(def_index=2, log_prob=-0.5, accepted=False)
    )

    assert trace.def_index == 2
    assert trace.log_prob == -0.5
    assert not trace.accepted


def test_step_trace_from_traced_sample_drops_action_space_handle():
    traced = sample_step_with_events(
        actionable_state(),
        PreferenceScorer(),
        __import__("numpy").random.default_rng(0),
    )
    step = step_trace_from_traced_sample(
        step_index=0,
        state_snapshot=actionable_state().snapshot(),
        traced=traced,
    )

    assert isinstance(step, StepTrace)
    assert not step.stopped
    assert step.action_space_snapshot is not None
    assert step.decision is not None
    assert step.sample_log_prob == pytest.approx(traced.sample.log_prob)
    assert step.token_events == traced.events
    assert not hasattr(step, "action_space")


def test_episode_trace_validates_reward_and_steps():
    step = StepTrace(
        step_index=0,
        state_snapshot={"definitions": []},
        stopped=True,
        def_attempts=(),
        def_index=None,
        action_space_snapshot=None,
        decision=None,
        decision_tokens=(T("STOP"),),
        token_events=(
            TokenChoiceEvent(
                sequence_tokens=(T("STATE_START"), T("STATE_END")),
                legal_next_tokens=(T("STOP"),),
                chosen_index=0,
                phase="def",
                step_index=0,
            ),
        ),
        sample_log_prob=0.0,
    )

    episode = EpisodeTrace(
        episode_index=0,
        episode_seed=123,
        steps=(step,),
        final_snapshot={"definitions": []},
        final_log_flops=7.0,
        reward=-7.0,
        terminal_reason="stop",
    )

    assert episode.reward == -episode.final_log_flops


def test_episode_trace_rejects_invalid_terminal_reason():
    with pytest.raises(ValueError, match="terminal_reason must be"):
        EpisodeTrace(
            episode_index=0,
            episode_seed=0,
            steps=(),
            final_snapshot={},
            final_log_flops=1.0,
            reward=-1.0,
            terminal_reason="done",
        )
```

- [ ] **Step 2: Run scaffold tests to verify they fail**

From `python/`, run:

```bash
uv run pytest tests/test_reinforce_package.py tests/test_reinforce_trace.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'reinforce_training'`.

- [ ] **Step 3: Add package exports and pyproject metadata**

Create `python/reinforce_training/__init__.py`:

```python
"""Naive REINFORCE training for transformer rewrite policies."""

from reinforce_training.trace import EpisodeTrace, Stage1AttemptTrace, StepTrace

__all__ = (
    "EpisodeTrace",
    "Stage1AttemptTrace",
    "StepTrace",
)
```

Modify `python/pyproject.toml`:

```toml
exclude = [
    "gristmill_rl/**/__pycache__",
    "gristmill_rl/**/*.pyc",
    "reinforce_training/**/__pycache__",
    "reinforce_training/**/*.pyc",
    "transformer_policy/**/__pycache__",
    "transformer_policy/**/*.pyc",
]
python-packages = ["gristmill_rl", "transformer_policy", "reinforce_training"]
```

- [ ] **Step 4: Implement serializable trace records**

Create `python/reinforce_training/trace.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from transformer_policy.trace import TokenChoiceEvent
from transformer_policy.types import Stage1Attempt, Token


TerminalReason = Literal["stop", "max_steps"]


@dataclass(frozen=True)
class Stage1AttemptTrace:
    def_index: int
    log_prob: float
    accepted: bool

    @staticmethod
    def from_policy_attempt(attempt: Stage1Attempt) -> "Stage1AttemptTrace":
        return Stage1AttemptTrace(
            def_index=attempt.def_index,
            log_prob=float(attempt.log_prob),
            accepted=attempt.accepted,
        )


@dataclass(frozen=True)
class StepTrace:
    step_index: int
    state_snapshot: dict[str, Any]
    stopped: bool
    def_attempts: tuple[Stage1AttemptTrace, ...]
    def_index: int | None
    action_space_snapshot: dict[str, Any] | None
    decision: dict[str, Any] | None
    decision_tokens: tuple[Token, ...]
    token_events: tuple[TokenChoiceEvent, ...]
    sample_log_prob: float

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        if not self.token_events:
            raise ValueError("step trace must contain token events")
        if not np.isfinite(self.sample_log_prob):
            raise ValueError("sample_log_prob must be finite")
        if self.stopped:
            if self.def_index is not None or self.action_space_snapshot is not None:
                raise ValueError("stopped step must not contain rewrite data")
            if self.decision is not None:
                raise ValueError("stopped step must not contain decision")
        else:
            if self.def_index is None:
                raise ValueError("rewrite step requires def_index")
            if self.action_space_snapshot is None:
                raise ValueError("rewrite step requires action_space_snapshot")
            if self.decision is None:
                raise ValueError("rewrite step requires decision")


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
        if self.episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        if self.terminal_reason not in {"stop", "max_steps"}:
            raise ValueError("terminal_reason must be 'stop' or 'max_steps'")
        if not np.isfinite(self.final_log_flops):
            raise ValueError("final_log_flops must be finite")
        if not np.isfinite(self.reward):
            raise ValueError("reward must be finite")


def step_trace_from_traced_sample(
    *,
    step_index: int,
    state_snapshot: dict[str, Any],
    traced,
) -> StepTrace:
    sample = traced.sample
    action_space_snapshot = None
    if sample.action_space is not None:
        action_space_snapshot = sample.action_space.snapshot()
    return StepTrace(
        step_index=step_index,
        state_snapshot=state_snapshot,
        stopped=sample.stopped,
        def_attempts=tuple(
            Stage1AttemptTrace.from_policy_attempt(attempt)
            for attempt in sample.def_attempts
        ),
        def_index=sample.def_index,
        action_space_snapshot=action_space_snapshot,
        decision=sample.decision,
        decision_tokens=sample.decision_tokens,
        token_events=traced.events,
        sample_log_prob=float(sample.log_prob),
    )
```

- [ ] **Step 5: Run package and trace tests**

From `python/`, run:

```bash
uv run pytest tests/test_reinforce_package.py tests/test_reinforce_trace.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit scaffold**

```bash
git add python/pyproject.toml python/reinforce_training/__init__.py python/reinforce_training/trace.py python/tests/test_reinforce_package.py python/tests/test_reinforce_trace.py
git commit -m "feat: scaffold reinforce training package"
```

---

### Task 4: REINFORCE Objective And Training Update

**Files:**
- Create: `python/reinforce_training/objective.py`
- Test: `python/tests/test_reinforce_objective.py`

- [ ] **Step 1: Write failing objective tests**

Create `python/tests/test_reinforce_objective.py`:

```python
import numpy as np
from flax import nnx

from transformer_policy.batch import pad_token_choice_events
from transformer_policy.decoder import sample_step_with_events
from transformer_policy.sequence_model import CausalTransformerScorer
from reinforce_training.objective import (
    TrainConfig,
    create_optimizer,
    rewards_and_advantages,
    reinforce_loss,
    train_step,
)

from .test_transformer_policy_decoder import PreferenceScorer
from .transformer_policy_fixtures import actionable_state


def _event_batch():
    events = sample_step_with_events(
        actionable_state(),
        PreferenceScorer(),
        np.random.default_rng(0),
    ).events
    return pad_token_choice_events(events, episode_ids=np.zeros(len(events), dtype=np.int32))


def _flat_param_values(scorer):
    state = nnx.state(scorer, nnx.Param)
    values = []
    for leaf in __import__("jax").tree_util.tree_leaves(state):
        value = getattr(leaf, "value", leaf)
        values.append(np.asarray(value).copy())
    return values


def test_rewards_and_advantages_use_negative_final_log_flops_and_batch_mean():
    rewards, advantages = rewards_and_advantages(np.asarray([2.0, 4.0], dtype=np.float32))

    np.testing.assert_allclose(rewards, [-2.0, -4.0])
    np.testing.assert_allclose(advantages, [1.0, -1.0])
    assert np.sum(advantages) == np.float32(0.0)


def test_reinforce_loss_is_finite():
    scorer = CausalTransformerScorer(
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        mlp_dim=32,
        rngs=nnx.Rngs(0),
    )
    loss, aux = reinforce_loss(
        scorer,
        _event_batch(),
        advantages=np.asarray([1.0], dtype=np.float32),
        episode_count=1,
    )

    assert np.isfinite(float(loss))
    assert np.isfinite(float(aux["mean_trajectory_log_prob"]))


def test_train_step_changes_params_for_nonzero_advantage():
    scorer = CausalTransformerScorer(
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        mlp_dim=32,
        rngs=nnx.Rngs(0),
    )
    before = _flat_param_values(scorer)
    optimizer = create_optimizer(scorer, TrainConfig(learning_rate=1e-2))

    metrics = train_step(
        scorer,
        optimizer=optimizer,
        batch=_event_batch(),
        advantages=np.asarray([1.0], dtype=np.float32),
        episode_count=1,
    )

    after = _flat_param_values(scorer)
    assert metrics["params_changed"]
    assert np.isfinite(metrics["loss"])
    assert any(
        not np.array_equal(left, right)
        for left, right in zip(before, after, strict=True)
    )
```

- [ ] **Step 2: Run objective tests to verify they fail**

From `python/`, run:

```bash
uv run pytest tests/test_reinforce_objective.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'reinforce_training.objective'`.

- [ ] **Step 3: Implement objective helpers**

Create `python/reinforce_training/objective.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from transformer_policy.batch import (
    PaddedTokenChoiceBatch,
    chosen_event_log_probs,
    score_event_batch,
    trajectory_log_probs,
)


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 1e-3


def create_optimizer(scorer, config: TrainConfig) -> nnx.Optimizer:
    if config.learning_rate <= 0.0 or not np.isfinite(config.learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    return nnx.Optimizer(
        scorer,
        optax.adam(config.learning_rate),
        wrt=nnx.Param,
    )


def rewards_and_advantages(final_log_flops: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(final_log_flops, dtype=np.float32)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("final_log_flops must be a nonempty 1-D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("final_log_flops must be finite")
    rewards = -values
    advantages = rewards - np.mean(rewards, dtype=np.float32)
    return rewards.astype(np.float32), advantages.astype(np.float32)


def reinforce_loss(
    scorer,
    batch: PaddedTokenChoiceBatch,
    *,
    advantages: np.ndarray,
    episode_count: int,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    advantage_values = jax.lax.stop_gradient(jnp.asarray(advantages, dtype=jnp.float32))
    logits = score_event_batch(scorer, batch)
    chosen_log_probs = chosen_event_log_probs(
        logits,
        jnp.asarray(batch.legal_mask, dtype=bool),
        jnp.asarray(batch.chosen_index, dtype=jnp.int32),
    )
    per_episode = trajectory_log_probs(
        chosen_log_probs,
        jnp.asarray(batch.episode_id, dtype=jnp.int32),
        episode_count=episode_count,
    )
    loss = -jnp.mean(advantage_values * per_episode)
    return loss, {
        "loss": loss,
        "mean_trajectory_log_prob": jnp.mean(per_episode),
        "mean_event_log_prob": jnp.mean(chosen_log_probs),
    }


def _flat_param_values(module) -> list[np.ndarray]:
    state = nnx.state(module, nnx.Param)
    leaves = jax.tree_util.tree_leaves(state)
    values = []
    for leaf in leaves:
        value = getattr(leaf, "value", leaf)
        values.append(np.asarray(value).copy())
    return values


def train_step(
    scorer,
    *,
    optimizer: nnx.Optimizer,
    batch: PaddedTokenChoiceBatch,
    advantages: np.ndarray,
    episode_count: int,
) -> dict[str, float | bool]:
    before = _flat_param_values(scorer)
    grad_fn = nnx.value_and_grad(reinforce_loss, has_aux=True)
    (loss, aux), grads = grad_fn(
        scorer,
        batch,
        advantages=advantages,
        episode_count=episode_count,
    )
    optimizer.update(scorer, grads)
    after = _flat_param_values(scorer)
    params_changed = any(
        not np.array_equal(left, right)
        for left, right in zip(before, after, strict=True)
    )
    return {
        "loss": float(loss),
        "mean_trajectory_log_prob": float(aux["mean_trajectory_log_prob"]),
        "mean_event_log_prob": float(aux["mean_event_log_prob"]),
        "params_changed": params_changed,
    }
```

- [ ] **Step 4: Run objective tests**

From `python/`, run:

```bash
uv run pytest tests/test_reinforce_objective.py tests/test_transformer_policy_batch.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit objective**

```bash
git add python/reinforce_training/objective.py python/tests/test_reinforce_objective.py
git commit -m "feat: add reinforce objective"
```

---

### Task 5: Rollout And Process-Parallel Episode Collection

**Files:**
- Create: `python/reinforce_training/rollout.py`
- Test: `python/tests/test_reinforce_rollout.py`

- [ ] **Step 1: Write failing rollout tests**

Create `python/tests/test_reinforce_rollout.py`:

```python
import numpy as np
import pytest
from flax import nnx

from reinforce_training.rollout import (
    PolicyConfig,
    RolloutConfig,
    collect_episode_batch,
    sample_episode,
)
from transformer_policy.sequence_model import CausalTransformerScorer

from .transformer_policy_fixtures import actionable_json


def test_sample_episode_returns_serializable_trace():
    scorer = CausalTransformerScorer(
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        mlp_dim=32,
        rngs=nnx.Rngs(0),
    )

    episode = sample_episode(
        input_json=actionable_json(),
        scorer=scorer,
        config=RolloutConfig(max_steps=1),
        episode_index=0,
        episode_seed=0,
    )

    assert episode.episode_index == 0
    assert episode.episode_seed == 0
    assert episode.reward == -episode.final_log_flops
    assert episode.terminal_reason in {"stop", "max_steps"}
    assert len(episode.steps) >= 1
    assert episode.steps[0].token_events


def test_collect_episode_batch_returns_sorted_full_batch_with_one_worker():
    config = PolicyConfig(hidden_dim=16, num_heads=4, num_layers=1, mlp_dim=32)
    scorer = config.create_scorer(seed=0)

    episodes = collect_episode_batch(
        input_json=actionable_json(),
        scorer=scorer,
        policy_config=config,
        rollout_config=RolloutConfig(max_steps=1),
        update_index=0,
        batch_size=2,
        num_workers=1,
        seed=10,
    )

    assert [episode.episode_index for episode in episodes] == [0, 1]
    assert [episode.episode_seed for episode in episodes] == [10, 11]


def test_collect_episode_batch_returns_sorted_full_batch_with_two_workers():
    config = PolicyConfig(hidden_dim=16, num_heads=4, num_layers=1, mlp_dim=32)
    scorer = config.create_scorer(seed=0)

    episodes = collect_episode_batch(
        input_json=actionable_json(),
        scorer=scorer,
        policy_config=config,
        rollout_config=RolloutConfig(max_steps=1),
        update_index=0,
        batch_size=2,
        num_workers=2,
        seed=10,
    )

    assert [episode.episode_index for episode in episodes] == [0, 1]
    assert len(episodes) == 2


def test_rollout_config_validates_positive_values():
    with pytest.raises(ValueError, match="max_steps must be positive"):
        RolloutConfig(max_steps=0)
```

- [ ] **Step 2: Run rollout tests to verify they fail**

From `python/`, run:

```bash
uv run pytest tests/test_reinforce_rollout.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'reinforce_training.rollout'`.

- [ ] **Step 3: Implement rollout collection**

Create `python/reinforce_training/rollout.py`:

```python
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
from flax import nnx

from gristmill_symbolics import RewriteState, TensorComputation
from transformer_policy.decoder import sample_step_with_events
from transformer_policy.sequence_model import CausalTransformerScorer
from reinforce_training.trace import EpisodeTrace, step_trace_from_traced_sample


@dataclass(frozen=True)
class PolicyConfig:
    hidden_dim: int = 32
    num_heads: int = 4
    num_layers: int = 1
    mlp_dim: int = 64

    def create_scorer(self, *, seed: int) -> CausalTransformerScorer:
        return CausalTransformerScorer(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            mlp_dim=self.mlp_dim,
            rngs=nnx.Rngs(seed),
        )


@dataclass(frozen=True)
class RolloutConfig:
    max_steps: int = 4

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")


def _model_state(scorer) -> Any:
    return nnx.state(scorer, nnx.Param)


def _restore_scorer(policy_config: PolicyConfig, state: Any) -> CausalTransformerScorer:
    scorer = policy_config.create_scorer(seed=0)
    nnx.update(scorer, state)
    return scorer


def sample_episode(
    *,
    input_json: str,
    scorer,
    config: RolloutConfig,
    episode_index: int,
    episode_seed: int,
) -> EpisodeTrace:
    comp = TensorComputation.from_json_string(input_json)
    state = RewriteState.from_computation(comp)
    rng = np.random.default_rng(episode_seed)
    steps = []
    terminal_reason = "max_steps"
    for step_index in range(config.max_steps):
        state_snapshot = state.snapshot()
        traced = sample_step_with_events(
            state,
            scorer,
            rng,
            step_index=step_index,
        )
        steps.append(
            step_trace_from_traced_sample(
                step_index=step_index,
                state_snapshot=state_snapshot,
                traced=traced,
            )
        )
        if traced.sample.stopped:
            terminal_reason = "stop"
            break
        state.step_with_space(traced.sample.action_space, traced.sample.decision)
    final_log_flops = float(state.log_total_flops())
    reward = -final_log_flops
    return EpisodeTrace(
        episode_index=episode_index,
        episode_seed=episode_seed,
        steps=tuple(steps),
        final_snapshot=state.snapshot(),
        final_log_flops=final_log_flops,
        reward=reward,
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
    scorer = _restore_scorer(policy_config, model_state)
    return sample_episode(
        input_json=input_json,
        scorer=scorer,
        config=rollout_config,
        episode_index=episode_index,
        episode_seed=episode_seed,
    )


def collect_episode_batch(
    *,
    input_json: str,
    scorer,
    policy_config: PolicyConfig,
    rollout_config: RolloutConfig,
    update_index: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[EpisodeTrace, ...]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    model_state = _model_state(scorer)

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
uv run pytest tests/test_reinforce_rollout.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit rollout**

```bash
git add python/reinforce_training/rollout.py python/tests/test_reinforce_rollout.py
git commit -m "feat: add reinforce rollout collection"
```

---

### Task 6: Checkpoint Save And Load

**Files:**
- Create: `python/reinforce_training/checkpoint.py`
- Test: `python/tests/test_reinforce_checkpoint.py`

- [ ] **Step 1: Write failing checkpoint tests**

Create `python/tests/test_reinforce_checkpoint.py`:

```python
import json

import numpy as np
import pytest
from flax import nnx

from reinforce_training.checkpoint import load_checkpoint, save_checkpoint
from reinforce_training.objective import TrainConfig, create_optimizer
from reinforce_training.rollout import PolicyConfig, RolloutConfig


def _score_vector(scorer):
    from transformer_policy.types import T

    return np.asarray(
        scorer.score_next(
            (T("STATE_START"), T("STATE_END")),
            (),
            (T("STOP"), T("DEF", def_index=0)),
        )
    )


def test_checkpoint_round_trip_restores_model_outputs(tmp_path):
    policy_config = PolicyConfig(hidden_dim=16, num_heads=4, num_layers=1, mlp_dim=32)
    scorer = policy_config.create_scorer(seed=123)
    optimizer = create_optimizer(scorer, TrainConfig(learning_rate=1e-3))
    expected = _score_vector(scorer)

    save_checkpoint(
        tmp_path / "checkpoint",
        scorer=scorer,
        optimizer=optimizer,
        policy_config=policy_config,
        train_config=TrainConfig(learning_rate=1e-3),
        rollout_config=RolloutConfig(max_steps=2),
        update_count=3,
        seed=9,
    )

    loaded = load_checkpoint(tmp_path / "checkpoint")

    assert loaded.policy_config == policy_config
    assert loaded.train_config == TrainConfig(learning_rate=1e-3)
    assert loaded.rollout_config == RolloutConfig(max_steps=2)
    assert loaded.update_count == 3
    assert loaded.seed == 9
    assert loaded.optimizer is not None
    np.testing.assert_allclose(_score_vector(loaded.scorer), expected)


def test_checkpoint_refuses_existing_without_overwrite(tmp_path):
    policy_config = PolicyConfig(hidden_dim=16, num_heads=4, num_layers=1, mlp_dim=32)
    scorer = policy_config.create_scorer(seed=0)
    optimizer = create_optimizer(scorer, TrainConfig())
    save_checkpoint(
        tmp_path / "checkpoint",
        scorer=scorer,
        optimizer=optimizer,
        policy_config=policy_config,
        train_config=TrainConfig(),
        rollout_config=RolloutConfig(),
        update_count=0,
        seed=0,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        save_checkpoint(
            tmp_path / "checkpoint",
            scorer=scorer,
            optimizer=optimizer,
            policy_config=policy_config,
            train_config=TrainConfig(),
            rollout_config=RolloutConfig(),
            update_count=0,
            seed=0,
        )


def test_checkpoint_metadata_is_json(tmp_path):
    policy_config = PolicyConfig(hidden_dim=16, num_heads=4, num_layers=1, mlp_dim=32)
    scorer = policy_config.create_scorer(seed=0)
    optimizer = create_optimizer(scorer, TrainConfig())
    save_checkpoint(
        tmp_path / "checkpoint",
        scorer=scorer,
        optimizer=optimizer,
        policy_config=policy_config,
        train_config=TrainConfig(),
        rollout_config=RolloutConfig(),
        update_count=0,
        seed=0,
    )

    metadata = json.loads((tmp_path / "checkpoint" / "metadata.json").read_text())

    assert metadata["schema_version"] == 1
    assert metadata["package"] == "reinforce_training"
    assert metadata["model_class"] == "CausalTransformerScorer"
```

- [ ] **Step 2: Run checkpoint tests to verify they fail**

From `python/`, run:

```bash
uv run pytest tests/test_reinforce_checkpoint.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'reinforce_training.checkpoint'`.

- [ ] **Step 3: Implement checkpointing**

Create `python/reinforce_training/checkpoint.py`:

```python
from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from flax import nnx
import orbax.checkpoint as ocp

from reinforce_training.objective import TrainConfig, create_optimizer
from reinforce_training.rollout import PolicyConfig, RolloutConfig


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LoadedCheckpoint:
    scorer: Any
    optimizer: Any
    policy_config: PolicyConfig
    train_config: TrainConfig
    rollout_config: RolloutConfig
    update_count: int
    seed: int
    metadata: dict[str, Any]


def _metadata_path(path: Path) -> Path:
    return path / "metadata.json"


def _state_path(path: Path) -> Path:
    return path / "state"


def _checkpoint_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _temporary_checkpoint_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _write_metadata(path: Path, payload: dict[str, Any]) -> None:
    _metadata_path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))


def _metadata_payload(
    *,
    policy_config: PolicyConfig,
    train_config: TrainConfig,
    rollout_config: RolloutConfig,
    update_count: int,
    seed: int,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    if update_count < 0:
        raise ValueError("update_count must be non-negative")
    return {
        "schema_version": SCHEMA_VERSION,
        "package": "reinforce_training",
        "model_class": "CausalTransformerScorer",
        "policy_config": asdict(policy_config),
        "train_config": asdict(train_config),
        "rollout_config": asdict(rollout_config),
        "update_count": update_count,
        "seed": seed,
        "seed_scheme": "seed + update_index * batch_size + episode_index",
        "metadata": metadata or {},
    }


def save_checkpoint(
    path: str | Path,
    *,
    scorer,
    optimizer,
    policy_config: PolicyConfig,
    train_config: TrainConfig,
    rollout_config: RolloutConfig,
    update_count: int,
    seed: int,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    checkpoint_path = _checkpoint_path(path)
    if checkpoint_path.exists() and not overwrite:
        raise FileExistsError(f"checkpoint path already exists: {checkpoint_path}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_checkpoint_path(checkpoint_path)
    payload = _metadata_payload(
        policy_config=policy_config,
        train_config=train_config,
        rollout_config=rollout_config,
        update_count=update_count,
        seed=seed,
        metadata=metadata,
    )
    try:
        temp_path.mkdir()
        state = {
            "model": nnx.state(scorer, nnx.Param),
            "optimizer": nnx.state(optimizer),
        }
        ocp.PyTreeCheckpointer().save(_state_path(temp_path), state, force=True)
        _write_metadata(temp_path, payload)
        if checkpoint_path.exists():
            _remove_path(checkpoint_path)
        temp_path.rename(checkpoint_path)
    except Exception:
        _remove_path(temp_path)
        raise


def load_checkpoint(path: str | Path) -> LoadedCheckpoint:
    checkpoint_path = _checkpoint_path(path)
    payload = json.loads(_metadata_path(checkpoint_path).read_text())
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema_version {payload.get('schema_version')}")
    if payload.get("package") != "reinforce_training":
        raise ValueError("checkpoint package must be reinforce_training")
    policy_config = PolicyConfig(**payload["policy_config"])
    train_config = TrainConfig(**payload["train_config"])
    rollout_config = RolloutConfig(**payload["rollout_config"])
    scorer = policy_config.create_scorer(seed=0)
    optimizer = create_optimizer(scorer, train_config)
    abstract_state = {
        "model": nnx.state(scorer, nnx.Param),
        "optimizer": nnx.state(optimizer),
    }
    restore_args = ocp.checkpoint_utils.construct_restore_args(abstract_state)
    restored_state = ocp.PyTreeCheckpointer().restore(
        _state_path(checkpoint_path),
        item=abstract_state,
        restore_args=restore_args,
    )
    nnx.update(scorer, restored_state["model"])
    nnx.update(optimizer, restored_state["optimizer"])
    return LoadedCheckpoint(
        scorer=scorer,
        optimizer=optimizer,
        policy_config=policy_config,
        train_config=train_config,
        rollout_config=rollout_config,
        update_count=int(payload["update_count"]),
        seed=int(payload["seed"]),
        metadata=payload.get("metadata", {}),
    )
```

- [ ] **Step 4: Run checkpoint tests**

From `python/`, run:

```bash
uv run pytest tests/test_reinforce_checkpoint.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit checkpointing**

```bash
git add python/reinforce_training/checkpoint.py python/tests/test_reinforce_checkpoint.py
git commit -m "feat: add reinforce checkpoints"
```

---

### Task 7: End-To-End Training CLI

**Files:**
- Create: `python/reinforce_training/train.py`
- Test: `python/tests/test_reinforce_train.py`

- [ ] **Step 1: Write failing CLI tests**

Create `python/tests/test_reinforce_train.py`:

```python
import json
import subprocess
import sys

from .transformer_policy_fixtures import actionable_json


def test_reinforce_train_cli_completes_tiny_run(tmp_path):
    input_path = tmp_path / "input.json"
    input_path.write_text(actionable_json())

    result = subprocess.run(
        [
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
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(result.stdout.strip().splitlines()[-1])
    assert metrics["updates"] == 1
    assert metrics["batch_size"] == 2
    assert metrics["num_workers"] == 1
    assert metrics["params_changed"]
    assert metrics["checkpoint_out"] is None


def test_reinforce_train_cli_writes_checkpoint(tmp_path):
    input_path = tmp_path / "input.json"
    checkpoint_path = tmp_path / "checkpoint"
    input_path.write_text(actionable_json())

    result = subprocess.run(
        [
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
            "--hidden-dim",
            "16",
            "--num-heads",
            "4",
            "--num-layers",
            "1",
            "--mlp-dim",
            "32",
            "--checkpoint-out",
            str(checkpoint_path),
            "--seed",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(result.stdout.strip().splitlines()[-1])
    assert metrics["checkpoint_out"] == str(checkpoint_path)
    assert (checkpoint_path / "metadata.json").exists()
    assert (checkpoint_path / "state").exists()
```

- [ ] **Step 2: Run CLI tests to verify they fail**

From `python/`, run:

```bash
uv run pytest tests/test_reinforce_train.py -q
```

Expected: FAIL with `No module named reinforce_training.train`.

- [ ] **Step 3: Implement CLI and training loop**

Create `python/reinforce_training/train.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from transformer_policy.batch import pad_token_choice_events
from reinforce_training.checkpoint import load_checkpoint, save_checkpoint
from reinforce_training.objective import (
    TrainConfig,
    create_optimizer,
    rewards_and_advantages,
    train_step,
)
from reinforce_training.rollout import (
    PolicyConfig,
    RolloutConfig,
    collect_episode_batch,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0 or not np.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="Train a naive REINFORCE rewrite policy.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--updates", type=_positive_int, default=1)
    parser.add_argument("--batch-size", type=_positive_int, default=4)
    parser.add_argument("--max-steps", type=_positive_int, default=4)
    parser.add_argument("--num-workers", type=_positive_int, default=1)
    parser.add_argument("--learning-rate", type=_positive_float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-in", type=Path, default=None)
    parser.add_argument("--checkpoint-out", type=Path, default=None)
    parser.add_argument("--checkpoint-overwrite", action="store_true")
    parser.add_argument("--hidden-dim", type=_positive_int, default=32)
    parser.add_argument("--num-heads", type=_positive_int, default=4)
    parser.add_argument("--num-layers", type=_positive_int, default=1)
    parser.add_argument("--mlp-dim", type=_positive_int, default=64)
    return parser.parse_args(argv)


def _episode_events(episodes):
    events = []
    episode_ids = []
    for episode_index, episode in enumerate(episodes):
        for step in episode.steps:
            for event in step.token_events:
                events.append(event)
                episode_ids.append(episode_index)
    return tuple(events), np.asarray(episode_ids, dtype=np.int32)


def run(args) -> dict[str, object]:
    input_json = args.input.read_text()
    checkpoint_in = None
    if args.checkpoint_in is not None:
        loaded = load_checkpoint(args.checkpoint_in)
        scorer = loaded.scorer
        optimizer = loaded.optimizer
        policy_config = loaded.policy_config
        train_config = loaded.train_config
        rollout_config = loaded.rollout_config
        start_update = loaded.update_count
        checkpoint_in = str(args.checkpoint_in)
    else:
        policy_config = PolicyConfig(
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            mlp_dim=args.mlp_dim,
        )
        scorer = policy_config.create_scorer(seed=args.seed)
        train_config = TrainConfig(learning_rate=args.learning_rate)
        optimizer = create_optimizer(scorer, train_config)
        rollout_config = RolloutConfig(max_steps=args.max_steps)
        start_update = 0

    last_metrics: dict[str, object] = {}
    for offset in range(args.updates):
        update_index = start_update + offset
        episodes = collect_episode_batch(
            input_json=input_json,
            scorer=scorer,
            policy_config=policy_config,
            rollout_config=rollout_config,
            update_index=update_index,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        final_log_flops = np.asarray(
            [episode.final_log_flops for episode in episodes],
            dtype=np.float32,
        )
        rewards, advantages = rewards_and_advantages(final_log_flops)
        events, episode_ids = _episode_events(episodes)
        batch = pad_token_choice_events(events, episode_ids=episode_ids)
        update_metrics = train_step(
            scorer,
            optimizer=optimizer,
            batch=batch,
            advantages=advantages,
            episode_count=len(episodes),
        )
        checkpoint_out = str(args.checkpoint_out) if args.checkpoint_out else None
        last_metrics = {
            "update": update_index + 1,
            "updates": start_update + args.updates,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "mean_reward": float(np.mean(rewards)),
            "mean_final_log_flops": float(np.mean(final_log_flops)),
            "best_final_log_flops": float(np.min(final_log_flops)),
            "mean_steps": float(np.mean([len(episode.steps) for episode in episodes])),
            "stop_count": int(sum(episode.terminal_reason == "stop" for episode in episodes)),
            "max_steps_count": int(sum(episode.terminal_reason == "max_steps" for episode in episodes)),
            "loss": float(update_metrics["loss"]),
            "mean_sample_log_prob": float(
                np.mean(
                    [
                        step.sample_log_prob
                        for episode in episodes
                        for step in episode.steps
                    ]
                )
            ),
            "mean_trajectory_log_prob": float(update_metrics["mean_trajectory_log_prob"]),
            "params_changed": bool(update_metrics["params_changed"]),
            "checkpoint_in": checkpoint_in,
            "checkpoint_out": checkpoint_out,
        }
        print(json.dumps(last_metrics, sort_keys=True), flush=True)

    if args.checkpoint_out is not None:
        save_checkpoint(
            args.checkpoint_out,
            scorer=scorer,
            optimizer=optimizer,
            policy_config=policy_config,
            train_config=train_config,
            rollout_config=rollout_config,
            update_count=start_update + args.updates,
            seed=args.seed,
            overwrite=args.checkpoint_overwrite,
        )
    return last_metrics


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run CLI tests**

From `python/`, run:

```bash
uv run pytest tests/test_reinforce_train.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit CLI**

```bash
git add python/reinforce_training/train.py python/tests/test_reinforce_train.py
git commit -m "feat: add reinforce training cli"
```

---

### Task 8: Full Verification And Boundary Checks

**Files:**
- Read: `python/transformer_policy/*.py`
- Read: `python/reinforce_training/*.py`
- Read: `python/tests/test_transformer_policy_*.py`
- Read: `python/tests/test_reinforce_*.py`

- [ ] **Step 1: Check no legacy RL imports**

From `python/`, run:

```bash
rg -n "gristmill_rl" reinforce_training tests/test_reinforce_*.py
```

Expected: no matches.

- [ ] **Step 2: Run transformer policy tests**

From `python/`, run:

```bash
uv run pytest tests/test_transformer_policy_package.py tests/test_transformer_policy_types.py tests/test_transformer_policy_tokenize.py tests/test_transformer_policy_embed.py tests/test_transformer_policy_sequence_model.py tests/test_transformer_policy_decoder.py tests/test_transformer_policy_policy.py tests/test_transformer_policy_trace.py tests/test_transformer_policy_batch.py -q
```

Expected: PASS.

- [ ] **Step 3: Run reinforce tests**

From `python/`, run:

```bash
uv run pytest tests/test_reinforce_package.py tests/test_reinforce_trace.py tests/test_reinforce_objective.py tests/test_reinforce_rollout.py tests/test_reinforce_checkpoint.py tests/test_reinforce_train.py -q
```

Expected: PASS.

- [ ] **Step 4: Run PyO3 binding and rewrite-adjacent tests**

From `python/`, run:

```bash
uv run pytest tests/test_bindings.py -q
```

Expected: PASS.

From repo root, run:

```bash
cargo test rewrite
```

Expected: PASS.

- [ ] **Step 5: Run all Python tests**

From `python/`, run:

```bash
uv run pytest -q
```

Expected: PASS.

- [ ] **Step 6: Commit final verification if any cleanup was needed**

If Step 1-5 required code cleanup, commit it:

```bash
git add python/transformer_policy python/reinforce_training python/tests python/pyproject.toml
git commit -m "test: verify reinforce training"
```

If no cleanup was needed, do not create an empty commit.

---

## Self-Review Checklist

- Spec coverage: The plan adds the required `transformer_policy` trace/batch hooks, creates the standalone `reinforce_training` package, implements process-parallel rollout, uses final reward with batch-mean baseline, keeps gradients only through token log-probs, adds padded JAX rescoring, adds checkpoints, and adds an end-to-end CLI.
- Dependency boundary: `reinforce_training` never imports `gristmill_rl`; `transformer_policy` remains independent of `reinforce_training`.
- STOP handling: The traced sampling path records STOP as a normal token event; STOP-only states naturally produce one legal token and log-prob zero.
- Parallelism: Rollout uses `ProcessPoolExecutor` across episodes; JAX batching is reserved for rescoring token events.
- Type consistency: `TokenChoiceEvent`, `TracedPolicySample`, `PaddedTokenChoiceBatch`, `EpisodeTrace`, `StepTrace`, `PolicyConfig`, `RolloutConfig`, and `TrainConfig` are defined before downstream tasks use them.
- Verification: The plan includes transformer-policy tests, reinforce tests, no-legacy-import checks, binding tests, and Rust rewrite tests.
