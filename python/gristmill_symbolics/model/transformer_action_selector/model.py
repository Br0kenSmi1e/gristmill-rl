from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from gristmill_symbolics import ActionSpace
from gristmill_symbolics import TensorComputation
from gristmill_symbolics import action_spaces_for_batch
from gristmill_symbolics import apply_decisions_for_batch
from gristmill_symbolics import validate_decisions_for_batch
from gristmill_symbolics.model.protocols import StepwiseModel
from gristmill_symbolics.model.tokenizer import SIDE
from gristmill_symbolics.model.tokenizer import TOKEN_FIELDS, TOKEN_KIND
from gristmill_symbolics.model.tokenizer import stack_token_arrays
from gristmill_symbolics.model.tokenizer import tokenize_action_space_snapshot
from gristmill_symbolics.model.tokenizer import tokenize_computation_snapshot

from .nn import LogitDecoder, TokenEmbedder, TransformerEncoder


@dataclass(frozen=True)
class SelectorState:
    comp: TensorComputation
    target_mask: jax.Array | None = None


@dataclass(frozen=True)
class SelectorChoice:
    action_space: ActionSpace
    decision: dict[str, object]
    logp: float


@dataclass(frozen=True)
class SelectorTransitions:
    state: TensorComputation
    choices: Sequence[SelectorChoice]


BatchedState = Sequence[SelectorState]
BatchedTransitions = Sequence[SelectorTransitions]


class TransformerActionSelectorModel(
    StepwiseModel[BatchedState, BatchedTransitions]
):
    def __init__(
        self,
        *,
        state_token_pad_to: int,
        action_token_pad_to: int,
        definition_pad_to: int,
        candidate_pad_to: int,
        side_term_pad_to: int,
        d_model: int = 32,
        num_attention_layers: int = 1,
        num_attention_heads: int = 4,
        id_vocab_size: int = 128,
        init_scale: float = 0.02,
    ):
        self.state_token_pad_to = state_token_pad_to
        self.action_token_pad_to = action_token_pad_to
        self.definition_pad_to = definition_pad_to
        self.candidate_pad_to = candidate_pad_to
        self.side_term_pad_to = side_term_pad_to
        self.d_model = d_model
        self.embedder = TokenEmbedder(
            d_model=d_model,
            id_vocab_size=id_vocab_size,
            init_scale=init_scale,
        )
        self.encoder = TransformerEncoder(
            d_model=d_model,
            num_layers=num_attention_layers,
            num_heads=num_attention_heads,
            init_scale=init_scale,
        )
        self.target_decoder = LogitDecoder(
            d_model=d_model,
            output_size=1 + definition_pad_to,
            init_scale=init_scale,
        )
        self.candidate_decoder = LogitDecoder(
            d_model=d_model,
            output_size=candidate_pad_to,
            init_scale=init_scale,
        )
        self.mask_decoder = LogitDecoder(
            d_model=d_model,
            output_size=side_term_pad_to,
            init_scale=init_scale,
        )
        self._sample_target_batch = jax.jit(
            jax.vmap(
                self._sample_target_from_state_tokens,
                in_axes=(None, 0, 0, 0, 0),
            )
        )
        self._score_target_batch = jax.jit(
            jax.vmap(
                jax.value_and_grad(
                    self._score_target_from_state_tokens,
                ),
                in_axes=(None, 0, 0, 0, 0),
            )
        )
        self._sample_decision_batch = jax.jit(
            jax.vmap(
                self._sample_decision_from_tokens,
                in_axes=(None, 0, 0, 0, 0, 0, 0),
            )
        )
        self._score_decision_batch = jax.jit(
            jax.vmap(
                jax.value_and_grad(self._score_decision_from_tokens),
                in_axes=(None, 0, 0, 0, 0, 0, 0),
            )
        )

    def init_params(self, rng):
        embed_key, encoder_key, target_key, cand_key, mask_key = (
            jax.random.split(rng, 5)
        )
        example_tokens = {
            field: jnp.zeros((1,), dtype=jnp.int32)
            for field in TOKEN_FIELDS
        }
        example_mask = jnp.ones((1,), dtype=jnp.bool_)
        example_vectors = jnp.zeros((1, self.d_model), dtype=jnp.bfloat16)
        return {
            "embedder": self.embedder.init(
                embed_key,
                example_tokens,
            )["params"],
            "encoder": self.encoder.init(
                encoder_key,
                example_vectors,
                example_mask,
            )["params"],
            "target_decoder": self.target_decoder.init(
                target_key,
                example_vectors,
                example_mask,
            )["params"],
            "candidate_decoder": self.candidate_decoder.init(
                cand_key,
                example_vectors,
                example_mask,
            )["params"],
            "mask_decoder": self.mask_decoder.init(
                mask_key,
                example_vectors,
                example_mask,
            )["params"],
        }

    def sample_step(self, params, rng, states):
        if not states:
            raise ValueError("sample_step requires at least one state")

        batch = self._sample_transition_batch(params, rng, states)
        next_comps, applied = self._apply_sampled_decisions(batch)
        logp, grad_logp = self._score_sampled_transition(params, batch)
        next_masks = self._updated_target_masks(
            batch["target_mask"],
            batch["targets"],
            batch["spaces"],
            applied,
        )
        next_states = [
            SelectorState(comp=comp, target_mask=mask)
            for comp, mask in zip(next_comps, next_masks)
        ]
        return next_states, logp, grad_logp

    def score_step(self, params, transitions):
        batch = self._flatten_transitions(transitions)
        state_tokens, state_mask = self._computation_token_batch(
            batch["comps"],
        )
        action_tokens, action_mask = self._action_space_token_batch(
            batch["spaces"],
        )
        target_logp, target_grad = self._score_target_batch(
            params,
            state_tokens,
            state_mask,
            batch["target_mask"],
            batch["targets"],
        )
        decision_logp, decision_grad = self._score_decision_batch(
            params,
            state_tokens,
            state_mask,
            action_tokens,
            action_mask,
            batch["decisions"],
            batch["active"],
        )
        return (
            target_logp + decision_logp,
            _tree_add(target_grad, decision_grad),
        )

    def _sample_transition_batch(self, params, rng, states):
        comps = [state.comp for state in states]
        state_tokens, state_mask = self._computation_token_batch(comps)
        target_mask = self._target_mask_from_states(states)
        target_rng, decision_rng = jax.random.split(rng)
        targets = self._sample_targets(
            params,
            target_rng,
            state_tokens,
            state_mask,
            target_mask,
            len(states),
        )
        spaces = _query_action_spaces(comps, targets)
        active = _active_action_mask(targets, spaces)
        action_tokens, action_mask = self._optional_space_tokens(spaces)
        decisions = self._sample_decisions(
            params,
            decision_rng,
            state_tokens,
            state_mask,
            action_tokens,
            action_mask,
            active,
            len(states),
        )
        return {
            "comps": comps,
            "state_tokens": state_tokens,
            "state_mask": state_mask,
            "target_mask": target_mask,
            "targets": targets,
            "spaces": spaces,
            "active": active,
            "action_tokens": action_tokens,
            "action_mask": action_mask,
            "decisions": decisions,
        }

    def _sample_targets(
        self,
        params,
        rng,
        state_tokens,
        state_mask,
        target_mask,
        batch_size,
    ):
        return self._sample_target_batch(
            params,
            jax.random.split(rng, batch_size),
            state_tokens,
            state_mask,
            target_mask,
        )

    def _sample_decisions(
        self,
        params,
        rng,
        state_tokens,
        state_mask,
        action_tokens,
        action_mask,
        active,
        batch_size,
    ):
        return self._sample_decision_batch(
            params,
            jax.random.split(rng, batch_size),
            state_tokens,
            state_mask,
            action_tokens,
            action_mask,
            jnp.asarray(active, dtype=jnp.bool_),
        )

    def _apply_sampled_decisions(self, batch):
        python_decisions = self._decision_batch_to_python(
            batch["spaces"],
            batch["decisions"],
            batch["active"],
        )
        validate_decisions_for_batch(batch["spaces"], python_decisions)
        next_comps = [comp.clone() for comp in batch["comps"]]
        applied = apply_decisions_for_batch(
            next_comps,
            batch["spaces"],
            python_decisions,
        )
        return next_comps, applied

    def _score_sampled_transition(self, params, batch):
        target_logp, target_grad = self._score_target_batch(
            params,
            batch["state_tokens"],
            batch["state_mask"],
            batch["target_mask"],
            batch["targets"],
        )
        decision_logp, decision_grad = self._score_decision_batch(
            params,
            batch["state_tokens"],
            batch["state_mask"],
            batch["action_tokens"],
            batch["action_mask"],
            batch["decisions"],
            jnp.asarray(batch["active"], dtype=jnp.bool_),
        )
        return (
            target_logp + decision_logp,
            _tree_add(target_grad, decision_grad),
        )

    def _computation_token_batch(
        self,
        comps: Sequence[TensorComputation],
    ):
        items = [
            tokenize_computation_snapshot(comp.snapshot())
            for comp in comps
        ]
        return stack_token_arrays(items, pad_to=self.state_token_pad_to)

    def _action_space_token_batch(
        self,
        spaces: Sequence[ActionSpace],
    ):
        snapshots = [space.snapshot() for space in spaces]
        return self._action_space_snapshot_token_batch(snapshots)

    def _optional_action_space_token_batch(
        self,
        spaces: Sequence[ActionSpace | None],
    ):
        snapshots = [
            _action_space_snapshot_or_dummy(space)
            for space in spaces
        ]
        return self._action_space_snapshot_token_batch(snapshots)

    def _optional_space_tokens(self, spaces):
        return self._optional_action_space_token_batch(spaces)

    def _action_space_snapshot_token_batch(self, snapshots: Sequence[dict]):
        items = [
            tokenize_action_space_snapshot(snapshot)
            for snapshot in snapshots
        ]
        return stack_token_arrays(items, pad_to=self.action_token_pad_to)

    def _target_mask_from_states(self, states: BatchedState):
        rows = [
            _normalized_target_mask(state.target_mask, self.definition_pad_to)
            for state in states
        ]
        return jnp.asarray(rows, dtype=jnp.bool_)

    def _updated_target_masks(
        self,
        target_mask,
        targets,
        spaces,
        applied,
    ):
        rows = []
        target_rows = jax.device_get(target_mask).tolist()
        target_slots = jax.device_get(targets).tolist()
        for row, target, space, did_apply in zip(
            target_rows,
            target_slots,
            spaces,
            applied,
        ):
            rows.append(
                _updated_target_mask(
                    row,
                    int(target),
                    space is None,
                    bool(did_apply),
                    self.definition_pad_to,
                )
            )
        return jnp.asarray(rows, dtype=jnp.bool_)

    def _decision_batch_to_python(self, spaces, decisions, active):
        candidates = jax.device_get(decisions["candidate_index"]).tolist()
        left = jax.device_get(decisions["left_mask"]).tolist()
        right = jax.device_get(decisions["right_mask"]).tolist()
        out = []
        for sample, space in enumerate(spaces):
            if not active[sample] or space is None:
                out.append(None)
                continue
            template = space.snapshot()["candidate_templates"][
                int(candidates[sample])
            ]
            out.append({
                "candidate_index": int(candidates[sample]),
                "left_mask": _bool_prefix(
                    left[sample],
                    _term_count(template, "left_definition"),
                ),
                "right_mask": _bool_prefix(
                    right[sample],
                    _term_count(template, "right_definition"),
                ),
            })
        return out

    def _flatten_transitions(self, transitions):
        comps = []
        spaces = []
        decisions = []
        targets = []
        for group in transitions:
            for choice in group.choices:
                comps.append(group.state)
                spaces.append(choice.action_space)
                decisions.append(choice.decision)
                targets.append(1 + choice.action_space.def_index)
        if not comps:
            raise ValueError("score_step requires at least one transition")
        return {
            "comps": comps,
            "spaces": spaces,
            "decisions": self._decision_list_to_arrays(decisions),
            "targets": jnp.asarray(targets, dtype=jnp.int32),
            "target_mask": jnp.ones(
                (len(comps), 1 + self.definition_pad_to),
                dtype=jnp.bool_,
            ),
            "active": jnp.ones((len(comps),), dtype=jnp.bool_),
        }

    def _decision_list_to_arrays(self, decisions):
        candidates = []
        left = []
        right = []
        for decision in decisions:
            candidates.append(int(decision["candidate_index"]))
            left.append(_pad_bool_mask(
                decision["left_mask"],
                self.side_term_pad_to,
            ))
            right.append(_pad_bool_mask(
                decision["right_mask"],
                self.side_term_pad_to,
            ))
        return {
            "candidate_index": jnp.asarray(candidates, dtype=jnp.int32),
            "left_mask": jnp.asarray(left, dtype=jnp.bool_),
            "right_mask": jnp.asarray(right, dtype=jnp.bool_),
        }

    def _encode_tokens(self, params, tokens, token_mask):
        vectors = self.embedder.apply(
            {"params": params["embedder"]},
            tokens,
        )
        return self.encoder.apply(
            {"params": params["encoder"]},
            vectors,
            token_mask,
        )

    def _target_logits_from_encoded(self, params, encoded, token_mask):
        return self.target_decoder.apply(
            {"params": params["target_decoder"]},
            encoded,
            token_mask,
        )

    def _candidate_logits_from_encoded(self, params, encoded, token_mask):
        return self.candidate_decoder.apply(
            {"params": params["candidate_decoder"]},
            encoded,
            token_mask,
        )

    def _mask_logits_from_encoded(self, params, encoded, token_mask):
        return self.mask_decoder.apply(
            {"params": params["mask_decoder"]},
            encoded,
            token_mask,
        )

    def _target_logits_from_state_tokens(
        self,
        params,
        state_tokens,
        state_mask,
    ):
        encoded = self._encode_tokens(params, state_tokens, state_mask)
        return self._target_logits_from_encoded(
            params,
            encoded,
            state_mask,
        )

    def _sample_target_from_state_tokens(
        self,
        params,
        rng,
        state_tokens,
        state_mask,
        target_mask,
    ):
        logits = self._target_logits_from_state_tokens(
            params,
            state_tokens,
            state_mask,
        )
        structural = self._target_mask_from_state_tokens(
            state_tokens,
            state_mask,
        )
        return _sample_categorical(rng, logits, structural & target_mask)

    def _score_target_from_state_tokens(
        self,
        params,
        state_tokens,
        state_mask,
        target_mask,
        target,
    ):
        logits = self._target_logits_from_state_tokens(
            params,
            state_tokens,
            state_mask,
        )
        structural = self._target_mask_from_state_tokens(
            state_tokens,
            state_mask,
        )
        active = jnp.any(target_mask)
        valid = _valid_when_active(structural & target_mask, active)
        score = _score_categorical(logits, valid, target)
        return jnp.where(active, score, 0.0)

    def _sample_decision_from_tokens(
        self,
        params,
        rng,
        state_tokens,
        state_mask,
        action_tokens,
        action_mask,
        active,
    ):
        cand_rng, left_rng, right_rng = jax.random.split(rng, 3)
        encoded, token_mask = self._encode_state_action_tokens(
            params,
            state_tokens,
            state_mask,
            action_tokens,
            action_mask,
        )
        candidate_context = token_mask
        candidate_logits = self._candidate_logits_from_encoded(
            params,
            encoded,
            candidate_context,
        )
        candidate_mask = self._candidate_mask_from_action_tokens(
            action_tokens,
            action_mask,
        )
        candidate = _sample_categorical(
            cand_rng,
            candidate_logits,
            candidate_mask & active,
        )

        left_context = self._chosen_candidate_context_mask(
            state_mask,
            action_tokens,
            action_mask,
            candidate,
        )
        mask_logits = self._mask_logits_from_encoded(
            params,
            encoded,
            left_context,
        )
        left_valid = self._side_mask_from_action_tokens(
            action_tokens,
            action_mask,
            candidate,
            SIDE.LEFT,
        ) & active
        left_mask = _sample_nonempty_mask(left_rng, mask_logits, left_valid)

        right_context = self._chosen_left_context_mask(
            state_mask,
            action_tokens,
            action_mask,
            candidate,
            left_mask,
        )
        mask_logits = self._mask_logits_from_encoded(
            params,
            encoded,
            right_context,
        )
        right_valid = self._side_mask_from_action_tokens(
            action_tokens,
            action_mask,
            candidate,
            SIDE.RIGHT,
        ) & active
        right_mask = _sample_nonempty_mask(
            right_rng,
            mask_logits,
            right_valid,
        )
        return {
            "candidate_index": candidate,
            "left_mask": left_mask,
            "right_mask": right_mask,
        }

    def _score_decision_from_tokens(
        self,
        params,
        state_tokens,
        state_mask,
        action_tokens,
        action_mask,
        decision,
        active,
    ):
        encoded, token_mask = self._encode_state_action_tokens(
            params,
            state_tokens,
            state_mask,
            action_tokens,
            action_mask,
        )
        candidate_context = token_mask
        candidate_logits = self._candidate_logits_from_encoded(
            params,
            encoded,
            candidate_context,
        )
        candidate_mask = self._candidate_mask_from_action_tokens(
            action_tokens,
            action_mask,
        )
        candidate_valid = _valid_when_active(candidate_mask, active)
        candidate = decision["candidate_index"]
        candidate_logp = _score_categorical(
            candidate_logits,
            candidate_valid,
            candidate,
        )

        left_context = self._chosen_candidate_context_mask(
            state_mask,
            action_tokens,
            action_mask,
            candidate,
        )
        mask_logits = self._mask_logits_from_encoded(
            params,
            encoded,
            left_context,
        )
        left_valid = self._side_mask_from_action_tokens(
            action_tokens,
            action_mask,
            candidate,
            SIDE.LEFT,
        ) & active
        left_logp = _score_nonempty_mask(
            mask_logits,
            left_valid,
            decision["left_mask"],
        )
        right_context = self._chosen_left_context_mask(
            state_mask,
            action_tokens,
            action_mask,
            candidate,
            decision["left_mask"],
        )
        mask_logits = self._mask_logits_from_encoded(
            params,
            encoded,
            right_context,
        )
        right_valid = self._side_mask_from_action_tokens(
            action_tokens,
            action_mask,
            candidate,
            SIDE.RIGHT,
        ) & active
        right_logp = _score_nonempty_mask(
            mask_logits,
            right_valid,
            decision["right_mask"],
        )
        total = candidate_logp + left_logp + right_logp
        return jnp.where(active, total, 0.0)

    def _encode_state_action_tokens(
        self,
        params,
        state_tokens,
        state_mask,
        action_tokens,
        action_mask,
    ):
        tokens = _concat_token_arrays(state_tokens, action_tokens)
        token_mask = jnp.concatenate([state_mask, action_mask], axis=0)
        return self._encode_tokens(params, tokens, token_mask), token_mask

    def _chosen_candidate_context_mask(
        self,
        state_mask,
        action_tokens,
        action_mask,
        candidate,
    ):
        chosen = (
            action_mask
            & (action_tokens["candidate_index"] == candidate)
        )
        return _concat_masks(state_mask, chosen)

    def _chosen_left_context_mask(
        self,
        state_mask,
        action_tokens,
        action_mask,
        candidate,
        left_mask,
    ):
        left_term_content = (
            action_mask
            & (action_tokens["candidate_index"] == candidate)
            & (action_tokens["side"] == int(SIDE.LEFT))
            & (action_tokens["term_index"] >= 0)
        )
        selected = _selected_slots(
            action_tokens["term_index"],
            left_mask,
        )
        return _concat_masks(state_mask, left_term_content & selected)

    def _target_mask_from_state_tokens(self, state_tokens, state_mask):
        is_def = state_mask & (
            state_tokens["token_kind"] == int(TOKEN_KIND.DEF_START)
        )
        defs = _slot_mask(
            state_tokens["def_index"],
            is_def,
            self.definition_pad_to,
        )
        return jnp.concatenate([jnp.ones((1,), dtype=jnp.bool_), defs])

    def _candidate_mask_from_action_tokens(self, action_tokens, action_mask):
        is_candidate = action_mask & (
            action_tokens["token_kind"] == int(TOKEN_KIND.CANDIDATE_START)
        )
        return _slot_mask(
            action_tokens["candidate_index"],
            is_candidate,
            self.candidate_pad_to,
        )

    def _side_mask_from_action_tokens(
        self,
        action_tokens,
        action_mask,
        candidate,
        side,
    ):
        is_term = (
            action_mask
            & (action_tokens["token_kind"] == int(TOKEN_KIND.TERM_START))
            & (action_tokens["candidate_index"] == candidate)
            & (action_tokens["side"] == int(side))
        )
        return _slot_mask(
            action_tokens["term_index"],
            is_term,
            self.side_term_pad_to,
        )


def _concat_token_arrays(left, right):
    return {
        field: jnp.concatenate([left[field], right[field]], axis=0)
        for field in TOKEN_FIELDS
    }


def _concat_masks(left, right):
    return jnp.concatenate([left, right], axis=0)


def _dummy_action_space_snapshot():
    return {"def_index": 0, "candidate_templates": []}


def _action_space_snapshot_or_dummy(space):
    if space is None:
        return _dummy_action_space_snapshot()
    return space.snapshot()


def _target_slots_to_defs(targets):
    return [
        None if int(target) == 0 else int(target) - 1
        for target in jax.device_get(targets).tolist()
    ]


def _query_action_spaces(comps, targets):
    return list(action_spaces_for_batch(comps, _target_slots_to_defs(targets)))


def _active_action_mask(targets, spaces):
    target_values = jax.device_get(targets).tolist()
    return [
        int(target) > 0 and space is not None
        for target, space in zip(target_values, spaces)
    ]


def _normalized_target_mask(mask, definition_pad_to: int):
    if mask is None:
        return [True] * (1 + definition_pad_to)
    values = [bool(value) for value in jax.device_get(mask).tolist()]
    if len(values) == definition_pad_to:
        return [True, *values]
    if len(values) == 1 + definition_pad_to:
        return values
    raise ValueError("target_mask has incompatible length")


def _updated_target_mask(
    mask,
    target: int,
    no_action_space: bool,
    applied: bool,
    definition_pad_to: int,
):
    out = [bool(value) for value in mask]
    if target <= 0:
        return [False] * len(out)
    if no_action_space:
        out[target] = False
        return out
    if not applied:
        return out
    def_index = target - 1
    defs = out[1:]
    defs = defs[:def_index] + [True, True, True] + defs[def_index + 1:]
    defs = defs[:definition_pad_to]
    defs += [False] * (definition_pad_to - len(defs))
    return [out[0], *defs]


def _term_count(template, side: str):
    return len(template[side]["terms"])


def _bool_prefix(values, length: int):
    return [bool(value) for value in values[:length]]


def _pad_bool_mask(values, width: int):
    values = [bool(value) for value in values]
    if len(values) > width:
        raise ValueError("decision mask exceeds side_term_pad_to")
    return values + [False] * (width - len(values))


def _tree_add(left, right):
    return jax.tree_util.tree_map(lambda x, y: x + y, left, right)


def _slot_mask(item_index, item_mask, width: int):
    slots = jnp.arange(width, dtype=jnp.int32)
    in_range = (item_index >= 0) & (item_index < width)
    keep = item_mask & in_range
    by_slot = keep[None, :] & (item_index[None, :] == slots[:, None])
    return jnp.any(by_slot, axis=1)


def _selected_slots(item_index, selected):
    safe = jnp.clip(item_index, 0, selected.shape[0] - 1)
    in_range = (item_index >= 0) & (item_index < selected.shape[0])
    return in_range & jnp.take(selected, safe)


def _sample_categorical(rng, logits, valid):
    valid, _ = _valid_with_fallback(valid)
    masked = jnp.where(valid, logits, -jnp.inf)
    return jax.random.categorical(rng, masked).astype(jnp.int32)


def _score_categorical(logits, valid, choice_index):
    valid, has_valid = _valid_with_fallback(valid)
    masked = jnp.where(valid, logits, -jnp.inf)
    log_probs = jax.nn.log_softmax(masked)
    safe = jnp.clip(choice_index, 0, logits.shape[0] - 1)
    logp = jnp.take(log_probs, safe)
    legal = jnp.take(valid, safe)
    in_range = (choice_index >= 0) & (choice_index < logits.shape[0])
    return jnp.where(has_valid & legal & in_range, logp, -jnp.inf)


def _valid_with_fallback(valid):
    has_valid = jnp.any(valid)
    fallback = jnp.arange(valid.shape[0]) == 0
    return jnp.where(has_valid, valid, fallback), has_valid


def _valid_when_active(valid, active):
    fallback = jnp.arange(valid.shape[0]) == 0
    return jnp.where(active, valid, fallback)


def _last_valid_mask(valid):
    remaining = jnp.cumsum(valid.astype(jnp.int32)[::-1])[::-1]
    return valid & (remaining == 1)


def _bit_logp(logit, keep):
    return jnp.where(
        keep,
        jax.nn.log_sigmoid(logit),
        jax.nn.log_sigmoid(-logit),
    )


def _sample_nonempty_mask(rng, logits, valid):
    has_valid = jnp.any(valid)
    sampled = jax.random.bernoulli(rng, jax.nn.sigmoid(logits))
    last_valid = _last_valid_mask(valid)

    def step(seen_keep, inputs):
        is_valid, is_last_valid, raw_keep = inputs
        forced = is_valid & is_last_valid & (~seen_keep)
        keep = jnp.where(is_valid, forced | raw_keep, False)
        return seen_keep | keep, keep

    _, mask = jax.lax.scan(
        step,
        jnp.asarray(False),
        (valid, last_valid, sampled),
    )
    return jnp.where(has_valid, mask, False)


def _score_nonempty_mask(logits, valid, mask):
    has_valid = jnp.any(valid)
    last_valid = _last_valid_mask(valid)

    def step(seen_keep, inputs):
        is_valid, is_last_valid, keep, logit = inputs
        forced = is_valid & is_last_valid & (~seen_keep)
        legal = jnp.where(is_valid, jnp.where(forced, keep, True), ~keep)
        logp = jnp.where(is_valid & (~forced), _bit_logp(logit, keep), 0.0)
        return seen_keep | (is_valid & keep), (logp, legal)

    _, (logps, legal_bits) = jax.lax.scan(
        step,
        jnp.asarray(False),
        (valid, last_valid, mask, logits),
    )
    ok = (
        jnp.all(legal_bits)
        & has_valid
        & jnp.any(mask & valid)
        & (~jnp.any(mask & (~valid)))
    )
    return jnp.where(has_valid, jnp.where(ok, jnp.sum(logps), -jnp.inf), 0.0)
