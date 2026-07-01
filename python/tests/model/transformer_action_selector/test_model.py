import jax
import jax.numpy as jnp
import pytest
from flax.core import freeze, unfreeze

from gristmill_symbolics import TensorComputation, action_space_for_def
from gristmill_symbolics.model.transformer_action_selector import (
    TransformerActionSelectorModel,
)
from gristmill_symbolics.model.transformer_action_selector.model import (
    SelectorChoice,
    SelectorState,
    SelectorTransitions,
)
from tests.test_bindings import (
    BASIC_FIXTURE,
    actionable_json,
    first_full_decision,
)


def _model(**overrides):
    values = {
        "state_token_pad_to": 256,
        "action_token_pad_to": 512,
        "definition_pad_to": 8,
        "candidate_pad_to": 16,
        "side_term_pad_to": 16,
        "d_model": 8,
        "num_attention_heads": 1,
        "id_vocab_size": 32,
    }
    values.update(overrides)
    return TransformerActionSelectorModel(**values)


def _actionable_comp():
    return TensorComputation.from_json_string(actionable_json())


def _basic_comp():
    return TensorComputation.load_json(BASIC_FIXTURE)


def _actionable_transition():
    comp = _actionable_comp()
    space = action_space_for_def(comp, 0)
    return SelectorTransitions(
        state=comp,
        choices=[
            SelectorChoice(
                action_space=space,
                decision=first_full_decision(space),
                logp=-1.0,
            )
        ],
    )


def _assert_batched_grad_tree(grad, params, batch_size):
    assert jax.tree_util.tree_structure(grad) == (
        jax.tree_util.tree_structure(params)
    )
    for leaf in jax.tree_util.tree_leaves(grad):
        assert leaf.shape[0] == batch_size
        assert jnp.all(jnp.isfinite(leaf))


def _force_target_slot(params, slot):
    mutable = unfreeze(params)
    bias = mutable["target_decoder"]["logits"]["bias"]
    forced = jnp.full_like(bias, -100.0).at[slot].set(100.0)
    mutable["target_decoder"]["logits"]["bias"] = forced
    return freeze(mutable)


def test_init_params_returns_current_network_tree():
    model = _model(definition_pad_to=5, candidate_pad_to=7, side_term_pad_to=9)

    params = model.init_params(jax.random.PRNGKey(0))

    assert set(params) == {
        "embedder",
        "encoder",
        "target_decoder",
        "candidate_decoder",
        "mask_decoder",
    }
    assert params["target_decoder"]["logits"]["kernel"].shape[-1] == 6
    assert params["candidate_decoder"]["logits"]["kernel"].shape[-1] == 7
    assert params["mask_decoder"]["logits"]["kernel"].shape[-1] == 9


def test_score_step_scores_grouped_transitions_with_per_sample_grads():
    model = _model()
    params = model.init_params(jax.random.PRNGKey(1))
    transitions = [_actionable_transition()]

    logp, grad = model.score_step(params, transitions)

    assert logp.shape == (1,)
    assert jnp.all(jnp.isfinite(logp))
    _assert_batched_grad_tree(grad, params, batch_size=1)


def test_score_step_flattens_multiple_choices_per_state():
    model = _model()
    params = model.init_params(jax.random.PRNGKey(2))
    transition = _actionable_transition()
    transitions = [
        SelectorTransitions(
            state=transition.state,
            choices=[transition.choices[0], transition.choices[0]],
        )
    ]

    logp, grad = model.score_step(params, transitions)

    assert logp.shape == (2,)
    _assert_batched_grad_tree(grad, params, batch_size=2)


def test_score_step_flattens_separate_transition_groups():
    model = _model()
    params = model.init_params(jax.random.PRNGKey(21))
    transitions = [_actionable_transition(), _actionable_transition()]

    logp, grad = model.score_step(params, transitions)

    assert logp.shape == (2,)
    assert jnp.all(jnp.isfinite(logp))
    _assert_batched_grad_tree(grad, params, batch_size=2)


def test_score_step_rejects_empty_transition_batch():
    model = _model()
    params = model.init_params(jax.random.PRNGKey(3))

    with pytest.raises(ValueError, match="score_step requires"):
        model.score_step(params, [])


def test_sample_step_returns_next_selector_states_and_grads():
    model = _model()
    params = model.init_params(jax.random.PRNGKey(4))
    states = [
        SelectorState(comp=_actionable_comp()),
        SelectorState(comp=_basic_comp()),
    ]

    next_states, logp, grad = model.sample_step(
        params,
        jax.random.PRNGKey(5),
        states,
    )

    assert len(next_states) == 2
    assert all(isinstance(state, SelectorState) for state in next_states)
    assert all(state.comp is not original.comp for state, original in zip(
        next_states,
        states,
    ))
    assert logp.shape == (2,)
    assert jnp.all(jnp.isfinite(logp))
    _assert_batched_grad_tree(grad, params, batch_size=2)


def test_sample_step_stop_target_leaves_comp_and_mask_unchanged():
    model = _model()
    params = _force_target_slot(model.init_params(jax.random.PRNGKey(41)), 0)
    comp = _actionable_comp()
    before = comp.snapshot()
    target_mask = jnp.asarray(
        [True, True, False, True, False, False, False, False, False],
        dtype=jnp.bool_,
    )
    states = [SelectorState(comp=comp, target_mask=target_mask)]

    next_states, logp, grad = model.sample_step(
        params,
        jax.random.PRNGKey(42),
        states,
    )

    assert next_states[0].comp.snapshot() == before
    assert jax.device_get(next_states[0].target_mask).tolist() == (
        jax.device_get(target_mask).tolist()
    )
    assert jnp.all(jnp.isfinite(logp))
    _assert_batched_grad_tree(grad, params, batch_size=1)


def test_sample_step_respects_disabled_target_mask_slot():
    model = _model()
    params = _force_target_slot(model.init_params(jax.random.PRNGKey(51)), 1)
    comp = _actionable_comp()
    before = comp.snapshot()
    target_mask = jnp.asarray(
        [True, False, False, False, False, False, False, False, False],
        dtype=jnp.bool_,
    )
    states = [SelectorState(comp=comp, target_mask=target_mask)]

    next_states, logp, _ = model.sample_step(
        params,
        jax.random.PRNGKey(52),
        states,
    )

    assert next_states[0].comp.snapshot() == before
    assert jax.device_get(next_states[0].target_mask).tolist() == (
        jax.device_get(target_mask).tolist()
    )
    assert jnp.all(jnp.isfinite(logp))


def test_sample_step_marks_missing_action_space_target_unavailable():
    model = _model()
    params = _force_target_slot(model.init_params(jax.random.PRNGKey(6)), 1)
    states = [SelectorState(comp=_basic_comp())]

    next_states, logp, _ = model.sample_step(
        params,
        jax.random.PRNGKey(7),
        states,
    )

    mask = jax.device_get(next_states[0].target_mask).tolist()
    assert logp.shape == (1,)
    assert mask[0] is True
    assert mask[1] is False


def test_sample_step_successful_rewrite_expands_target_mask():
    model = _model()
    params = _force_target_slot(model.init_params(jax.random.PRNGKey(61)), 1)
    comp = _actionable_comp()
    before = comp.snapshot()
    target_mask = jnp.asarray(
        [True, True, False, False, False, False, False, False, False],
        dtype=jnp.bool_,
    )

    next_states, logp, grad = model.sample_step(
        params,
        jax.random.PRNGKey(62),
        [SelectorState(comp=comp, target_mask=target_mask)],
    )

    assert next_states[0].comp.snapshot() != before
    assert len(next_states[0].comp.snapshot()["definitions"]) == 3
    assert jax.device_get(next_states[0].target_mask).tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert jnp.all(jnp.isfinite(logp))
    _assert_batched_grad_tree(grad, params, batch_size=1)


def test_sample_step_rejects_empty_state_batch():
    model = _model()
    params = model.init_params(jax.random.PRNGKey(8))

    with pytest.raises(ValueError, match="sample_step requires"):
        model.sample_step(params, jax.random.PRNGKey(9), [])


def test_sample_step_reports_state_token_padding_too_small():
    model = _model(state_token_pad_to=1)
    params = model.init_params(jax.random.PRNGKey(10))

    with pytest.raises(ValueError, match="cannot pad token arrays"):
        model.sample_step(
            params,
            jax.random.PRNGKey(11),
            [SelectorState(comp=_actionable_comp())],
        )


def test_score_step_reports_action_token_padding_too_small():
    model = _model(action_token_pad_to=1)
    params = model.init_params(jax.random.PRNGKey(12))

    with pytest.raises(ValueError, match="cannot pad token arrays"):
        model.score_step(params, [_actionable_transition()])
