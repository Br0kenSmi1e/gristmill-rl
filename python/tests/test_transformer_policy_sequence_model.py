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
    with_prefix = np.asarray(
        scorer.score_next(context, (T("CAND", candidate_index=0),), legal)
    )

    assert without_prefix.shape == with_prefix.shape
    assert not np.array_equal(without_prefix, with_prefix)


def test_sequence_model_scores_legal_tokens_independent_of_legal_order():
    scorer = CausalTransformerScorer(
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        mlp_dim=32,
        rngs=nnx.Rngs(2),
    )
    context = (T("STATE_START"), T("STATE_END"))
    stop = T("STOP")
    rewrite = T("DEF", def_index=0)

    original_order = (stop, rewrite)
    reversed_order = (rewrite, stop)

    original_scores = dict(
        zip(original_order, np.asarray(scorer.score_next(context, (), original_order)))
    )
    reversed_scores = dict(
        zip(reversed_order, np.asarray(scorer.score_next(context, (), reversed_order)))
    )

    for token in original_order:
        np.testing.assert_allclose(original_scores[token], reversed_scores[token])


def test_sequence_model_rejects_empty_legal_set():
    scorer = CausalTransformerScorer(
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        mlp_dim=32,
        rngs=nnx.Rngs(3),
    )

    with pytest.raises(ValueError, match="legal_next_tokens must not be empty"):
        scorer.score_next((T("STATE_START"),), (), ())


def test_sequence_model_rejects_empty_context_plus_prefix():
    scorer = CausalTransformerScorer(
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        mlp_dim=32,
        rngs=nnx.Rngs(4),
    )

    with pytest.raises(ValueError, match="context plus prefix must not be empty"):
        scorer.score_next((), (), (T("STOP"),))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"num_heads": 0}, "num_heads must be positive"),
        ({"hidden_dim": 18, "num_heads": 4}, "hidden_dim must be divisible by num_heads"),
        ({"num_layers": 0}, "num_layers must be positive"),
    ),
)
def test_sequence_model_validates_constructor_arguments(kwargs, message):
    params = dict(hidden_dim=16, num_heads=4, num_layers=1, mlp_dim=32)
    params.update(kwargs)

    with pytest.raises(ValueError, match=message):
        CausalTransformerScorer(**params, rngs=nnx.Rngs(5))
