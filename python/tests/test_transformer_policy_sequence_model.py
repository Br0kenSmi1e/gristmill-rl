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
