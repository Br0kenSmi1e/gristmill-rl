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
