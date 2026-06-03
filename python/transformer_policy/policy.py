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
