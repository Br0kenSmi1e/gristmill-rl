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
    assert not hasattr(episode.steps[0], "action_space")


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
    assert all(
        not hasattr(step, "action_space")
        for episode in episodes
        for step in episode.steps
    )


def test_collect_episode_batch_uses_deterministic_update_offset_seeds():
    config = PolicyConfig(hidden_dim=16, num_heads=4, num_layers=1, mlp_dim=32)
    scorer = config.create_scorer(seed=0)

    episodes = collect_episode_batch(
        input_json=actionable_json(),
        scorer=scorer,
        policy_config=config,
        rollout_config=RolloutConfig(max_steps=1),
        update_index=3,
        batch_size=2,
        num_workers=1,
        seed=10,
    )

    assert [episode.episode_seed for episode in episodes] == [16, 17]


def test_policy_config_validates_scorer_shape_before_launch():
    with pytest.raises(ValueError, match="hidden_dim must be positive integer"):
        PolicyConfig(hidden_dim=0)

    with pytest.raises(ValueError, match="hidden_dim must be divisible by num_heads"):
        PolicyConfig(hidden_dim=15, num_heads=4)


def test_rollout_config_validates_positive_values():
    with pytest.raises(ValueError, match="max_steps must be positive"):
        RolloutConfig(max_steps=0)


def test_collect_episode_batch_validates_batch_arguments():
    config = PolicyConfig(hidden_dim=16, num_heads=4, num_layers=1, mlp_dim=32)
    scorer = config.create_scorer(seed=0)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        collect_episode_batch(
            input_json=actionable_json(),
            scorer=scorer,
            policy_config=config,
            rollout_config=RolloutConfig(max_steps=1),
            update_index=0,
            batch_size=0,
            num_workers=1,
            seed=10,
        )

    with pytest.raises(ValueError, match="num_workers must be positive"):
        collect_episode_batch(
            input_json=actionable_json(),
            scorer=scorer,
            policy_config=config,
            rollout_config=RolloutConfig(max_steps=1),
            update_index=0,
            batch_size=1,
            num_workers=0,
            seed=10,
        )

    with pytest.raises(ValueError, match="update_index must be non-negative"):
        collect_episode_batch(
            input_json=actionable_json(),
            scorer=scorer,
            policy_config=config,
            rollout_config=RolloutConfig(max_steps=1),
            update_index=-1,
            batch_size=1,
            num_workers=1,
            seed=10,
        )


def test_collect_episode_batch_adds_episode_context_to_worker_failures():
    config = PolicyConfig(hidden_dim=16, num_heads=4, num_layers=1, mlp_dim=32)
    scorer = config.create_scorer(seed=0)

    with pytest.raises(RuntimeError, match="episode_index=0.*episode_seed=10"):
        collect_episode_batch(
            input_json="not json",
            scorer=scorer,
            policy_config=config,
            rollout_config=RolloutConfig(max_steps=1),
            update_index=0,
            batch_size=1,
            num_workers=2,
            seed=10,
        )
