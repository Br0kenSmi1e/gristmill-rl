import inspect
import json

import numpy as np
import pytest
from flax import nnx

import reinforce_training.checkpoint as checkpoint_module
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


def _policy_config() -> PolicyConfig:
    return PolicyConfig(hidden_dim=16, num_heads=4, num_layers=1, mlp_dim=32)


def _save_minimal_checkpoint(path, *, metadata=None, overwrite=False, update_count=0):
    policy_config = _policy_config()
    train_config = TrainConfig()
    scorer = policy_config.create_scorer(seed=0)
    optimizer = create_optimizer(scorer, train_config)
    save_checkpoint(
        path,
        scorer=scorer,
        optimizer=optimizer,
        policy_config=policy_config,
        train_config=train_config,
        rollout_config=RolloutConfig(),
        update_count=update_count,
        seed=0,
        metadata=metadata,
        overwrite=overwrite,
    )
    return scorer


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


def test_checkpoint_overwrite_replaces_metadata(tmp_path):
    path = tmp_path / "checkpoint"
    _save_minimal_checkpoint(path, metadata={"tag": "first"})

    _save_minimal_checkpoint(path, metadata={"tag": "second"}, overwrite=True)

    loaded = load_checkpoint(path)
    assert loaded.metadata == {"tag": "second"}
    metadata = json.loads((path / "metadata.json").read_text())
    assert metadata["metadata"] == {"tag": "second"}


def test_checkpoint_overwrite_state_failure_preserves_existing_checkpoint(
    tmp_path, monkeypatch
):
    path = tmp_path / "checkpoint"
    scorer = _save_minimal_checkpoint(path, metadata={"tag": "first"})
    expected = _score_vector(scorer)

    def fail_save(*args, **kwargs):
        raise RuntimeError("state save failed")

    monkeypatch.setattr(checkpoint_module.ocp.PyTreeCheckpointer, "save", fail_save)

    with pytest.raises(RuntimeError, match="state save failed"):
        _save_minimal_checkpoint(path, metadata={"tag": "second"}, overwrite=True)

    loaded = load_checkpoint(path)
    assert loaded.metadata == {"tag": "first"}
    np.testing.assert_allclose(_score_vector(loaded.scorer), expected)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 999, "Unsupported checkpoint schema_version 999"),
        ("package", "other", "Unsupported checkpoint package 'other'"),
        ("model_class", "OtherModel", "Unsupported checkpoint model_class 'OtherModel'"),
    ],
)
def test_load_checkpoint_rejects_bad_identity_metadata(
    tmp_path, field, value, message
):
    path = tmp_path / "checkpoint"
    _save_minimal_checkpoint(path)
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match=message):
        load_checkpoint(path)


def test_save_checkpoint_rejects_negative_update_count_before_writing(tmp_path):
    path = tmp_path / "missing-parent" / "checkpoint"
    policy_config = _policy_config()
    scorer = policy_config.create_scorer(seed=0)
    optimizer = create_optimizer(scorer, TrainConfig())

    with pytest.raises(ValueError, match="update_count must be non-negative"):
        save_checkpoint(
            path,
            scorer=scorer,
            optimizer=optimizer,
            policy_config=policy_config,
            train_config=TrainConfig(),
            rollout_config=RolloutConfig(),
            update_count=-1,
            seed=0,
        )

    assert not path.exists()
    assert not path.parent.exists()


def test_save_checkpoint_rejects_non_json_metadata_before_writing(tmp_path):
    path = tmp_path / "missing-parent" / "checkpoint"
    policy_config = _policy_config()
    scorer = policy_config.create_scorer(seed=0)
    optimizer = create_optimizer(scorer, TrainConfig())

    with pytest.raises(ValueError, match="checkpoint metadata must be JSON serializable"):
        save_checkpoint(
            path,
            scorer=scorer,
            optimizer=optimizer,
            policy_config=policy_config,
            train_config=TrainConfig(),
            rollout_config=RolloutConfig(),
            update_count=0,
            seed=0,
            metadata={"bad": {1, 2}},
        )

    assert not path.exists()
    assert not path.parent.exists()


def test_reinforce_checkpoint_does_not_import_gristmill_rl():
    assert "gristmill_rl" not in inspect.getsource(checkpoint_module)
