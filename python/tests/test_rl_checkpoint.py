import json

import numpy as np
import pytest

from gristmill_rl.actions import first_full_mask_action
from gristmill_rl.checkpoint import load_checkpoint, save_checkpoint
from gristmill_rl.features import FeatureConfig, extract_features
from gristmill_rl.model import PolicyValueModel, TrainConfig, train_step

from .rl_fixtures import actionable_space


def checkpoint_features(config: FeatureConfig | None = None):
    feature_config = config or FeatureConfig(
        max_candidates=4, max_left_terms=1, max_right_terms=2
    )
    comp, space = actionable_space()
    return (
        extract_features(
            comp_snapshot=comp.snapshot(),
            action_space_snapshot=space.snapshot(),
            start_from=0,
            log_total_flops=comp.log_total_flops(),
            config=feature_config,
        ),
        first_full_mask_action(space.snapshot()),
        feature_config,
    )


def assert_outputs_close(left, right):
    np.testing.assert_allclose(left.candidate_logits, right.candidate_logits)
    np.testing.assert_allclose(left.left_logits, right.left_logits)
    np.testing.assert_allclose(left.right_logits, right.right_logits)
    np.testing.assert_allclose(left.value, right.value)


def test_checkpoint_load_restores_model_outputs_on_fixed_features(tmp_path):
    features, _, feature_config = checkpoint_features()
    model = PolicyValueModel(hidden_dim=16, rng_seed=123)
    expected = model(features)

    save_checkpoint(
        tmp_path / "checkpoint",
        model=model,
        feature_config=feature_config,
        hidden_dim=16,
        metadata={"tag": "initial"},
    )

    loaded = load_checkpoint(tmp_path / "checkpoint")

    assert loaded.feature_config == feature_config
    assert loaded.metadata == {"tag": "initial"}
    assert_outputs_close(loaded.model(features), expected)


def test_checkpoint_load_restores_trained_parameters(tmp_path):
    features, action, feature_config = checkpoint_features()
    model = PolicyValueModel(hidden_dim=16, rng_seed=0)
    train_step(
        model,
        batch=[
            {
                "features": features,
                "actions": [action],
                "policy_target": np.asarray([1.0], dtype=np.float32),
                "value_target": 0.5,
            }
        ],
        config=TrainConfig(learning_rate=1e-2),
    )
    expected = model(features)

    save_checkpoint(
        tmp_path / "checkpoint",
        model=model,
        feature_config=feature_config,
        hidden_dim=16,
    )

    fresh = PolicyValueModel(hidden_dim=16, rng_seed=0)
    assert not np.allclose(fresh(features).value, expected.value)

    loaded = load_checkpoint(tmp_path / "checkpoint")

    assert_outputs_close(loaded.model(features), expected)


def test_save_checkpoint_refuses_existing_directory_without_overwrite(tmp_path):
    features, _, feature_config = checkpoint_features()
    model = PolicyValueModel(hidden_dim=16, rng_seed=0)
    path = tmp_path / "checkpoint"
    save_checkpoint(
        path,
        model=model,
        feature_config=feature_config,
        hidden_dim=16,
        metadata={"tag": "first"},
    )

    with pytest.raises(FileExistsError, match="already exists"):
        save_checkpoint(
            path,
            model=model,
            feature_config=feature_config,
            hidden_dim=16,
            metadata={"tag": "second"},
        )

    loaded = load_checkpoint(path)
    assert loaded.metadata == {"tag": "first"}
    assert_outputs_close(loaded.model(features), model(features))


def test_save_checkpoint_overwrite_replaces_metadata(tmp_path):
    _, _, feature_config = checkpoint_features()
    model = PolicyValueModel(hidden_dim=16, rng_seed=0)
    path = tmp_path / "checkpoint"
    save_checkpoint(
        path,
        model=model,
        feature_config=feature_config,
        hidden_dim=16,
        metadata={"tag": "first"},
    )

    save_checkpoint(
        path,
        model=model,
        feature_config=feature_config,
        hidden_dim=16,
        metadata={"tag": "second"},
        overwrite=True,
    )

    metadata = json.loads((path / "metadata.json").read_text())
    assert metadata["metadata"] == {"tag": "second"}
    assert load_checkpoint(path).metadata == {"tag": "second"}


def test_load_checkpoint_rejects_unknown_schema(tmp_path):
    _, _, feature_config = checkpoint_features()
    model = PolicyValueModel(hidden_dim=16, rng_seed=0)
    path = tmp_path / "checkpoint"
    save_checkpoint(
        path,
        model=model,
        feature_config=feature_config,
        hidden_dim=16,
    )
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["schema_version"] = 999
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="Unsupported checkpoint schema_version 999"):
        load_checkpoint(path)
