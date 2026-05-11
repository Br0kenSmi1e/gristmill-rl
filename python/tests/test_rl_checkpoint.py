import json

import numpy as np
import pytest

import gristmill_rl.checkpoint as checkpoint_module
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
    assert loaded.metadata.schema_version == 1
    assert loaded.metadata.hidden_dim == 16
    assert loaded.metadata.feature_config == feature_config
    assert loaded.metadata.metadata == {"tag": "initial"}
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
    assert loaded.metadata.schema_version == 1
    assert loaded.metadata.hidden_dim == 16
    assert loaded.metadata.feature_config == feature_config
    assert loaded.metadata.metadata == {"tag": "first"}
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
    loaded = load_checkpoint(path)
    assert loaded.metadata.schema_version == 1
    assert loaded.metadata.hidden_dim == 16
    assert loaded.metadata.feature_config == feature_config
    assert loaded.metadata.metadata == {"tag": "second"}


def test_save_checkpoint_overwrite_state_failure_preserves_existing_checkpoint(
    tmp_path, monkeypatch
):
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

    def fail_save(*args, **kwargs):
        raise RuntimeError("state save failed")

    monkeypatch.setattr(checkpoint_module.ocp.PyTreeCheckpointer, "save", fail_save)

    with pytest.raises(RuntimeError, match="state save failed"):
        save_checkpoint(
            path,
            model=model,
            feature_config=feature_config,
            hidden_dim=16,
            metadata={"tag": "second"},
            overwrite=True,
        )

    loaded = load_checkpoint(path)
    assert loaded.metadata.metadata == {"tag": "first"}


def test_save_checkpoint_overwrite_metadata_failure_preserves_existing_checkpoint(
    tmp_path, monkeypatch
):
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

    def fail_write_metadata(*args, **kwargs):
        raise RuntimeError("metadata write failed")

    monkeypatch.setattr(checkpoint_module, "_write_metadata", fail_write_metadata)

    with pytest.raises(RuntimeError, match="metadata write failed"):
        save_checkpoint(
            path,
            model=model,
            feature_config=feature_config,
            hidden_dim=16,
            metadata={"tag": "second"},
            overwrite=True,
        )

    loaded = load_checkpoint(path)
    assert loaded.metadata.metadata == {"tag": "first"}


def test_save_checkpoint_removes_new_checkpoint_when_metadata_write_fails(
    tmp_path, monkeypatch
):
    _, _, feature_config = checkpoint_features()
    model = PolicyValueModel(hidden_dim=16, rng_seed=0)
    path = tmp_path / "checkpoint"

    def fail_write_metadata(*args, **kwargs):
        raise RuntimeError("metadata write failed")

    monkeypatch.setattr(checkpoint_module, "_write_metadata", fail_write_metadata)

    with pytest.raises(RuntimeError, match="metadata write failed"):
        save_checkpoint(
            path,
            model=model,
            feature_config=feature_config,
            hidden_dim=16,
        )

    assert not path.exists()


def test_save_checkpoint_rejects_hidden_dim_mismatch_before_writing(tmp_path):
    _, _, feature_config = checkpoint_features()
    model = PolicyValueModel(hidden_dim=16, rng_seed=0)
    path = tmp_path / "checkpoint"

    with pytest.raises(ValueError, match="hidden_dim 32 does not match model hidden_dim 16"):
        save_checkpoint(
            path,
            model=model,
            feature_config=feature_config,
            hidden_dim=32,
        )

    assert not path.exists()


@pytest.mark.parametrize(
    ("metadata_payload", "message"),
    [
        ([], "checkpoint metadata must be an object"),
        (
            {
                "schema_version": 1,
                "model": [],
                "features": {
                    "max_candidates": 4,
                    "max_left_terms": 1,
                    "max_right_terms": 2,
                },
                "metadata": {},
            },
            "checkpoint metadata.model must be an object",
        ),
        (
            {
                "schema_version": 1,
                "model": {"class": "PolicyValueModel", "hidden_dim": 0},
                "features": {
                    "max_candidates": 4,
                    "max_left_terms": 1,
                    "max_right_terms": 2,
                },
                "metadata": {},
            },
            "checkpoint metadata.model.hidden_dim must be a positive integer",
        ),
        (
            {
                "schema_version": 1,
                "model": {"class": "PolicyValueModel", "hidden_dim": True},
                "features": {
                    "max_candidates": 4,
                    "max_left_terms": 1,
                    "max_right_terms": 2,
                },
                "metadata": {},
            },
            "checkpoint metadata.model.hidden_dim must be a positive integer",
        ),
        (
            {
                "schema_version": 1,
                "model": {"class": "PolicyValueModel", "hidden_dim": 16},
                "features": [],
                "metadata": {},
            },
            "checkpoint metadata.features must be an object",
        ),
        (
            {
                "schema_version": 1,
                "model": {"class": "PolicyValueModel", "hidden_dim": 16},
                "features": {
                    "max_candidates": 0,
                    "max_left_terms": 1,
                    "max_right_terms": 2,
                },
                "metadata": {},
            },
            "checkpoint metadata.features.max_candidates must be a positive integer",
        ),
        (
            {
                "schema_version": 1,
                "model": {"class": "PolicyValueModel", "hidden_dim": 16},
                "features": {
                    "max_candidates": 4,
                    "max_left_terms": False,
                    "max_right_terms": 2,
                },
                "metadata": {},
            },
            "checkpoint metadata.features.max_left_terms must be a positive integer",
        ),
        (
            {
                "schema_version": 1,
                "model": {"class": "PolicyValueModel", "hidden_dim": 16},
                "features": {
                    "max_candidates": 4,
                    "max_left_terms": 1,
                    "max_right_terms": "2",
                },
                "metadata": {},
            },
            "checkpoint metadata.features.max_right_terms must be a positive integer",
        ),
    ],
)
def test_load_checkpoint_rejects_malformed_metadata_with_value_error(
    tmp_path, metadata_payload, message
):
    path = tmp_path / "checkpoint"
    path.mkdir()
    (path / "metadata.json").write_text(json.dumps(metadata_payload))

    with pytest.raises(ValueError, match=message):
        load_checkpoint(path)


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
