import importlib

import pytest


def test_new_public_import_surface_exists():
    model = importlib.import_module("gristmill_symbolics.model")
    model_protocols = importlib.import_module("gristmill_symbolics.model.protocols")
    trainer = importlib.import_module("gristmill_symbolics.trainer")
    trainer_protocols = importlib.import_module("gristmill_symbolics.trainer.protocols")
    model_pkg = importlib.import_module(
        "gristmill_symbolics.model.transformer_action_selector"
    )
    trainer_pkg = importlib.import_module("gristmill_symbolics.trainer.reinforce")
    train_state = importlib.import_module("gristmill_symbolics.cli.train_state")
    checkpoint = importlib.import_module("gristmill_symbolics.cli.checkpoint")

    assert model.__all__ == ("ExpressionModel",)
    assert trainer.__all__ == ("Trainer",)
    assert hasattr(model, "ExpressionModel")
    assert hasattr(model_protocols, "ExpressionModel")
    assert hasattr(trainer, "Trainer")
    assert hasattr(trainer_protocols, "Trainer")
    assert hasattr(model_pkg, "TransformerActionSelectorModel")
    assert hasattr(trainer_pkg, "ReinforceTrainer")
    assert hasattr(train_state, "init_train_state")
    assert hasattr(train_state, "advance_train_state")
    assert hasattr(checkpoint, "save_checkpoint")
    assert hasattr(checkpoint, "load_checkpoint")


@pytest.mark.parametrize(
    "module_name",
    [
        "gristmill_symbolics.policy",
        "gristmill_symbolics.reinforce",
    ],
)
def test_old_public_training_packages_are_not_supported(module_name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_no_public_config_dataclasses_are_exported():
    model_pkg = importlib.import_module(
        "gristmill_symbolics.model.transformer_action_selector"
    )
    trainer_pkg = importlib.import_module("gristmill_symbolics.trainer.reinforce")

    assert model_pkg.__all__ == ("TransformerActionSelectorModel",)
    assert trainer_pkg.__all__ == ("ReinforceTrainer",)

    forbidden = {
        "PolicyConfig",
        "CurrentTransformerModelConfig",
        "TransformerActionSelectorConfig",
        "OptimizerConfig",
        "RewardConfig",
        "BaselineConfig",
        "ReinforceTrainerConfig",
    }
    assert forbidden.isdisjoint(set(getattr(model_pkg, "__all__", ())))
    assert forbidden.isdisjoint(set(getattr(trainer_pkg, "__all__", ())))
    for name in forbidden:
        assert not hasattr(model_pkg, name)
        assert not hasattr(trainer_pkg, name)
