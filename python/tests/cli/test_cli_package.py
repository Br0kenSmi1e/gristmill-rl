import gristmill_symbolics.cli as cli
from gristmill_symbolics.cli import checkpoint, train, train_state


def test_cli_package_is_orchestration_marker_not_training_api_export():
    assert cli.__doc__ == "Command-line training orchestration."
    for name in (
        "PolicyConfig",
        "CurrentTransformerModelConfig",
        "OptimizerConfig",
        "ReinforceTrainerConfig",
        "RewardConfig",
        "BaselineConfig",
    ):
        assert not hasattr(cli, name)


def test_cli_modules_expose_current_checkpoint_and_train_state_surface():
    assert checkpoint.CHECKPOINT_SCHEMA_VERSION == 3
    assert hasattr(checkpoint, "save_checkpoint")
    assert hasattr(checkpoint, "load_checkpoint")
    assert hasattr(train, "main")
    assert hasattr(train_state, "init_train_state")
    assert hasattr(train_state, "advance_train_state")
