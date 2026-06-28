import pytest

from gristmill_symbolics._training import TrainingError
import gristmill_symbolics.trainer.reinforce as reinforce
from gristmill_symbolics.trainer.reinforce import ReinforceTrainer


def test_reinforce_trainer_constructor_owns_batch_reward_baseline_and_optimizer():
    trainer = ReinforceTrainer(
        batch_size=2,
        learning_rate=1.0e-2,
        standardize_baseline=True,
        baseline_epsilon=1.0e-12,
    )

    assert trainer.batch_size == 2
    assert trainer.constructor_kwargs()["reward_kind"] == "log_flops_improvement"
    assert trainer.constructor_kwargs()["standardize_baseline"] is True
    assert trainer.constructor_kwargs()["baseline_epsilon"] == pytest.approx(1.0e-12)
    assert trainer.constructor_kwargs()["learning_rate"] == pytest.approx(1.0e-2)


@pytest.mark.parametrize(
    ("kwargs", "field_name"),
    [
        ({"batch_size": 0}, "batch_size"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"b1": 1.0}, "b1"),
        ({"b2": float("nan")}, "b2"),
        ({"eps": 0.0}, "eps"),
        ({"standardize_baseline": "yes"}, "standardize_baseline"),
        ({"baseline_epsilon": 0.0}, "baseline_epsilon"),
    ],
)
def test_reinforce_trainer_constructor_rejects_invalid_values(kwargs, field_name):
    values = {"batch_size": 2}
    values.update(kwargs)

    with pytest.raises(TrainingError, match=field_name):
        ReinforceTrainer(**values)


def test_reinforce_package_exports_only_concrete_trainer():
    assert reinforce.__all__ == ("ReinforceTrainer",)
    assert reinforce.ReinforceTrainer is ReinforceTrainer
    assert not hasattr(reinforce, "Reinforce" + "Trainer" + "Config")
