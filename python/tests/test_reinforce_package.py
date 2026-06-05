import importlib
import sys


def test_reinforce_training_imports_without_legacy_rl():
    sys.modules.pop("gristmill_rl", None)

    module = importlib.import_module("reinforce_training")

    assert module.__all__ == (
        "EpisodeTrace",
        "Stage1AttemptTrace",
        "StepTrace",
    )
    assert "gristmill_rl" not in sys.modules
