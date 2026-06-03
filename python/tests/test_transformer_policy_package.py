import importlib
import sys


def test_transformer_policy_imports_without_legacy_rl():
    sys.modules.pop("gristmill_rl", None)

    module = importlib.import_module("transformer_policy")

    assert module.__all__ == ()
    assert "gristmill_rl" not in sys.modules
