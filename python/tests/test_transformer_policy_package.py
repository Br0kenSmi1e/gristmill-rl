import importlib
import sys


def test_transformer_policy_imports_without_legacy_rl(monkeypatch):
    monkeypatch.delitem(sys.modules, "gristmill_rl", raising=False)
    monkeypatch.delitem(sys.modules, "transformer_policy", raising=False)

    module = importlib.import_module("transformer_policy")

    assert module.__all__ == ()
    assert "gristmill_rl" not in sys.modules
