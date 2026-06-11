def test_existing_extension_exports_still_import_from_package_root():
    import gristmill_symbolics

    assert hasattr(gristmill_symbolics, "TensorComputation")
    assert hasattr(gristmill_symbolics, "RewriteState")
    assert hasattr(gristmill_symbolics, "RewriteStateRow")
    assert hasattr(gristmill_symbolics, "validate_decision")


def test_policy_package_imports_without_training_modules(monkeypatch):
    import sys

    monkeypatch.delitem(sys.modules, "reinforce_training", raising=False)
    monkeypatch.delitem(sys.modules, "gristmill_symbolics.policy", raising=False)

    import gristmill_symbolics.policy as policy

    assert "reinforce_training" not in sys.modules
    assert policy.__all__ == (
        "tokenize_state_snapshot",
        "tokenize_action_space_snapshot",
        "sample_target",
        "score_target",
        "sample_action",
        "score_action",
    )
