EXPECTED_POLICY_EXPORTS = (
    "ACTION_TOKEN_FIELDS",
    "SENTINEL",
    "STATE_TOKEN_FIELDS",
    "ActionChoiceTree",
    "PolicyConfig",
    "action_choice_to_python",
    "make_action_choice",
    "pad_token_tree",
    "stack_token_trees",
    "tokenize_state_snapshot",
    "tokenize_action_space_snapshot",
    "init_policy_params",
    "sample_target",
    "score_target",
)


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
    assert policy.__all__ == EXPECTED_POLICY_EXPORTS


def test_policy_package_star_import_exports_bound_names():
    namespace: dict[str, object] = {}

    exec("from gristmill_symbolics.policy import *", namespace)

    for name in EXPECTED_POLICY_EXPORTS:
        assert name in namespace
