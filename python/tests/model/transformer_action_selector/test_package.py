EXPECTED_MODEL_EXPORTS = ("TransformerActionSelectorModel",)
LEGACY_CONFIG_EXPORT = "Policy" + "Config"


def test_existing_extension_exports_still_import_from_package_root():
    import gristmill_symbolics

    assert hasattr(gristmill_symbolics, "TensorComputation")
    assert hasattr(gristmill_symbolics, "RewriteState")
    assert hasattr(gristmill_symbolics, "RewriteStateRow")
    assert hasattr(gristmill_symbolics, "validate_decision")


def test_transformer_action_selector_package_exports_only_concrete_model():
    import gristmill_symbolics.model.transformer_action_selector as model_pkg

    assert model_pkg.__all__ == EXPECTED_MODEL_EXPORTS
    assert hasattr(model_pkg, "TransformerActionSelectorModel")
    assert not hasattr(model_pkg, LEGACY_CONFIG_EXPORT)


def test_transformer_action_selector_star_import_exports_bound_names():
    namespace: dict[str, object] = {}

    exec("from gristmill_symbolics.model.transformer_action_selector import *", namespace)

    for name in EXPECTED_MODEL_EXPORTS:
        assert name in namespace
    assert LEGACY_CONFIG_EXPORT not in namespace
