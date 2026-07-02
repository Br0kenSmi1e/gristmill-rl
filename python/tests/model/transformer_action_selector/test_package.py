def test_selector_package_exports_current_model_only():
    import gristmill_symbolics.model.transformer_action_selector as selector

    assert selector.__all__ == (
        "BatchedState",
        "BatchedTransitions",
        "SelectorChoice",
        "SelectorState",
        "SelectorTransitions",
        "TransformerActionSelectorModel",
    )
    assert hasattr(selector, "SelectorState")
    assert hasattr(selector, "SelectorChoice")
    assert hasattr(selector, "SelectorTransitions")
    assert hasattr(selector, "TransformerActionSelectorModel")
    assert not hasattr(selector, "RewriteState")
    assert not hasattr(selector, "RewriteStateRow")


def test_root_package_keeps_thin_rust_rewrite_surface():
    import gristmill_symbolics

    assert hasattr(gristmill_symbolics, "TensorComputation")
    assert hasattr(gristmill_symbolics, "ActionSpace")
    assert hasattr(gristmill_symbolics, "action_space_for_def")
    assert not hasattr(gristmill_symbolics, "RewriteState")
    assert not hasattr(gristmill_symbolics, "RewriteStateRow")
