def test_module_exports_core_types():
    import gristmill_symbolics

    assert hasattr(gristmill_symbolics, "TensorComputation")
    assert hasattr(gristmill_symbolics, "ActionSpace")
    assert hasattr(gristmill_symbolics, "GristmillSymbolicsError")
