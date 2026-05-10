def test_rl_package_exports_version():
    import gristmill_rl

    assert isinstance(gristmill_rl.__version__, str)
    assert gristmill_rl.__version__ == "0.1.0"
