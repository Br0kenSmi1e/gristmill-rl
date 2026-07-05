import ast
import importlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "gristmill_symbolics" / "direct_optimizer"

FORBIDDEN_IMPORTS = {
    "gristmill_symbolics.model.tokenizer",
    "gristmill_symbolics.model.transformer_action_selector",
    "gristmill_symbolics.trainer.reinforce",
    "gristmill_symbolics.cli.checkpoint",
}


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_direct_optimizer_package_imports():
    package = importlib.import_module("gristmill_symbolics.direct_optimizer")

    assert package.__all__ == ()


def test_orbax_checkpoint_is_declared_as_direct_dependency():
    pyproject = (ROOT / "pyproject.toml").read_text()

    assert '"orbax-checkpoint>=0.11"' in pyproject


def test_direct_optimizer_modules_do_not_import_forbidden_training_paths():
    assert PACKAGE.exists()
    for path in PACKAGE.glob("*.py"):
        imported = _imported_modules(path)
        forbidden = {
            module
            for module in imported
            if any(
                module == blocked or module.startswith(blocked + ".")
                for blocked in FORBIDDEN_IMPORTS
            )
        }
        assert forbidden == set(), f"{path} imports {sorted(forbidden)}"
