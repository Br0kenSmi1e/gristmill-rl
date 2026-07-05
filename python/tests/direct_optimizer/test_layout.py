import ast
import importlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "gristmill_symbolics" / "direct_optimizer"

FORBIDDEN_IMPORTS = {
    "gristmill_symbolics.model.tokenizer",
    "gristmill_symbolics.model.transformer_action_selector",
    "gristmill_symbolics.trainer.reinforce",
    "gristmill_symbolics.cli.checkpoint",
}


def _package_name_for(path: Path) -> str | None:
    try:
        relative = path.resolve().relative_to(ROOT)
    except ValueError:
        return None

    if relative.parts[:2] != ("gristmill_symbolics", "direct_optimizer"):
        return None

    return ".".join(relative.with_suffix("").parts[:-1])


def _resolved_import_from_module(path: Path, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module

    package_name = _package_name_for(path)
    if package_name is None:
        return node.module

    package_parts = package_name.split(".")
    if node.level > len(package_parts):
        return node.module

    module_parts = package_parts[: len(package_parts) - node.level + 1]
    if node.module:
        module_parts.extend(node.module.split("."))
    return ".".join(module_parts)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = _resolved_import_from_module(path, node)
            if module:
                modules.add(module)
                modules.update(f"{module}.{alias.name}" for alias in node.names)
    return modules


def test_direct_optimizer_package_imports():
    package = importlib.import_module("gristmill_symbolics.direct_optimizer")

    assert package.__all__ == (
        "DirectOptimizerTransformer",
        "optimize_from_checkpoint",
        "optimize_with_model",
    )


def test_sample_module_help_runs_without_runtime_warning():
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "error",
            "-m",
            "gristmill_symbolics.direct_optimizer.sample",
            "--help",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stderr == ""


def test_orbax_checkpoint_is_declared_as_direct_dependency():
    pyproject = (ROOT / "pyproject.toml").read_text()

    assert '"orbax-checkpoint>=0.11"' in pyproject


def test_imported_modules_records_import_from_aliases(tmp_path):
    path = tmp_path / "imports.py"
    path.write_text(
        "\n".join(
            [
                "from gristmill_symbolics.model import tokenizer",
                "from gristmill_symbolics.trainer import reinforce",
                "from gristmill_symbolics.cli import checkpoint",
            ]
        )
    )

    assert {
        "gristmill_symbolics.model.tokenizer",
        "gristmill_symbolics.trainer.reinforce",
        "gristmill_symbolics.cli.checkpoint",
    }.issubset(_imported_modules(path))


def test_imported_modules_resolves_relative_import_from_aliases():
    path = PACKAGE / "_temp_boundary_import.py"
    path.write_text("from ..model import tokenizer")
    try:
        assert "gristmill_symbolics.model.tokenizer" in _imported_modules(path)
    finally:
        path.unlink()


def test_imported_modules_resolves_package_initializer_relative_import_from_alias():
    tree = ast.parse("from ..model import tokenizer")
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))

    module = _resolved_import_from_module(PACKAGE / "__init__.py", node)
    imported = set()
    if module:
        imported.add(module)
        imported.update(f"{module}.{alias.name}" for alias in node.names)

    assert "gristmill_symbolics.model.tokenizer" in imported


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


def test_trainer_does_not_import_symbolic_or_sampler_boundaries():
    modules = _imported_modules(PACKAGE / "trainer.py")

    assert "gristmill_symbolics" not in modules
    assert "gristmill_symbolics.direct_optimizer.sample" not in modules


def test_sampler_does_not_import_trainer_module():
    modules = _imported_modules(PACKAGE / "sample.py")

    assert "gristmill_symbolics.direct_optimizer.trainer" not in modules


def test_direct_optimizer_does_not_register_existing_cli_checkpoint():
    modules = _imported_modules(PACKAGE / "checkpoint.py")

    assert "gristmill_symbolics.cli.checkpoint" not in modules
    assert "gristmill_symbolics.direct_optimizer.trainer" not in modules
