"""
The BreezeML dependency contract.

Core `import breezeml` must work with ONLY these third-party packages:

    scikit-learn, pandas, numpy, joblib

This test enforces the contract with AST analysis: every module-level
import in breezeml/ must be stdlib, breezeml-internal, or one of the four.
Optional integrations (sentence-transformers, shap, matplotlib, xgboost,
lightgbm, seaborn, fastapi, skl2onnx, mcp) must be imported lazily inside
functions so they never break a plain `pip install breezeml`.

If this test fails, you added a hard dependency. Don't. Make it lazy or an
optional extra instead. "4 dependencies. Always."
"""
import ast
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "breezeml"

ALLOWED_THIRD_PARTY = {"sklearn", "pandas", "numpy", "joblib"}


def _stdlib_names():
    names = getattr(sys, "stdlib_module_names", None)  # Python 3.10+
    if names is not None:
        return set(names)
    # Python 3.9 fallback: sys.stdlib_module_names does not exist yet.
    # Derive the set from the stdlib directories plus builtins. C-extension
    # stdlib modules (math, _json, ...) live in lib-dynload, not the main
    # stdlib dir and not always in sys.builtin_module_names, so scan both
    # and merge a curated core set as a final safety net.
    import sysconfig

    found = set(sys.builtin_module_names)
    paths = sysconfig.get_paths()
    for key in ("stdlib", "platstdlib"):
        base = paths.get(key)
        if not base:
            continue
        base_dir = Path(base)
        if not base_dir.exists():
            continue
        for item in base_dir.iterdir():
            found.add(item.name.split(".")[0])
        dynload = base_dir / "lib-dynload"
        if dynload.exists():
            for item in dynload.iterdir():
                found.add(item.name.split(".")[0])
    # Core stdlib names that must always count, regardless of build layout.
    found |= {
        "math", "cmath", "json", "csv", "os", "sys", "io", "re", "time",
        "datetime", "collections", "itertools", "functools", "warnings",
        "pathlib", "typing", "abc", "argparse", "subprocess", "random",
        "statistics", "decimal", "hashlib", "pickle", "copy", "enum",
        "contextlib", "dataclasses", "logging", "operator", "textwrap",
    }
    return found


def _module_level_imports(tree: ast.Module):
    """Yield (root_module, lineno) for imports at module scope only."""
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:  # relative import: breezeml-internal
                continue
            if node.module:
                yield node.module.split(".")[0], node.lineno


def test_core_has_only_four_dependencies():
    stdlib = _stdlib_names()
    violations = []

    for py_file in sorted(PACKAGE_DIR.glob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for root, lineno in _module_level_imports(tree):
            if root in stdlib or root == "breezeml" or root == "__future__":
                continue
            if root not in ALLOWED_THIRD_PARTY:
                violations.append(f"{py_file.name}:{lineno} imports '{root}'")

    assert not violations, (
        "Dependency contract broken! Module-level imports outside "
        f"{sorted(ALLOWED_THIRD_PARTY)}:\n  " + "\n  ".join(violations)
        + "\nMake the import lazy (inside the function) or an optional extra."
    )


def test_import_breezeml_pulls_no_optional_packages():
    """Importing breezeml must not import any optional dependency."""
    import subprocess

    forbidden = [
        "sentence_transformers", "shap", "matplotlib", "xgboost",
        "lightgbm", "seaborn", "fastapi", "skl2onnx", "mcp",
    ]
    code = (
        "import sys; import breezeml; "
        f"bad = [m for m in {forbidden!r} if m in sys.modules]; "
        "print(','.join(bad))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(PACKAGE_DIR.parent),
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    leaked = result.stdout.strip()
    assert leaked == "", f"import breezeml pulled optional packages: {leaked}"
