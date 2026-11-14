"""Mechanical guards for the module boundaries.

File size is a ratchet: existing monoliths may shrink, never grow, and a
new file over the cap fails. When a file shrinks, lower its BASELINE
entry in the same commit.
"""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "stmmlib"
MAX_LINES = 300

# Total line counts at 0.2.2, including blanks. Only files already over
# the cap are listed; they are the refactor backlog, not an exemption.
BASELINE = {
    "beta/solvers.py": 365,
    "beta/moran_panel.py": 733,
    "beta/result.py": 712,
    "diagnostics_suite.py": 663,
    "beta/fit.py": 599,
    "gaussian/covariance.py": 607,
    "beta/lrt_fixed.py": 551,
    "gaussian/models.py": 535,
    "gaussian/transforms.py": 419,
    "beta/inference.py": 340,
    "beta/report.py": 373,
    "gaussian/optimization.py": 361,
    "beta/simulation.py": 344,
    "registry.py": 326,
    "gaussian/design.py": 312,
    "gaussian/diagnostics.py": 308,
}


def _lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


@pytest.mark.parametrize("rel, cap", sorted(BASELINE.items()))
def test_existing_file_does_not_grow(rel, cap):
    current = _lines(PACKAGE / rel)
    assert current <= cap, (
        f"{rel} grew from {cap} to {current} lines. Split it instead of "
        f"adding to the monolith, then lower BASELINE['{rel}'] in this test.")


def test_new_file_stays_small():
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE).as_posix()
        if rel in BASELINE:
            continue
        n = _lines(path)
        if n > MAX_LINES:
            offenders.append(f"{rel} ({n} lines)")
    assert not offenders, (
        "New files must stay under "
        f"{MAX_LINES} lines: {', '.join(offenders)}")


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name


@pytest.mark.parametrize("family, other", [
    ("gaussian", "beta"),
    ("beta", "gaussian"),
])
def test_families_do_not_import_each_other(family, other):
    prefix = f"stmmlib.{other}"
    offenders = []
    for path in sorted((PACKAGE / family).rglob("*.py")):
        for module in _imports(path):
            if module == prefix or module.startswith(prefix + "."):
                offenders.append(
                    f"{path.relative_to(PACKAGE).as_posix()} imports {module}")
    assert not offenders, (
        f"{family}/ must not depend on {other}/: {', '.join(offenders)}")
