"""Arrange-Act-Assert pattern checker for test files.

Enforces that test functions with 4+ logical statements use blank lines
to separate Arrange, Act, and Assert phases. Also checks that every
test has at least one assert.

Exit code 0 = all tests pass, 1 = violations found.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

MIN_LINES_FOR_AAA = 4  # tests with fewer statements are exempt
ASSERT_FUNCS = frozenset({"assertEqual", "assertIn", "assertTrue", "assertFalse",
                           "assertRaises", "assertIsNone", "assertIsNotNone"})


def _is_assert_stmt(node: ast.stmt) -> bool:
    """Check if a statement is an assertion."""
    if isinstance(node, ast.Assert):
        return True
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        func = node.value.func
        if isinstance(func, ast.Attribute) and func.attr in ASSERT_FUNCS:
            return True
    if isinstance(node, ast.With):
        # pytest.raises context manager counts as assert
        for item in node.items:
            ctx = item.context_expr
            if isinstance(ctx, ast.Call) and isinstance(ctx.func, ast.Attribute) and ctx.func.attr == "raises":
                return True
    return False


def _is_non_assert_statement(node: ast.stmt) -> bool:
    """Check if a statement is a non-assert action/arrangement."""
    return not _is_assert_stmt(node) and not isinstance(node, (ast.Pass, ast.Return))


def _count_logical_statements(body: list[ast.stmt]) -> int:
    """Count statements excluding docstrings and comments."""
    count = 0
    for i, node in enumerate(body):
        if i == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # skip docstring
        count += 1
    return count


def _get_source_lines(filepath: Path) -> list[str]:
    return filepath.read_text().splitlines()


def _has_blank_line_between(source_lines: list[str], node_before: ast.stmt, node_after: ast.stmt) -> bool:
    """Check if there's at least one blank line between two AST nodes."""
    end_line = node_before.end_lineno or node_before.lineno
    start_line = node_after.lineno
    for lineno in range(end_line, start_line - 1):
        if lineno < len(source_lines) and source_lines[lineno].strip() == "":
            return True
    return False


def check_function(func: ast.FunctionDef, source_lines: list[str], filepath: str) -> list[str]:
    """Check a single test function for AAA violations."""
    violations = []
    body = func.body

    # Skip docstring
    start = 0
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        start = 1

    stmts = body[start:]
    logical_count = _count_logical_statements(body)

    # Rule 1: Every test must have at least one assert
    has_assert = any(_is_assert_stmt(s) for s in stmts)
    if not has_assert:
        violations.append(
            f"  {filepath}:{func.lineno}: {func.name} — no assert found (AAA: every test must assert)"
        )
        return violations

    # Rule 2: For non-trivial tests, require blank line separation between phases
    if logical_count < MIN_LINES_FOR_AAA:
        return violations

    # Find the transition point: last non-assert before first assert
    first_assert_idx = None
    last_arrange_idx = None
    for i, stmt in enumerate(stmts):
        if _is_assert_stmt(stmt) and first_assert_idx is None:
            first_assert_idx = i
        if _is_non_assert_statement(stmt) and first_assert_idx is None:
            last_arrange_idx = i

    if first_assert_idx is not None and last_arrange_idx is not None and first_assert_idx > 0:
        prev_stmt = stmts[first_assert_idx - 1]
        assert_stmt = stmts[first_assert_idx]
        if not _has_blank_line_between(source_lines, prev_stmt, assert_stmt):
            violations.append(
                f"  {filepath}:{assert_stmt.lineno}: {func.name} — "
                f"missing blank line before assert phase (AAA: separate Act from Assert)"
            )

    return violations


def check_file(filepath: Path) -> list[str]:
    """Check all test functions in a file."""
    source = filepath.read_text()
    source_lines = source.splitlines()
    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return [f"  {filepath}: SyntaxError — could not parse"]

    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            violations.extend(check_function(node, source_lines, str(filepath)))

    return violations


def main() -> int:
    test_dir = Path("tests")
    if not test_dir.exists():
        print("ERROR: tests/ directory not found")
        return 1

    all_violations: list[str] = []
    for path in sorted(test_dir.rglob("test_*.py")):
        all_violations.extend(check_file(path))

    if all_violations:
        print(f"AAA pattern violations ({len(all_violations)}):\n")
        for v in all_violations:
            print(v)
        print("\nFix: add a blank line between Arrange/Act and Assert phases.")
        return 1

    print("AAA pattern check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
