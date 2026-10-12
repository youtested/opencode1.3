"""Print (and check) the project-size numbers quoted in README.md.

The README quotes tool/test/module counts. They went stale before because
nothing kept them honest. Run:

    python tools/readme_stats.py          # print the real numbers
    python tools/readme_stats.py --check  # exit 1 if README disagrees

The test count needs pytest, so without it the test line is reported as
"unknown" and --check skips that figure rather than failing.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = ROOT / "opencode_py"
TESTS = ROOT / "tests"
README = ROOT / "README.md"


def _lines(paths) -> int:
    total = 0
    for p in paths:
        try:
            total += len(p.read_text(encoding="utf-8").splitlines())
        except OSError:
            pass
    return total


def builtin_tools() -> int:
    """Count the names the tool registry actually exports."""
    init = PKG / "tools" / "__init__.py"
    try:
        tree = ast.parse(init.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return 0
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "TOOL_NAMES":
                    try:
                        return len(ast.literal_eval(node.value))
                    except (ValueError, SyntaxError):
                        return 0
    return 0


def test_count() -> int | None:
    """Ask pytest how many tests exist. None when pytest is unavailable."""
    import subprocess
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", str(TESTS)],
            cwd=str(ROOT), capture_output=True, text=True, timeout=180,
        ).stdout
    except Exception:
        return None
    m = re.search(r"(\d+)\s+tests? collected", out)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s+tests? collected", out)
    return None


def collect() -> dict[str, int | None]:
    src_files = [p for p in PKG.rglob("*.py")]
    return {
        "modules": len([p for p in src_files if p.name != "__init__.py"]),
        "source_lines": _lines(src_files),
        "tools": builtin_tools(),
        "test_files": len(list(TESTS.glob("test_*.py"))),
        "test_lines": _lines(TESTS.rglob("*.py")),
        "tests": test_count(),
    }


def _fmt(n: int | None) -> str:
    return "unknown" if n is None else str(n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if README is out of date")
    args = ap.parse_args()
    s = collect()

    print(f"tools/  builtins      : {s['tools']}")
    print(f"tests/  files         : {s['test_files']}")
    print(f"tests/  tests         : {_fmt(s['tests'])}")
    print(f"python  modules       : {s['modules']}")
    print(f"python  source lines  : {s['source_lines']} (~{s['source_lines'] // 1000}k)")
    print(f"tests   lines         : {s['test_lines']} (~{s['test_lines'] / 1000:.1f}k)")

    if not args.check:
        return 0

    try:
        text = README.read_text(encoding="utf-8")
    except OSError as e:
        print(f"cannot read README: {e}", file=sys.stderr)
        return 1

    problems = []

    m = re.search(r"tools/\s+(\d+) builtins", text)
    if m and int(m.group(1)) != s["tools"]:
        problems.append(f"tools builtins: README says {m.group(1)}, actual {s['tools']}")

    m = re.search(r"tests/\s+(\d+) test files,\s*(\d+) tests", text)
    if m:
        if int(m.group(1)) != s["test_files"]:
            problems.append(f"test files: README says {m.group(1)}, actual {s['test_files']}")
        if s["tests"] is not None and int(m.group(2)) != s["tests"]:
            problems.append(f"tests: README says {m.group(2)}, actual {s['tests']}")
    else:
        problems.append("could not find the 'tests/  N test files, N tests' line in README")

    m = re.search(r"(\d+) Python modules:\s*~(\d+)k lines of source plus ~([\d.]+)k lines of tests", text)
    if m:
        if int(m.group(1)) != s["modules"]:
            problems.append(f"modules: README says {m.group(1)}, actual {s['modules']}")
        if abs(int(m.group(2)) - round(s["source_lines"] / 1000)) > 0:
            problems.append(f"source k: README says {m.group(2)}k, actual {round(s['source_lines']/1000)}k")
        if abs(float(m.group(3)) - s["test_lines"] / 1000) > 0.05:
            problems.append(f"test k: README says {m.group(3)}k, actual {s['test_lines']/1000:.1f}k")
    else:
        problems.append("could not find the 'N Python modules: ~Nk lines ...' line in README")

    if problems:
        print("\nREADME is out of date:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print("\nREADME numbers are up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
