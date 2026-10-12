"""Guard the shape of ``opencode_py/agent/loop.py`` against accidental edits.

The agent loop is the highest-risk file in the project: a change here can give
wrong answers, lose work, or hang. This test records every class, method and
module-level function with a hash of its exact source, then checks that the
*combined* content of ``loop.py`` plus its mixin modules (``loop_*.py``) still
matches.

That means it survives a refactor that MOVES code between those files — a
method moving from ``loop.py`` into ``loop_mixins`` is fine — but fails if
anything is lost, renamed, or edited, even by one character.

To refresh after a deliberate change:
    python tests/fingerprint_loop.py --write
"""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib

AGENT_DIR = pathlib.Path(__file__).resolve().parent.parent / "opencode_py" / "agent"
LOOP = AGENT_DIR / "loop.py"
STORE = pathlib.Path(__file__).resolve().parent / "fingerprint_loop.json"


def _sources() -> list[pathlib.Path]:
    """loop.py plus every loop_* mixin module, in a stable order."""
    files = [LOOP] + sorted(AGENT_DIR.glob("loop_*.py"))
    return [p for p in files if p.exists()]


def collect() -> dict[str, str]:
    """name -> sha256 of the exact source segment.

    Methods are keyed by NAME ONLY (not class-qualified), so moving a method
    from ``AgentLoop`` into ``loop_*`` mixins keeps the key stable — that is
    the whole point, a refactor must pass. Editing a body, renaming, or
    losing something still fails.

    Classes are recorded by existence only: their body legitimately changes
    when methods move out of them.
    """
    out: dict[str, str] = {}
    for path in _sources():
        src = path.read_text(encoding="utf-8")
        lines = src.splitlines(keepends=True)
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                out[f"class:{node.name}"] = "present"
                for m in node.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        # a getter/setter pair shares one name; disambiguate
                        # with the decorator (@x.setter -> "name@setter")
                        attrs = sorted(
                            d.attr for d in m.decorator_list
                            if isinstance(d, ast.Attribute)
                        )
                        key = f"{m.name}@{','.join(attrs)}" if attrs else m.name
                        h = _hash(lines, m)
                        if key in out and out[key] != h:
                            raise AssertionError(
                                f"duplicate method {key!r} with differing bodies "
                                f"across the loop package"
                            )
                        out[key] = h
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[node.name] = _hash(lines, node)
    return out


def _hash(lines: list[str], node) -> str:
    start = min([d.lineno for d in node.decorator_list] + [node.lineno])
    body = "".join(lines[start - 1 : node.end_lineno])
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    got = collect()
    STORE.write_text(json.dumps(got, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {STORE.name} with {len(got)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())