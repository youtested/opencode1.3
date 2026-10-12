"""find_symbols tool: code navigation (definitions, callers, deps).

RULES (full guidance for maintainers; the sent schema is short):
- Single-hop verbs: def/callers/refs/deps/imports/symbols (+ bare word = def).
- Multi-hop verbs: impact/path/dead/cycles/communities/graphstats — these walk
  the resolved graph in index/graph.py, so they follow a relationship across
  files instead of reporting one file's contents at a time.
- Indexed once, cached, refreshed incrementally — faster than grep+read.
- Use BEFORE editing: jump to definition, check callers, then `impact X` to
  see what a change would break.
"""
from __future__ import annotations

from pathlib import Path

from .registry import Tool, schema_with

# ponytail: index.engine (~80ms: ast walkers + ctags probe) loads on first
# find_symbols call, not at registry build. _DEFAULT_LIMIT mirrors
# engine.MAX_RESULTS but stays import-light; the engine import below is the
# single source of truth at call time.
_DEFAULT_LIMIT = 60


def tool() -> Tool:
    description = """Name search across files, plus multi-hop graph queries. SECOND choice: if you have file+line use lsp first (0.4ms exact). Single-hop: def/callers/refs/deps/imports/symbols. MULTI-HOP (follows relationships across files): 'impact X' = everything that breaks if X changes, 'path A to B' = the call chain, 'dead' = unreferenced defs, 'cycles' = circular imports, 'communities' = subsystems, 'graphstats' = graph health. Use this for name-only search, then read symbol= for the block. grep is plain text only."""

    def run(input: dict) -> dict:
        q = str(input.get("query") or "").strip()
        if not q:
            return {
                "output": (
                    "Empty query. Single-hop: 'def run', "
                    "'callers _atomic_write', 'deps tools/edit.py', "
                    "'imports write', 'symbols main.py'. "
                    "Multi-hop: 'impact run_tool' (what breaks if it changes), "
                    "'path run_tool to send' (the call chain), 'dead', "
                    "'cycles', 'communities', 'graphstats'."
                ),
                "error": True,
            }
        root = input.get("root")
        kind = str(input.get("kind") or "")
        try:
            limit = int(input.get("limit") or _DEFAULT_LIMIT)
        except (TypeError, ValueError):
            limit = _DEFAULT_LIMIT
        ignore = input.get("ignore")
        if isinstance(ignore, str):
            ignore = [ignore]
        force = bool(input.get("fresh", False))
        from ..index.engine import query as _query
        return _query(
            q,
            root=Path(root) if root else None,
            kind=kind,
            limit=max(1, min(limit, 200)),
            ignore_extra=[str(x) for x in (ignore or [])],
            force=force,
        )

    return Tool(
        name="find_symbols",
        description=description,
        parameters=schema_with(
            {
                "query": {
                    "type": "string",
                    "description": (
                        'What to find, with an optional leading verb. '
                        'Single-hop: "def run", "callers _atomic_write", '
                        '"refs Registry", "deps open/opencode_py/tools/edit.py", '
                        '"imports opencode_py.tools.write", "symbols bash.py". '
                        'Multi-hop: "impact run_tool" (blast radius, everything '
                        'that breaks if it changes), "path run_tool to send" '
                        '(the call chain between two things), "dead" (defs '
                        'nothing calls), "cycles" (circular imports), '
                        '"communities" (which files group into which subsystem), '
                        '"graphstats" (node/edge counts and confidence).'
                    ),
                },
                "root": {
                    "type": "string",
                    "description": (
                        "Project root to index (default: the current worktree)."
                    ),
                    "optional": True,
                },
                "kind": {
                    "type": "string",
                    "description": (
                        'Optional definition-kind filter for "def" queries: '
                        "function, method, class, variable, constant."
                    ),
                    "optional": True,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to show (default 60).",
                    "optional": True,
                },
                "ignore": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extra directory names to skip while indexing.",
                    "optional": True,
                },
                "fresh": {
                    "type": "boolean",
                    "description": "Force a full rebuild of the index (default false).",
                    "optional": True,
                },
            },
            ["query"],
        ),
        run=run,
        permission="find_symbols",
    )