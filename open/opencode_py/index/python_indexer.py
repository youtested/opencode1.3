"""Python indexer: a real ``ast`` walk, the gold standard for Python.

Produces exact definitions — functions/methods/classes with signatures,
parameter lists, line ranges — plus module- and class-level constants,
call sites (with the enclosing container), and import edges. Because it uses
``ast`` nothing is guessed, so "who calls X" returns exactly the call sites,
never look-alike text.
"""

from __future__ import annotations

from ..util.log import swallow
import ast
import sys
from pathlib import Path

from .model import FileIndex, ImportRecord, Ref, Symbol

# Names so ubiquitous in Python that indexing them as refs would swamp every
# "who uses X" answer with noise: builtins plus special names that appear in
# nearly every file.
_SKIP_BUILTINS = {
    "self", "cls",
    "None", "True", "False",
    "abs", "all", "any", "ascii", "bin", "bool", "breakpoint", "bytearray",
    "bytes", "callable", "chr", "classmethod", "compile", "complex", "delattr",
    "dict", "dir", "divmod", "enumerate", "eval", "exec", "filter", "float",
    "format", "frozenset", "getattr", "globals", "hasattr", "hash", "help",
    "hex", "id", "input", "int", "isinstance", "issubclass", "iter", "len",
    "list", "locals", "map", "max", "memoryview", "min", "next", "object",
    "oct", "open", "ord", "pow", "print", "property", "range", "repr",
    "reversed", "round", "set", "setattr", "slice", "sorted", "staticmethod",
    "str", "sum", "tuple", "type", "vars", "zip", "__import__",
}

try:
    _STDLIB_MODULES = frozenset(sys.stdlib_module_names)
except AttributeError:  # pragma: no cover - pre-3.10 runtime
    _STDLIB_MODULES = frozenset(
        "abc argparse array asyncio base64 bisect builtins collections contextlib "
        "copy csv dataclasses datetime decimal difflib enum errno fnmatch functools "
        "gc glob gzip hashlib heapq html http importlib inspect io itertools json "
        "logging math mmap multiprocessing os pathlib pickle platform pprint queue "
        "random re select shutil signal socket sqlite3 ssl statistics string struct "
        "subprocess sys tempfile threading time traceback types typing unittest urllib "
        "uuid warnings weakref xml zipfile".split()
    )


def _parse(src: str, filename: str) -> ast.Module | None:
    """Parse with a graceful syntax downgrade for brand-new language features."""
    try:
        return ast.parse(src, filename=filename, type_comments=True)
    except SyntaxError:
        try:
            return ast.parse(src, filename=filename, feature_version=(3, 8))
        except (SyntaxError, ValueError):
            return None


def _signature_for_def(node: ast.AST) -> str:
    """Human-facing declaration text for a function/class definition."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = node.args
        parts = [a.arg for a in list(args.posonlyargs) + list(args.args)]
        if args.vararg:
            parts.append("*" + args.vararg.arg)
        elif args.kwonlyargs:
            parts.append("*")
        if args.kwonlyargs:
            parts += [a.arg for a in args.kwonlyargs]
        if args.kwarg:
            parts.append("**" + args.kwarg.arg)
        prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
        return f"{prefix}{node.name}({', '.join(parts)})"
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(b) for b in node.bases) if node.bases else ""
        return f"class {node.name}({bases})" if bases else f"class {node.name}"
    return node.__class__.__name__


def _dotted(node: ast.expr | None) -> str | None:
    """Best-effort dotted path for a Call/Attribute/Name expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


class _PythonWalker:
    def __init__(self, path: str) -> None:
        self.path = path
        self.stack: list[str] = []  # enclosing class names
        self.depth = 0             # function-nesting depth (refs inside still count)
        self.symbols: list[Symbol] = []
        self.refs: list[Ref] = []
        self.imports: list[ImportRecord] = []

    def _container(self) -> str:
        return ".".join(self.stack)

    # -- statements -------------------------------------------------------
    def _stmt(self, node: ast.stmt) -> None:
        method = getattr(self, "s_" + node.__class__.__name__, None)
        if method is not None:
            method(node)
            return
        self._stmt_generic(node)

    def _stmt_generic(self, node: ast.stmt) -> None:
        # Descend into EVERY non-stmt child, not just ast.expr. Statements wrap
        # their expressions in other node kinds — ExceptHandler.type (the name
        # in `except ContextOverflowError:`), withitem.context_expr, the
        # match_case / MatchClass.cls of a `match` block — and restricting this
        # to stmt/expr silently dropped those names, so "callers X" under-
        # reported and callers of a real function looked dead.
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                self._stmt(child)
            else:
                self._expr(child, set(), "use")

    def s_Module(self, node: ast.Module) -> None:
        for child in node.body:
            self._stmt(child)

    def s_ClassDef(self, node: ast.ClassDef) -> None:
        self.symbols.append(
            Symbol(node.name, "class", self.path, node.lineno,
                   node.end_lineno or node.lineno, _signature_for_def(node),
                   self._container(), "python")
        )
        self.stack.append(node.name)
        for child in node.body:
            self._stmt(child)
        self.stack.pop()
        # decorators run BEFORE the class exists, so they belong to the
        # enclosing scope, not to the class they decorate
        for deco in node.decorator_list:
            self._expr(deco, set(), "attribute")

    def s_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._fn(node, "function")

    def s_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._fn(node, "function")

    def _fn(self, node: ast.AST, kind: str) -> None:
        is_method = bool(self.stack)
        self.symbols.append(
            Symbol(node.name, "method" if is_method else "function",
                   self.path, node.lineno, node.end_lineno or node.lineno,
                   _signature_for_def(node), self._container(), "python")
        )
        self.depth += 1
        # parameter + return annotations are real references (a type used only
        # in a signature is still a use), and without this a class that is only
        # ever mentioned as a type looked unreferenced
        # arg nodes carry `annotation` only; the *defaults* live on the
        # arguments node (positional) and as kw_defaults (keyword-only)
        argsnode = getattr(node, "args", None)
        for a in getattr(argsnode, "args", []) or []:
            for piece in (a.annotation, getattr(a, "type_params", None)):
                if isinstance(piece, ast.expr):
                    self._expr(piece, set(), "use")
        for d in getattr(argsnode, "defaults", []) or []:
            if isinstance(d, ast.expr):
                self._expr(d, set(), "use")
        for kw in getattr(argsnode, "kw_defaults", []) or []:
            if isinstance(kw, ast.expr):
                self._expr(kw, set(), "use")
        ret = getattr(node, "returns", None)
        if isinstance(ret, ast.expr):
            self._expr(ret, set(), "use")
        for child in node.body:
            self._stmt(child)
        for other in getattr(node, "decorator_list", []):
            self._expr(other, set(), "attribute")
        self.depth -= 1

    def s_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(
                ImportRecord(alias.name, self.path, node.lineno, None,
                             [alias.asname] if alias.asname else [])
            )

    def s_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            base = "." * node.level + (node.module or "")
            self.imports.append(
                ImportRecord(base, self.path, node.lineno, True,
                             [a.asname or a.name for a in node.names])
            )
            return
        mod = node.module or ""
        aliases = [a.asname or a.name for a in node.names]
        is_std = mod.split(".")[0] in _STDLIB_MODULES
        for a in node.names:
            if a.name == "*" and not a.asname:
                continue
            self.symbols.append(
                Symbol(a.asname or a.name, "import", self.path, node.lineno,
                       node.lineno, f"from {mod} import {a.name}",
                       self._container(), "python")
            )
        self.imports.append(
            # stdlib is definitively external; anything else stays None here so
            # _classify_local_imports can resolve it against the repo layout
            ImportRecord(mod, self.path, node.lineno, False if is_std else None, aliases)
        )

    def _target_reads(self, target: ast.AST) -> None:
        """Index the part of an assignment target that is *read*.

        ``loop.cfg.compaction_enabled = True`` writes only
        ``compaction_enabled``; it still reads ``loop.cfg`` on the way there,
        and those are real references. A bare ``x = 1`` is a pure Store and
        contributes nothing, which is why this is not just ``_expr(target)``.
        """
        node = target
        while isinstance(node, ast.Attribute):
            node = node.value
            if isinstance(node, ast.Attribute):
                if node.attr not in _SKIP_BUILTINS:
                    self.refs.append(
                        Ref(node.attr, self.path, node.lineno, "attribute",
                            self._container())
                    )
            else:
                self._expr(node, set(), "use")
                return
        if isinstance(node, ast.Subscript):
            # a[key] = v reads both a and key
            self._expr(node, set(), "use")

    def s_Assign(self, node: ast.Assign) -> None:
        if self.depth == 0:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    is_const = target.id.isupper()
                    self.symbols.append(
                        Symbol(target.id, "constant" if is_const else "variable",
                               self.path, node.lineno, node.end_lineno or node.lineno,
                               f"{target.id} = ...", self._container(), "python")
                    )
                    break
        for target in node.targets:
            self._target_reads(target)
        self._expr(node.value, set(), "use")

    def s_AugAssign(self, node: ast.AugAssign) -> None:
        # x += 1 reads x before writing it
        self._target_reads(node.target)
        self._expr(node.value, set(), "use")

    def s_AnnAssign(self, node: ast.AnnAssign) -> None:
        if self.depth == 0 and isinstance(node.target, ast.Name):
            is_const = node.target.id.isupper()
            self.symbols.append(
                Symbol(node.target.id, "constant" if is_const else "variable",
                       self.path, node.lineno, node.end_lineno or node.lineno,
                       f"{node.target.id}: ...", self._container(), "python")
            )
        if node.value is not None:
            self._expr(node.value, set(), "use")
        self._target_reads(node.target)
        # the annotation is a type reference: `count: int = 0` / `cb: Callable`
        if isinstance(getattr(node, "annotation", None), ast.expr):
            self._expr(node.annotation, set(), "use")

    # -- expressions ------------------------------------------------------
    def _expr(self, node: ast.AST, visited: set[int], role: str) -> None:
        """Record Name/Attribute usages under this expression tree.

        ``visited`` de-duplicates shared subtrees when visitors overlap (a Call
        already walks its args, etc.). ``role`` annotates why a name was used:
        "call", "use", "attribute".
        """
        if id(node) in visited or node is None:
            return
        visited.add(id(node))
        if isinstance(node, ast.Name):
            # ctx is None for an `except SomeError:` type and a MatchClass
            # subject; those are real references. Store/Del are not, so they
            # stay excluded.
            ctx = getattr(node, "ctx", None)
            if node.id not in _SKIP_BUILTINS and (ctx is None or isinstance(ctx, ast.Load)):
                self.refs.append(
                    Ref(node.id, self.path, node.lineno, role, self._container())
                )
            return
        if isinstance(node, ast.Attribute):
            if node.attr not in _SKIP_BUILTINS and isinstance(getattr(node, "ctx", None), ast.Load):
                self.refs.append(
                    Ref(node.attr, self.path, node.lineno, role, self._container())
                )
            self._expr(node.value, visited, "use")
            return
        if isinstance(node, ast.Call):
            self._expr(node.func, visited, "call")
            for arg in node.args:
                self._expr(arg, visited, "use")
            for kw in node.keywords:
                self._expr(kw.value, visited, "use")
            return
        if isinstance(node, (ast.Constant, ast.Name)):
            return
        for child in ast.iter_child_nodes(node):
            self._expr(child, visited, "use")


def index_file(root: Path, rel: str, src: str, size: int, mtime: float) -> FileIndex:
    """Index one Python file. Never raises on bad syntax (falls back to empty)."""
    tree = _parse(src, rel)
    walker = _PythonWalker(rel)
    if tree is not None:
        try:
            walker.s_Module(tree)
        except Exception as _oc_exc:  # pragma: no cover - ast quirks must not kill a build
            swallow(_oc_exc, "index/python_indexer.py:321")
            # Keep whatever was collected before the failure. Re-running with a
            # fresh walker used to silently drop every symbol and ref for the
            # file, turning a crash into an empty index with no trace of the cause.
            pass
    _classify_local_imports(walker.imports, root)
    return FileIndex(
        rel, mtime, size, "python",
        symbols=walker.symbols, refs=walker.refs, imports=walker.imports,
    )


def _classify_local_imports(imports: list[ImportRecord], root: Path) -> None:
    """Fill in ``local`` for absolute imports by checking the repo layout.

    Namespace packages (PEP 420) have no ``__init__.py``, so a bare directory
    containing ``*.py`` files counts too.
    """
    for imp in imports:
        if imp.local is not None:
            continue
        first = imp.module.split(".")[0]
        pkg_dir = root / first
        imp.local = bool(
            (root / f"{first}.py").exists()
            or (pkg_dir / "__init__.py").exists()
            or (pkg_dir.is_dir() and any(pkg_dir.glob("*.py")))
        )