"""Two data-integrity bugs found by audit, and their fixes.

1. Saving a file threw away its permissions: the temp file is created 0600 and
   os.replace keeps the new file's mode, so every edit turned 0644 into 0600 and
   stripped the execute bit off a 0755 script.
2. A missing (or null) newString was read as "delete this text", so a model that
   forgot the field silently deleted what oldString matched and was told it
   succeeded.
"""

import os
import stat
from pathlib import Path

import pytest

from opencode_py.tools.edit import _edit, _require_new_string, tool
from opencode_py.tools.write import _atomic_write, _existing_mode, _write


def mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


# --- fix 1: permissions survive a save -----------------------------------

def test_atomic_write_keeps_the_original_mode(tmp_path):
    p = tmp_path / "f.py"
    p.write_text("x = 1\n")
    os.chmod(p, 0o644)
    _atomic_write(p, "x = 2\n")
    assert mode(p) == 0o644


def test_edit_keeps_the_execute_bit(tmp_path):
    p = tmp_path / "s.sh"
    p.write_text("f() {\n  echo hi\n}\n")
    os.chmod(p, 0o755)
    r = _edit(str(p), "  echo hi", "  echo there")
    assert r.get("error") is not True
    assert mode(p) == 0o755
    assert os.stat(p).st_mode & 0o111, "must still be executable"


@pytest.mark.parametrize("bits", [0o600, 0o644, 0o640, 0o755, 0o750, 0o664])
def test_every_common_mode_round_trips(tmp_path, bits):
    p = tmp_path / f"m{bits}.py"
    p.write_text("x = 1\n")
    os.chmod(p, bits)
    _atomic_write(p, "x = 2\n")
    assert mode(p) == bits


def test_group_and_other_reads_survive(tmp_path):
    """The bug hit shared files hardest: after an edit others could not read it."""
    p = tmp_path / "shared.conf"
    p.write_text("a = 1\n")
    os.chmod(p, 0o644)
    _edit(str(p), "a = 1", "a = 2")
    assert mode(p) & 0o044, "group/other must still be able to read it"


def test_brand_new_file_gets_no_inherited_mode(tmp_path):
    p = tmp_path / "brand_new.py"
    _write(str(p), "x = 1\n")
    assert p.exists()
    # nothing existed to copy from, so it keeps mkstemp's private default
    assert _existing_mode(p) == mode(p)
    assert _existing_mode(tmp_path / "nope.py") is None


def test_permissions_survive_crlf_edits_and_line_range(tmp_path):
    p = tmp_path / "crlf.py"
    p.write_bytes(b"a = 1\r\nb = 2\r\n")
    os.chmod(p, 0o644)
    _edit(str(p), "    a = 1", "    a = 9")
    assert mode(p) == 0o644


def test_line_range_edit_keeps_the_mode(tmp_path):
    p = tmp_path / "lr.py"
    p.write_text("a = 1\nb = 2\n")
    os.chmod(p, 0o755)
    r = _edit(str(p), "", "B = 2", offset=2, limit=1)
    assert r.get("error") is not True
    assert mode(p) == 0o755


def test_refused_edit_never_touches_the_file_at_all(tmp_path):
    p = tmp_path / "ref.py"
    p.write_text("x = 1\n")
    os.chmod(p, 0o644)
    r = _edit(str(p), "x = 1", "x = (")
    assert r.get("error") is True
    assert mode(p) == 0o644
    assert p.read_text() == "x = 1\n"


def _skip_without_permission_enforcement(tmp_path):
    probe = tmp_path / "probe.txt"
    probe.write_text("x\n")
    os.chmod(probe, 0o444)
    try:
        with open(probe, "a"):
            pass
    except OSError:
        return
    else:
        pytest.skip("this environment ignores read-only bits")


def test_read_only_file_is_refused_and_left_alone(tmp_path):
    """Renaming a temp file over the target only needs write access to the
    DIRECTORY, so before this fix a file the user had deliberately locked was
    edited anyway. Both edit paths must refuse it."""
    _skip_without_permission_enforcement(tmp_path)
    p = tmp_path / "ro.py"
    p.write_text("x = 1\n")
    os.chmod(p, 0o444)
    r = _edit(str(p), "x = 1", "x = 2")
    assert r.get("error") is True
    assert r["metadata"]["guard"] == "read_only"
    assert "read-only" in r["output"]
    assert "chmod u+w" in r["output"]  # says how to proceed
    assert p.read_text() == "x = 1\n"
    assert mode(p) == 0o444


def test_read_only_refusal_also_covers_line_range_mode(tmp_path):
    _skip_without_permission_enforcement(tmp_path)
    p = tmp_path / "ro2.py"
    p.write_text("a = 1\nb = 2\n")
    os.chmod(p, 0o444)
    r = _edit(str(p), "", "B = 2", offset=2, limit=1)
    assert r.get("error") is True
    assert r["metadata"]["guard"] == "read_only"
    assert p.read_text() == "a = 1\nb = 2\n"


def test_force_overrides_a_read_only_file(tmp_path):
    _skip_without_permission_enforcement(tmp_path)
    p = tmp_path / "ro3.py"
    p.write_text("x = 1\n")
    os.chmod(p, 0o444)
    r = _edit(str(p), "x = 1", "x = 2", force=True)
    assert r.get("error") is not True
    assert p.read_text() == "x = 2\n"
    assert mode(p) == 0o444  # the flag is still respected afterwards


@pytest.mark.parametrize("bits", [0o644, 0o664, 0o666, 0o755, 0o775, 0o600, 0o400 + 0o200])
def test_a_writable_file_is_never_treated_as_read_only(tmp_path, bits):
    """Only a file nobody can write is read-only. 0644 is the normal case."""
    p = tmp_path / f"w{bits}.py"
    p.write_text("x = 1\n")
    os.chmod(p, bits)
    r = _edit(str(p), "x = 1", "x = 2")
    assert r.get("error") is not True
    assert p.read_text() == "x = 2\n"


@pytest.mark.parametrize("bits", [0o444, 0o400, 0o555, 0o111])
def test_is_read_only_detects_every_locked_mode(tmp_path, bits):
    from opencode_py.tools.write import is_read_only

    p = tmp_path / f"l{bits}.py"
    p.write_text("x = 1\n")
    os.chmod(p, bits)
    assert is_read_only(p) is True


def test_is_read_only_is_false_for_a_missing_file(tmp_path):
    from opencode_py.tools.write import is_read_only

    assert is_read_only(tmp_path / "does_not_exist.py") is False


def test_a_read_only_refusal_is_not_tracked_as_an_edit(tmp_path):
    from opencode_py.tools import verify as V

    _skip_without_permission_enforcement(tmp_path)
    p = tmp_path / "ro4.py"
    p.write_text("x = 1\n")
    os.chmod(p, 0o444)
    V._TRACKED.clear()
    V._TRACK_ORDER.clear()
    _edit(str(p), "x = 1", "x = 2")
    assert V.tracked() == []
    assert p.read_text() == "x = 1\n"  # refused, so untouched


def test_write_tool_also_preserves_mode(tmp_path):
    p = tmp_path / "w.py"
    p.write_text("x = 1\n")
    os.chmod(p, 0o644)
    _write(str(p), "x = 2\n")
    assert mode(p) == 0o644


# --- fix 2: a missing newString is an error, never a deletion ------------

ORIGINAL = "keep = 1\nmid = 2\nkeep2 = 3\n"


def test_missing_newstring_raises_instead_of_deleting(tmp_path):
    p = tmp_path / "g.py"
    p.write_text(ORIGINAL)
    t = tool()
    with pytest.raises(KeyError):
        t.run({"filePath": str(p), "oldString": "mid = 2"})
    assert p.read_text() == ORIGINAL


def test_null_newstring_raises(tmp_path):
    p = tmp_path / "g.py"
    p.write_text(ORIGINAL)
    t = tool()
    with pytest.raises(KeyError):
        t.run({"filePath": str(p), "oldString": "mid = 2", "newString": None})
    assert p.read_text() == ORIGINAL


def test_non_string_newstring_raises(tmp_path):
    p = tmp_path / "g.py"
    p.write_text(ORIGINAL)
    t = tool()
    with pytest.raises(KeyError):
        t.run({"filePath": str(p), "oldString": "mid = 2", "newString": 42})
    assert p.read_text() == ORIGINAL


def test_empty_newstring_is_still_allowed(tmp_path):
    """Blanking a line out is a real edit; only a MISSING field is refused."""
    p = tmp_path / "g.py"
    p.write_text(ORIGINAL)
    r = _edit(str(p), "mid = 2", "")
    assert r.get("error") is not True
    assert p.read_text() == "keep = 1\n\nkeep2 = 3\n"


def test_require_new_string_returns_a_string():
    assert _require_new_string({"newString": "x"}) == "x"
    assert _require_new_string({"newString": ""}) == ""


def test_the_refusal_explains_how_to_delete_a_line():
    with pytest.raises(KeyError) as err:
        _require_new_string({})
    message = str(err.value)
    assert "nothing was changed" in message
    # the advice must be one that actually works
    assert "offset+limit" in message


def test_deleting_a_line_the_way_the_message_advises_works(tmp_path):
    """Follow the tool's own instructions and check they are true."""
    p = tmp_path / "d.py"
    p.write_text(ORIGINAL)
    r = _edit(str(p), "", "keep2 = 3", offset=2, limit=2)
    assert r.get("error") is not True
    assert p.read_text() == "keep = 1\nkeep2 = 3\n"


def test_the_agent_loop_turns_it_into_a_readable_error():
    """The loop catches the exception; the model must still learn why.

    Searched across loop.py *and* its loop_* mixins: the split moved this
    code into loop_tools.py, and the guarantee is about behaviour, not about
    which file it happens to sit in.
    """
    import inspect

    from opencode_py.agent import loop as loop_mod

    root = Path(inspect.getfile(loop_mod)).parent
    src = "".join(
        p.read_text(encoding="utf-8") for p in sorted(root.glob("loop*.py"))
    )
    assert 'f"{name} failed: {e}"' in src
