"""tests/model_fuzz.py is a hand-run tool (real model, not in CI), but its reporting
is plain code: a violation must be printed the moment it is found, not only in the
final summary (a run that stalls or is stopped used to lose them all; Nov 9).

Run in a subprocess with a stubbed model: importing the tool points the store at a
throw-away database and forces mock email, which must not leak into this process."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_CODE = """
import sys
sys.path.insert(0, "tests")
import model_fuzz
from app import main
main.run_turn = lambda messages, **kw: ("stub reply", messages)
calls = {"n": 0}
def fake(session_id, message, reply, before, history):
    calls["n"] += 1
    return ["FAKE problem %d" % calls["n"]] if calls["n"] in (2, 6) else []
model_fuzz.invariants._violations = fake
violations = model_fuzz.run(sessions=2, seed=1, turns=4)
print("RETURNED", len(violations))
"""


def test_violations_are_printed_as_they_are_found_not_only_at_the_end():
    result = subprocess.run([sys.executable, "-c", _CODE], cwd=ROOT, capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
    assert result.returncode == 0, result.stderr[-800:]
    out = result.stdout
    assert "VIOLATION (session 1): FAKE problem 2" in out
    assert "VIOLATION (session 2): FAKE problem 6" in out
    assert "RETURNED 2" in out
    # the first one is shown BEFORE the session that found it is reported finished
    assert out.index("VIOLATION (session 1)") < out.index("session 1/2 done")
