"""Real bug: every read_text()/write_text() in app/ relied on the platform's
default encoding, while 8 of the 10 knowledge-base files contain non-ASCII
text (em-dashes). On a machine whose default isn't UTF-8 — a Windows laptop
running this demo is the realistic case — load_documents() raised
UnicodeDecodeError on the first em-dash, and since retrieval loads the
knowledge base at import, the whole app couldn't start. (rag_eval's HTML
report, which contains "—", "→" and "§", had the mirror problem on write.)

The behavioral test forces a non-UTF-8 default in a subprocess so it fails
the same way on any machine, including CI. The subprocess imports only
app.data_ingest, which never touches app.db.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_the_knowledge_base_loads_under_a_non_utf8_default_encoding():
    code = (
        "import sys\n"
        "from app import data_ingest\n"
        "docs = data_ingest.load_documents()\n"
        "assert len(docs) >= 1, 'no documents loaded'\n"
        "assert any('\\u2014' in text for _, text in docs), 'em-dash was not preserved'\n"
    )
    env = {**os.environ, "LC_ALL": "C", "PYTHONUTF8": "0"}
    env.pop("PYTHONIOENCODING", None)
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-800:]


def test_every_text_read_and_write_in_app_names_its_encoding():
    # Static guard for the sites the behavioral test doesn't import (store.py's
    # JSON reads, retrieval.py, rag_eval.py's report): nothing may fall back to
    # the platform default.
    unspecified = []
    for path in sorted((ROOT / "app").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("read_text", "write_text")
                and not any(kw.arg == "encoding" for kw in node.keywords)
            ):
                unspecified.append(f"{path.name}:{node.lineno} .{node.func.attr}()")
    assert not unspecified, f"missing encoding=: {unspecified}"
