"""Shared pytest fixtures.

app/store.py hardcodes a single DB_PATH (app/data/app.db) — the same file
the live demo actually uses. Every test file added before this one
deliberately avoided anything that writes through store.py for exactly that
reason (see test_tools.py's and test_retrieval.py's module docstrings).
isolated_db closes that gap: it points store.DB_PATH at a fresh temp file
for the duration of one test, so store's own functions (create_order,
cancel_order, save_address, etc.) can finally be exercised directly without
ever touching the real demo data.
"""

import pytest

from app import store


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test_app.db")
    store._init_db()
    yield
