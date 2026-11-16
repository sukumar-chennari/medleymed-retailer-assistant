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


@pytest.fixture(autouse=True)
def _never_touch_the_real_database(tmp_path, monkeypatch):
    """Every test gets a throw-away database, whether or not it asks for isolated_db.

    Found Nov 16: two test classes in test_contraindications.py used the store without
    isolated_db, so EVERY run of the suite wrote two session rows (and a pending order)
    into the real demo database (app/data/app.db); and a fixture of mine that wrote to
    the store ran before isolated_db was applied (autouse fixtures run first) and put 17
    test orders there. Making the isolation unconditional removes that whole class of
    mistake. Tests that want the explicit name can still use isolated_db: it re-points
    DB_PATH at its own file.
    """
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "autouse_app.db")
    store._init_db()
    yield
