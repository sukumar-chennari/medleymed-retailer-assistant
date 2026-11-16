"""No test may touch the real demo database (app/data/app.db).

conftest.py makes that unconditional with an autouse fixture. This test deliberately
asks for NO fixture: it is what proves a test that forgets isolated_db is still safe
(two classes in test_contraindications.py forgot it, and every run of the suite wrote
session rows into the real database until Nov 16)."""

import sqlite3
from pathlib import Path

from app import store

REAL_DB = Path(store.__file__).resolve().parent / "data" / "app.db"


def test_a_test_with_no_fixtures_is_pointed_at_a_throwaway_database():
    assert Path(store.DB_PATH).resolve() != REAL_DB
    assert "autouse_app.db" in str(store.DB_PATH)


def test_writes_from_such_a_test_do_not_reach_the_real_database():
    def snapshot():
        if not REAL_DB.exists():
            return None
        con = sqlite3.connect(REAL_DB)
        try:
            return {t: con.execute(f"select * from {t}").fetchall()
                    for (t,) in con.execute("select name from sqlite_master where type='table'")}
        finally:
            con.close()

    before = snapshot()
    store.save_address("demo_user", "99 Isolation Test Road")
    store.save_session_messages("isolation-probe", [{"role": "user", "content": "x"}])
    assert store.get_address("demo_user") == "99 Isolation Test Road"      # it did write... somewhere
    assert snapshot() == before                                            # ...but not there
