import json
import sqlite3
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "app.db"

_catalog: list[dict] = json.loads((DATA_DIR / "catalog.json").read_text())
_catalog_by_id: dict[str, dict] = {p["id"]: p for p in _catalog}
_catalog_by_id_normalized: dict[str, dict] = {p["id"].strip().lower(): p for p in _catalog}


def _connect() -> sqlite3.Connection:
    """A fresh short-lived connection per call rather than one shared
    connection — FastAPI's sync routes run in a thread pool, and sqlite3
    connections aren't safe to share across threads. SQLite itself already
    serializes writes, and this app's traffic is a single live demo user,
    so the per-call connection cost is irrelevant here."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                address TEXT,
                email TEXT
            );
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                product_id TEXT NOT NULL,
                product_name TEXT NOT NULL,
                price_usd REAL NOT NULL,
                quantity INTEGER NOT NULL,
                total_price_usd REAL NOT NULL,
                address TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS session_state (
                session_id TEXT PRIMARY KEY,
                messages_json TEXT NOT NULL DEFAULT '[]',
                pending_order_json TEXT,
                pending_email TEXT,
                pending_clarification TEXT,
                pending_address_confirmation_json TEXT,
                last_products_json TEXT,
                last_recommended_product TEXT
            );
            CREATE TABLE IF NOT EXISTS metrics_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                session_id TEXT,
                event_type TEXT NOT NULL,
                name TEXT NOT NULL,
                value REAL,
                detail TEXT
            );
            """
        )
        # Seed the demo user once — on every later startup the table is
        # already non-empty, so a previously saved address/email survives a
        # restart instead of resetting to the demo_user.json defaults.
        if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
            demo_user = json.loads((DATA_DIR / "demo_user.json").read_text())
            conn.execute(
                "INSERT INTO users (user_id, name, address, email) VALUES (?, ?, ?, ?)",
                (demo_user["user_id"], demo_user.get("name"), demo_user.get("address"), demo_user.get("email")),
            )


_init_db()


def get_catalog() -> list[dict]:
    return _catalog


def find_product(product_id: str) -> Optional[dict]:
    """Matches case/whitespace-insensitively — small local models occasionally
    emit product ids with different casing or stray whitespace."""
    if not product_id:
        return None
    return _catalog_by_id.get(product_id) or _catalog_by_id_normalized.get(product_id.strip().lower())


def lookup_by_category(category: str) -> list[dict]:
    category = category.strip().lower()
    return [p for p in _catalog if p["category"] == category]


def get_user(user_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def get_address(user_id: str) -> Optional[str]:
    user = get_user(user_id)
    return user["address"] if user else None


def save_address(user_id: str, address: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO users (user_id, address) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET address = excluded.address",
            (user_id, address),
        )


def get_email(user_id: str) -> Optional[str]:
    user = get_user(user_id)
    return user["email"] if user else None


def save_email(user_id: str, email: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO users (user_id, email) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET email = excluded.email",
            (user_id, email),
        )


def _next_order_id(conn: sqlite3.Connection) -> str:
    """Derived from the highest existing numeric suffix rather than a row
    count, so a restart never collides with — or reuses — an id from before
    the restart even if an order was ever deleted."""
    max_n = 0
    for (order_id,) in conn.execute("SELECT order_id FROM orders").fetchall():
        try:
            max_n = max(max_n, int(order_id.split("-")[-1]))
        except ValueError:
            continue
    return f"ord-{max_n + 1:04d}"


def create_order(user_id: str, product_id: str, address: str, quantity: int = 1) -> dict:
    product = find_product(product_id)
    if product is None:
        raise ValueError(f"Unknown product_id '{product_id}'")
    order = {
        "user_id": user_id,
        "product_id": product["id"],
        "product_name": product["name"],
        "price_usd": product["price_usd"],
        "quantity": quantity,
        "total_price_usd": round(product["price_usd"] * quantity, 2),
        "address": address,
    }
    with _connect() as conn:
        order_id = _next_order_id(conn)
        order["order_id"] = order_id
        conn.execute(
            "INSERT INTO orders (order_id, user_id, product_id, product_name, price_usd, "
            "quantity, total_price_usd, address) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                order_id, order["user_id"], order["product_id"], order["product_name"],
                order["price_usd"], order["quantity"], order["total_price_usd"], order["address"],
            ),
        )
    return order


def list_orders(user_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM orders WHERE user_id = ?", (user_id,)).fetchall()
        return [dict(row) for row in rows]


def get_order(order_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
        return dict(row) if row else None


def _ensure_session_row(conn: sqlite3.Connection, session_id: str) -> None:
    conn.execute("INSERT OR IGNORE INTO session_state (session_id) VALUES (?)", (session_id,))


def _get_session_column(session_id: str, column: str):
    with _connect() as conn:
        row = conn.execute(f"SELECT {column} FROM session_state WHERE session_id = ?", (session_id,)).fetchone()
        return row[0] if row else None


def _set_session_column(session_id: str, column: str, value) -> None:
    with _connect() as conn:
        _ensure_session_row(conn, session_id)
        conn.execute(f"UPDATE session_state SET {column} = ? WHERE session_id = ?", (value, session_id))


def set_pending_order(session_id: str, product_id: str, quantity: int = 1) -> None:
    _set_session_column(session_id, "pending_order_json", json.dumps({"product_id": product_id, "quantity": quantity}))


def get_pending_order(session_id: str) -> Optional[dict]:
    raw = _get_session_column(session_id, "pending_order_json")
    return json.loads(raw) if raw else None


def clear_pending_order(session_id: str) -> None:
    _set_session_column(session_id, "pending_order_json", None)


def set_pending_address_confirmation(session_id: str, product_id: str, quantity: int = 1) -> None:
    _set_session_column(
        session_id, "pending_address_confirmation_json", json.dumps({"product_id": product_id, "quantity": quantity})
    )


def get_pending_address_confirmation(session_id: str) -> Optional[dict]:
    raw = _get_session_column(session_id, "pending_address_confirmation_json")
    return json.loads(raw) if raw else None


def clear_pending_address_confirmation(session_id: str) -> None:
    _set_session_column(session_id, "pending_address_confirmation_json", None)


def set_pending_email(session_id: str, order_id: str) -> None:
    _set_session_column(session_id, "pending_email", order_id)


def get_pending_email(session_id: str) -> Optional[str]:
    return _get_session_column(session_id, "pending_email")


def clear_pending_email(session_id: str) -> None:
    _set_session_column(session_id, "pending_email", None)


def set_pending_clarification(session_id: str, trigger: str) -> None:
    _set_session_column(session_id, "pending_clarification", trigger)


def get_pending_clarification(session_id: str) -> Optional[str]:
    return _get_session_column(session_id, "pending_clarification")


def clear_pending_clarification(session_id: str) -> None:
    _set_session_column(session_id, "pending_clarification", None)


def set_last_products(session_id: str, products: list[dict]) -> None:
    _set_session_column(session_id, "last_products_json", json.dumps(products))


def get_last_products(session_id: str) -> Optional[list[dict]]:
    raw = _get_session_column(session_id, "last_products_json")
    return json.loads(raw) if raw else None


def set_last_recommended_product(session_id: str, product_id: str) -> None:
    _set_session_column(session_id, "last_recommended_product", product_id)


def get_last_recommended_product(session_id: str) -> Optional[str]:
    return _get_session_column(session_id, "last_recommended_product")


def clear_last_recommended_product(session_id: str) -> None:
    _set_session_column(session_id, "last_recommended_product", None)


def get_session_messages(session_id: str) -> list:
    raw = _get_session_column(session_id, "messages_json")
    return json.loads(raw) if raw else []


def save_session_messages(session_id: str, messages: list) -> None:
    _set_session_column(session_id, "messages_json", json.dumps(messages))


def log_metric_event(
    session_id: Optional[str], event_type: str, name: str, value: Optional[float] = None, detail: Optional[str] = None
) -> None:
    """A single append-only observability log — every guardrail trigger,
    tool call, and retrieval score gets one row here. Deliberately simple
    (no aggregation at write time, no separate metrics service) since the
    read side (get_metrics_summary) just aggregates with SQL on demand,
    which is plenty at this traffic scale and keeps this from becoming its
    own subsystem to maintain."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO metrics_events (session_id, event_type, name, value, detail) VALUES (?, ?, ?, ?, ?)",
            (session_id, event_type, name, value, detail),
        )


def get_metrics_summary() -> dict:
    """Aggregates the raw event log into the numbers actually worth
    showing: how confident retrieval has been on average, and how often
    each guardrail and tool has fired — turning "we have guardrails" from
    a claim in a doc into a real, queryable count."""
    with _connect() as conn:
        total_events = conn.execute("SELECT COUNT(*) FROM metrics_events").fetchone()[0]

        retrieval_rows = conn.execute(
            "SELECT value FROM metrics_events WHERE event_type = 'retrieval' AND value IS NOT NULL"
        ).fetchall()
        retrieval_scores = [row[0] for row in retrieval_rows]
        avg_retrieval_confidence = (
            round(sum(retrieval_scores) / len(retrieval_scores), 3) if retrieval_scores else None
        )

        guardrail_counts = [
            {"name": row[0], "count": row[1]}
            for row in conn.execute(
                "SELECT name, COUNT(*) FROM metrics_events WHERE event_type = 'guardrail' "
                "GROUP BY name ORDER BY COUNT(*) DESC"
            ).fetchall()
        ]
        tool_call_counts = [
            {"name": row[0], "count": row[1]}
            for row in conn.execute(
                "SELECT name, COUNT(*) FROM metrics_events WHERE event_type = 'tool_call' "
                "GROUP BY name ORDER BY COUNT(*) DESC"
            ).fetchall()
        ]

        return {
            "total_events": total_events,
            "avg_retrieval_confidence": avg_retrieval_confidence,
            "retrieval_sample_count": len(retrieval_scores),
            "guardrail_counts": guardrail_counts,
            "guardrail_total": sum(g["count"] for g in guardrail_counts),
            "tool_call_counts": tool_call_counts,
        }
