import itertools
import json
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent / "data"

_catalog: list[dict] = json.loads((DATA_DIR / "catalog.json").read_text())
_catalog_by_id: dict[str, dict] = {p["id"]: p for p in _catalog}
_catalog_by_id_normalized: dict[str, dict] = {p["id"].strip().lower(): p for p in _catalog}

_users: dict[str, dict] = {}
_demo_user = json.loads((DATA_DIR / "demo_user.json").read_text())
_users[_demo_user["user_id"]] = _demo_user

_orders: dict[str, dict] = {}
_order_id_counter = itertools.count(1)

_sessions: dict[str, list] = {}

_pending_orders: dict[str, str] = {}
_pending_emails: dict[str, str] = {}


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
    return _users.get(user_id)


def get_address(user_id: str) -> Optional[str]:
    user = _users.get(user_id)
    return user["address"] if user else None


def save_address(user_id: str, address: str) -> None:
    user = _users.setdefault(user_id, {"user_id": user_id, "address": None})
    user["address"] = address


def get_email(user_id: str) -> Optional[str]:
    user = _users.get(user_id)
    return user["email"] if user else None


def save_email(user_id: str, email: str) -> None:
    user = _users.setdefault(user_id, {"user_id": user_id, "email": None})
    user["email"] = email


def create_order(user_id: str, product_id: str, address: str) -> dict:
    order_id = f"ord-{next(_order_id_counter):04d}"
    product = find_product(product_id)
    if product is None:
        raise ValueError(f"Unknown product_id '{product_id}'")
    order = {
        "order_id": order_id,
        "user_id": user_id,
        "product_id": product["id"],
        "product_name": product["name"],
        "price_usd": product["price_usd"],
        "address": address,
    }
    _orders[order_id] = order
    return order


def list_orders(user_id: str) -> list[dict]:
    return [o for o in _orders.values() if o["user_id"] == user_id]


def get_order(order_id: str) -> Optional[dict]:
    return _orders.get(order_id)


def set_pending_order(session_id: str, product_id: str) -> None:
    _pending_orders[session_id] = product_id


def get_pending_order(session_id: str) -> Optional[str]:
    return _pending_orders.get(session_id)


def clear_pending_order(session_id: str) -> None:
    _pending_orders.pop(session_id, None)


def set_pending_email(session_id: str, order_id: str) -> None:
    _pending_emails[session_id] = order_id


def get_pending_email(session_id: str) -> Optional[str]:
    return _pending_emails.get(session_id)


def clear_pending_email(session_id: str) -> None:
    _pending_emails.pop(session_id, None)


def get_session_messages(session_id: str) -> list:
    return _sessions.setdefault(session_id, [])


def save_session_messages(session_id: str, messages: list) -> None:
    _sessions[session_id] = messages
