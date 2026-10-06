"""static/app.js pre-checks a chosen photo before sending it, because the
backend rejects a bad one with a 422 that the chat can only show as a
generic, retry-forever "something went wrong". Those client-side limits are
hand-copied from app/schemas.py, and there is no JS test runner in this
project, so this keeps the two from silently drifting apart."""

import re
from pathlib import Path

from app import schemas

ROOT = Path(__file__).resolve().parent.parent
APP_JS = (ROOT / "static" / "app.js").read_text()
INDEX_HTML = (ROOT / "static" / "index.html").read_text()


def _js_allowed_types() -> set[str]:
    match = re.search(r"const ALLOWED_IMAGE_TYPES = \[(.*?)\];", APP_JS, re.S)
    assert match, "ALLOWED_IMAGE_TYPES not found in static/app.js"
    return set(re.findall(r'"(image/[a-z0-9.+-]+)"', match.group(1)))


def _schema_allowed_types() -> set[str]:
    group = re.search(r"image/\((.*?)\)", schemas.IMAGE_MEDIA_TYPE_PATTERN).group(1)
    return {f"image/{subtype}" for subtype in group.split("|")}


def test_js_allowed_types_match_the_backend_pattern():
    assert _js_allowed_types() == _schema_allowed_types()


def test_file_picker_accept_list_matches_the_backend_pattern():
    accept = re.search(r'id="image-input"[^>]*accept="([^"]*)"', INDEX_HTML).group(1)
    assert set(accept.split(",")) == _schema_allowed_types()


def test_every_js_allowed_type_actually_passes_backend_validation():
    for media_type in _js_allowed_types():
        schemas.ChatRequest(session_id="s1", text="hi", image_media_type=media_type)


def test_js_size_limit_stays_under_the_backend_base64_cap():
    raw_limit = int(re.search(r"const MAX_IMAGE_BYTES = ([\d_]+);", APP_JS).group(1).replace("_", ""))
    base64_chars = (raw_limit + 2) // 3 * 4  # base64 length for that many raw bytes
    assert base64_chars <= schemas.MAX_IMAGE_B64_LEN


def _app_js_without_comments_and_helpers(*helper_names: str) -> str:
    source = re.sub(r"//[^\n]*", "", APP_JS)
    for name in helper_names:
        source = re.sub(rf"function {name}\([^)]*\) \{{.*?\n\}}\n", "", source, flags=re.S)
    return source


def test_localstorage_is_only_touched_inside_the_guarded_helpers():
    # Real bug: localStorage.getItem/setItem ran at the top level of app.js
    # with no try/catch, so blocked storage (SecurityError) killed the whole
    # script before any listener attached — dashboard stuck on "Loading…",
    # chat button dead. Reproduced in a real browser; there's no JS runner
    # here, so this keeps unguarded access from being reintroduced.
    remainder = _app_js_without_comments_and_helpers("loadStoredSessionId", "storeSessionId")
    assert "localStorage" not in remainder


def test_randomuuid_is_only_called_inside_the_fallback_helper():
    # crypto.randomUUID doesn't exist on non-secure origins (http://<lan-ip>),
    # so a bare call at top level or in the reset handler threw there too.
    remainder = _app_js_without_comments_and_helpers("newSessionId")
    assert "randomUUID" not in remainder
