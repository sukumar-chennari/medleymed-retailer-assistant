"""Random sessions through the REAL chat model, with the same invariants as the
stubbed stateful fuzz (tests/test_state_invariants.py).

Not collected by pytest (the name has no test_ prefix) and not run in CI: it needs
the local Ollama chat model and takes about 25 seconds per session. Run it by hand:

    PYTHONPATH=. python tests/model_fuzz.py [sessions=25] [seed=1] [turns=7]

Why it exists: the stubbed fuzz makes the chat model unavailable, so the model
paths (free-text answers, tool calls, the guardrail middleware) are unreachable
there. The first run of this tool (25 sessions, Nov 6) found a bug the stub fuzz
could not see: with an order waiting for its address, "my email is x@y.com" made
the model call save_address("x@y.com"), saving the email as the shipping address.

Everything runs against a throw-away database and with email forced to mock mode.
A model reply is nondeterministic, so the checks are the invariants (orders match
receipts, no junk addresses, the safety rules, consistency), never exact wording.
"""

import contextlib
import importlib.util
import io
import pathlib
import random
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config, main, store  # noqa: E402  (after the path tweak)

config.SMTP_CONFIGURED = False
store.DB_PATH = pathlib.Path(tempfile.mkdtemp(prefix="model_fuzz_")) / "fuzz.db"
store._init_db()

_spec = importlib.util.spec_from_file_location("state_invariants", ROOT / "tests" / "test_state_invariants.py")
invariants = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(invariants)

from fastapi.testclient import TestClient  # noqa: E402

# Messages that make no sense to the model (pure noise) only slow the run down.
POOL = [m for m in invariants.POOL if m not in ("💊", "?!", "...", "hmm")]


def run(sessions: int, seed: int, turns: int) -> list[tuple[str, list[str], str]]:
    client = TestClient(main.app)
    rng = random.Random(seed)
    violations = []
    log = io.StringIO()
    started = time.time()
    for sequence in range(sessions):
        store.reset_demo_data()
        session_id = f"model-fuzz-{seed}-{sequence}"
        plan = [rng.choice(POOL) for _ in range(turns)]
        if rng.random() < 0.7:
            at = rng.randrange(0, turns - 2)
            plan[at: at + 3] = rng.choice(invariants.RISKY_CHUNKS)
        history: list[str] = []
        for message in plan:
            history.append(message)
            before = store.list_orders("demo_user")
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                res = client.post("/api/chat", json={"session_id": session_id, "text": message})
            reply = res.json().get("reply", "") if res.status_code == 200 else f"HTTP {res.status_code}"
            for problem in invariants._violations(session_id, message, reply, before, history):
                violations.append((problem, history[:], reply[:160]))
                # Printed the moment it is found: a run takes ~25 s per session, and
                # one that stalls or is stopped used to lose every violation it had
                # already seen (the Nov 9 run showed "violations so far 4" and never
                # said what they were).
                print(f"VIOLATION (session {sequence + 1}): {problem}\n    after: {history}\n    reply: {reply[:160]}", flush=True)
        print(f"session {sequence + 1}/{sessions} done, violations so far {len(violations)}, {int(time.time() - started)}s", flush=True)
    return violations


def main_cli(argv: list[str]) -> int:
    sessions = int(argv[1]) if len(argv) > 1 else 25
    seed = int(argv[2]) if len(argv) > 2 else 1
    turns = int(argv[3]) if len(argv) > 3 else 7
    violations = run(sessions, seed, turns)
    print(f"=== MODEL FUZZ RESULT === sessions: {sessions} seed: {seed} violations: {len(violations)}", flush=True)
    seen = set()
    for problem, history, reply in violations:
        if problem[:70] in seen:
            continue
        seen.add(problem[:70])
        print(f"- {problem}\n    after: {history}\n    reply: {reply}", flush=True)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main_cli(sys.argv))
