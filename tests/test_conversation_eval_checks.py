"""app/conversation_eval.py needs a real model and isn't in CI, so nothing
ever tested the harness's own CHECKS — and they turned out to be weak enough
to score a fabricated reply as a pass. On Oct 7 a run scored the model's
invented receipt as passing "confirms the address already on file", because
the check was just `"123 First Rd" in reply`. These tests pin that the checks
now reject that exact reply and accept the real deterministic one, and that
the state checks read real database state (which a made-up reply can't fake).

Uses the isolated_db fixture (see conftest.py)."""

import pytest

from app import conversation_eval as ce
from app import guardrails, store

pytestmark = pytest.mark.usefixtures("isolated_db")

# The exact reply the model produced on Oct 7 after skipping reorder_last:
# no order existed, and the id and the product name were invented.
FABRICATED_RECEIPT = (
    "Order ID: ord-0002\nProduct: Cough Suppressant Syrup (Dextromorphorphan)\n"
    "Quantity: 1\nTotal: $6.29\nShipping to: 123 First Rd"
)


class TestAddressConfirmationCheck:
    def test_the_fabricated_receipt_no_longer_passes(self):
        assert not ce._asks_to_confirm_address("123 First Rd")(FABRICATED_RECEIPT)

    def test_the_real_deterministic_reply_passes(self):
        real = guardrails.reply_for_deferred_order(
            {"needs_address_confirmation": True, "address_on_file": "123 First Rd"}
        )
        assert ce._asks_to_confirm_address("123 First Rd")(real)

    def test_the_wrong_address_does_not_pass(self):
        real = guardrails.reply_for_deferred_order(
            {"needs_address_confirmation": True, "address_on_file": "999 Other Rd"}
        )
        assert not ce._asks_to_confirm_address("123 First Rd")(real)


class TestStateChecks:
    def test_order_count_reads_the_real_orders_table(self):
        store.save_address("demo_user", "1 Test Way")
        assert ce._order_count_is(0)("s1")
        store.create_order(user_id="demo_user", product_id="fev-001", address="1 Test Way")
        assert ce._order_count_is(1)("s1")
        assert not ce._order_count_is(2)("s1")

    def test_order_status_reads_the_real_order(self):
        store.save_address("demo_user", "1 Test Way")
        order = store.create_order(user_id="demo_user", product_id="fev-001", address="1 Test Way")
        assert not ce._order_status_is(order["order_id"], "cancelled")("s1")
        store.cancel_order(order["order_id"], "demo_user")
        assert ce._order_status_is(order["order_id"], "cancelled")("s1")
        assert not ce._order_status_is("ord-9999", "cancelled")("s1")  # no such order

    def test_awaiting_confirmation_reads_the_real_session_state(self):
        assert not ce._awaiting_address_confirmation("s1")
        store.set_pending_address_confirmation("s1", "fev-001", 1)
        assert ce._awaiting_address_confirmation("s1")
        assert not ce._awaiting_address_confirmation("another-session")

    def test_a_fabricated_receipt_cannot_satisfy_the_state_checks(self):
        # The point of state checks: no order row exists, whatever the reply says.
        assert FABRICATED_RECEIPT  # the text the reply-only check was fooled by
        assert not ce._order_count_is(1)("s1")
        assert not ce._awaiting_address_confirmation("s1")


class TestEveryAddressConfirmationTurnChecksRealState:
    def test_no_confirms_the_address_turn_relies_on_reply_text_alone(self):
        weak = []
        for case in ce.CONVERSATION_CASES:
            for turn in case["turns"]:
                names = " ".join(name for name, _ in turn["checks"]).lower()
                if "address" in names and "on file" in names and not turn.get("state_checks"):
                    weak.append((case["name"], turn["text"]))
        assert not weak, f"turns that only check reply wording: {weak}"

    def test_every_state_check_is_a_named_callable(self):
        for case in ce.CONVERSATION_CASES:
            for turn in case["turns"]:
                for name, fn in turn.get("state_checks", []):
                    assert isinstance(name, str) and name and callable(fn), (case["name"], name)
