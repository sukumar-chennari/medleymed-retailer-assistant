"""Integration tests for the /api/chat route's deterministic dispatch
branches in main.py's chat() — the ones that never call the LLM at all
(pending clarification/address-confirmation/email/order resolution, bare
product selection, the last-recommended-product affirmative shortcut).
This is the actual endpoint every one of those pending-state branches
lives in, previously only exercised end-to-end by the slow, LLM-dependent
app/conversation_eval.py script (or live curl testing) — these tests get
the same real dispatch logic under fast, reliable pytest by seeding
session state directly via store.set_pending_*/set_last_* rather than
driving a full multi-turn conversation through the LLM to reach it.

Uses the isolated_db fixture (see conftest.py) and forces
config.SMTP_CONFIGURED to False, same reasoning as test_tools_orders.py/
test_main_completion.py: this machine's real .env has live-demo SMTP
credentials, and a test suite must never risk a real send.

Anything that falls through to run_turn (a plain symptom description
with no pending state, a genuinely ambiguous reply) still needs the real
LLM and stays in conversation_eval.py / manual testing.
"""

import pytest
from fastapi.testclient import TestClient

from app import config, main, store, tools

pytestmark = pytest.mark.usefixtures("isolated_db")

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def _force_mock_email(monkeypatch):
    monkeypatch.setattr(config, "SMTP_CONFIGURED", False)


def _chat(session_id: str, text: str) -> str:
    res = client.post("/api/chat", json={"session_id": session_id, "text": text})
    res.raise_for_status()
    return res.json()["reply"]


class TestMissingInput:
    def test_no_text_and_no_image_is_rejected(self):
        res = client.post("/api/chat", json={"session_id": "s1", "text": ""})
        assert res.status_code == 400


class TestUnhandledErrorSafetyNet:
    def test_an_unexpected_exception_returns_a_friendly_200_not_a_500(self, monkeypatch):
        # Last-resort safety net (main.py's own comment on it): whatever
        # goes wrong, the user gets a real 200 with a friendly reply
        # instead of a raw 500 that skips this reply entirely.
        def _boom(*args, **kwargs):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(store, "save_session_messages", _boom)
        store.set_pending_clarification("s1", "cough")  # any deterministic path reaches save_session_messages
        res = client.post("/api/chat", json={"session_id": "s1", "text": "dry"})
        assert res.status_code == 200
        assert res.json()["reply"] == main.GENERIC_FAILURE_REPLY


class TestPendingClarification:
    def test_a_clear_answer_resolves_deterministically(self):
        store.set_pending_clarification("s1", "cough")
        reply = _chat("s1", "dry")
        assert "Cough Suppressant" in reply
        assert store.get_pending_clarification("s1") is None


class TestPendingAddressConfirmation:
    def _seed(self, session_id: str):
        store.save_address("demo_user", "123 Main St")
        store.set_pending_address_confirmation(session_id, "fev-001", quantity=1)

    def test_affirmative_completes_the_order_with_the_address_on_file(self):
        self._seed("s1")
        reply = _chat("s1", "yes")
        assert "Order confirmed!" in reply
        assert "123 Main St" in reply
        assert store.get_pending_address_confirmation("s1") is None

    def test_a_new_address_replaces_the_one_on_file(self):
        self._seed("s1")
        reply = _chat("s1", "456 New Ave")
        assert "Order confirmed!" in reply
        assert "456 New Ave" in reply

    def test_wanting_a_different_address_asks_for_it_without_completing(self):
        self._seed("s1")
        reply = _chat("s1", "no, a different one please")
        assert "new shipping address" in reply.lower()
        # Still pending — the order isn't placed until the real address arrives.
        assert store.get_pending_address_confirmation("s1") is not None
        assert store.list_orders("demo_user") == []

    def test_declining_cancels_the_pending_order_without_placing_it(self):
        self._seed("s1")
        reply = _chat("s1", "cancel that")
        assert "won't place that order" in reply.lower()
        assert store.get_pending_address_confirmation("s1") is None
        assert store.list_orders("demo_user") == []

    def test_an_unrelated_message_reaches_a_real_turn_instead_of_repeating_the_question(self, monkeypatch):
        # Real bug: this used to unconditionally repeat "Just to confirm —
        # should I ship to X..." for ANY reply that wasn't yes/an address/
        # "different address" — even a completely unrelated message ("who
        # is narendra modi") — trapping the conversation instead of just
        # answering it. Same trapping bug already fixed for
        # pending_product_id/pending_email_order_id, just missing here.
        # Mirrors those siblings' own tests: stub run_turn instead of
        # needing a real LLM call, since this file stays fast/deterministic.
        self._seed("s1")
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("stubbed reply", messages))
        reply = _chat("s1", "who is narendra modi")
        assert reply == "stubbed reply"
        # Stays set, so a genuine answer given afterward still completes it.
        assert store.get_pending_address_confirmation("s1") is not None


class TestPendingEmail:
    def _seed(self, session_id: str) -> str:
        store.save_address("demo_user", "123 Main St")
        order = store.create_order(user_id="demo_user", product_id="fev-001", address="123 Main St")
        store.set_pending_email(session_id, order["order_id"])
        return order["order_id"]

    def test_a_real_email_completes_it(self):
        self._seed("s1")
        reply = _chat("s1", "a@example.com")
        assert "sent the confirmation" in reply
        assert store.get_email("demo_user") == "a@example.com"
        assert store.get_pending_email("s1") is None

    def test_declining_leaves_the_order_confirmed_without_an_email(self):
        self._seed("s1")
        reply = _chat("s1", "no thanks")
        assert "already confirmed" in reply.lower()
        assert store.get_pending_email("s1") is None
        assert store.get_email("demo_user") is None


class TestPendingOrderNoAddressYet:
    def _seed(self, session_id: str):
        store.set_pending_order(session_id, "fev-001", quantity=1)

    def test_supplying_an_address_completes_the_order(self):
        self._seed("s1")
        reply = _chat("s1", "123 First Time Rd")
        assert "Order confirmed!" in reply
        assert "123 First Time Rd" in reply

    def test_declining_cancels_the_pending_order_without_placing_it(self):
        self._seed("s1")
        reply = _chat("s1", "no, forget it")
        assert "won't place that order" in reply.lower()
        assert store.list_orders("demo_user") == []
        assert store.get_pending_order("s1") is None


class TestBareSelectionAndLastRecommended:
    def test_bare_numeric_selection_starts_the_real_order(self):
        products = [
            {"id": "fev-001", "name": "Paracetamol 500mg Tablets", "description": "x"},
            {"id": "fev-002", "name": "Paracetamol Extra Strength 650mg", "description": "y"},
        ]
        store.set_last_products("s1", products)
        reply = _chat("s1", "2")
        # No address on file yet in this isolated DB — start_order defers.
        assert "shipping address" in reply.lower()
        assert store.get_pending_order("s1") == {"product_id": "fev-002", "quantity": 1}

    def test_bare_affirmative_after_a_single_recommendation_starts_the_real_order(self):
        store.set_last_recommended_product("s1", "fev-001")
        reply = _chat("s1", "yes")
        assert "shipping address" in reply.lower()
        assert store.get_pending_order("s1") == {"product_id": "fev-001", "quantity": 1}
        assert store.get_last_recommended_product("s1") is None


class TestYesAfterAnInformationalAnswer:
    """Real bug: any single product named in any free-text model reply was
    recorded as "the product just recommended", so after a dosage answer a
    bare "yes" (to "anything else?") started an order for a product the
    customer only asked about. Reproduced end to end; no model runs, since a
    bare "yes" with a remembered product is handled deterministically."""

    def test_yes_after_a_dosage_answer_does_not_start_an_order(self, monkeypatch):
        from app import guardrails

        guardrails.remember_recommended_product(
            "s1", "For Paracetamol 500mg Tablets, adults take 1-2 tablets every 4-6 hours."
        )
        # "yes" is now free text for the model (nothing deterministic claims it).
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("stubbed", messages))
        assert _chat("s1", "yes") == "stubbed"
        assert store.get_pending_order("s1") is None

    def test_yes_after_a_real_recommendation_still_starts_the_order(self):
        from app import guardrails

        guardrails.remember_recommended_product(
            "s1", "Paracetamol 500mg Tablets (fev-001) would be a good fit. Would you like to order this?"
        )
        reply = _chat("s1", "yes")
        assert "shipping address" in reply.lower()
        assert store.get_pending_order("s1") == {"product_id": "fev-001", "quantity": 1}


class TestPlainReorderIsDeterministic:
    """Real bug: "reorder that" was left to the model, which in 3 of ~8 live
    conversation_eval.py runs skipped its reorder_last tool call (inventing a
    receipt, or replying generically) so no address confirmation was ever set
    up. A plain request is now handled in code. run_turn is stubbed to RAISE,
    proving the model is never reached for these."""

    def _seed_one_order(self, session_id="s1"):
        store.save_address("demo_user", "123 First Rd")
        return store.create_order(user_id="demo_user", product_id="col-004", address="123 First Rd")

    def _model_must_not_run(self, monkeypatch):
        def boom(*args, **kwargs):
            raise AssertionError("the model must not be reached for a plain reorder request")

        monkeypatch.setattr(main, "run_turn", boom)

    def test_a_plain_request_sets_up_the_address_confirmation(self, monkeypatch):
        self._seed_one_order()
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", "reorder that for me")
        assert "address on file" in reply.lower() and "123 First Rd" in reply
        assert store.get_pending_address_confirmation("s1") == {"product_id": "col-004", "quantity": 1}
        assert len(store.list_orders("demo_user")) == 1  # nothing placed yet

    def test_it_works_with_an_email_confirmation_still_pending(self, monkeypatch):
        # THE scenario that flaked: an order was just placed with no email on
        # file, so pending_email is set, which used to route this to the model.
        order = self._seed_one_order()
        store.set_pending_email("s1", order["order_id"])
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", "can you order that again")
        assert "address on file" in reply.lower()
        assert store.get_pending_address_confirmation("s1") is not None
        assert store.get_pending_email("s1") == order["order_id"]  # still there

    def test_a_cancelled_last_order_can_be_reordered(self, monkeypatch):
        order = self._seed_one_order()
        store.cancel_order(order["order_id"], "demo_user")
        self._model_must_not_run(monkeypatch)
        _chat("s1", "reorder that")
        assert store.get_pending_address_confirmation("s1") == {"product_id": "col-004", "quantity": 1}

    def test_with_no_previous_order_it_says_so_instead_of_inventing_one(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", "reorder that")
        assert "No previous orders" in reply
        assert store.get_pending_address_confirmation("s1") is None

    def test_a_follow_up_rejecting_the_address_now_asks_for_a_new_one(self, monkeypatch):
        # The turn that fell through to the model when the reorder flaked.
        self._seed_one_order()
        self._model_must_not_run(monkeypatch)
        _chat("s1", "reorder that")
        reply = _chat("s1", "no, i will give a different one")
        assert "new shipping address" in reply.lower()

    def test_anything_beyond_a_plain_request_is_still_left_to_the_model(self, monkeypatch):
        self._seed_one_order()
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        for text in [
            "reorder that, 2 of them", "reorder another", "reorder it twice", "reorder fev-001",
            "don't reorder that", "reorder that and email me at a@b.com",
            "I would like to reorder that same medicine for my whole extended family tomorrow",
        ]:
            assert _chat("s1", text) == "from the model", text
            assert store.get_pending_address_confirmation("s1") is None, text

    def test_a_pending_address_confirmation_is_not_hijacked(self, monkeypatch):
        self._seed_one_order()
        store.set_pending_address_confirmation("s1", "fev-001", 1)
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        assert _chat("s1", "reorder that") == "from the model"
        assert store.get_pending_address_confirmation("s1") == {"product_id": "fev-001", "quantity": 1}


class TestNamedProductOrder:
    """The dashboard's "Ask Assistant" button sends "I'd like to order <name>".
    That used to reach the model, which asked an age/dry-or-wet clarification
    or claimed the product wasn't in the catalog: 0 of 10 catalog products
    started an order. A message that clearly orders exactly one named
    product now starts the order deterministically."""

    @staticmethod
    def _model_must_not_run(monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("the model must not be reached for a plainly named order")

        monkeypatch.setattr(main, "run_turn", _boom)

    @pytest.mark.parametrize("product", store.get_catalog(), ids=lambda p: p["id"])
    def test_the_ask_assistant_button_text_starts_that_products_order(self, monkeypatch, product):
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", f"I'd like to order {product['name']}")
        assert "address" in reply.lower()
        assert store.get_pending_order("s1")["product_id"] == product["id"]

    @pytest.mark.parametrize("text, quantity", [
        ("I'd like to order 2 Ibuprofen 200mg Tablets", 2),
        ("order Ibuprofen 200mg Tablets twice", 2),
        ("buy a pair of Ibuprofen 200mg Tablets", 2),
        ("order one Ibuprofen 200mg Tablets", 1),
        ("order Ibuprofen 200mg Tablets x2", 2),
        ("order fev-003 x 2", 2),
    ])
    def test_an_explicit_quantity_is_carried_into_the_order(self, monkeypatch, text, quantity):
        self._model_must_not_run(monkeypatch)
        _chat("s1", text)
        assert store.get_pending_order("s1") == {"product_id": "fev-003", "quantity": quantity}

    def test_a_quantity_over_the_limit_is_capped_and_the_customer_is_told(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", "order 5 Ibuprofen 200mg Tablets")
        assert "capped at our per-order limit" in reply
        assert store.get_pending_order("s1")["quantity"] == 2

    def test_a_product_id_alone_is_enough_to_name_the_product(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        _chat("s1", "please order col-005")
        assert store.get_pending_order("s1")["product_id"] == "col-005"

    def test_it_works_with_an_email_confirmation_still_pending(self, monkeypatch):
        store.set_pending_email("s1", "ord-previous")
        self._model_must_not_run(monkeypatch)
        _chat("s1", "I'd like to order Ibuprofen 200mg Tablets")
        assert store.get_pending_order("s1")["product_id"] == "fev-003"
        assert store.get_pending_email("s1") == "ord-previous"

    def test_anything_beyond_a_plain_named_order_is_left_to_the_model(self, monkeypatch):
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        for text in [
            "order another Ibuprofen 200mg Tablets", "order more Ibuprofen 200mg Tablets",
            "order 2 Ibuprofen 200mg Tablets, actually three", "order two Ibuprofen 200mg Tablets x 3",
            "don't order Ibuprofen 200mg Tablets", "dosage of Ibuprofen 200mg Tablets",
            "what are the side effects of Ibuprofen 200mg Tablets before I order",
            "I'd like to order Ibuprofen 200mg Tablets or Cetirizine 10mg Antihistamine Tablets",
            "Ibuprofen 200mg Tablets", "order paracetamol",
            "order Ibuprofen 200mg Tablets and email me at a@b.com",
            "I would really like to order Ibuprofen 200mg Tablets for the whole entire family please today",
        ]:
            assert _chat("s1", text) == "from the model", text
            assert store.get_pending_order("s1") is None, text

    def test_naming_another_product_while_one_awaits_its_address_switches_the_order(self, monkeypatch):
        # Previously the whole sentence was saved as the shipping address
        # and the FIRST product was ordered to it.
        store.set_pending_order("s1", "fev-001", 1)
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", "I'd like to order Ibuprofen 200mg Tablets")
        assert "address" in reply.lower()
        assert store.get_pending_order("s1")["product_id"] == "fev-003"
        assert store.list_orders("demo_user") == []  # nothing was placed


class TestProductNameIsNotTakenAsAnAddress:
    def test_a_product_phrase_with_an_order_waiting_never_becomes_the_address(self, monkeypatch):
        store.set_pending_order("s1", "fev-001", 1)
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        for text in ["Ibuprofen 200mg Tablets", "paracetamol 500mg please"]:
            assert _chat("s1", text) == "from the model", text
            assert store.list_orders("demo_user") == [], text
            assert store.get_pending_order("s1")["product_id"] == "fev-001", text
        # "I want Cetirizine 10mg" is a clear request for that product: it
        # switches the waiting order (see TestSwitchingProductWhileAnOrderWaits)
        # but is never taken as the address either.
        _chat("s1", "I want Cetirizine 10mg")
        assert store.list_orders("demo_user") == []
        assert store.get_address("demo_user") in (None, "")

    def test_a_real_address_still_completes_the_order(self, monkeypatch):
        store.set_pending_order("s1", "fev-001", 1)
        reply = _chat("s1", "Flat 4, 10 Paracetamol Lane")
        assert "Order confirmed!" in reply and "Flat 4, 10 Paracetamol Lane" in reply


class TestNamingAnOrderIsNotADecline:
    """Real bug: with a pending email / address question, a short message
    naming an existing order ("cancel ord-0002 please") matched the decline
    words and was answered "the order's already confirmed without an email
    receipt" — nothing was cancelled and the customer was told it was fine."""

    def _model_reply(self, monkeypatch):
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))

    @pytest.mark.parametrize("text", ["cancel ord-0001 please", "no, cancel ord-0001", "cancel ORD-0001"])
    def test_pending_email(self, monkeypatch, text):
        # Handled in code since Oct 27 (TestCancelRequestsTheModelGotWrong):
        # the invariant is unchanged — the message is NOT swallowed as "no, skip
        # the email receipt" — and the order is now really cancelled.
        store.save_address("demo_user", "10 High St, Leeds")
        order = store.create_order("demo_user", "fev-003", "10 High St, Leeds", 1)
        store.set_pending_email("s1", order["order_id"])
        self._model_reply(monkeypatch)
        reply = _chat("s1", text)
        assert "without an email receipt" not in reply
        assert "Order ord-0001" in reply and "has been cancelled" in reply
        assert store.get_order("ord-0001")["status"] == "cancelled"

    @pytest.mark.parametrize("text", ["cancel ord-0001 please", "no, cancel ord-0001"])
    def test_pending_address_confirmation(self, monkeypatch, text):
        store.set_pending_address_confirmation("s1", "fev-001", 1)
        self._model_reply(monkeypatch)
        assert _chat("s1", text) == "from the model"
        assert store.get_pending_address_confirmation("s1") == {"product_id": "fev-001", "quantity": 1}

    @pytest.mark.parametrize("text", ["cancel ord-0001 please", "no, cancel ord-0001"])
    def test_pending_order_address(self, monkeypatch, text):
        store.set_pending_order("s1", "fev-001", 1)
        self._model_reply(monkeypatch)
        assert _chat("s1", text) == "from the model"
        assert store.get_pending_order("s1")["product_id"] == "fev-001"

    def test_plain_declines_still_work_in_each_context(self, monkeypatch):
        self._model_reply(monkeypatch)
        store.set_pending_email("s1", "ord-0001")
        assert "without an email receipt" in _chat("s1", "no thanks")
        store.set_pending_address_confirmation("s2", "fev-001", 1)
        assert "won't place that order" in _chat("s2", "cancel that")
        store.set_pending_order("s3", "fev-001", 1)
        assert "without a shipping address" in _chat("s3", "no thanks")


class TestSavedAddressRequests:
    """Real bug (live probe): "change my shipping address to X" made the model
    call save_address and then invent an "Order updated!" receipt for orders it
    never touched; the guard replaced that with "what symptom is this for
    (fever or cold)?". "What address do you have on file?" got the same
    non-answer. Both are now answered from the real saved address."""

    @staticmethod
    def _model_must_not_run(monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("the model must not be reached")

        monkeypatch.setattr(main, "run_turn", _boom)

    def test_changing_the_address_saves_it_for_future_orders(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", "change my shipping address to 99 New Rd, York")
        assert store.get_address("demo_user") == "99 New Rd, York"
        assert "99 New Rd, York" in reply and "future orders" in reply
        assert "already placed" not in reply  # no orders exist yet

    def test_it_says_plainly_that_existing_orders_keep_their_address(self, monkeypatch):
        store.save_address("demo_user", "10 High St, Leeds")
        store.create_order("demo_user", "fev-001", "10 High St, Leeds", 1)
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", "Please update my address to 99 New Rd, York.")
        assert store.get_address("demo_user") == "99 New Rd, York"
        assert "still ship to the address they were placed with" in reply
        assert store.list_orders("demo_user")[0]["address"] == "10 High St, Leeds"

    def test_asking_for_the_saved_address_reads_it_back(self, monkeypatch):
        store.save_address("demo_user", "10 High St, Leeds")
        self._model_must_not_run(monkeypatch)
        assert "10 High St, Leeds" in _chat("s1", "what address do you have on file for me?")

    def test_asking_with_no_address_saved_says_so(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        assert "don't have a shipping address" in _chat("s1", "what's my shipping address")

    def test_a_product_name_is_never_saved_as_the_new_address(self, monkeypatch):
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        assert _chat("s1", "change my address to Ibuprofen 200mg Tablets") == "from the model"
        assert store.get_address("demo_user") in (None, "")

    def test_questions_about_something_else_are_left_to_the_model(self, monkeypatch):
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        for text in ["what is the address of your pharmacy?", "what are the side effects, address me politely"]:
            assert _chat("s1", text) == "from the model", text

    def test_a_pending_address_question_is_not_hijacked(self, monkeypatch):
        store.set_pending_order("s1", "fev-001", 1)
        reply = _chat("s1", "99 New Rd, York")
        assert "Order confirmed!" in reply and "99 New Rd, York" in reply


class TestPriceQuestions:
    """Prices live in the catalog, not the knowledge base. Real bug: "how much
    does Ibuprofen 200mg Tablets cost?" got "the price is not explicitly
    stated in our knowledge base", and "what's the price of col-003" got the
    out-of-scope decline."""

    @staticmethod
    def _model_must_not_run(monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("the model must not be reached")

        monkeypatch.setattr(main, "run_turn", _boom)

    @pytest.mark.parametrize("product", store.get_catalog(), ids=lambda p: p["id"])
    def test_every_catalog_price_is_answered_exactly(self, monkeypatch, product):
        self._model_must_not_run(monkeypatch)
        by_name = _chat("s1", f"how much does {product['name']} cost?")
        assert by_name == f"{product['name']} costs ${product['price_usd']:.2f}."
        assert f"${product['price_usd']:.2f}" in _chat("s2", f"what's the price of {product['id']}")

    def test_an_ingredient_lists_every_product_containing_it(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", "price of paracetamol")
        for name in ["Paracetamol 500mg Tablets", "Paracetamol Extra Strength 650mg", "Children's Paracetamol Syrup"]:
            assert name in reply
        assert "Ibuprofen" not in reply

    def test_cheapest_is_computed_from_the_catalog(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        catalog = store.get_catalog()
        cheapest_fever = min((p for p in catalog if p["category"] == "fever"), key=lambda p: p["price_usd"])
        reply = _chat("s1", "what is your cheapest fever medicine?")
        assert cheapest_fever["name"] in reply and f"${cheapest_fever['price_usd']:.2f}" in reply

    def test_a_price_question_mid_order_leaves_the_order_waiting(self, monkeypatch):
        store.set_pending_order("s1", "fev-001", 1)
        self._model_must_not_run(monkeypatch)
        assert "$5.29" in _chat("s1", "how much is Ibuprofen 200mg Tablets?")
        assert store.get_pending_order("s1")["product_id"] == "fev-001"
        assert store.list_orders("demo_user") == []

    def test_dosing_and_unrelated_how_much_questions_still_go_to_the_model(self, monkeypatch):
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        for text in ["how much ibuprofen should I take", "how much is shipping", "price of guaifenesin dosage",
                     "how much is the weather"]:
            assert _chat("s1", text) == "from the model", text


class TestCatalogListing:
    """Real bug: "show me all your products" got the out-of-scope decline — a
    customer asking what the shop carries was told the shop can't help."""

    @staticmethod
    def _model_must_not_run(monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("the model must not be reached")

        monkeypatch.setattr(main, "run_turn", _boom)

    @pytest.mark.parametrize("text", [
        "show me all your products", "what do you sell?", "what medicines do you have",
        "list your catalog", "can I see all the medicines", "what products do you have",
    ])
    def test_the_whole_real_catalog_is_listed_with_prices(self, monkeypatch, text):
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", text)
        for product in store.get_catalog():
            assert f"- {product['name']} — ${product['price_usd']:.2f}" in reply

    def test_naming_a_product_from_the_list_then_starts_its_order(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        _chat("s1", "show me all your products")
        _chat("s1", "I'd like to order Guaifenesin Expectorant Syrup")
        assert store.get_pending_order("s1")["product_id"] == "col-005"

    def test_symptoms_products_and_orders_are_left_to_the_model(self, monkeypatch):
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        for text in ["what do you have for a cold", "show me products for fever", "what is paracetamol",
                     "tell me about your products and the dosage of ibuprofen", "order all your products",
                     "show me the ibuprofen product"]:
            assert _chat("s1", text) == "from the model", text

    def test_with_an_order_awaiting_its_address_the_question_is_not_saved_as_the_address(self, monkeypatch):
        store.set_pending_order("s1", "fev-001", 1)
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        assert _chat("s1", "what do you sell?") == "from the model"
        assert store.list_orders("demo_user") == []
        assert store.get_pending_order("s1")["product_id"] == "fev-001"


class TestEmailRequestsOutsideThePendingPrompt:
    """Live probe: "my email is X" before ordering got a greeting and was not
    saved; "what email do you have for me?" got the "what symptom is this for"
    non-answer; "send me the receipt" just reprinted the order."""

    @staticmethod
    def _model_must_not_run(monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("the model must not be reached")

        monkeypatch.setattr(main, "run_turn", _boom)

    @staticmethod
    def _place_order():
        store.save_address("demo_user", "10 High St, Leeds")
        return store.create_order("demo_user", "fev-003", "10 High St, Leeds", 1)

    @pytest.mark.parametrize("text", [
        "my email is sukumar@example.com", "use sukumar@example.com as my email",
        "sukumar@example.com is my email.",
    ])
    def test_an_unprompted_email_is_saved(self, monkeypatch, text):
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", text)
        assert store.get_email("demo_user") == "sukumar@example.com"
        assert "sukumar@example.com" in reply

    def test_the_saved_email_is_read_back(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        assert "don't have an email" in _chat("s1", "what email do you have for me?")
        store.save_email("demo_user", "a@b.com")
        assert "a@b.com" in _chat("s1", "what email do you have for me?")

    def test_a_receipt_request_with_an_email_on_file_sends_it_for_the_latest_order(self, monkeypatch):
        order = self._place_order()
        store.save_email("demo_user", "a@b.com")
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", "send me the receipt")
        assert order["order_id"] in reply and "a@b.com" in reply and "sent the confirmation" in reply

    def test_a_receipt_request_with_no_email_asks_for_one_and_captures_the_reply(self, monkeypatch):
        order = self._place_order()
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", "please resend my order confirmation")
        assert "what email address" in reply.lower()
        assert store.get_pending_email("s1") == order["order_id"]
        assert "sent the confirmation" in _chat("s1", "a@b.com")
        assert store.get_email("demo_user") == "a@b.com"

    def test_a_receipt_request_with_no_orders_says_so(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        assert "no receipt to send" in _chat("s1", "send me the receipt")

    def test_a_receipt_request_naming_an_email_saves_it_and_sends(self, monkeypatch):
        order = self._place_order()
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", "email me the receipt at a@b.com")
        assert store.get_email("demo_user") == "a@b.com"
        assert order["order_id"] in reply

    def test_a_cancelled_order_is_never_the_one_receipted(self, monkeypatch):
        order = self._place_order()
        store.cancel_order(order["order_id"], "demo_user")
        self._model_must_not_run(monkeypatch)
        assert "no receipt to send" in _chat("s1", "send me the receipt")

    def test_other_email_messages_are_left_alone(self, monkeypatch):
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        for text in ["order Ibuprofen 200mg Tablets, my email is a@b.com", "reorder that and email me at a@b.com",
                     "what is the email of the pharmacy support@shop.com?", "what is your email", "how do I get a receipt"]:
            assert _chat("s1", text) == "from the model", text
        assert store.get_email("demo_user") is None

    def test_the_pending_email_prompt_still_owns_its_own_reply(self, monkeypatch):
        order = self._place_order()
        store.set_pending_email("s1", order["order_id"])
        reply = _chat("s1", "my email is a@b.com")
        assert "sent the confirmation" in reply and store.get_pending_email("s1") is None


class TestBrandsWeDoNotStock:
    """Real bug (Oct 22 probe): "do you have Advil", "I want Tylenol", "I need
    Crocin" and "do you have Dolo 650" all got "This demo only handles fever and
    cold OTC guidance — I can't help with that here". They are fever medicines,
    and each brand is just an ingredient we do stock."""

    @staticmethod
    def _model_must_not_run(monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("the model must not be reached")

        monkeypatch.setattr(main, "run_turn", _boom)

    @pytest.mark.parametrize("text, expected, absent", [
        ("do you have Advil", ["fev-003"], ["fev-001", "col-003"]),
        ("I want Tylenol", ["fev-001", "fev-002", "fev-004"], ["fev-003", "col-003"]),
        ("I need Crocin", ["fev-001", "fev-002", "fev-004"], ["col-003"]),
        ("do you have Dolo 650", ["fev-001", "fev-002"], ["fev-003"]),
        ("any Zyrtec?", ["col-001"], ["col-002"]),
        ("do you have Sudafed", ["col-002"], ["col-003"]),
        ("do you have acetaminophen", ["fev-001"], ["fev-003"]),
    ])
    def test_a_brand_gets_the_products_with_the_same_active_ingredient(self, monkeypatch, text, expected, absent):
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", text)
        assert "can't help with that here" not in reply
        for product_id in expected:
            assert f"({product_id})" in reply
        for product_id in absent:
            assert f"({product_id})" not in reply

    def test_the_equivalents_can_be_picked_by_number_and_the_brand_is_not_ordered(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        _chat("s1", "do you have Advil")
        assert [p["id"] for p in store.get_last_products("s1")] == ["fev-003"]
        _chat("s1", "3")
        assert store.get_pending_order("s1")["product_id"] == "fev-003"

    @pytest.mark.parametrize("text", ["do you sell aspirin", "any NyQuil?", "can I get amoxicillin", "do you have Benadryl"])
    def test_a_medicine_with_no_equivalent_gets_an_honest_not_stocked(self, monkeypatch, text):
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", text)
        assert reply.startswith("We don't carry") and "show me all your products" in reply

    def test_questions_about_a_brand_are_left_to_the_model(self, monkeypatch):
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        for text in ["what's better for fever, Tylenol or Advil", "is Tylenol safe in pregnancy",
                     "how much Tylenol can I take", "I took Tylenol"]:
            assert _chat("s1", text) == "from the model", text

    def test_a_stated_condition_still_screens_the_equivalents(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        store.save_session_messages("s1", [{"role": "user", "content": "I have a stomach ulcer"}])
        reply = _chat("s1", "do you have Advil")
        assert reply.startswith("I can't recommend Ibuprofen 200mg Tablets")


class TestChangeOfMindIsNotSavedAsTheAddress:
    @pytest.mark.parametrize("text", ["wait, I changed my mind about the colour", "let's do something else", "actually can you help me with something"])
    def test_a_change_of_mind_is_never_the_shipping_address(self, monkeypatch, text):
        store.set_pending_order("s1", "fev-003", 1)
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        assert _chat("s1", text) == "from the model"
        assert store.list_orders("demo_user") == []
        assert store.get_address("demo_user") in (None, "")
        assert store.get_pending_order("s1")["product_id"] == "fev-003"

    def test_the_real_address_afterwards_still_completes_the_order(self, monkeypatch):
        store.set_pending_order("s1", "fev-003", 1)
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        _chat("s1", "wait, I changed my mind about the colour")
        assert "Order confirmed!" in _chat("s1", "5 Mid Rd, Pune")

    def test_the_same_holds_for_an_address_confirmation(self, monkeypatch):
        store.save_address("demo_user", "10 High St, Leeds")
        store.set_pending_address_confirmation("s1", "fev-003", 1)
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        _chat("s1", "wait, I changed my mind about the colour")
        assert store.get_address("demo_user") == "10 High St, Leeds"
        assert store.list_orders("demo_user") == []


class TestQuantityChangeWhileAnOrderWaits:
    """"make it 2" used to be saved as the address; once refused it reached the
    model, which ignored the pending order. It now changes that order."""

    @staticmethod
    def _model_must_not_run(monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("the model must not be reached")

        monkeypatch.setattr(main, "run_turn", _boom)

    @pytest.mark.parametrize("text, quantity", [
        ("make it 2", 2), ("actually just 1", 1), ("make that two please", 2), ("change it to 2", 2),
        ("let's do 2", 2), ("I want 2 of them", 2), ("no wait, make it twice", 2),
    ])
    def test_the_waiting_order_changes_quantity(self, monkeypatch, text, quantity):
        store.set_pending_order("s1", "fev-003", 1)
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", text)
        assert reply.startswith(f"Got it — {quantity} x Ibuprofen 200mg Tablets.") and "address" in reply
        assert store.get_pending_order("s1") == {"product_id": "fev-003", "quantity": quantity}
        assert store.list_orders("demo_user") == []

    def test_the_new_quantity_reaches_the_real_order(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        store.set_pending_order("s1", "fev-003", 1)
        _chat("s1", "make it 2")
        reply = _chat("s1", "5 Mid Rd, Pune")
        assert "Quantity: 2" in reply and "5 Mid Rd, Pune" in reply

    def test_over_the_limit_is_capped_and_the_customer_is_told(self, monkeypatch):
        store.set_pending_order("s1", "fev-003", 1)
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", "make it 5")
        assert "capped at our per-order limit" in reply
        assert store.get_pending_order("s1")["quantity"] == 2

    def test_it_works_on_an_address_confirmation_too(self, monkeypatch):
        store.save_address("demo_user", "10 High St, Leeds")
        store.set_pending_address_confirmation("s1", "fev-003", 1)
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", "make it 2")
        assert "10 High St, Leeds" in reply and "Got it — 2 x Ibuprofen" in reply
        assert store.get_pending_address_confirmation("s1") == {"product_id": "fev-003", "quantity": 2}
        assert "Quantity: 2" in _chat("s1", "yes")

    def test_with_nothing_waiting_it_is_left_to_the_model(self, monkeypatch):
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        assert _chat("s1", "make it 2") == "from the model"
        assert store.get_pending_order("s1") is None

    def test_a_condition_refusal_still_applies_on_a_quantity_change(self, monkeypatch):
        store.save_session_messages("s1", [{"role": "user", "content": "I have a stomach ulcer"}])
        store.set_pending_order("s1", "fev-003", 1)
        self._model_must_not_run(monkeypatch)
        assert "I can't set up an order" in _chat("s1", "make it 2")


class TestRepeatTheLastOrder:
    """Real bug (Oct 23 probe): "one more of the same" after an order reached
    the model, which asked "what symptom is this for?" — and the follow-up "yes"
    was then declined as out of scope."""

    @staticmethod
    def _place_order(quantity=1):
        store.save_address("demo_user", "10 High St, Leeds")
        return store.create_order("demo_user", "fev-003", "10 High St, Leeds", quantity)

    @staticmethod
    def _model_must_not_run(monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("the model must not be reached")

        monkeypatch.setattr(main, "run_turn", _boom)

    @pytest.mark.parametrize("text, quantity", [
        ("one more of the same", 1), ("same again", 1), ("ok one more please", 1), ("two more", 2),
        ("the same again", 1), ("another one of that", 1), ("again please", 1),
    ])
    def test_it_reorders_the_last_product_with_the_quantity_asked(self, monkeypatch, text, quantity):
        self._place_order()
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", text)
        assert "10 High St, Leeds" in reply  # asks to confirm the address on file
        assert store.get_pending_address_confirmation("s1") == {"product_id": "fev-003", "quantity": quantity}
        assert len(store.list_orders("demo_user")) == 1

    def test_it_works_right_after_an_order_with_the_email_prompt_pending(self, monkeypatch):
        order = self._place_order()
        store.set_pending_email("s1", order["order_id"])
        self._model_must_not_run(monkeypatch)
        _chat("s1", "one more of the same")
        assert store.get_pending_address_confirmation("s1")["product_id"] == "fev-003"
        assert store.get_pending_email("s1") == order["order_id"]

    def test_confirming_places_the_second_order(self, monkeypatch):
        self._place_order()
        self._model_must_not_run(monkeypatch)
        _chat("s1", "two more")
        assert "Quantity: 2" in _chat("s1", "yes")
        assert len(store.list_orders("demo_user")) == 2

    def test_with_no_previous_order_it_is_left_to_the_model(self, monkeypatch):
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        assert _chat("s1", "one more of the same") == "from the model"

    def test_ambiguous_wording_is_left_to_the_model(self, monkeypatch):
        self._place_order()
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        for text in ["another one", "another", "same", "show me another", "one more paracetamol", "another symptom"]:
            assert _chat("s1", text) == "from the model", text
        assert store.get_pending_address_confirmation("s1") is None


class TestSwitchingProductWhileAnOrderWaits:
    """Real gap (Oct 23 probe): "actually give me paracetamol 500mg instead",
    said while an order waited for its address, got the child-or-adult question
    and left the old order pending. It needs natural verbs ("give me", "I want")
    and partial names ("paracetamol 500mg", a unique ingredient)."""

    @staticmethod
    def _model_must_not_run(monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("the model must not be reached")

        monkeypatch.setattr(main, "run_turn", _boom)

    @pytest.mark.parametrize("text, product_id", [
        ("actually give me paracetamol 500mg instead", "fev-001"), ("give me paracetamol 650mg", "fev-002"),
        ("switch to Paracetamol Extra Strength 650mg", "fev-002"), ("I want cetirizine", "col-001"),
        ("i'd like guaifenesin", "col-005"), ("send me pseudoephedrine please", "col-002"),
        ("let's go with ibuprofen 200mg", "fev-003"), ("order ibuprofen", "fev-003"),
    ])
    def test_the_waiting_order_switches_to_the_named_product(self, monkeypatch, text, product_id):
        store.set_pending_order("s1", "col-004", 1)
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", text)
        assert "address" in reply.lower()
        assert store.get_pending_order("s1")["product_id"] == product_id
        assert store.list_orders("demo_user") == []

    def test_a_quantity_is_carried_through_a_switch(self, monkeypatch):
        store.set_pending_order("s1", "col-004", 1)
        self._model_must_not_run(monkeypatch)
        _chat("s1", "order 2 ibuprofen")
        assert store.get_pending_order("s1") == {"product_id": "fev-003", "quantity": 2}

    @pytest.mark.parametrize("text", [
        "I want to know about ibuprofen", "give me paracetamol", "order dextromethorphan", "I need ibuprofen 400mg",
        "order ibuprofen 400mg", "give me ibuprofen and cetirizine", "is ibuprofen safe", "I want ibuprofen dosage",
        "do you have ibuprofen", "I wonder if ibuprofen works", "don't order ibuprofen",
    ])
    def test_ambiguous_questions_and_other_strengths_are_left_to_the_model(self, monkeypatch, text):
        store.set_pending_order("s1", "col-004", 1)
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        _chat("s1", text)
        assert store.get_pending_order("s1")["product_id"] == "col-004", text


class TestAStaleClarificationDoesNotSwallowTheAddress:
    """Real bug (Oct 24): mid-order, "order ibuprofen 400mg" reached the model,
    which asked child-or-adult. The customer's next message — the shipping
    address — was taken as the answer, fell through to the model, and the model
    invented "Ibuprofen 400mg Tablets"."""

    @staticmethod
    def _model_must_not_run(monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("the model must not be reached")

        monkeypatch.setattr(main, "run_turn", _boom)

    def test_the_address_completes_the_waiting_order(self, monkeypatch):
        store.set_pending_order("s1", "fev-001", 1)
        store.set_pending_clarification("s1", "fever")
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", "3 Switch Street, Pune")
        assert "Order confirmed!" in reply and "Paracetamol 500mg Tablets" in reply
        assert "3 Switch Street, Pune" in reply
        assert store.get_pending_clarification("s1") is None

    def test_a_real_answer_to_the_question_is_still_used(self, monkeypatch):
        store.set_pending_order("s1", "fev-001", 1)
        store.set_pending_clarification("s1", "fever")
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", "adult")
        assert "(fev-001)" in reply and "Which one would you like to try?" in reply
        assert store.get_pending_order("s1")["product_id"] == "fev-001"

    def test_it_works_for_an_address_confirmation_too(self, monkeypatch):
        store.save_address("demo_user", "10 High St, Leeds")
        store.set_pending_address_confirmation("s1", "fev-001", 1)
        store.set_pending_clarification("s1", "fever")
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", "yes")
        assert "Order confirmed!" in reply and "10 High St, Leeds" in reply

    def test_without_an_order_waiting_an_unrecognised_reply_still_reaches_the_model(self, monkeypatch):
        store.set_pending_clarification("s1", "fever")
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        assert _chat("s1", "hmm not sure") == "from the model"


class TestAnAffirmativeWithNothingWaiting:
    """Real bug (Oct 24): a bare "yes" / "place the order" / "confirm" with
    nothing in progress reached the model, which invented a context ("I'll need
    you to confirm your shipping address before we can proceed") and tripped a
    tool-argument validation error behind the scenes."""

    @pytest.mark.parametrize("text", [
        "place the order", "confirm", "go ahead", "yes please order it", "ok order", "proceed", "do it",
    ])
    def test_it_says_there_is_nothing_to_confirm_without_the_model(self, monkeypatch, text):
        def _boom(*args, **kwargs):
            raise AssertionError("the model must not be reached")

        monkeypatch.setattr(main, "run_turn", _boom)
        assert _chat("s1", text) == main.NOTHING_TO_CONFIRM_REPLY
        assert store.list_orders("demo_user") == []

    @pytest.mark.parametrize("text", ["yes", "place the order", "confirm", "go ahead", "do it"])
    def test_a_just_recommended_product_still_starts_its_order(self, monkeypatch, text):
        store.set_last_recommended_product("s1", "fev-003")
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        reply = _chat("s1", text)
        assert reply != main.NOTHING_TO_CONFIRM_REPLY and "address" in reply.lower()
        assert store.get_pending_order("s1")["product_id"] == "fev-003"

    def test_each_open_context_keeps_its_own_handling(self, monkeypatch):
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        store.set_pending_email("e1", "ord-0001")
        assert _chat("e1", "yes") != main.NOTHING_TO_CONFIRM_REPLY
        store.set_pending_clarification("c1", "fever")
        assert _chat("c1", "yes") != main.NOTHING_TO_CONFIRM_REPLY
        store.save_address("demo_user", "10 High St, Leeds")
        store.set_pending_address_confirmation("a1", "fev-003", 1)
        assert "Order confirmed!" in _chat("a1", "yes")

    @pytest.mark.parametrize("text", ["yes", "ok", "sure", "yeah", "yes please"])
    def test_a_bare_yes_is_left_alone_it_often_answers_anything_else(self, monkeypatch, text):
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        assert _chat("s1", text) == "from the model"

    def test_a_non_affirmative_message_is_unaffected(self, monkeypatch):
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        assert _chat("s1", "who is the prime minister") == "from the model"


class TestRepliesToTheReceiptEmailPrompt:
    """Real bug (Oct 25 probe): "yes"/"sure"/"ok send it" got "Thank you! Your
    address has been saved" (nothing like that happened), "later" got "You've
    saved your address", "don't send" got "Is the address still correct?",
    "not now" got the out-of-scope decline, and "send it to my email" got "what
    symptom is this for?"."""

    @staticmethod
    def _place_an_order_through_the_prompt(monkeypatch, sid="s1"):
        def _boom(*args, **kwargs):
            raise AssertionError("the model must not be reached")

        monkeypatch.setattr(main, "run_turn", _boom)
        store.save_address("demo_user", "10 High St, Leeds")
        _chat(sid, "order Ibuprofen 200mg Tablets")
        reply = _chat(sid, "yes")
        assert "reply with your email address" in reply
        assert store.get_pending_email(sid) == "ord-0001"

    @pytest.mark.parametrize("text", ["yes", "yes please", "sure", "ok send it", "send it to my email", "send me the receipt"])
    def test_an_affirmative_asks_for_the_address_and_keeps_waiting(self, monkeypatch, text):
        self._place_an_order_through_the_prompt(monkeypatch)
        reply = _chat("s1", text)
        assert "what email address should I send the confirmation for order ord-0001" in reply
        assert store.get_pending_email("s1") == "ord-0001"

    def test_the_email_that_follows_is_captured_and_sent(self, monkeypatch):
        self._place_an_order_through_the_prompt(monkeypatch)
        _chat("s1", "yes")
        reply = _chat("s1", "a@b.com")
        assert "sent the confirmation for order ord-0001 to a@b.com" in reply
        assert store.get_email("demo_user") == "a@b.com" and store.get_pending_email("s1") is None

    @pytest.mark.parametrize("text", ["later", "not now", "don't send", "skip it", "no need", "maybe later", "nah"])
    def test_a_skip_closes_the_prompt(self, monkeypatch, text):
        self._place_an_order_through_the_prompt(monkeypatch)
        reply = _chat("s1", text)
        assert "already confirmed without an email receipt" in reply
        assert store.get_pending_email("s1") is None

    def test_with_an_email_on_file_the_confirmation_is_sent_straight_away(self, monkeypatch):
        self._place_an_order_through_the_prompt(monkeypatch)
        store.save_email("demo_user", "x@y.com")
        reply = _chat("s1", "yes")
        assert "sent the confirmation for order ord-0001 to x@y.com" in reply
        assert store.get_pending_email("s1") is None

    def test_a_yes_to_something_else_is_left_to_the_model(self, monkeypatch):
        self._place_an_order_through_the_prompt(monkeypatch)
        def _answers(messages, **kwargs):
            # like the real run_turn: the turn's user text and reply are appended
            messages = [*messages, {"role": "user", "content": kwargs["user_text"]},
                        {"role": "assistant", "content": "Anything else I can help with?"}]
            return "Anything else I can help with?", messages

        monkeypatch.setattr(main, "run_turn", _answers)
        _chat("s1", "what are the side effects of ibuprofen")
        assert _chat("s1", "yes") == "Anything else I can help with?"
        assert store.get_pending_email("s1") == "ord-0001"


class TestCancelRequestsTheModelGotWrong:
    """Real bugs (Oct 27 probe, two orders on file): "cancel my last order" ->
    "I don't have any information about an order with that ID" (none was
    given); "cancel ord-0001 and ord-0002" and "cancel all my orders" -> the
    out-of-scope decline."""

    @staticmethod
    def _two_orders():
        store.save_address("demo_user", "10 High St, Leeds")
        store.create_order("demo_user", "fev-003", "10 High St, Leeds", 1)
        store.create_order("demo_user", "col-001", "10 High St, Leeds", 1)

    @staticmethod
    def _statuses():
        return {o["order_id"]: o["status"] for o in store.list_orders("demo_user")}

    @staticmethod
    def _model_must_not_run(monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("the model must not be reached")

        monkeypatch.setattr(main, "run_turn", _boom)

    @pytest.mark.parametrize("text", ["cancel my last order", "please cancel the latest order", "cancel my most recent order"])
    def test_last_cancels_only_the_newest_active_order(self, monkeypatch, text):
        self._two_orders()
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", text)
        assert "Order ord-0002 (Cetirizine 10mg Antihistamine Tablets) has been cancelled." in reply
        assert self._statuses() == {"ord-0001": "placed", "ord-0002": "cancelled"}

    @pytest.mark.parametrize("text", ["cancel ord-0001 and ord-0002", "cancel all my orders", "I want to cancel both orders"])
    def test_all_or_several_are_cancelled(self, monkeypatch, text):
        self._two_orders()
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", text)
        assert "ord-0001" in reply and "ord-0002" in reply and reply.count("has been cancelled.") == 2
        assert set(self._statuses().values()) == {"cancelled"}

    def test_a_numbered_reference_names_the_order(self, monkeypatch):
        self._two_orders()
        self._model_must_not_run(monkeypatch)
        _chat("s1", "cancel order 1")
        assert self._statuses() == {"ord-0001": "cancelled", "ord-0002": "placed"}

    def test_a_plain_cancel_with_several_orders_asks_which(self, monkeypatch):
        self._two_orders()
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", "cancel my order")
        assert "You have 2 active orders" in reply and "ord-0001" in reply and "ord-0002" in reply
        assert set(self._statuses().values()) == {"placed"}  # nothing cancelled until they choose

    def test_a_plain_cancel_with_one_order_cancels_it(self, monkeypatch):
        store.save_address("demo_user", "10 High St, Leeds")
        store.create_order("demo_user", "fev-003", "10 High St, Leeds", 1)
        self._model_must_not_run(monkeypatch)
        assert "Order ord-0001" in _chat("s1", "cancel my order")

    def test_unknown_and_already_cancelled_orders_are_reported_not_invented(self, monkeypatch):
        self._two_orders()
        store.cancel_order("ord-0002", "demo_user")
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", "cancel ord-9999 and ord-0002")
        assert "couldn't find an order with the ID ord-9999" in reply and "Order ord-0002 was already cancelled." in reply
        assert self._statuses()["ord-0001"] == "placed"

    def test_with_nothing_to_cancel_it_says_so(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        assert "don't have any active orders to cancel" in _chat("s1", "cancel all my orders")

    def test_it_works_with_the_email_prompt_pending(self, monkeypatch):
        self._two_orders()
        store.set_pending_email("s1", "ord-0002")
        self._model_must_not_run(monkeypatch)
        _chat("s1", "cancel my last order")
        assert self._statuses()["ord-0002"] == "cancelled"

    @pytest.mark.parametrize("text", [
        "don't cancel my order", "can I cancel my order?", "cancel", "cancel that", "cancel my appointment",
        "what happens if I cancel my order",
    ])
    def test_negated_questions_and_vague_messages_cancel_nothing(self, monkeypatch, text):
        self._two_orders()
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        _chat("s1", text)
        assert set(self._statuses().values()) == {"placed"}, text

    def test_an_order_awaiting_its_address_keeps_its_own_handling(self, monkeypatch):
        self._two_orders()
        store.set_pending_order("s1", "fev-001", 1)
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        _chat("s1", "cancel my last order")
        assert set(self._statuses().values()) == {"placed"}


class TestShopPolicyQuestionsAreAnsweredHonestly:
    """Real bugs (Oct 27 probe): "how long does delivery take", "what is your
    refund policy", "how do I pay", "do I need a prescription" got "This demo
    only handles fever and cold OTC guidance — I can't help with that here";
    "is there a delivery charge" and "when will my order arrive" just printed the
    order list; "can I return the medicine" listed the whole catalog; "can I
    change my delivery address for ord-0001" was answered with the saved
    address. There is no policy data, so a specific answer would be invented."""

    @staticmethod
    def _model_must_not_run(monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("the model must not be reached")

        monkeypatch.setattr(main, "run_turn", _boom)

    @staticmethod
    def _an_order():
        store.save_address("demo_user", "10 High St, Leeds")
        store.create_order("demo_user", "fev-003", "10 High St, Leeds", 1)

    @pytest.mark.parametrize("text, topic", [
        ("how long does delivery take", "delivery times or tracking"), ("when will my order arrive", "delivery times or tracking"),
        ("track my order", "delivery times or tracking"), ("where is my order", "delivery times or tracking"),
        ("is there a delivery charge", "delivery or shipping charges"), ("is delivery free", "delivery or shipping charges"),
        ("what is your refund policy", "refunds or returns"), ("can I return the medicine", "refunds or returns"),
        ("how do I pay", "payment methods"), ("do you accept credit cards", "payment methods"),
        ("what is your phone number", "opening hours or contact details"), ("are you open on sunday", "opening hours or contact details"),
    ])
    def test_a_policy_question_says_it_has_no_information(self, monkeypatch, text, topic):
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", text)
        assert reply.startswith(f"I don't have information about {topic}") and "rather not guess" in reply
        assert "can't help with that here" not in reply

    def test_a_delivery_question_still_shows_the_real_order_status(self, monkeypatch):
        self._an_order()
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", "when will my order arrive")
        assert "delivery times or tracking" in reply and "Order ID: ord-0001" in reply and "Status: placed" in reply

    def test_a_refund_question_does_not_dump_the_order_list(self, monkeypatch):
        self._an_order()
        self._model_must_not_run(monkeypatch)
        assert "Order ID" not in _chat("s1", "what is your refund policy")

    def test_prescriptions_are_answered_from_the_catalog_not_invented(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", "do I need a prescription")
        assert "over-the-counter" in reply and "pharmacist" in reply

    def test_returning_the_medicine_no_longer_lists_the_catalog(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        assert "Here's what we carry" not in _chat("s1", "can I return the medicine")

    def test_a_real_catalog_listing_request_still_works(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        assert "Here's what we carry" in _chat("s1", "show me all your products")

    def test_changing_the_address_of_a_placed_order_is_explained(self, monkeypatch):
        self._an_order()
        self._model_must_not_run(monkeypatch)
        for text in ["can I change my delivery address for ord-0001", "update the shipping address on my last order"]:
            assert _chat("s1", text) == main.ORDER_ADDRESS_CHANGE_REPLY
        assert store.list_orders("demo_user")[0]["address"] == "10 High St, Leeds"

    def test_changing_the_saved_address_still_works(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        assert "saved 99 New Rd, York" in _chat("s1", "change my shipping address to 99 New Rd, York")

    @pytest.mark.parametrize("text", [
        "order Ibuprofen 200mg Tablets", "cancel my order", "what is the price of ibuprofen", "I paid twice",
        "what are the side effects of ibuprofen",
    ])
    def test_other_messages_are_not_taken_for_policy_questions(self, monkeypatch, text):
        store.save_address("demo_user", "10 High St, Leeds")
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        reply = _chat("s1", text)
        assert "rather not guess" not in reply

    def test_an_order_awaiting_its_address_is_left_alone(self, monkeypatch):
        store.set_pending_order("s1", "fev-003", 1)
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        assert "rather not guess" not in _chat("s1", "how do I pay")


class TestPickingFromTheListThatWasShown:
    """Real bugs (Oct 28 probe, after a list): "the one with ibuprofen" -> the
    child-or-adult question; "number 2 please" -> the model claimed "before we
    proceed, confirm your address" with no order started; "the second one and
    the third one" -> an order for TWO of the second product; "which one is
    cheapest" -> the cheapest product in the whole catalog, not the list."""

    @staticmethod
    def _shown(*ids):
        store.set_last_products("s1", [store.find_product(i) for i in ids])

    @staticmethod
    def _model_must_not_run(monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("the model must not be reached")

        monkeypatch.setattr(main, "run_turn", _boom)

    @pytest.mark.parametrize("text, product_id", [
        ("the ibuprofen one", "fev-003"), ("the one with ibuprofen", "fev-003"), ("ibuprofen please", "fev-003"),
        ("number 2 please", "fev-002"), ("number 3", "fev-003"), ("option 1", "fev-001"), ("#2", "fev-002"),
        ("i'll take the first", "fev-001"), ("the second one please", "fev-002"), ("the last one", "fev-003"),
        ("the extra strength one", "fev-002"),
    ])
    def test_the_named_or_numbered_product_starts_its_order(self, monkeypatch, text, product_id):
        self._shown("fev-001", "fev-002", "fev-003")
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", text)
        assert "address" in reply.lower()
        assert store.get_pending_order("s1") == {"product_id": product_id, "quantity": 1}

    @pytest.mark.parametrize("text", [
        "both of them", "the second one and the third one", "can I get the first two", "number 1 and number 3",
    ])
    def test_several_references_ask_which_first_and_order_nothing(self, monkeypatch, text):
        self._shown("fev-001", "fev-002", "fev-003")
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", text)
        assert reply.startswith("I can start one order at a time. Which would you like first?")
        assert store.get_pending_order("s1") is None and store.list_orders("demo_user") == []

    @pytest.mark.parametrize("text", ["which one is cheapest", "the cheapest one", "which of these is the cheapest"])
    def test_cheapest_means_the_list_on_screen(self, monkeypatch, text):
        self._shown("col-001", "col-002", "col-004")   # $5.99, $6.99, $6.29: the cheapest is col-001
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", text)
        assert reply == "Of the products I just showed you, the lowest-priced is Cetirizine 10mg Antihistamine Tablets at $5.99."

    @pytest.mark.parametrize("text", [
        "tell me more about the first one", "which one is safest",
        "how much is the second one", "what are the side effects of the third one",
    ])
    def test_questions_about_the_list_are_left_to_the_model(self, monkeypatch, text):
        self._shown("fev-001", "fev-002", "fev-003")
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        assert _chat("s1", text) == "from the model"
        assert store.get_pending_order("s1") is None

    def test_symptom_sentences_are_not_selections(self, monkeypatch):
        self._shown("fev-001", "fev-002", "fev-003")
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        for text in ["both my kids have a fever", "I have all the symptoms", "my second child has a cough"]:
            assert main._resolve_list_reference(text, store.get_last_products("s1")) is None, text

    def test_without_a_list_on_screen_nothing_is_selected(self):
        assert main._resolve_list_reference("number 2 please", None) is None
        assert main._resolve_list_reference("number 2 please", [store.find_product("fev-001")]) is None

    def test_a_reference_outside_the_list_is_not_guessed(self):
        shown = [store.find_product("fev-001"), store.find_product("fev-002")]
        assert main._resolve_list_reference("number 5", shown) is None

    def test_the_catalog_wide_cheapest_still_works_with_no_list_reference(self, monkeypatch):
        self._shown("col-001", "col-002", "col-004")
        self._model_must_not_run(monkeypatch)
        assert "Paracetamol 500mg Tablets at $4.99" in _chat("s1", "what is your cheapest fever medicine")


class TestAnEmailTypedWhereAnAddressIsExpected:
    """Real weakness (Oct 29, a flaky harness check): a bare email typed while an
    order waited for its address went to the model, which sometimes asked "is your
    shipping address still sukumar123@example.com?"."""

    @staticmethod
    def _model_must_not_run(monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("the model must not be reached")

        monkeypatch.setattr(main, "run_turn", _boom)

    def test_awaiting_an_address_the_email_is_saved_and_the_address_asked_again(self, monkeypatch):
        store.set_pending_order("s1", "fev-003", 1)
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", "sukumar123@example.com")
        assert reply == "Thanks — I've saved sukumar123@example.com for your order confirmation. What's your shipping address so I can send that out?"
        assert store.get_email("demo_user") == "sukumar123@example.com"
        assert store.get_address("demo_user") in (None, "") and store.get_pending_order("s1")["product_id"] == "fev-003"
        assert "Order confirmed!" in _chat("s1", "5 Mid Rd, Pune")

    def test_at_an_address_confirmation_the_question_is_asked_again(self, monkeypatch):
        store.save_address("demo_user", "10 High St, Leeds")
        store.set_pending_address_confirmation("s1", "fev-003", 1)
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", "a@b.com")
        assert "saved a@b.com" in reply and "We have this address on file: 10 High St, Leeds" in reply
        assert store.get_address("demo_user") == "10 High St, Leeds"
        assert store.get_pending_address_confirmation("s1") is not None
        assert "Order confirmed!" in _chat("s1", "yes")

    def test_an_address_with_an_email_alongside_is_still_an_address(self, monkeypatch):
        store.set_pending_order("s1", "fev-003", 1)
        reply = _chat("s1", "5 Mid Rd, Pune, a@b.com")
        assert "Order confirmed!" in reply and "5 Mid Rd, Pune" in reply


class TestAnOpenQuestionDoesNotBlockAnExplicitCommand:
    """Real bug (Oct 29): after "I have a fever" (child-or-adult question open),
    "order Ibuprofen 200mg Tablets" got the same question again instead of
    starting the order."""

    @staticmethod
    def _model_must_not_run(monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("the model must not be reached")

        monkeypatch.setattr(main, "run_turn", _boom)

    def test_an_explicit_order_starts_and_the_stale_question_is_dropped(self, monkeypatch):
        store.set_pending_clarification("s1", "fever")
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", "order Ibuprofen 200mg Tablets")
        assert "address" in reply.lower() and "child, or for an adult" not in reply
        assert store.get_pending_order("s1")["product_id"] == "fev-003"
        assert store.get_pending_clarification("s1") is None

    def test_a_cancel_request_runs(self, monkeypatch):
        store.save_address("demo_user", "10 High St, Leeds")
        store.create_order("demo_user", "fev-003", "10 High St, Leeds", 1)
        store.set_pending_clarification("s1", "fever")
        self._model_must_not_run(monkeypatch)
        assert "has been cancelled" in _chat("s1", "cancel my last order")

    def test_a_plain_reorder_runs(self, monkeypatch):
        store.save_address("demo_user", "10 High St, Leeds")
        store.create_order("demo_user", "fev-003", "10 High St, Leeds", 1)
        store.set_pending_clarification("s1", "fever")
        self._model_must_not_run(monkeypatch)
        _chat("s1", "reorder that")
        assert store.get_pending_address_confirmation("s1")["product_id"] == "fev-003"

    def test_a_real_answer_to_the_question_is_still_used(self, monkeypatch):
        store.set_pending_clarification("s1", "fever")
        self._model_must_not_run(monkeypatch)
        assert "(fev-001)" in _chat("s1", "adult")

    def test_an_unrelated_message_still_goes_through_the_normal_path(self, monkeypatch):
        store.set_pending_clarification("s1", "fever")
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        assert _chat("s1", "hmm not sure") == "from the model"


class TestOneMessageOrders:
    """Real bugs (Oct 30 probe): "order Ibuprofen 200mg Tablets and ship it to 5 Mid
    Rd, Pune" read the 5 in the address as the QUANTITY (5, capped to 2) and
    dropped the address; "order 2 Cetirizine ... to 12 High St, Leeds" and the
    versions with an email were bounced into the child-or-adult question or
    model-written prose (address saved silently, no order, email ignored)."""

    @staticmethod
    def _model_must_not_run(monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("the model must not be reached")

        monkeypatch.setattr(main, "run_turn", _boom)

    @staticmethod
    def _orders():
        return [(o["product_id"], o["quantity"], o["address"]) for o in store.list_orders("demo_user")]

    @pytest.mark.parametrize("text, expected", [
        ("order Ibuprofen 200mg Tablets and ship it to 5 Mid Rd, Pune", ("fev-003", 1, "5 Mid Rd, Pune")),
        ("order 2 Cetirizine 10mg Antihistamine Tablets to 12 High St, Leeds", ("col-001", 2, "12 High St, Leeds")),
        ("buy Paracetamol 500mg Tablets, deliver to 9 Oak Ave", ("fev-001", 1, "9 Oak Ave")),
        ("please order col-005 send to Flat 4, Rose Lane, Delhi", ("col-005", 1, "Flat 4, Rose Lane, Delhi")),
    ])
    def test_product_quantity_and_address_in_one_message(self, monkeypatch, text, expected):
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", text)
        assert "Order confirmed!" in reply and f"Shipping to: {expected[2]}" in reply
        assert self._orders() == [expected] and store.get_address("demo_user") == expected[2]

    def test_the_number_in_the_address_is_never_the_quantity(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        _chat("s1", "order Ibuprofen 200mg Tablets and ship it to 5 Mid Rd, Pune")
        assert self._orders()[0][1] == 1

    def test_an_email_in_the_same_message_is_saved_and_the_confirmation_sent(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", "I want Guaifenesin Expectorant Syrup, deliver to Flat 4, Rose Lane, Delhi and email me at a@b.com")
        assert "Order confirmed!" in reply and "A confirmation email has been sent." in reply
        assert store.get_email("demo_user") == "a@b.com" and store.get_pending_email("s1") is None
        assert self._orders() == [("col-005", 1, "Flat 4, Rose Lane, Delhi")]

    def test_the_per_order_quantity_limit_still_applies_and_is_announced(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", "order 5 Ibuprofen 200mg Tablets ship to 9 Oak Ave")
        assert "capped at our per-order limit of 2" in reply and self._orders() == [("fev-003", 2, "9 Oak Ave")]

    def test_a_stated_condition_still_refuses_the_product_and_places_nothing(self, monkeypatch):
        store.save_session_messages("s1", [{"role": "user", "content": "I have a stomach ulcer"}])
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", "order Ibuprofen 200mg Tablets ship to 9 Oak Ave")
        assert "I can't set up an order" in reply and self._orders() == []
        assert store.get_address("demo_user") in (None, "")

    @pytest.mark.parametrize("text", [
        "order Ibuprofen 200mg Tablets ship to home", "order ibuprofen 400mg ship to 9 Oak Ave",
        "ship to 9 Oak Ave", "order something ship to 9 Oak Ave", "order Ibuprofen 200mg Tablets and Cetirizine 10mg Antihistamine Tablets to 9 Oak Ave",
        "can I order Ibuprofen 200mg Tablets and ship to 9 Oak Ave?", "don't order Ibuprofen 200mg Tablets ship to 9 Oak Ave",
    ])
    def test_anything_incomplete_or_unclear_places_nothing(self, monkeypatch, text):
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        _chat("s1", text)
        assert self._orders() == [], text

    def test_a_message_without_an_address_keeps_the_normal_flow(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        assert "address" in _chat("s1", "order Ibuprofen 200mg Tablets").lower() and self._orders() == []

    def test_an_order_awaiting_its_address_is_left_alone(self, monkeypatch):
        store.set_pending_order("s1", "col-004", 1)
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        _chat("s1", "order Ibuprofen 200mg Tablets ship to 9 Oak Ave")
        assert self._orders() == []
        assert store.get_pending_order("s1")["product_id"] == "col-004"

    def test_an_open_question_does_not_block_it(self, monkeypatch):
        store.set_pending_clarification("s1", "fever")
        self._model_must_not_run(monkeypatch)
        assert "Order confirmed!" in _chat("s1", "order Ibuprofen 200mg Tablets ship to 9 Oak Ave")


class TestAboutTheAssistant:
    """Real gaps (Oct 30 probe): "who are you", "what can you do", "help", "I need
    help", "menu" got the out-of-scope decline; so did "are you a doctor", "can you
    diagnose me", "can you speak telugu" and "sorry"; "start over" / "reset" did
    nothing."""

    @staticmethod
    def _model_must_not_run(monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("the model must not be reached")

        monkeypatch.setattr(main, "run_turn", _boom)

    @pytest.mark.parametrize("text", [
        "who are you", "what can you do", "help", "I need help", "menu", "what do you do", "how does this work",
        "hi, what can you do?", "can you help me", "what is this",
    ])
    def test_the_capabilities_reply(self, monkeypatch, text):
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", text)
        assert reply == main.CAPABILITIES_REPLY and "can't help with that here" not in reply
        assert "not a doctor or pharmacist" in reply and "112" in reply

    def test_the_capabilities_are_true_of_the_assistant(self, monkeypatch):
        # every capability it advertises is something that works without the model
        self._model_must_not_run(monkeypatch)
        assert "Here's what we carry" in _chat("s1", "show me all your products")
        assert "address" in _chat("s2", "order Ibuprofen 200mg Tablets").lower()
        assert "don't have any active orders" in _chat("s3", "cancel my last order")

    @pytest.mark.parametrize("text", ["are you a doctor", "are you a real person", "can you diagnose me", "are you a bot", "are you a pharmacist"])
    def test_who_it_is_and_what_it_is_not(self, monkeypatch, text):
        self._model_must_not_run(monkeypatch)
        assert _chat("s1", text) == main.ABOUT_REPLY

    @pytest.mark.parametrize("text", ["can you speak telugu", "do you understand hindi", "can you talk in tamil"])
    def test_language_questions(self, monkeypatch, text):
        self._model_must_not_run(monkeypatch)
        assert _chat("s1", text) == main.LANGUAGE_REPLY

    @pytest.mark.parametrize("text", ["sorry", "my bad", "oops", "so sorry"])
    def test_an_apology(self, monkeypatch, text):
        self._model_must_not_run(monkeypatch)
        assert _chat("s1", text) == main.SORRY_REPLY

    @pytest.mark.parametrize("text", ["start over", "reset", "restart", "new chat", "please start over"])
    def test_start_over_really_clears_what_was_pending(self, monkeypatch, text):
        store.set_pending_order("s1", "fev-003", 1)
        store.set_pending_clarification("s1", "fever")
        store.set_last_products("s1", [store.find_product("fev-001"), store.find_product("fev-002")])
        store.save_address("demo_user", "10 High St, Leeds")
        order = store.create_order("demo_user", "fev-001", "10 High St, Leeds", 1)
        self._model_must_not_run(monkeypatch)
        assert _chat("s1", text) == main.RESTART_REPLY
        assert store.get_pending_order("s1") is None and store.get_pending_clarification("s1") is None
        assert not store.get_last_products("s1")
        assert store.get_order(order["order_id"])["status"] == "placed"      # real orders are untouched
        assert store.get_address("demo_user") == "10 High St, Leeds"

    def test_with_an_order_waiting_help_is_not_saved_as_the_address(self, monkeypatch):
        store.set_pending_order("s1", "fev-003", 1)
        self._model_must_not_run(monkeypatch)
        assert _chat("s1", "help") == main.CAPABILITIES_REPLY
        assert store.get_address("demo_user") in (None, "") and store.get_pending_order("s1")["product_id"] == "fev-003"

    def test_with_a_question_open_help_still_answers(self, monkeypatch):
        store.set_pending_clarification("s1", "fever")
        self._model_must_not_run(monkeypatch)
        assert _chat("s1", "what can you do") == main.CAPABILITIES_REPLY

    @pytest.mark.parametrize("text", [
        "can you help me with my cough", "help me, I have a fever", "who is the prime minister", "what is paracetamol used for",
        "I am sorry to say my fever is back", "reset my password", "Help Street 5, Pune",
    ])
    def test_real_requests_are_left_alone(self, monkeypatch, text):
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        reply = _chat("s1", text)
        assert reply not in {main.CAPABILITIES_REPLY, main.ABOUT_REPLY, main.LANGUAGE_REPLY, main.RESTART_REPLY, main.SORRY_REPLY}


class TestDoYouHaveAProductWeCarry:
    """Real gaps (Nov 1 probe): "do you have ibuprofen", "is cetirizine available"
    got the child-or-adult question; "do you carry anything with guaifenesin" the
    out-of-scope decline; "is col-005 in stock" the model's invention that "Col-005
    is not currently available for ordering, as there are no orders to check on"."""

    @staticmethod
    def _model_must_not_run(monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("the model must not be reached")

        monkeypatch.setattr(main, "run_turn", _boom)

    @pytest.mark.parametrize("text, name, price", [
        ("do you have ibuprofen", "Ibuprofen 200mg Tablets", "$5.29"),
        ("do you sell Paracetamol 500mg Tablets", "Paracetamol 500mg Tablets", "$4.99"),
        ("do you carry anything with guaifenesin", "Guaifenesin Expectorant Syrup", "$6.79"),
        ("have you got Cetirizine 10mg Antihistamine Tablets", "Cetirizine 10mg Antihistamine Tablets", "$5.99"),
    ])
    def test_a_stocked_product_is_confirmed_with_its_price(self, monkeypatch, text, name, price):
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", text)
        assert reply.startswith(f"Yes, {name} is in our catalog at {price}.") and reply.endswith("Would you like to order it?")

    @pytest.mark.parametrize("text", ["is col-005 in stock", "is cetirizine available"])
    def test_a_stock_question_says_honestly_that_levels_are_not_tracked(self, monkeypatch, text):
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", text)
        assert "I don't track live stock levels" in reply and "not currently available" not in reply

    def test_an_ingredient_in_several_products_lists_them_all(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", "do you stock paracetamol")
        for product_id in ("fev-001", "fev-002", "fev-004"):
            assert f"({product_id})" in reply
        assert "(fev-003)" not in reply and [p["id"] for p in store.get_last_products("s1")][:3] == ["fev-001", "fev-002", "fev-004"]

    def test_yes_then_starts_the_order_for_a_single_product(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        _chat("s1", "do you have ibuprofen")
        assert "address" in _chat("s1", "yes").lower()
        assert store.get_pending_order("s1")["product_id"] == "fev-003"

    def test_a_listed_product_can_be_picked_by_number(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        _chat("s1", "do you stock paracetamol")
        _chat("s1", "the second one")
        assert store.get_pending_order("s1")["product_id"] == "fev-002"

    def test_a_stated_condition_adds_its_caution(self, monkeypatch):
        store.save_session_messages("s1", [{"role": "user", "content": "I have a stomach ulcer"}])
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", "do you have ibuprofen")
        assert "Please be careful with:" in reply and "you mentioned a stomach ulcer" in reply

    def test_an_unstocked_brand_keeps_its_own_answer(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        assert _chat("s1", "do you have Tylenol").startswith("We don't carry Tylenol")

    @pytest.mark.parametrize("text", [
        "do you have something for a cold", "do you have anything for fever", "order ibuprofen", "what are the side effects of ibuprofen",
        "is my order available", "do you have my address",
    ])
    def test_everything_else_keeps_its_own_path(self, monkeypatch, text):
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        assert "is in our catalog at" not in _chat("s1", text)

    def test_an_order_awaiting_its_address_is_left_alone(self, monkeypatch):
        store.set_pending_order("s1", "col-004", 1)
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        assert "is in our catalog at" not in _chat("s1", "do you have ibuprofen")
        assert store.get_pending_order("s1")["product_id"] == "col-004"


class TestOrderingAMedicineWeDoNotCarry:
    """Real bugs (Nov 2 probe): "order 2 Advil" had the model claim "I'll start the
    process, confirm your shipping address" with nothing started; "add Dolo 650 to
    my cart" said "I've added Dolo 650 to your cart" (there is no cart); "order
    Tylenol" got a confused explanation; "buy aspirin" the out-of-scope decline."""

    @staticmethod
    def _model_must_not_run(monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("the model must not be reached")

        monkeypatch.setattr(main, "run_turn", _boom)

    @pytest.mark.parametrize("text, brand, equivalent", [
        ("order Tylenol", "Tylenol", "fev-001"), ("order 2 Advil", "Advil", "fev-003"), ("I'd like to order Crocin", "Crocin", "fev-001"),
        ("add Dolo 650 to my cart", "Dolo", "fev-001"), ("buy Zyrtec", "Zyrtec", "col-001"), ("purchase Sudafed", "Sudafed", "col-002"),
    ])
    def test_a_brand_order_gets_the_same_ingredient_products_and_starts_nothing(self, monkeypatch, text, brand, equivalent):
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", text)
        assert reply.startswith(f"We don't carry {brand}") and f"({equivalent})" in reply
        assert store.get_pending_order("s1") is None and store.list_orders("demo_user") == []

    @pytest.mark.parametrize("text", ["buy aspirin", "purchase Benadryl", "order amoxicillin", "get me some NyQuil"])
    def test_a_medicine_with_no_equivalent_gets_an_honest_not_stocked(self, monkeypatch, text):
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", text)
        assert reply.startswith("We don't carry") and "show me all your products" in reply and store.get_pending_order("s1") is None

    def test_the_equivalents_can_then_be_picked_and_ordered(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        _chat("s1", "order Advil")
        assert "address" in _chat("s1", "yes").lower()
        assert store.get_pending_order("s1")["product_id"] == "fev-003"

    @pytest.mark.parametrize("text", [
        "order Ibuprofen 200mg Tablets", "order col-001", "I ordered Tylenol yesterday and it helped", "how much does Tylenol cost to order",
    ])
    def test_stocked_products_and_unrelated_messages_keep_their_own_path(self, monkeypatch, text):
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        assert not _chat("s1", text).startswith("We don't carry")

    def test_cancelling_an_order_of_a_brand_still_cancels_not_a_not_stocked_reply(self, monkeypatch):
        store.save_address("demo_user", "10 High St, Leeds")
        store.create_order("demo_user", "fev-001", "10 High St, Leeds", 1)
        self._model_must_not_run(monkeypatch)
        assert "has been cancelled" in _chat("s1", "cancel my Tylenol order")


class TestFillerWhileAnOrderAwaitsItsAddress:
    @pytest.mark.parametrize("text", ["hmm", "...", "thanks", "?!", "idk", "what"])
    def test_filler_is_never_saved_as_the_address_and_the_order_keeps_waiting(self, monkeypatch, text):
        store.set_pending_order("s1", "fev-003", 1)
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        _chat("s1", text)
        assert store.get_address("demo_user") in (None, "") and store.list_orders("demo_user") == []
        assert store.get_pending_order("s1")["product_id"] == "fev-003"

    @pytest.mark.parametrize("text", ["later", "not now", "maybe later"])
    def test_later_declines_the_waiting_order(self, monkeypatch, text):
        store.set_pending_order("s1", "fev-003", 1)
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        reply = _chat("s1", text)
        assert "won't place that order without a shipping address" in reply
        assert store.get_pending_order("s1") is None and store.list_orders("demo_user") == []

    def test_the_real_address_afterwards_still_works(self, monkeypatch):
        store.set_pending_order("s1", "fev-003", 1)
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        _chat("s1", "hmm")
        assert "Shipping to: 5 Mid Rd, Pune" in _chat("s1", "5 Mid Rd, Pune")


class TestAConditionStatedAfterTheOrderStartedStillStopsIt:
    """Real safety gap (stateful fuzz, Nov 3): the condition check ran only when
    an order STARTED. "order Ibuprofen 200mg Tablets" ... "I'm on warfarin" ... then
    the address -> the ibuprofen order was placed."""

    @staticmethod
    def _model_must_not_run(monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("the model must not be reached")

        monkeypatch.setattr(main, "run_turn", _boom)

    @pytest.mark.parametrize("statement", ["I have a stomach ulcer", "I'm on warfarin", "I'm pregnant", "I'm allergic to ibuprofen"])
    def test_a_condition_said_while_the_address_is_awaited_blocks_the_order(self, monkeypatch, statement):
        self._model_must_not_run(monkeypatch)
        _chat("s1", "order Ibuprofen 200mg Tablets")
        store.save_session_messages("s1", store.get_session_messages("s1") + [{"role": "user", "content": statement}])
        reply = _chat("s1", "5 Mid Rd, Pune")
        assert reply.startswith("Sorry, I can't set up an order for Ibuprofen 200mg Tablets") and "Order confirmed!" not in reply
        assert store.list_orders("demo_user") == [] and store.get_pending_order("s1") is None

    def test_the_same_at_the_address_confirmation_step(self, monkeypatch):
        store.save_address("demo_user", "10 High St, Leeds")
        self._model_must_not_run(monkeypatch)
        _chat("s1", "order Ibuprofen 200mg Tablets")
        assert store.get_pending_address_confirmation("s1") is not None
        store.save_session_messages("s1", store.get_session_messages("s1") + [{"role": "user", "content": "I have a stomach ulcer"}])
        reply = _chat("s1", "yes")
        assert "I can't set up an order for Ibuprofen 200mg Tablets" in reply and store.list_orders("demo_user") == []
        assert store.get_pending_address_confirmation("s1") is None

    def test_a_condition_in_the_completing_message_itself_counts(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        _chat("s1", "order Ibuprofen 200mg Tablets")
        reply = _chat("s1", "5 Mid Rd, Pune, I'm allergic to ibuprofen")
        assert "I can't set up an order" in reply and store.list_orders("demo_user") == []

    def test_another_product_is_unaffected_and_a_clean_order_still_completes(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        _chat("s1", "order Paracetamol 500mg Tablets")
        store.save_session_messages("s1", store.get_session_messages("s1") + [{"role": "user", "content": "I have a stomach ulcer"}])
        assert "Order confirmed!" in _chat("s1", "5 Mid Rd, Pune")
        _chat("s2", "order Ibuprofen 200mg Tablets")
        assert "Order confirmed!" in _chat("s2", "5 Mid Rd, Pune")


class TestAStatementAboutTheCustomerIsNotAnAddress:
    """Real bug (stateful fuzz, Nov 3): with an order waiting for its address, "I
    have a stomach ulcer" was accepted as the shipping address and the order placed
    to it."""

    @staticmethod
    def _stub_model(monkeypatch):
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))

    @pytest.mark.parametrize("text", [
        "I have a stomach ulcer", "I'm pregnant", "I am allergic to ibuprofen", "I can't breathe", "I don't have a house number",
        "my son is allergic to penicillin", "my mother has diabetes", "it's for my wife", "he has a fever", "I think I have the flu",
        "I need it urgently", "I'm on warfarin", "im sick", "im on warfarin", "im allergic to ibuprofen",
    ])
    def test_it_is_never_saved_as_the_address(self, monkeypatch, text):
        store.set_pending_order("s1", "fev-003", 1)
        self._stub_model(monkeypatch)
        _chat("s1", text)
        assert store.get_address("demo_user") in (None, "") and store.list_orders("demo_user") == []
        assert store.get_pending_order("s1")["product_id"] == "fev-003"

    @pytest.mark.parametrize("address", [
        "I.T. Park Road, Hyderabad", "Im Lane 4, Pune", "Itza Nagar 5", "Hemant Colony, Delhi", "My Home Apartments, Block 4, Pune",
        "Issac Newton Road 12", "He Man Street 3",
    ])
    def test_real_addresses_that_start_with_such_letters_still_pass(self, address):
        assert main._looks_like_a_first_time_address(address) is True


class TestWhileAnOrderWaitsNothingButARealAddressBecomesTheAddress:
    @pytest.mark.parametrize("text", [
        "adult", "dry", "order it", "reorder that", "good morning", "send me the receipt", "for myself", "show me all your products",
        "my 4 year old has a cold", "ordr ibuprofin 200 mg", "naaku jvaram undi",
    ])
    def test_no_order_is_shipped_to_it(self, monkeypatch, text):
        store.set_pending_order("s1", "fev-003", 1)
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        _chat("s1", text)
        assert store.list_orders("demo_user") == [] and store.get_address("demo_user") in (None, "")

    def test_the_real_address_still_completes_it_afterwards(self, monkeypatch):
        store.set_pending_order("s1", "fev-003", 1)
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        for filler in ["adult", "order it", "good morning"]:
            _chat("s1", filler)
        assert "Shipping to: 5 Mid Rd, Pune" in _chat("s1", "5 Mid Rd, Pune")


class TestChangeMyAddressToXWhileAnOrderWaits:
    """Real bug (fuzz, Nov 5): with an order waiting for its address, "change my
    address to 99 New Rd, York" was saved as the address -- the whole sentence --
    and the order placed to it. The sentence GIVES the address: complete the order
    with "99 New Rd, York"."""

    @staticmethod
    def _model_must_not_run(monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("the model must not be reached")

        monkeypatch.setattr(main, "run_turn", _boom)

    @pytest.mark.parametrize("text", [
        "change my address to 99 New Rd, York", "update my shipping address to 99 New Rd, York", "please set my address as 99 New Rd, York",
    ])
    def test_the_order_ships_to_the_address_in_the_sentence_only(self, monkeypatch, text):
        store.set_pending_order("s1", "fev-003", 1)
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", text)
        assert "Order confirmed!" in reply and "Shipping to: 99 New Rd, York" in reply
        assert store.get_address("demo_user") == "99 New Rd, York" and store.get_pending_order("s1") is None

    def test_at_the_address_confirmation_step_it_is_the_different_address(self, monkeypatch):
        store.save_address("demo_user", "10 High St, Leeds")
        store.set_pending_address_confirmation("s1", "fev-003", 1)
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", "change my address to 99 New Rd, York")
        assert "Order confirmed!" in reply and "Shipping to: 99 New Rd, York" in reply
        assert store.get_address("demo_user") == "99 New Rd, York" and store.get_pending_address_confirmation("s1") is None

    def test_the_quantity_and_product_of_the_waiting_order_are_kept(self, monkeypatch):
        store.set_pending_order("s1", "col-001", 2)
        self._model_must_not_run(monkeypatch)
        _chat("s1", "change my address to 99 New Rd, York")
        assert [(o["product_id"], o["quantity"], o["address"]) for o in store.list_orders("demo_user")] == [("col-001", 2, "99 New Rd, York")]

    def test_a_stated_condition_still_stops_it(self, monkeypatch):
        store.save_session_messages("s1", [{"role": "user", "content": "I have a stomach ulcer"}])
        store.set_pending_order("s1", "fev-003", 1)
        self._model_must_not_run(monkeypatch)
        assert "I can't set up an order" in _chat("s1", "change my address to 99 New Rd, York")
        assert store.list_orders("demo_user") == []

    def test_with_nothing_waiting_it_only_saves_the_address_as_before(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", "change my address to 99 New Rd, York")
        assert "saved 99 New Rd, York" in reply and store.list_orders("demo_user") == []

    def test_a_phrase_that_is_not_a_real_address_is_not_accepted(self, monkeypatch):
        store.set_pending_order("s1", "fev-003", 1)
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        _chat("s1", "change my address to later")   # "later" is also a decline word: the waiting order is dropped
        assert store.list_orders("demo_user") == [] and store.get_address("demo_user") in (None, "")


class TestAnEmailSentenceWhileAnOrderAwaitsItsAddress:
    """Real bug (a fuzz of the REAL chat model, Nov 6): with an order waiting for its
    address, "my email is x@y.com" reached the model, which called
    save_address("x@y.com") -- the email became the shipping address."""

    @staticmethod
    def _model_must_not_run(monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("the model must not be reached")

        monkeypatch.setattr(main, "run_turn", _boom)

    @pytest.mark.parametrize("text", ["my email is x@y.com", "use x@y.com as my email", "email me at x@y.com", "x@y.com is my email"])
    def test_the_email_is_saved_and_the_address_asked_again(self, monkeypatch, text):
        store.set_pending_order("s1", "fev-003", 1)
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", text)
        assert "saved x@y.com" in reply and "shipping address" in reply
        assert store.get_email("demo_user") == "x@y.com" and store.get_address("demo_user") in (None, "")
        assert store.get_pending_order("s1")["product_id"] == "fev-003" and store.list_orders("demo_user") == []

    def test_at_the_address_confirmation_step_too(self, monkeypatch):
        store.save_address("demo_user", "10 High St, Leeds")
        store.set_pending_address_confirmation("s1", "fev-003", 1)
        self._model_must_not_run(monkeypatch)
        reply = _chat("s1", "my email is x@y.com")
        assert "saved x@y.com" in reply and "10 High St, Leeds" in reply
        assert store.get_address("demo_user") == "10 High St, Leeds"

    @pytest.mark.parametrize("text", ["5 Mid Rd, Pune, my email is a@b.com", "5 Mid Rd, Pune and email me at a@b.com", "5 Mid Rd, Pune, a@b.com"])
    def test_a_real_address_with_an_email_is_still_an_address(self, monkeypatch, text):
        store.set_pending_order("s1", "fev-003", 1)
        reply = _chat("s1", text)
        assert "Order confirmed!" in reply and store.get_email("demo_user") == "a@b.com"


class TestTheEmailsIntroducingWordsAreNotPartOfTheAddress:
    """Real bug (found Nov 6): "5 Mid Rd, Pune, my email is a@b.com" shipped to --
    and the receipt said -- "5 Mid Rd, Pune, my email is"."""

    @pytest.mark.parametrize("text", [
        "5 Mid Rd, Pune, my email is a@b.com", "5 Mid Rd, Pune and email me at a@b.com",
        "5 Mid Rd, Pune. Send the confirmation to a@b.com", "5 Mid Rd, Pune, email: a@b.com", "5 Mid Rd, Pune, a@b.com",
    ])
    def test_the_address_is_stored_and_printed_without_the_lead_in(self, text):
        store.set_pending_order("s1", "fev-003", 1)
        reply = _chat("s1", text)
        assert "Shipping to: 5 Mid Rd, Pune\n" in reply + "\n"
        assert store.get_address("demo_user") == "5 Mid Rd, Pune" and store.get_email("demo_user") == "a@b.com"


class TestIWantToOrderWithNoProduct:
    """Real bug (Nov 8): "I want to order" made the model invent "I'll need you to
    confirm your shipping address before we can proceed" (with a hidden tool-
    argument error); the follow-up "ibuprofen" then got the child-or-adult question
    instead of being taken as the product; and the address typed after that was
    saved silently with no order placed."""

    @staticmethod
    def _model_must_not_run(monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("the model must not be reached")

        monkeypatch.setattr(main, "run_turn", _boom)

    @pytest.mark.parametrize("text", [
        "I want to order", "order", "I'd like to buy something", "I need to order medicine", "place an order", "buy", "hello, I want to order",
    ])
    def test_it_asks_which_product(self, monkeypatch, text):
        self._model_must_not_run(monkeypatch)
        assert _chat("s1", text) == main.ORDER_WHICH_REPLY
        assert store.get_pending_clarification("s1") == main.ORDER_WHICH_TRIGGER and store.get_pending_order("s1") is None

    @pytest.mark.parametrize("answer, product_id", [
        ("ibuprofen", "fev-003"), ("Ibuprofen 200mg Tablets", "fev-003"), ("cetirizine", "col-001"), ("col-005", "col-005"),
        ("the guaifenesin one", "col-005"),
    ])
    def test_the_next_message_naming_a_product_starts_that_order(self, monkeypatch, answer, product_id):
        self._model_must_not_run(monkeypatch)
        _chat("s1", "I want to order")
        reply = _chat("s1", answer)
        assert "address" in reply.lower() and store.get_pending_order("s1")["product_id"] == product_id
        assert store.get_pending_clarification("s1") is None

    def test_an_ingredient_in_several_products_asks_which_one_and_remembers_the_list(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        _chat("s1", "I want to order")
        reply = _chat("s1", "paracetamol")
        assert reply.startswith("Which one would you like?") and "(fev-001)" in reply and "(fev-002)" in reply
        _chat("s1", "the second one")
        assert store.get_pending_order("s1")["product_id"] == "fev-002"

    def test_the_whole_flow_ends_in_a_real_order(self, monkeypatch):
        self._model_must_not_run(monkeypatch)
        _chat("s1", "I want to order")
        _chat("s1", "ibuprofen")
        assert "Order confirmed!" in _chat("s1", "5 Mid Rd, Pune")
        assert [o["product_id"] for o in store.list_orders("demo_user")] == ["fev-003"]

    def test_an_answer_that_names_no_product_goes_to_the_normal_path(self, monkeypatch):
        _chat_stub = lambda messages, **kwargs: ("from the model", messages)
        monkeypatch.setattr(main, "run_turn", _chat_stub)
        _chat("s1", "I want to order")
        assert _chat("s1", "I have a fever") != main.ORDER_WHICH_REPLY

    @pytest.mark.parametrize("text", [
        "order ibuprofen", "how do I order", "can I order", "I want to order it", "ok order", "cancel my order", "I want to know about ordering",
    ])
    def test_other_messages_do_not_start_it(self, monkeypatch, text):
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        assert _chat("s1", text) != main.ORDER_WHICH_REPLY

    def test_with_an_order_already_waiting_it_is_left_alone(self, monkeypatch):
        store.set_pending_order("s1", "col-004", 1)
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        assert _chat("s1", "I want to order") != main.ORDER_WHICH_REPLY
        assert store.get_pending_order("s1")["product_id"] == "col-004"


class TestHowManyCanIOrder:
    """Real bug (Nov 8): "what is the maximum quantity" made the model INVENT a limit
    ("Paracetamol 500mg Tablets: 20 tablets per order"); "how many can I order"
    got an apology for a mistake it had not made (with a hidden tool error). The
    real limit is tools.MAX_QUANTITY_PER_ORDER."""

    @pytest.mark.parametrize("text", [
        "what is the maximum quantity", "how many can I order", "what's the order limit", "can I order more than 2",
        "is there a limit on how many I can buy", "maximum per order", "how much can I order",
    ])
    def test_the_real_limit_is_stated(self, monkeypatch, text):
        def _boom(*args, **kwargs):
            raise AssertionError("the model must not be reached")

        monkeypatch.setattr(main, "run_turn", _boom)
        reply = _chat("s1", text)
        assert reply.startswith(f"You can order up to {tools.MAX_QUANTITY_PER_ORDER} of each product in one order.")
        assert "20 tablets" not in reply and "placeholder limit" in reply

    def test_the_stated_limit_is_what_is_actually_enforced(self, monkeypatch):
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        limit = tools.MAX_QUANTITY_PER_ORDER
        assert f"{limit} of each product" in _chat("s1", "what is the maximum quantity")
        _chat("s2", f"order {limit + 3} Ibuprofen 200mg Tablets")
        assert store.get_pending_order("s2")["quantity"] == limit

    @pytest.mark.parametrize("text", [
        "what is the maximum dose of ibuprofen per day", "how many tablets can I take", "how many can I take per day",
        "order 2 ibuprofen", "how many orders do I have", "limit",
    ])
    def test_dosing_and_other_messages_are_left_alone(self, monkeypatch, text):
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        assert "of each product in one order" not in _chat("s1", text)

    def test_an_order_awaiting_its_address_is_left_alone(self, monkeypatch):
        store.set_pending_order("s1", "col-004", 1)
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        assert "of each product in one order" not in _chat("s1", "what is the maximum quantity")


class TestCurlyApostropheNegation:
    """Real bug (Nov 9): phones type "don’t" with a curly apostrophe, and the
    cancel/reorder negation check only knew the straight one — so "don’t
    cancel my order" was read as a cancel request. Every curly quote is now
    normalised at the door."""

    def _order(self):
        store.save_address("demo_user", "10 High St, Leeds")
        return store.create_order("demo_user", "fev-003", "10 High St, Leeds", 1)

    @pytest.mark.parametrize("text", ["don’t cancel ord-0001", "Please don’t cancel ord-0001 yet"])
    def test_a_negated_cancel_with_a_curly_apostrophe_cancels_nothing(self, monkeypatch, text):
        self._order()
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        _chat("s1", text)
        assert store.get_order("ord-0001")["status"] != "cancelled"

    def test_the_straight_apostrophe_form_behaves_the_same(self, monkeypatch):
        self._order()
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        _chat("s1", "don't cancel ord-0001")
        assert store.get_order("ord-0001")["status"] != "cancelled"

    def test_the_model_sees_the_normalised_text(self, monkeypatch):
        seen = []
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: (seen.append(kwargs["user_text"]) or "ok", messages))
        _chat("s1", "I’ve been feeling “off” since yesterday, any ideas")
        assert seen and "’" not in seen[0] and "“" not in seen[0]


class TestDozenAndNegativeQuantityRoute:
    @pytest.mark.parametrize("text", ["order a dozen ibuprofen", "order half a dozen ibuprofen", "order two dozen ibuprofen"])
    def test_a_dozen_is_capped_at_the_real_limit_and_says_so(self, text):
        reply = _chat("s1", text)
        assert f"per-order limit of {tools.MAX_QUANTITY_PER_ORDER}" in reply
        assert store.get_pending_order("s1")["quantity"] == tools.MAX_QUANTITY_PER_ORDER

    def test_a_negative_quantity_says_it_was_not_valid(self):
        reply = _chat("s1", "order -1 ibuprofen")
        assert "wasn't valid" in reply
        assert store.get_pending_order("s1")["quantity"] == 1


class TestPendingOrderAddressReplies:
    def _pending(self):
        store.set_pending_order("s1", "fev-003", 1)

    @pytest.mark.parametrize("text", ["in 3 days", "for 2 weeks", "5 years old", "650 mg tablets"])
    def test_a_quantity_is_not_shipped_to_as_an_address(self, monkeypatch, text):
        self._pending()
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        assert _chat("s1", text) == "from the model"
        assert store.get_pending_order("s1") is not None and store.list_orders("demo_user") == []

    @pytest.mark.parametrize("text", [
        "Deliver to: 12 Baker Street, London", "my address is 12 Baker Street, London",
        "ship it to 12 Baker Street, London please",
    ])
    def test_the_lead_in_and_the_please_are_not_part_of_the_saved_address(self, text):
        self._pending()
        reply = _chat("s1", text)
        assert "Shipping to: 12 Baker Street, London\n" in reply
        assert store.get_address("demo_user") == "12 Baker Street, London"


class TestReorderPhrasings:
    """Real gap (Nov 11): with the real model, "same as last time" listed the
    order instead of reordering it. Common ways of saying "again" are now
    recognised in code."""

    def _placed(self):
        store.save_address("demo_user", "10 High St, Leeds")
        store.create_order("demo_user", "fev-003", "10 High St, Leeds", 1)

    @pytest.mark.parametrize("text", [
        "same as last time", "repeat my last order", "order the same", "same again please",
        "get the same thing again", "same as before", "Same again!",
    ])
    def test_these_start_a_reorder(self, monkeypatch, text):
        self._placed()
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        reply = _chat("s1", text)
        assert reply != "from the model", text
        assert "10 High St, Leeds" in reply
        assert store.get_pending_address_confirmation("s1") == {"product_id": "fev-003", "quantity": 1}

    @pytest.mark.parametrize("text", [
        "what is the same as before in ibuprofen", "how do I repeat my last order", "which is the same as last time",
        "don't repeat my last order", "order it",
    ])
    def test_questions_and_retractions_do_not(self, monkeypatch, text):
        self._placed()
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))
        _chat("s1", text)
        assert store.get_pending_address_confirmation("s1") is None
        assert len(store.list_orders("demo_user")) == 1


class TestPriceQuestionsAnswerWhatWasAsked:
    """Real gaps (Nov 11): "how much for 2 ibuprofen" answered the single-unit
    price; "price of paracetamol 500mg" listed all the paracetamol products;
    "how much for ibuprofen 400mg" quoted the 200mg price as if it were the
    400mg one."""

    @pytest.fixture(autouse=True)
    def _no_model(self, monkeypatch):
        monkeypatch.setattr(main, "run_turn", lambda messages, **kwargs: ("from the model", messages))

    @pytest.mark.parametrize("text, reply", [
        ("how much for 2 ibuprofen", "2 × Ibuprofen 200mg Tablets = $10.58 ($5.29 each)."),
        ("how much for two ibuprofen tablets", "2 × Ibuprofen 200mg Tablets = $10.58 ($5.29 each)."),
        ("total for 2 paracetamol 500mg", "2 × Paracetamol 500mg Tablets = $9.98 ($4.99 each)."),
        ("how much for 2 fev-003", "2 × Ibuprofen 200mg Tablets = $10.58 ($5.29 each)."),
    ])
    def test_a_quantity_gets_a_total(self, text, reply):
        assert _chat("s1", text) == reply

    def test_a_quantity_over_the_limit_is_capped_and_says_so(self):
        reply = _chat("s1", "price of 5 ibuprofen")
        assert f"per-order limit is {tools.MAX_QUANTITY_PER_ORDER}" in reply
        assert "= $10.58" in reply

    @pytest.mark.parametrize("text", ["how much is ibuprofen", "cost of 1 ibuprofen"])
    def test_a_single_unit_keeps_the_plain_price(self, text):
        assert _chat("s1", text) == "Ibuprofen 200mg Tablets costs $5.29."

    @pytest.mark.parametrize("text, reply", [
        ("price of paracetamol 500mg", "Paracetamol 500mg Tablets costs $4.99."),
        ("price of paracetamol 650mg", "Paracetamol Extra Strength 650mg costs $6.49."),
    ])
    def test_a_named_strength_names_one_product(self, text, reply):
        assert _chat("s1", text) == reply

    def test_two_strengths_still_list_the_range(self):
        reply = _chat("s1", "price of paracetamol 500mg and 650mg")
        assert "Paracetamol 500mg Tablets" in reply and "Paracetamol Extra Strength 650mg" in reply

    def test_a_strength_we_do_not_stock_is_not_priced_as_if_we_did(self):
        reply = _chat("s1", "how much for ibuprofen 400mg")
        assert reply.startswith("We don't carry that strength") and "$5.29" in reply


class TestAgesAreNotQuantitiesRoute:
    @pytest.mark.parametrize("text", ["order paracetamol 500mg for my 5 year old", "my 2 year old has a fever, I want paracetamol 500mg"])
    def test_the_pending_quantity_is_one(self, text):
        _chat("s1", text)
        assert store.get_pending_order("s1")["quantity"] == 1


class TestIbuprofenForAChildRoute:
    @pytest.fixture(autouse=True)
    def _no_model(self, monkeypatch):
        def boom(*args, **kwargs):
            raise AssertionError("the model must not be reached")
        monkeypatch.setattr(main, "run_turn", boom)

    @pytest.mark.parametrize("text", [
        "can I give ibuprofen to my 5 year old", "is ibuprofen safe for kids", "ibuprofen dose for my son",
        "my daughter has fever can she take ibuprofen", "can children take ibuprofen", "ibuprofen for baby",
        "can I give advil to my toddler",
    ])
    def test_the_question_gets_an_honest_adult_only_answer(self, text):
        reply = _chat("s1", text)
        assert "adult dose only" in reply and "pharmacist or doctor" in reply
        assert "Children's Paracetamol Syrup (fev-004)" in reply
        assert store.get_pending_order("s1") is None

    @pytest.mark.parametrize("text", [
        "order ibuprofen for my 5 year old", "I want ibuprofen for my son", "my 2 year old has fever I want ibuprofen",
        "order ibuprofen for my 6 month old",
    ])
    def test_an_order_for_a_child_is_refused_and_nothing_starts(self, text):
        reply = _chat("s1", text)
        assert reply.startswith("Sorry, I can't set up an order for Ibuprofen 200mg Tablets")
        assert "a child" in reply
        assert store.get_pending_order("s1") is None

    def test_an_adult_can_still_order_ibuprofen(self):
        assert store.get_pending_order("s1") is None
        _chat("s1", "order ibuprofen")
        assert store.get_pending_order("s1")["product_id"] == "fev-003"


class TestTwoEmailAddressesInOneMessage:
    """Real gap (Nov 12): "bob@gmail.com and alice@x.org" sent the receipt to the
    first address without asking which was meant."""

    def _waiting_for_an_email(self):
        store.save_address("demo_user", "10 High St, Leeds")
        order = store.create_order("demo_user", "fev-001", "10 High St, Leeds", 1)
        store.set_pending_email("s1", order["order_id"])

    @pytest.mark.parametrize("text", [
        "bob@gmail.com and alice@x.org", "bob@gmail.com, alice@x.org", "my email is bob@gmail.com or alice@x.org",
    ])
    def test_it_asks_which_and_sends_nothing(self, monkeypatch, text):
        self._waiting_for_an_email()
        sent = []
        monkeypatch.setattr(tools, "send_confirmation_email", lambda *a, **k: sent.append(a) or "{}")
        reply = _chat("s1", text)
        assert "which one should I use" in reply and "bob@gmail.com" in reply and "alice@x.org" in reply
        assert sent == [] and store.get_email("demo_user") in (None, "")
        assert store.get_pending_email("s1") is not None   # still waiting for the answer

    @pytest.mark.parametrize("text", ["bob@gmail.com", "BOB@gmail.com bob@gmail.com", "bob@gmail.com, bob@gmail.com"])
    def test_one_address_repeated_is_still_one_address(self, text):
        self._waiting_for_an_email()
        reply = _chat("s1", text)
        assert "I've sent the confirmation" in reply
