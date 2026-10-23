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

from app import config, main, store

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
            "Ibuprofen 200mg Tablets", "order ibuprofen",
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
        for text in ["Ibuprofen 200mg Tablets", "I want Cetirizine 10mg", "paracetamol 500mg please"]:
            assert _chat("s1", text) == "from the model", text
            assert store.list_orders("demo_user") == [], text
            assert store.get_pending_order("s1")["product_id"] == "fev-001", text

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
        store.set_pending_email("s1", "ord-0001")
        self._model_reply(monkeypatch)
        assert _chat("s1", text) == "from the model"
        assert store.get_pending_email("s1") == "ord-0001"

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
