"""A recommendation of a medicine we don't carry is replaced by what we do.

Real bug (Oct 22-26 probes): the model's free text said "I suggest taking
Acetaminophen (Tylenol)", "ibuprofen (Advil, Motrin)", "you might consider
Benadryl" — products that are not in the catalog, cannot be ordered and were
never vetted against the knowledge base."""

import pytest

from app import brands, main, store

pytestmark = pytest.mark.usefixtures("isolated_db")


class TestRecommendedOffCatalog:
    @pytest.mark.parametrize("reply, names", [
        ("I suggest taking Acetaminophen (Tylenol) as it can help reduce fever.", ["tylenol"]),
        ("For adults with fevers, you might consider ibuprofen (Advil, Motrin).", ["motrin", "advil"]),
        ("You could try Benadryl or Robitussin for the cough.", ["robitussin", "benadryl"]),
        ("Take Nurofen if needed.", ["nurofen"]),
        ("I recommend using Sudafed for the congestion.", ["sudafed"]),
    ])
    def test_a_recommended_brand_is_found(self, reply, names):
        assert sorted(brands.recommended_off_catalog(reply)) == sorted(names)

    @pytest.mark.parametrize("reply", [
        "Tylenol is a brand name of paracetamol.", "Aspirin is not suitable for children.",
        "I recommend Paracetamol 500mg Tablets.", "Advil contains ibuprofen, which we carry.", "",
    ])
    def test_a_mention_that_does_not_recommend_is_left_alone(self, reply):
        assert brands.recommended_off_catalog(reply) == []

    def test_a_brand_the_customer_named_themselves_is_not_flagged(self):
        assert brands.recommended_off_catalog("I suggest taking Tylenol.", "do you have Tylenol") == []


def _screen(reply, user_text="I have a fever", session="s1"):
    messages = [{"role": "user", "content": user_text}, {"role": "assistant", "content": reply}]
    return main._screen_recommendation(session, reply, messages), messages


class TestScreenRecommendation:
    def test_a_recommended_brand_becomes_the_matching_catalog_products(self):
        reply, messages = _screen("Since you have a fever, I suggest taking Acetaminophen (Tylenol) to reduce it.")
        assert reply.startswith("We don't carry Tylenol, but its active ingredient is paracetamol, which we do:")
        for product_id in ("fev-001", "fev-002", "fev-004"):
            assert f"({product_id})" in reply
        assert "(col-003)" not in reply and "Ibuprofen" not in reply
        assert messages[-1]["content"] == reply  # the stored session matches what was shown

    def test_the_shown_products_can_then_be_picked_by_number(self):
        _screen("You might consider ibuprofen (Advil) for the fever.")
        assert [p["id"] for p in store.get_last_products("s1")] == ["fev-003"]

    def test_a_medicine_with_no_catalog_equivalent_gets_an_honest_not_stocked(self):
        reply, _ = _screen("You could try Benadryl for the allergy symptoms.")
        assert reply.startswith("We don't carry Benadryl") and "show me all your products" in reply

    def test_an_informational_mention_is_untouched(self):
        text = "Tylenol is a brand name of paracetamol, which we carry as Paracetamol 500mg Tablets."
        assert _screen(text)[0] == text

    def test_a_catalog_recommendation_is_untouched(self):
        text = "Paracetamol 500mg Tablets (fev-001) would be a good fit. Would you like to order this?"
        assert _screen(text)[0] == text

    def test_the_condition_screen_still_applies_to_the_replacement(self):
        store.save_session_messages("s1", [{"role": "user", "content": "I have a stomach ulcer"}])
        messages = [{"role": "user", "content": "I have a stomach ulcer and a fever"},
                    {"role": "assistant", "content": "You might consider Advil for the fever."}]
        reply = main._screen_recommendation("s1", messages[-1]["content"], messages)
        assert reply.startswith("I can't recommend Ibuprofen 200mg Tablets")
