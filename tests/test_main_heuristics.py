"""Tests for the pure text-heuristic functions in app/main.py — the
deterministic pre-agent dispatch chain that decides whether a message is
an address, an email, a bare product selection, an affirmative/decline,
etc., BEFORE anything reaches the LLM. This is some of the most
bug-history-laden code in the project: every function here has a real,
previously-shipped failure mode documented in its own comment ("order
col-001" mistaken for an address, "hyderabad" as a bare place-name
address, "last one" fabricating an unrelated product list, "yes please"
and "ok continue bro" falling through to a stuck loop, "no new one" not
being recognized as rejecting the address on file) — none of which had a
pinned regression test until now, only ever verified by hand via curl.

No LLM or database needed — every function tested here is pure text
matching.
"""

from app import main


class TestExtractEmail:
    def test_extracts_email_and_removes_it_from_the_remainder(self):
        email, remainder = main._extract_email("123 Main St a@example.com")
        assert email == "a@example.com"
        assert remainder == "123 Main St"

    def test_strips_a_trailing_email_label(self):
        email, remainder = main._extract_email("123 Main St, email: a@example.com")
        assert email == "a@example.com"
        assert remainder == "123 Main St"

    def test_no_email_present(self):
        email, remainder = main._extract_email("123 Main St")
        assert email is None
        assert remainder == "123 Main St"


class TestEmailPunctuation:
    """Real bug: EMAIL_RE's tail `[\\w.-]+` also matched trailing punctuation,
    so "my email is bob@gmail.com." was captured as "bob@gmail.com." — saved as
    the customer's email and used as the confirmation recipient (a mail server
    rejects the trailing dot). Fixing the pattern frees that punctuation, so
    the remainder cleanup has to absorb it too, or a bare email stops being
    recognized as bare and the stray punctuation is saved as an ADDRESS."""

    def test_trailing_sentence_punctuation_is_not_part_of_the_email(self):
        for text in ["my email is bob@gmail.com.", "it's bob@gmail.com!", "bob@gmail.com?", "email: bob@gmail.com;"]:
            assert main._extract_email(text)[0] == "bob@gmail.com", text

    def test_a_real_address_with_a_trailing_period_after_the_email_keeps_the_address(self):
        email, remainder = main._extract_email("123 Main St, bob@gmail.com.")
        assert email == "bob@gmail.com"
        assert remainder == "123 Main St"

    def test_multi_part_domains_still_match_in_full(self):
        assert main._extract_email("a.b+tag@mail.co.uk")[0] == "a.b+tag@mail.co.uk"
        assert main._extract_email("a.b+tag@mail.co.uk.")[0] == "a.b+tag@mail.co.uk"
        assert main._extract_email("first.last@sub.domain.example.org")[0] == "first.last@sub.domain.example.org"

    def test_a_bare_email_stays_bare_with_trailing_or_wrapping_punctuation(self):
        for text in [
            "bob123@gmail.com.", "bob123@gmail.com!", "(bob123@gmail.com)",
            "<bob123@gmail.com>", '"bob123@gmail.com"', "[bob123@gmail.com]",
        ]:
            assert main._is_bare_email(text), text
            assert not main._looks_like_an_address(text), text
            assert not main._looks_like_a_first_time_address(text), text

    def test_text_without_a_real_email_is_untouched(self):
        assert main._extract_email("not an email") == (None, "not an email")
        assert main._extract_email("bob@localhost") == (None, "bob@localhost")


class TestLooksLikeAnEmail:
    def test_detects_a_real_email(self):
        assert main._looks_like_an_email("a@example.com")

    def test_plain_text_is_not_an_email(self):
        assert not main._looks_like_an_email("123 Main St")


class TestStripAddressFiller:
    def test_strips_ship_to_phrasing(self):
        assert main._strip_address_filler("ship to 456 New Ave") == "456 New Ave"
        assert main._strip_address_filler("actually please ship this to 456 New Ave") == "456 New Ave"

    def test_leaves_a_bare_address_unchanged(self):
        assert main._strip_address_filler("456 New Ave") == "456 New Ave"


class TestLooksLikeAnAddress:
    def test_a_plausible_address_passes(self):
        assert main._looks_like_an_address("123 Main St, Springfield")

    def test_rejects_a_product_id_even_with_digits(self):
        # Real bug: "order col-001" has a digit and is long enough to pass
        # the basic check, but is clearly a product reference, not an
        # address — it was getting silently saved as the shipping address.
        assert not main._looks_like_an_address("order col-001")
        assert not main._looks_like_an_address("fev-001 please")

    def test_rejects_text_with_no_digits(self):
        assert not main._looks_like_an_address("thank you")
        assert not main._looks_like_an_address("actually never mind")

    def test_rejects_text_that_is_too_short(self):
        assert not main._looks_like_an_address("no 5")

    def test_rejects_a_bare_email_even_with_a_digit(self):
        # Real bug: an email containing a digit (e.g. sukumar123@example.com)
        # has a digit and is long enough to pass the basic check, so this
        # used to be accepted as the address — _extract_email then stripped
        # the email out, left nothing behind, and the caller's
        # `address_only or address_text` fallback re-saved the raw email
        # text as the shipping address.
        assert not main._looks_like_an_address("sukumar123@example.com")

    def test_an_address_with_an_email_alongside_it_still_counts(self):
        assert main._looks_like_an_address("123 Main St, sukumar@example.com")


class TestIsBareEmail:
    def test_email_alone_is_bare(self):
        assert main._is_bare_email("sukumar@example.com")
        assert main._is_bare_email("sukumar123@example.com")

    def test_email_with_real_address_content_is_not_bare(self):
        assert not main._is_bare_email("123 Main St, sukumar@example.com")

    def test_no_email_is_not_bare(self):
        assert not main._is_bare_email("123 Main St")


class TestLooksLikeAFirstTimeAddress:
    def test_a_bare_place_name_with_no_digits_counts(self):
        # Real bug: a bare place name is a completely normal answer to a
        # fresh "what's your address?" ask but has no digit, so it used to
        # fall through to the LLM — which then claimed the address was
        # saved without ever actually calling save_address.
        assert main._looks_like_a_first_time_address("hyderabad")

    def test_a_normal_address_with_digits_counts(self):
        assert main._looks_like_a_first_time_address("123 Main St")

    def test_a_product_id_does_not_count(self):
        assert not main._looks_like_a_first_time_address("col-001")

    def test_a_question_does_not_count(self):
        assert not main._looks_like_a_first_time_address("what do you mean?")
        assert not main._looks_like_a_first_time_address("where should I send it")

    def test_a_decline_does_not_count(self):
        assert not main._looks_like_a_first_time_address("no thanks")

    def test_a_real_symptom_mention_does_not_count(self):
        # Guards against a fever/cold mention being misread as an address
        # just because it showed up during the address-collection turn.
        assert not main._looks_like_a_first_time_address("I have a fever")

    def test_too_long_does_not_count(self):
        assert not main._looks_like_a_first_time_address("x" * 100)

    def test_too_short_does_not_count(self):
        assert not main._looks_like_a_first_time_address("x")

    def test_a_bare_email_does_not_count_even_without_digits(self):
        # Real bug: this fell straight through to the generic fallback at
        # the end of the function (not a question, not a decline, not a
        # symptom mention) and was accepted as a first-time address even
        # with no digits at all — the digit-based check in
        # _looks_like_an_address never even got a chance to reject it.
        assert not main._looks_like_a_first_time_address("sukumar@example.com")
        assert not main._looks_like_a_first_time_address("sukumar123@example.com")

    def test_a_bare_filler_phrase_does_not_count(self):
        # Real bug: ADDRESS_FILLER_RE's own \s+ after "to" never matches
        # with nothing following it, so _strip_address_filler left a bare
        # "ship to"/"please ship it to" intact — this generic fallback (no
        # digit required) then accepted it as a real first-time address,
        # saving and showing back the literal filler phrase as the
        # shipping address.
        for text in ["ship to", "ship it to", "please ship it to", "send it to", "deliver this to"]:
            assert not main._looks_like_a_first_time_address(text), text


class TestIsFillerOnly:
    def test_bare_filler_phrases_are_filler_only(self):
        for text in ["ship to", "ship it to", "please ship it to", "send it to", "deliver this to", "actually ship to"]:
            assert main._is_filler_only(text), text

    def test_filler_with_a_real_address_after_it_is_not_filler_only(self):
        assert not main._is_filler_only("ship to 456 New Ave")
        assert not main._is_filler_only("please ship it to 10 Downing Street")

    def test_a_real_address_alone_is_not_filler_only(self):
        assert not main._is_filler_only("123 Main St")


class TestIsAffirmative:
    def test_simple_affirmatives(self):
        for text in ["yes", "yeah", "ok", "sure", "yes please", "go ahead"]:
            assert main._is_affirmative(text), text

    def test_casual_variants_found_in_real_testing(self):
        # Real bug: these fell through to a stuck re-ask before "continue",
        # "bro", etc. were added to AFFIRMATIVE_WORDS.
        assert main._is_affirmative("ok continue bro")
        assert main._is_affirmative("sounds good")

    def test_a_message_with_any_non_affirmative_word_is_not_affirmative(self):
        # Real bug this guards against (see AFFIRMATIVE_WORDS's own
        # comment): checking ANY word used to false-positive on messages
        # that clearly aren't a plain yes.
        assert not main._is_affirmative("ok but what about the ibuprofen instead")

    def test_symptom_text_is_not_affirmative(self):
        assert not main._is_affirmative("I have a fever")


class TestWantsDifferentAddress:
    def test_rejection_phrasings_found_in_real_testing(self):
        # Real bug: these all hit the same unhelpful re-ask verbatim,
        # producing a stuck loop with no progress.
        for text in ["no new one", "i will give another address", "i will give different one", "somewhere else"]:
            assert main._wants_different_address(text), text

    def test_the_conversation_eval_phrasing_still_counts(self):
        # Real regression: MAX_ADDRESS_REJECTION_WORDS=6 was picked by only
        # checking the phrasings in this test file — conversation_eval.py's
        # own established "no, i will give a different one" is 7 words and
        # got silently broken by that cap, regressing a previously-passing
        # scripted conversation (falling through to a real, nondeterministic
        # LLM turn instead of the deterministic re-ask).
        assert main._wants_different_address("no, i will give a different one")

    def test_a_plain_address_does_not_want_a_different_one(self):
        assert not main._wants_different_address("123 Main St, Springfield")

    def test_a_long_unrelated_message_containing_a_rejection_word_does_not_count(self):
        # Real bug: unlike _declines, this had no length cap at all, so a
        # longer message asking an unrelated real question ("do you have
        # any other cold medicine options") — which just happens to
        # contain "other" — got misread as rejecting the address on file.
        text = "actually, do you have any other cold medicine options"
        assert not main._wants_different_address(text)


class TestDeclines:
    def test_short_decline_phrasings(self):
        for text in ["no", "nope", "skip", "cancel", "no thanks"]:
            assert main._declines(text), text

    def test_a_long_message_containing_a_decline_word_does_not_count(self):
        # MAX_DECLINE_WORDS guards against a longer message that happens to
        # contain "no" somewhere being misread as a decline.
        text = "no I actually think the paracetamol one sounds like a reasonable fit for me"
        assert not main._declines(text)

    def test_a_plain_affirmative_is_not_a_decline(self):
        assert not main._declines("yes please")


class TestLooksLikeOrderCancellation:
    def test_cancel_and_order_together_counts(self):
        # Real bug this guards: found live via conversation_eval.py — this
        # exact message used to be swallowed as "no, skip the confirmation
        # email" (pending_email_order_id + _declines("cancel")) instead of
        # ever reaching the real cancel_order tool.
        assert main._looks_like_order_cancellation("please cancel my order")
        assert main._looks_like_order_cancellation("can you cancel order ord-0002")

    def test_bare_cancel_alone_does_not_count(self):
        # Genuinely ambiguous with no other context — stays interpreted as
        # declining whichever pending ask is actually in front of it.
        assert not main._looks_like_order_cancellation("cancel")

    def test_order_without_cancel_does_not_count(self):
        assert not main._looks_like_order_cancellation("order fev-001")

    def test_unrelated_decline_does_not_count(self):
        assert not main._looks_like_order_cancellation("no thanks")


class TestResolveBareSelection:
    PRODUCTS = [
        {"id": "fev-001", "name": "Paracetamol 500mg Tablets"},
        {"id": "fev-002", "name": "Paracetamol Extra Strength 650mg"},
        {"id": "fev-003", "name": "Ibuprofen 200mg Tablets"},
    ]

    def test_no_products_shown_yet_resolves_to_none(self):
        assert main._resolve_bare_selection("2", None) is None
        assert main._resolve_bare_selection("2", []) is None

    def test_bare_digit_matches_by_product_id_suffix_first(self):
        # "2" matches fev-002's numeric suffix directly, which happens to
        # also be list position 2 here — see the next test for a case where
        # the two disagree and the suffix match must win.
        assert main._resolve_bare_selection("2", self.PRODUCTS) == self.PRODUCTS[1]

    def test_bare_digit_falls_back_to_list_position_when_no_suffix_matches(self):
        # Neither col-004 nor col-005 has a suffix equal to 1 — "1" must
        # still resolve, by plain list position, to the first item shown.
        cold_products = [{"id": "col-004", "name": "Cough Suppressant Syrup"}, {"id": "col-005", "name": "Guaifenesin"}]
        assert main._resolve_bare_selection("1", cold_products) == cold_products[0]

    def test_bare_digit_out_of_range_resolves_to_none(self):
        assert main._resolve_bare_selection("99", self.PRODUCTS) is None

    def test_only_filler_words_resolves_to_none(self):
        assert main._resolve_bare_selection("the item please", self.PRODUCTS) is None

    def test_ordinal_word_resolves_by_position(self):
        assert main._resolve_bare_selection("the second one", self.PRODUCTS) == self.PRODUCTS[1]
        assert main._resolve_bare_selection("third item bro", self.PRODUCTS) == self.PRODUCTS[2]

    def test_last_resolves_to_the_final_item(self):
        # Real bug: the model used to fabricate an entirely unrelated
        # product list for "last one" instead of picking the actual last
        # item shown.
        assert main._resolve_bare_selection("last one", self.PRODUCTS) == self.PRODUCTS[-1]

    def test_bare_one_resolves_to_the_first_item(self):
        assert main._resolve_bare_selection("one", self.PRODUCTS) == self.PRODUCTS[0]

    def test_unrelated_text_resolves_to_none(self):
        assert main._resolve_bare_selection("actually never mind", self.PRODUCTS) is None
