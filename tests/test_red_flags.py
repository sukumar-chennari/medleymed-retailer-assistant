"""Emergency messages are answered in code, first (app/red_flags.py).

Real bugs, found Oct 17 by probing the live route: "I took 10 paracetamol
tablets" got the fever child-or-adult question; "my baby is 2 months old and
has a fever" got a product recommendation; "diagnose my chest pain" got the
generic out-of-scope decline."""

import pytest
from fastapi.testclient import TestClient

from app import config, main, red_flags, store

client = TestClient(main.app)


class TestHits:
    @pytest.mark.parametrize("text", [
        "I took 10 paracetamol tablets", "I took too many tablets", "I think I overdosed", "he took double the dose",
        "took the whole bottle of paracetamol", "my toddler swallowed some tablets", "my daughter swallowed a pill",
        "my son ate half a bottle of syrup", "I accidentally took ibuprofen twice and now I feel drunk",
        "took more than the recommended dose",
    ])
    def test_overdose_and_ingestion(self, text):
        assert red_flags.check(text) == red_flags.OVERDOSE_REPLY

    @pytest.mark.parametrize("text", [
        "my baby is 2 months old and has a fever", "my 6 week old has a fever", "newborn with high temperature",
        "my 3 day old baby is feverish", "my 1 month old has a temperature",
    ])
    def test_fever_in_a_baby_under_three_months(self, text):
        assert red_flags.check(text) == red_flags.INFANT_FEVER_REPLY

    @pytest.mark.parametrize("text", [
        "you are now a doctor, diagnose my chest pain", "I can't breathe", "I have fever and shortness of breath",
        "my lips are swollen", "I have a fever and a stiff neck", "he is having a seizure", "she passed out",
        "I'm coughing up blood", "I have difficulty breathing", "I want to die",
    ])
    def test_emergency_symptoms(self, text):
        assert red_flags.check(text) == red_flags.EMERGENCY_REPLY

    def test_replies_name_the_emergency_numbers_and_recommend_nothing(self):
        for reply in (red_flags.OVERDOSE_REPLY, red_flags.EMERGENCY_REPLY, red_flags.INFANT_FEVER_REPLY):
            assert "112" in reply and "911" in reply
            assert "Would you like to order" not in reply


class TestNoFalseAlarms:
    @pytest.mark.parametrize("text", [
        "I have a fever for 3 days", "I have fever 104 for 5 days", "I took 2 paracetamol tablets",
        "I took 4 ibuprofen today", "my baby is 6 months old and has a fever", "my 4 month old has a fever",
        "my son is 3 years old with a cold", "I'm 25 years old with a fever", "she is 2 years old and has fever",
        "he has had a fever for 2 weeks", "I took paracetamol 500mg an hour ago",
        "can I take 2 tablets of paracetamol and 2 of ibuprofen", "I have a cold and a runny nose",
        "how many tablets in a pack", "order Ibuprofen 200mg Tablets", "what is the overdose limit of paracetamol",
        "side effects of overdose",
    ])
    def test_ordinary_messages_are_not_escalated(self, text):
        assert red_flags.check(text) is None

    @pytest.mark.parametrize("text", [
        "no chest pain, just a fever", "I don't have trouble breathing", "no seizures, only a cold",
        "there is no stiff neck",
    ])
    def test_a_negated_symptom_is_not_escalated(self, text):
        assert red_flags.check(text) is None

    def test_one_negated_and_one_real_mention_still_escalates(self):
        assert red_flags.check("no chest pain earlier but now I can't breathe") == red_flags.EMERGENCY_REPLY

    def test_empty_text(self):
        assert red_flags.check("") is None


@pytest.mark.usefixtures("isolated_db")
class TestRoute:
    @pytest.fixture(autouse=True)
    def _no_model(self, monkeypatch):
        monkeypatch.setattr(config, "SMTP_CONFIGURED", False)

        def _boom(*args, **kwargs):
            raise AssertionError("an emergency must never reach the model")

        monkeypatch.setattr(main, "run_turn", _boom)

    def _chat(self, sid, text):
        res = client.post("/api/chat", json={"session_id": sid, "text": text})
        res.raise_for_status()
        return res.json()["reply"]

    def test_each_emergency_is_answered_without_the_model(self):
        assert self._chat("s1", "I took 10 paracetamol tablets") == red_flags.OVERDOSE_REPLY
        assert self._chat("s2", "my baby is 2 months old and has a fever") == red_flags.INFANT_FEVER_REPLY
        assert self._chat("s3", "diagnose my chest pain") == red_flags.EMERGENCY_REPLY

    def test_it_wins_over_a_pending_clarification_without_clearing_it(self):
        store.set_pending_clarification("s1", "fever")
        assert self._chat("s1", "my baby is 2 months old") != red_flags.INFANT_FEVER_REPLY  # no fever word: normal flow
        store.set_pending_clarification("s2", "fever")
        assert self._chat("s2", "it's for my 2 month old baby, the fever is high") == red_flags.INFANT_FEVER_REPLY
        assert store.get_pending_clarification("s2") == "fever"

    def test_it_leaves_an_order_awaiting_its_address_waiting(self):
        store.set_pending_order("s1", "fev-001", 1)
        assert self._chat("s1", "I can't breathe") == red_flags.EMERGENCY_REPLY
        assert store.get_pending_order("s1")["product_id"] == "fev-001"
        assert store.list_orders("demo_user") == []


class TestDoctorNote:
    """Real gap (Oct 17 probe): "I have fever 104 for 5 days" and "my fever has
    lasted a week" got the usual adult list with no mention of seeing a doctor."""

    @pytest.mark.parametrize("text", [
        "I have fever 104 for 5 days", "my fever has lasted a week", "fever of 39.8 C", "temperature 103.5 F",
        "fever since 4 days", "I've had a fever for two weeks", "fever for three days", "fever 104",
        "temperature is 40 degrees celsius", "my fever has been going for 5 days",
    ])
    def test_a_high_or_long_fever_gets_the_note(self, text):
        assert red_flags.doctor_note(text) == red_flags.DOCTOR_NOTE

    @pytest.mark.parametrize("text", [
        "I have a fever", "I have a fever for 2 days", "fever since yesterday", "I have a cold for 5 days",
        "temperature 99 F", "fever 38.5 C", "fever 101", "I took 500 mg of paracetamol for my fever",
        "fever for 1 day", "order fev-004", "my 3 year old has a fever", "I'm 40 and have a fever",
        "I'm 45 with a fever", "", "fever at 7pm",
    ])
    def test_an_ordinary_fever_does_not(self, text):
        assert red_flags.doctor_note(text) is None


@pytest.mark.usefixtures("isolated_db")
class TestDoctorNoteRoute:
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        from app import agent

        monkeypatch.setattr(config, "SMTP_CONFIGURED", False)
        monkeypatch.setattr(agent, "create_agent", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no model")))

    def _chat(self, sid, text):
        res = client.post("/api/chat", json={"session_id": sid, "text": text})
        res.raise_for_status()
        return res.json()["reply"]

    def test_the_note_is_added_to_the_recommendation_not_to_the_question(self):
        question = self._chat("s1", "I have fever 104 for 5 days")
        assert "child, or for an adult" in question and red_flags.DOCTOR_NOTE not in question
        recommendation = self._chat("s1", "for myself")
        assert "(fev-001)" in recommendation and recommendation.rstrip().endswith(red_flags.DOCTOR_NOTE)
        assert recommendation.count(red_flags.DOCTOR_NOTE) == 1

    def test_an_ordinary_fever_gets_no_note(self):
        self._chat("s1", "I have a fever")
        assert red_flags.DOCTOR_NOTE not in self._chat("s1", "for myself")

    def test_the_stored_session_matches_what_was_shown(self):
        self._chat("s1", "I have a fever for a week")
        shown = self._chat("s1", "for myself")
        assert store.get_session_messages("s1")[-1]["content"] == shown


class TestCurlyApostropheNegation:
    """Phones type ’ not '. "I don’t have chest pain" is a denial either way."""

    @pytest.mark.parametrize("text", [
        "I don’t have chest pain, just a cold", "it isn’t a seizure", "she doesn’t have trouble breathing",
    ])
    def test_a_negated_symptom_with_a_curly_apostrophe_is_not_escalated(self, text):
        assert red_flags.check(text) is None

    def test_a_real_mention_after_a_curly_negation_still_escalates(self):
        assert red_flags.check("I didn’t have chest pain earlier but now I can’t breathe") == red_flags.EMERGENCY_REPLY


class TestEverydayEmergencyPhrasings:
    """Found Nov 20 by trying 59 ordinary ways of describing an emergency: 25 were NOT
    recognised ("my chest hurts", "I think I'm having a heart attack", "she collapsed",
    "turning blue", "stroke", "won't stop bleeding", "thinking of ending my life"...). The
    bias must stay toward escalating."""

    @pytest.mark.parametrize("text", [
        "pain in my chest", "my chest hurts", "pain in chest radiating to arm", "squeezing pain across my chest",
        "I think I'm having a heart attack", "my father is having a heart attack", "I'm having a stroke", "stroke",
        "she collapsed", "he blacked out", "he is not responding", "he isn't breathing", "no pulse", "he is having a fit",
        "severe allergic reaction", "blood in my stool", "black stools", "blood in the vomit",
        "severe bleeding", "bleeding heavily", "won't stop bleeding", "I'm pregnant and bleeding",
        "severe abdominal pain", "severe headache worst of my life", "this is the worst headache I've ever had",
        "double vision and confusion", "weakness on one side of my body",
        "thinking of ending my life", "I want to take my own life", "I feel like I'd be better off dead",
        "turning blue", "baby not feeding and very sleepy", "my toddler is limp",
    ])
    def test_these_are_emergencies(self, text):
        assert red_flags.check(text) == red_flags.EMERGENCY_REPLY or red_flags.check(text) == red_flags.INFANT_FEVER_REPLY, text

    @pytest.mark.parametrize("text", [
        "child swallowed ibuprofen tablets", "my son ate some tablets", "the baby swallowed a pill", "my daughter drank the cough syrup",
    ])
    def test_a_child_who_swallowed_medicine_is_a_poisoning(self, text):
        assert red_flags.check(text) == red_flags.OVERDOSE_REPLY, text

    @pytest.mark.parametrize("text", [
        "I had a heart attack two years ago, can I take ibuprofen", "what is a stroke", "signs of a heart attack",
        "history of stroke in my family", "my father had a stroke last year", "what does a heart attack feel like",
        "no chest pain", "I don't have chest pain", "chest congestion", "my chest feels congested from the cold",
        "a cut that bled a little", "my period is heavy this month", "no bleeding", "she is not responding to my texts",
        "he is not responding to treatment", "my child ate breakfast", "my son took his medicine on time",
        "can a child take paracetamol", "my toddler likes the syrup", "worst day of my life", "I have the worst cold",
        "I took 2 paracetamol this morning", "headache and mild fever", "tight chest from coughing",
    ])
    def test_ordinary_messages_stay_ordinary(self, text):
        assert red_flags.check(text) is None, text


class TestEverydayOverdosePhrasings:
    """Found Nov 20: "took ten paracetamol", "I took a handful of tablets", "she swallowed her
    whole bottle of tablets", "I took 3000 mg of paracetamol at once", and a parent who "gave my
    baby double the dose" / "gave my child too much syrup" were not recognised as overdoses (only
    the digit count of 8 or more was)."""

    @pytest.mark.parametrize("text", [
        "took ten paracetamol", "I took twenty tablets", "took a dozen pills", "I took a handful of tablets",
        "she swallowed her whole bottle of tablets", "I took 3000 mg of paracetamol at once", "I took 5000 mg",
        "took 5 paracetamol 650 in one go", "he took 4 tablets all at once",
        "I accidentally gave my baby double the dose", "gave my child too much syrup", "I gave my 2 year old 3 times the dose",
        "I gave my 6 month old twice the dose",
    ])
    def test_these_are_overdoses(self, text):
        assert red_flags.check(text) == red_flags.OVERDOSE_REPLY, text

    @pytest.mark.parametrize("text", [
        "I took 2 paracetamol this morning", "took 1 tablet", "I take 2 tablets twice a day", "how many tablets can I take",
        "what is the max dose", "can I take 2 tablets", "I took 4 tablets over a whole day", "what happens if I take too much",
        "I took one tablet", "I took two tablets at once", "took 500mg", "I took 1000 mg", "I took 650mg at once",
        "I gave my baby the right dose", "gave my child the syrup as directed", "I gave my son one tablet",
        "I took 3 tablets in one go", "took six cold and flu tablets over two days", "I took one in the morning and one at night",
    ])
    def test_ordinary_dosing_stays_ordinary(self, text):
        assert red_flags.check(text) is None, text


class TestHindiAndTeluguEmergencies:
    """Found Nov 23: all 54 Hindi/Telugu emergency phrasings tried ("seene mein dard", "saans
    nahi aa rahi", "behosh ho gaya", "dil ka daura", "khoon ki ulti", "marna chahta hoon"...)
    were NOT recognised: the Hinglish layer knew symptom words only, so they went into the
    ordinary fever/cold flow. As elsewhere, these word lists are best-effort and need a native
    speaker's review."""

    @pytest.mark.parametrize("text", [
        "seene mein dard", "mere seene mein dard ho raha hai", "chhati mein dard", "chaati noppi", "chest mein dard",
        "saans nahi aa rahi", "saans lene mein takleef", "saans lene mein dikkat", "dam ghut raha hai", "oopiri aadatam ledu",
        "behosh ho gaya", "bebosh", "spruha ledu", "chakkar aakar gir gaya",
        "dil ka daura", "heart attack aa gaya", "gundepotu",
        "mirgi ka daura", "daura pad raha hai", "jhatke aa rahe hain",
        "khoon ki ulti", "ulti mein khoon", "raktham vanthulu",
        "bahut khoon beh raha hai", "khoon band nahi ho raha",
        "main marna chahta hoon", "mujhe jeena nahi hai", "aatmahatya", "nenu chanipovalani anukuntunnanu",
        "सीने में दर्द", "साँस लेने में तकलीफ", "बेहोश हो गया", "दिल का दौरा", "खून की उल्टी", "आत्महत्या",
        "ఛాతీ నొప్పి", "ఊపిరి ఆడటం లేదు", "గుండెపోటు",
    ])
    def test_these_are_emergencies(self, text):
        assert red_flags.check(text) == red_flags.EMERGENCY_REPLY, text

    @pytest.mark.parametrize("text", [
        "bahut zyada dawai kha li", "zyada goliyan kha li", "dawai ki bottle kha li", "bacche ne dawai kha li",
        "bachhe ne goli nigal li", "tablets ekkuva vesukunna", "ज्यादा दवा खा ली",
    ])
    def test_these_are_overdoses(self, text):
        assert red_flags.check(text) == red_flags.OVERDOSE_REPLY, text

    @pytest.mark.parametrize("text", [
        "chhati mein jakdan nahi hai", "seene mein dard nahi hai", "seene mein koi dard nahi", "dil ka daura nahi aaya",
        "khoon ki ulti nahi hui", "mujhe bukhar hai", "mera bachha ko khansi hai", "sar dard aur chakkar", "gale mein dard hai aur khansi",
        "sardi zukam hai", "naaku jvaram undi", "paracetamol chahiye", "ibuprofen ki kimat", "12 Dam Road, Hyderabad", "mere bete ko bukhar hai",
        "dawai kitni goli leni hai",
    ])
    def test_ordinary_hinglish_stays_ordinary(self, text):
        assert red_flags.check(text) is None, text

    def test_it_is_seen_even_when_chat_did_not_gloss_the_message(self, monkeypatch):
        # while an order waits for its address chat() skips the Hinglish gloss (an address is
        # stored exactly as typed); the emergency check glosses for itself
        from fastapi.testclient import TestClient
        monkeypatch.setattr(config, "SMTP_CONFIGURED", False)
        monkeypatch.setattr(main, "run_turn", lambda messages, **kw: ("from the model", messages))
        client = TestClient(main.app)
        client.post("/api/chat", json={"session_id": "hi1", "text": "order paracetamol 500mg"})
        assert store.get_pending_order("hi1") is not None
        reply = client.post("/api/chat", json={"session_id": "hi1", "text": "mere seene mein dard ho raha hai"}).json()["reply"]
        assert reply == red_flags.EMERGENCY_REPLY
        assert store.get_pending_order("hi1") is not None          # nothing was ordered or cleared
        assert store.list_orders("demo_user") == []


class TestLongFeverWithoutANumber:
    """Found Nov 24: "feverish for the past week", "fever for a few days", "fever all week" and
    "fever for days now" got no see-a-doctor note, while "fever for 3 days" did."""

    @pytest.mark.parametrize("text", [
        "feverish for the past week", "fever for a few days", "fever all week", "I've had a fever for days now",
        "fever since last week", "I have had fever for several days", "fever for the last week",
        "fever over the past few days", "fever for over a week", "this whole week I have had a fever",
    ])
    def test_the_doctor_note_is_added(self, text):
        assert red_flags.doctor_note(text) == red_flags.DOCTOR_NOTE, text

    @pytest.mark.parametrize("text", [
        "fever for a couple of days", "fever since this morning", "fever started yesterday", "I have a fever",
        "fever for two days", "fever last night", "fever for a day", "fever for 2 days",
        "a few days ago I felt fine but now I have a fever", "fever for a few hours", "fever 100",
    ])
    def test_a_short_or_unrelated_duration_gets_none(self, text):
        assert red_flags.doctor_note(text) is None, text


class TestMoreWaysToSayAnInfantHasAFever:
    """Found Nov 25: "baby is 10 days old and feels warm", "my infant has a fever, she is 7 weeks",
    "my baby (6 weeks) has a high temperature" and "baby under 3 months with fever" were not
    recognised as a fever in a baby under 3 months (the age had to say "old", and "under 3
    months" counted 3 as not under 3)."""

    @pytest.mark.parametrize("text", [
        "baby is 10 days old and feels warm", "my infant has a fever, she is 7 weeks", "my baby (6 weeks) has a high temperature",
        "baby under 3 months with fever", "baby less than 2 months feels hot", "my newborn is hot to the touch",
        "my baby is 6 weeks old and has a fever", "my 11 week old has a fever",
    ])
    def test_these_get_the_see_a_doctor_now_reply(self, text):
        assert red_flags.check(text) == red_flags.INFANT_FEVER_REPLY, text

    @pytest.mark.parametrize("text", [
        "my 4 month old has a fever", "my 1 year old has a fever", "I have had a fever for 3 weeks", "my baby has had a fever for 5 days",
        "my baby has a fever since 2 days", "my son has had fever for the past 2 weeks", "my son is 5 and has fever for 3 days",
        "my baby is fine but I have a fever, 3 days now", "my daughter is 10 weeks into her course and I have fever",
        "the soup feels hot", "my baby is fine, I have a fever for 3 days",
    ])
    def test_durations_older_children_and_unrelated_words_do_not(self, text):
        assert red_flags.check(text) != red_flags.INFANT_FEVER_REPLY, text


class TestDangerSignsThatGoWithAFever:
    """Found Nov 25: 17 of 20 danger signs that go with a fever were not recognised — a rash that
    doesn't fade, purple spots, fever with confusion / hard to wake, a bulging soft spot, no wet
    nappies, "can't keep fluids down", drooling with trouble swallowing, light hurting the eyes,
    "can't touch chin to chest", a temperature of 41. The product note itself already lists a
    rash that doesn't fade, confusion and a stiff neck as reasons for emergency care."""

    @pytest.mark.parametrize("text", [
        "fever and a rash that doesn't fade", "purple spots and fever", "fever with blotchy skin", "fever and confusion",
        "very drowsy and hard to wake with a fever", "my child has fever and is not drinking", "fever and not peeing at all",
        "can't keep any fluids down", "my baby has a bulging soft spot", "my child is having trouble waking up",
        "fever and severe headache with light hurting eyes", "fever and difficulty swallowing drooling", "no wet nappies for 8 hours",
        "fever with a sore neck and can't touch chin to chest", "he won't wake up", "I can't wake him",
        "my dad has fever and is hard to wake", "baby has not peed for 12 hours",
    ])
    def test_these_are_emergencies(self, text):
        assert red_flags.check(text) in (red_flags.EMERGENCY_REPLY, red_flags.INFANT_FEVER_REPLY), text

    @pytest.mark.parametrize("text", [
        "fever of 41", "temperature is 41", "fever 41.5", "temperature 105.9", "fever 106 F", "fever 41 C", "temp 42",
    ])
    def test_a_temperature_of_41_c_or_105_8_f_is_an_emergency_not_just_a_note(self, text):
        assert red_flags.check(text) == red_flags.EMERGENCY_REPLY, text

    @pytest.mark.parametrize("text", [
        "41 years old with a fever", "I am 41 and have a fever", "fever and 41 mg", "my 41 year old father has a fever",
        "fever 41 days", "fever 40", "fever 39.5", "temperature 104", "fever of 103",
        "I am confused about the dosage", "I'm not drinking alcohol these days", "my child is not drinking enough milk but no fever",
        "the rash is itchy", "a rash on my arm", "I feel drowsy after the cold tablet", "I'm sleepy", "light roast coffee",
        "fever for 2 days", "headache and fever", "my neck is sore from sleeping", "I haven't peed since the morning but I'm fine",
        "hard to wake up in the morning", "I can't wake up early",
    ])
    def test_ordinary_messages_stay_ordinary(self, text):
        assert red_flags.check(text) in (None,), text
