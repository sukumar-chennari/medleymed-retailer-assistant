"""The emergency detector must not cry wolf.

Found Nov 26: after three days of widening red_flags (emergency, overdose, infant fever, Hindi/Telugu),
about 150 ordinary shop messages were run through it and 16 were escalated to "get emergency care now" —
addresses with an emergency word in them ("7 Jaundice Avenue"), a pet, a past event, a general question,
an idiom, "a handful of almonds", "3000 mg of vitamin C", "trouble speaking in public", "my neck is stiff
from the pillow". They are kept here as a permanent corpus: anything that makes one of them escalate again
fails this test. (The reverse — real emergencies that must escalate — is in test_red_flags.py.)

Each message goes through the same normalisation chat() applies before the check."""

import pytest

from app import age_hints, hinglish, red_flags, spellfix

ORDINARY = """I have a fever and a runny nose
my head hurts and I feel tired
I have a cough since yesterday
sore throat and mild fever
my son has a cold
my daughter has a cough and runny nose
what is the dosage for paracetamol
how many tablets can I take per day
can I take ibuprofen on an empty stomach
is cetirizine safe during pregnancy
I'm allergic to penicillin
I have a stuffy nose and sneezing
my baby is 8 months old and has a cold
my 3 year old has a fever of 101
I have a fever of 102 for two days
my temperature is 100.4
my temperature is 38.5
I feel dizzy when I stand up
I feel nauseous after eating
I have a mild headache
my chest feels congested
I have a tickle in my throat
body ache and chills
I have been coughing at night
my nose is blocked
I feel weak and tired with fever
my throat hurts when I swallow
I have a dry cough
I have a wet cough with phlegm
my eyes are itchy and watery
I'm sneezing a lot
can I take paracetamol and cetirizine together
can I give paracetamol to my 4 year old
what is the max dose of ibuprofen
I took 2 tablets of paracetamol an hour ago
I took paracetamol this morning and again now
I forgot to take my tablet
I missed a dose
is it okay to drink alcohol with paracetamol
can I drive after taking the cold medicine
how long does paracetamol take to work
what are the side effects of ibuprofen
does cetirizine make you drowsy
I feel a bit drowsy after the tablet
my skin is itchy
I have a rash on my arm that's itchy
my rash is getting better
I have dry skin
I have a headache and light is a bit annoying
the tablet is white and round
please show me all products
order paracetamol 500mg
cancel my order
where is my order
my address is 12 Baker Street
Flat 41, Heart Lane, Leeds
41 Blood Bank Road, Pune
12 Stroke Road, Kolkata
Collapsed Bridge Road 5
22 Stroke Lane
Pulse Street 7, Chennai
10 Hives Close
7 Jaundice Avenue
Wheeler Street 3
Faint Road 9, Delhi
no I'm fine thanks
yes please
thank you so much
good night
hello
I'm 41 and I have a cold
I am 42 years old with a headache
my father is 65 and has a cough
my wife is pregnant and has a cold
I'm not pregnant
I don't have chest pain
no chest pain just a cough
my breathing is fine
I can breathe normally
no difficulty breathing
I haven't fainted
no seizures
I had chest pain years ago
what causes chest pain
what is jaundice
is yellow tablet safe
my cat has a fever
my dog is wheezing
the baby is sleeping
the baby is sleepy after the feed
my baby is 4 months and has a runny nose
my baby is feeding well
my baby has a fever of 100.5 and is 5 months old
my newborn niece is cute
a newborn baby is coming next month
my sister gave birth last week
I took ten minutes to walk here
I took twenty five minutes
I had a dozen eggs
I ate a handful of almonds
I took a handful of vitamins
I took a whole day off
I took five tablets of vitamin c
I took 5000 steps today
I took 4000 IU of vitamin D
I took 3000 mg of vitamin C at once
bleeding gums when I brush
a small cut that bled a little
nosebleed stopped quickly
my period is heavy
I'm not drinking enough water
my child is not eating much
my son is not drinking his milk today
hard to wake up on Mondays
I can't wake up without coffee
I feel faint sometimes when I skip meals
I can't pass the test
I can't speak to him
trouble speaking in public
my speech was fine
the rash on my arm doesn't itch
a rash that goes away on its own
my neck is stiff from the pillow
my neck is stiff after the gym
stiff neck from sleeping badly
numb fingers from typing
my arm hurts from tennis
jaw pain from a tooth
sweating a lot in summer
night sweats
my heart rate is 80
heart healthy diet
I want to die laughing
this exam will kill me
I could kill for a coffee
a fit of laughter
I'm fit and healthy
the fit of this shirt
stroke of luck
breast stroke swimming
my bed is bad for my back
blue pen
blue tablets
the pill is blue
my lips are dry
my lips are chapped
swollen ankles at night
swelling in my knee
my face is red from the sun
sunburn with blisters""".split("\n")


def _normalised(text):
    return age_hints.gloss(hinglish.gloss(spellfix.correct(text)))


def test_the_corpus_is_the_size_it_was_when_it_was_written():
    assert len(ORDINARY) == 150 and len(set(ORDINARY)) == 150


@pytest.mark.parametrize("text", ORDINARY)
def test_an_ordinary_message_is_not_escalated(text):
    assert red_flags.check(_normalised(text)) is None, text
