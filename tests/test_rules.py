"""Offline tests for the C1-C6 rule engine. Run: python -m pytest -q tests/test_rules.py

Safety tests are recall-oriented: every C1/C2 phrasing below must escalate no matter
what the model predicted (the fake probabilities say "other" or an unrelated class).
"""
from __future__ import annotations

import math
import socket
import time

import pytest

from src.classifier import train_model
from src.data import LABELS, generate_dataset
from src.rules import apply_rules, classify_ticket


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def guard(*args, **kwargs):
        raise RuntimeError("network access attempted during a test")
    monkeypatch.setattr(socket.socket, "connect", guard)


def make_probs(top: dict) -> dict:
    rest = [l for l in LABELS if l not in top]
    share = (1 - sum(top.values())) / len(rest)
    return {l: top.get(l, share) for l in LABELS}


P_OTHER = make_probs({"other": 0.9})
P_NOISE = make_probs({"noise_or_vibration": 0.8})
_FILL = "I have had this fridge for years and it has been good. "

C1_POSITIVE = [
    ("oven_or_cooktop", "There was a burning smell and a spark from the back when I switched it on, it is off now"),
    ("washing_machine", "SPARKS coming out of the plug"),
    ("refrigerator", "smoke came out of the back panel"),
    ("microwave", "i saw sparkk near the socket"),
    ("dishwasher", "sparking noise near the power cord, unplugged it immediately"),
    ("washing_machine", "burnt smell from the control board"),
    ("air_conditioner", "it smells like something is burning"),
    ("microwave", "smell of burning plastic"),
    ("water_heater", "got an electric shock when I touched the door handle"),
    ("washing_machine", "tripped the main breaker when I plugged it in"),
    ("dishwasher", "it trips the MCB every time I switch it on"),
    ("refrigerator", "blows the fuse each time"),
    ("washing_machine", "the whole house power trips as soon as the washer starts"),
    ("water_heater", "earth leakage trips as soon as the geyser is switched on"),
    ("oven_or_cooktop", "the thing caught fire"),
    ("oven_or_cooktop", "there is a fire inside the oven"),
    ("oven_or_cooktop", "small flames coming out of the vent"),
    ("washing_machine", "chingari nikli aur jalne ki smell aa rahi hai"),
    ("refrigerator", "dhuan nikal raha hai"),
    ("dishwasher", "salio una chispa y huele a quemado"),
    ("microwave", "ça sent le brûlé et il y a eu une étincelle"),
    ("water_heater", "mild shock from the casing"),
    ("water_heater", "I was electrocuted slightly"),
    ("washing_machine", "current laga jab haath lagaya, MCB bhi gir gaya"),
    ("microwave", "the unit sparks when plugged in"),
    ("oven_or_cooktop", "I noticed a burning odour coming from the oven"),
    ("air_conditioner", "keeps tripping the electricity supply"),
    ("dishwasher", "it tripped the power"),
    ("washing_machine", "sparkss everywhere"),
    ("refrigerator", "SMOKEEEE!!"),
    ("microwave", "spaark inside"),
    ("refrigerator", _FILL * 15 + "Yesterday there was a spark near the plug. " + _FILL * 15),
]

C1_NEGATIVE = [
    ("oven_or_cooktop", "no burning smell or sparks, just not heating"),
    ("oven_or_cooktop", "no sparks or smoke, but the oven is not heating"),
    ("dishwasher", "I did not see any sparks or smoke"),
    ("dishwasher", "the dishes are not sparkling clean"),
    ("washing_machine", "Where can I buy spare parts for my washer?"),
    ("refrigerator", "I was shocked at the price of the repair"),
    ("washing_machine", "shock absorbers are worn out and it is noisy"),
    ("oven_or_cooktop", "The flame is yellow and very weak"),
    ("refrigerator", "power cut yesterday and it never came back on"),
    ("washing_machine", "I keep tripping over the power cable"),
    ("washing_machine", "thanks, the engineer fixed the leak yesterday"),
    ("refrigerator", "ice cream melted and the freezer smells a bit"),
]

C2_POSITIVE = [
    ("oven_or_cooktop", "I can smell gas near the cooker"),
    ("water_heater", "gas smell from the geyser"),
    ("oven_or_cooktop", "smells of LPG when I switch it on"),
    ("oven_or_cooktop", "there's a gas leak"),
    ("water_heater", "gas is leaking from the pipe"),
    ("oven_or_cooktop", "leaking gas from the hose"),
    ("oven_or_cooktop", "gas ki smell aa rahi hai"),
    ("oven_or_cooktop", "olor a gas en la cocina"),
    ("water_heater", "odeur de gaz près du chauffe-eau"),
    ("oven_or_cooktop", "rotten egg smell from the hob"),
    ("oven_or_cooktop", "gass smell from the burner"),
    ("water_heater", "SMELL OF GAS"),
]


@pytest.mark.parametrize("product,text", C1_POSITIVE)
def test_c1_escalates_and_forces_electrical(product, text):
    d = apply_rules(P_OTHER, product, text)
    assert d.escalate is True
    assert d.fault_category == "electrical_or_control_board"
    assert "C1" in d.rules and d.confidence >= 0.90


@pytest.mark.parametrize("product,text", C1_NEGATIVE)
def test_c1_does_not_fire_on_negated_or_unrelated_text(product, text):
    d = apply_rules(P_NOISE, product, text)
    assert "C1" not in d.rules
    assert d.escalate is False


@pytest.mark.parametrize("product,text", C2_POSITIVE)
def test_c2_escalates_but_keeps_the_category(product, text):
    probs = make_probs({"not_heating": 0.7})
    d = apply_rules(probs, product, text)
    assert d.escalate is True and "C2" in d.rules
    if "C1" not in d.rules:
        assert d.fault_category == "not_heating"


def test_c2_category_comes_from_the_rest_of_the_text():
    d = apply_rules(make_probs({"not_heating": 0.7}), "oven_or_cooktop",
                    "burner won't light and I can smell gas")
    assert (d.fault_category, d.escalate) == ("not_heating", True)


@pytest.mark.parametrize("product,text", [
    ("oven_or_cooktop", "gas cooker flame is too low"),
    ("water_heater", "gas geyser leaking water onto the floor"),
    ("oven_or_cooktop", "no smell of gas"),
    ("microwave", "I can smell gas"),          # C2 applies to oven/cooktop and water heater only
    ("refrigerator", "smell of gas in the kitchen"),
])
def test_c2_does_not_fire(product, text):
    d = apply_rules(P_NOISE, product, text)
    assert "C2" not in d.rules and d.escalate is False


def test_c1_overrides_the_model_and_c3():
    d = apply_rules(make_probs({"no_power": 0.9}), "washing_machine",
                    "installed yesterday and there was a spark from the plug", 5)
    assert d.fault_category == "electrical_or_control_board"
    assert "C1" in d.rules and "C3" not in d.rules and d.escalate


# -------------------------------------------------------------------- C3
DOOR_TEXT = "delivered yesterday, the doors are not level and one does not close properly"


def test_c3_fit_complaint_within_30_days():
    probs = make_probs({"door_seal_or_latch": 0.6})
    for age in (0, 1, 30):
        d = apply_rules(probs, "refrigerator", DOOR_TEXT, age)
        assert d.fault_category == "installation_or_setup" and "C3" in d.rules
        assert d.escalate is False


def test_c3_does_not_apply_after_30_days_or_without_age():
    probs = make_probs({"door_seal_or_latch": 0.6})
    for age in (31, 400, None):
        d = apply_rules(probs, "refrigerator", DOOR_TEXT, age)
        assert d.fault_category == "door_seal_or_latch" and "C3" not in d.rules


def test_c3_ignores_invalid_ages():
    probs = make_probs({"door_seal_or_latch": 0.6})
    for age in (-5, float("nan"), True, "10"):
        assert "C3" not in apply_rules(probs, "refrigerator", DOOR_TEXT, age).rules


def test_c3_does_not_turn_a_sales_enquiry_into_installation():
    d = apply_rules(make_probs({"other": 0.8}), "air_conditioner",
                    "what are your installation charges for a new split AC", 5)
    assert d.fault_category == "other" and "C3" not in d.rules


# -------------------------------------------------------------------- C4
@pytest.mark.parametrize("text", ["display shows E21", "Error code F4", "E-05",
                                  "what does E33 mean on my washer?", "it just says Err 3"])
def test_c4_error_code_alone_is_control_panel(text):
    d = apply_rules(make_probs({"motor_or_drive": 0.4}), "washing_machine", text)
    assert d.fault_category == "control_panel_or_display" and "C4" in d.rules
    assert d.confidence == pytest.approx(0.65) and d.escalate is False


def test_c4_does_not_apply_when_a_symptom_is_described():
    d = apply_rules(make_probs({"motor_or_drive": 0.6}), "washing_machine",
                    "drum never turns and the display shows E21")
    assert d.fault_category == "motor_or_drive" and "C4" not in d.rules


# ---------------------------------------------------------------- C5 / C6
def test_c5_tie_picks_the_earlier_label_and_lowers_confidence():
    d = apply_rules(make_probs({"drainage_or_blockage": 0.31, "water_leak": 0.29}),
                    "washing_machine", "water everywhere and it will not empty")
    assert d.fault_category == "water_leak" and "C5" in d.rules
    assert d.confidence < 0.29


def test_c5_keeps_the_top_label_when_it_is_already_earlier():
    d = apply_rules(make_probs({"water_leak": 0.31, "drainage_or_blockage": 0.29}),
                    "washing_machine", "water everywhere")
    assert d.fault_category == "water_leak" and "C5" in d.rules
    assert d.confidence < 0.31


def test_c5_does_not_fire_on_a_clear_winner():
    d = apply_rules(make_probs({"water_leak": 0.6, "drainage_or_blockage": 0.2}),
                    "washing_machine", "puddle under the washer")
    assert d.fault_category == "water_leak" and "C5" not in d.rules
    assert d.confidence == pytest.approx(0.6)


def test_c6_other_has_confidence_below_0_3():
    d = apply_rules(P_OTHER, "microwave", "Thanks a lot for the quick repair")
    assert d.fault_category == "other" and "C6" in d.rules
    assert d.confidence < 0.3 and d.escalate is False


# ------------------------------------------------- full pipeline (real model)
@pytest.fixture(scope="module")
def model():
    return train_model(generate_dataset())


def test_brief_example_2_sparks_and_burning_smell(model):
    out = classify_ticket(model, "oven_or_cooktop",
                          "There was a burning smell and a spark from the back when I "
                          "switched it on, it is off now", 95)
    assert out["fault_category"] == "electrical_or_control_board"
    assert out["escalate"] is True


def test_brief_example_3_fit_complaint_under_30_days(model):
    out = classify_ticket(model, "refrigerator", DOOR_TEXT, 1)
    assert out["fault_category"] == "installation_or_setup"
    assert out["escalate"] is False


def test_brief_example_1_is_valid_and_not_escalated(model):
    out = classify_ticket(model, "washing_machine",
                          "machine fills with water then just sits there, drum never "
                          "turns, display shows E21", 410)
    assert out["fault_category"] in LABELS and out["escalate"] is False


@pytest.mark.parametrize("text", ["", "dead!", "x" * 2000, "मशीन चालू नहीं हो रही", "💥 it exploded 💥"])
def test_response_schema_is_always_valid(model, text):
    out = classify_ticket(model, "refrigerator", text, None)
    assert set(out) == {"fault_category", "confidence", "escalate", "reason"}
    assert out["fault_category"] in LABELS
    assert isinstance(out["confidence"], float)
    assert math.isfinite(out["confidence"]) and 0.0 <= out["confidence"] <= 1.0
    assert isinstance(out["escalate"], bool)
    assert isinstance(out["reason"], str) and out["reason"].endswith(".")
    assert ". " not in out["reason"]  # one sentence


def test_rules_are_fast_and_deterministic():
    text = " ".join(f"wordsmith{i}x" for i in range(200))[:2000]
    t0 = time.time()
    a = apply_rules(P_NOISE, "refrigerator", text)
    assert time.time() - t0 < 1.0
    assert a == apply_rules(P_NOISE, "refrigerator", text)