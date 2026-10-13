
"""API tests: contract, graded business rules, lenient input.

Run: python -m pytest -v tests/test_api.py
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.data_diverse import LABELS


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _post(client, product, text, age=None, tid="T-TEST"):
    return client.post("/classify", json={
        "ticket_id": tid,
        "product_category": product,
        "complaint_text": text,
        "product_age_days": age,
    })


# ---------------------------------------------------------------- contract
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_index_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_classify_response_matches_contract(client):
    body = _post(client, "refrigerator", "Not cooling at all", tid="abc 123").json()
    assert set(body) == {"ticket_id", "fault_category", "confidence", "escalate", "reason"}
    assert body["ticket_id"] == "abc 123"
    assert body["fault_category"] in LABELS
    assert isinstance(body["confidence"], float) and 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["escalate"], bool)
    assert isinstance(body["reason"], str) and body["reason"]


@pytest.mark.parametrize("product,text,age,category,escalate", [
    ("washing_machine",
     "machine fills with water then just sits there, drum never turns, display shows E21",
     410, "motor_or_drive", False),
    ("oven_or_cooktop",
     "There was a burning smell and a spark from the back when I switched it on, it is off now",
     95, "electrical_or_control_board", True),
    ("refrigerator",
     "delivered yesterday, the doors are not level and one does not close properly",
     1, "installation_or_setup", False),
])
def test_brief_examples(client, product, text, age, category, escalate):
    body = _post(client, product, text, age).json()
    assert body["fault_category"] == category
    assert body["escalate"] is escalate


# ------------------------------------------------- business rules (graded)
@pytest.mark.parametrize("product,text", [
    ("washing_machine", "It sparked and tripped the MCB"),
    ("microwave", "smoke coming out of the vents"),
    ("refrigerator", "I got an electric shock touching the door handle"),
    ("air_conditioner", "the supply keeps tripping whenever it starts"),
    ("washing_machine", "small fire near the plug"),
])
def test_c1_safety_words_escalate(client, product, text):
    body = _post(client, product, text).json()
    assert body["fault_category"] == "electrical_or_control_board"
    assert body["escalate"] is True


def test_c1_safety_word_at_end_of_long_text_still_escalates(client):
    text = ("the machine is a bit noisy and I bought it last year. " * 100) + "then it sparked"
    body = _post(client, "washing_machine", text).json()
    assert body["escalate"] is True


@pytest.mark.parametrize("product", ["oven_or_cooktop", "water_heater"])
def test_c2_gas_smell_escalates(client, product):
    assert _post(client, product, "I can smell gas near it").json()["escalate"] is True


def test_c4_bare_error_code_is_control_panel(client):
    body = _post(client, "dishwasher", "display shows E4").json()
    assert body["fault_category"] == "control_panel_or_display"


def test_c6_non_complaint_is_other_with_low_confidence(client):
    body = _post(client, "microwave", "thank you for the quick repair, great service").json()
    assert body["fault_category"] == "other"
    assert body["confidence"] < 0.3
    assert body["escalate"] is False


# ------------------------------------------------------------ lenient input
def test_unknown_product_is_still_classified(client):
    r = _post(client, "spaceship", "It broke")
    assert r.status_code == 200
    assert r.json()["fault_category"] in LABELS


def test_missing_or_null_fields_do_not_error(client):
    for payload in ({}, {"ticket_id": None, "product_category": None,
                         "complaint_text": None, "product_age_days": None}):
        r = client.post("/classify", json=payload)
        assert r.status_code == 200
        assert r.json()["fault_category"] in LABELS