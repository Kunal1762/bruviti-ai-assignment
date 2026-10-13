"""Tests for the FastAPI /classify endpoint.

Run: python -m pytest -v -s tests/test_api.py
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


# ------------------------------------------------------------------ /health
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ------------------------------------------------------------------ /labels
def test_labels_returns_all_12(client):
    r = client.get("/labels")
    assert r.status_code == 200
    assert r.json()["labels"] == LABELS
    assert len(r.json()["labels"]) == 12


# ------------------------------------------------------------------ / (UI)
def test_index_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


# ------------------------------------------------------------------ /classify schema
def test_classify_returns_correct_schema(client):
    r = client.post("/classify", json={
        "ticket_id": "T-1",
        "product_category": "refrigerator",
        "complaint_text": "Not cooling at all",
    })
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"ticket_id", "fault_category", "confidence", "escalate", "reason"}
    assert body["ticket_id"] == "T-1"
    assert body["fault_category"] in LABELS
    assert isinstance(body["confidence"], float)
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["escalate"], bool)
    assert body["reason"].endswith(".")


def test_ticket_id_is_echoed_unchanged(client):
    for tid in ("T-1001", "abc 123", "000123"):
        r = _post(client, "refrigerator", "not cooling at all", tid=tid)
        assert r.json()["ticket_id"] == tid


def test_numeric_ticket_id_is_accepted(client):
    r = client.post("/classify", json={
        "ticket_id": 4521,
        "product_category": "refrigerator",
        "complaint_text": "not cooling at all",
    })
    assert r.status_code == 200
    assert r.json()["ticket_id"] == "4521"


# ------------------------------------------------------------------ /classify logic
@pytest.mark.parametrize("product,text,expected_cat,must_escalate", [
    ("washing_machine",
     "It sparked and tripped the MCB",
     "electrical_or_control_board", True),
    ("refrigerator",
     "Not cooling at all, compressor runs but inside is warm",
     "not_cooling", False),
    ("washing_machine",
     "Water leaking from the bottom of the door every cycle",
     "water_leak", False),
    ("dishwasher",
     "display shows E4",
     "control_panel_or_display", False),
])
def test_classify_key_cases(client, product, text, expected_cat, must_escalate):
    r = _post(client, product, text)
    assert r.status_code == 200
    body = r.json()
    print(
        f"\n  product   : {product}"
        f"\n  message   : {text}"
        f"\n  classified: {body['fault_category']} "
        f"(conf={body['confidence']:.2f}, escalate={body['escalate']})"
        f"\n  reason    : {body['reason'][:100]}"
    )
    assert body["fault_category"] == expected_cat
    if must_escalate:
        assert body["escalate"] is True


def test_classify_c3_new_unit(client):
    r = _post(client, "refrigerator",
              "Delivered yesterday, the doors are not level and one does not close properly",
              age=1)
    assert r.status_code == 200
    assert r.json()["fault_category"] == "installation_or_setup"


# ------------------------------------------------------------------ safety rules (graded)
@pytest.mark.parametrize("product,text", [
    ("oven_or_cooktop", "There was a burning smell and a spark from the back when I switched it on"),
    ("microwave", "smoke coming out of the vents"),
    ("refrigerator", "I got an electric shock touching the door handle"),
    ("air_conditioner", "the supply keeps tripping whenever it starts"),
    ("washing_machine", "small fire near the plug"),
])
def test_c1_safety_words_always_escalate(client, product, text):
    body = _post(client, product, text).json()
    assert body["fault_category"] == "electrical_or_control_board"
    assert body["escalate"] is True


def test_c1_safety_word_at_end_of_long_text_still_escalates(client):
    text = ("the machine is a bit noisy and I bought it last year. " * 100) + "then it sparked"
    assert len(text) > 3000
    body = _post(client, "washing_machine", text).json()
    assert body["escalate"] is True
    assert body["fault_category"] == "electrical_or_control_board"


@pytest.mark.parametrize("product", ["oven_or_cooktop", "water_heater"])
def test_c2_gas_smell_escalates(client, product):
    body = _post(client, product, "I can smell gas near it").json()
    assert body["escalate"] is True


def test_c6_non_complaint_is_other_with_low_confidence(client):
    body = _post(client, "microwave", "thank you for the quick repair, great service").json()
    assert body["fault_category"] == "other"
    assert body["confidence"] < 0.3
    assert body["escalate"] is False


# ------------------------------------------------------------------ lenient input handling
def test_unknown_product_is_still_classified(client):
    r = _post(client, "spaceship", "It broke")
    assert r.status_code == 200
    assert r.json()["fault_category"] in LABELS


def test_unknown_product_with_safety_word_still_escalates(client):
    body = _post(client, "toaster", "there was smoke and sparks").json()
    assert body["escalate"] is True


@pytest.mark.parametrize("product", ["Washing Machine", "WASHING_MACHINE", " washing-machine "])
def test_product_name_is_normalised(client, product):
    r = _post(client, product, "drum never turns, display shows E21")
    assert r.status_code == 200
    assert r.json()["fault_category"] in LABELS


def test_missing_fields_still_return_valid_response(client):
    r = client.post("/classify", json={"complaint_text": "not cooling"})
    assert r.status_code == 200
    assert r.json()["fault_category"] in LABELS


def test_empty_body_does_not_error(client):
    r = client.post("/classify", json={})
    assert r.status_code == 200
    assert r.json()["fault_category"] in LABELS


def test_null_fields_do_not_error(client):
    r = client.post("/classify", json={
        "ticket_id": None, "product_category": None,
        "complaint_text": None, "product_age_days": None,
    })
    assert r.status_code == 200
    assert r.json()["fault_category"] in LABELS


@pytest.mark.parametrize("age", ["30", "", None, 12.5, "abc", -5, 0, 100000])
def test_age_tolerates_odd_values(client, age):
    r = _post(client, "refrigerator", "doors not level, delivered yesterday", age=age)
    assert r.status_code == 200
    assert r.json()["fault_category"] in LABELS


@pytest.mark.parametrize("text", ["", "a", "ok", "    "])
def test_empty_and_short_text_do_not_error(client, text):
    r = _post(client, "microwave", text)
    assert r.status_code == 200
    assert r.json()["fault_category"] in LABELS
    assert 0.0 <= r.json()["confidence"] <= 1.0


def test_very_long_text_is_accepted(client):
    r = _post(client, "refrigerator", "x" * 5000)
    assert r.status_code == 200


def test_non_english_and_odd_characters_do_not_error(client):
    for text in ("मशीन चालू नहीं हो रही", "fridge nahi chal raha 😡😡", "\u0000\u200b weird \n\t text"):
        r = _post(client, "refrigerator", text)
        assert r.status_code == 200
        assert r.json()["fault_category"] in LABELS