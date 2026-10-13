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


# ------------------------------------------------------------------ /health
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_loaded"] is True


# ------------------------------------------------------------------ /labels
def test_labels_returns_all_12(client):
    r = client.get("/labels")
    assert r.status_code == 200
    assert r.json()["labels"] == LABELS


# ------------------------------------------------------------------ /classify schema
def test_classify_returns_correct_schema(client):
    r = client.post("/classify", json={
        "product_category": "refrigerator",
        "complaint_text": "Not cooling at all",
    })
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"fault_category", "confidence", "escalate", "reason"}
    assert body["fault_category"] in LABELS
    assert isinstance(body["confidence"], float)
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["escalate"], bool)
    assert body["reason"].endswith(".")


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
    r = client.post("/classify", json={
        "product_category": product,
        "complaint_text": text,
    })
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
    r = client.post("/classify", json={
        "product_category": "refrigerator",
        "complaint_text": "Delivered yesterday, the doors are not level and one does not close properly",
        "product_age_days": 1,
    })
    assert r.status_code == 200
    assert r.json()["fault_category"] == "installation_or_setup"


# ------------------------------------------------------------------ validation
def test_invalid_product_returns_422(client):
    r = client.post("/classify", json={
        "product_category": "spaceship",
        "complaint_text": "It broke",
    })
    assert r.status_code == 422


def test_missing_fields_returns_422(client):
    r = client.post("/classify", json={"complaint_text": "not cooling"})
    assert r.status_code == 422


def test_very_long_text_is_accepted(client):
    r = client.post("/classify", json={
        "product_category": "refrigerator",
        "complaint_text": "x" * 5000,   # gets capped to 3000 internally
    })
    assert r.status_code == 200