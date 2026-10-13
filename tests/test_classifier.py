"""Offline tests for the data generator and the fault classifier.

Run from the repo root:   python -m pytest -q
See the held-out report:  python -m pytest -q -s -k heldout

Nothing here touches the network (a socket guard enforces it) or downloads data
or models. The model is deterministic, so results are stable between runs. These
tests deliberately do NOT assert that every synthetic example is classified
correctly; quality checks use loose floors well above chance (1/12 = 0.083).
"""
from __future__ import annotations

import math
import re
import socket
import time

import pytest
from sklearn.metrics import classification_report, f1_score

from src.classifier import evaluate, predict_fault, predict_proba_all, train_model
from src.data import LABELS, PRODUCTS, generate_dataset, load_csv, write_csv

EXPECTED_LABELS = [
    "no_power", "not_heating", "not_cooling", "water_leak", "drainage_or_blockage",
    "noise_or_vibration", "door_seal_or_latch", "control_panel_or_display",
    "electrical_or_control_board", "motor_or_drive", "installation_or_setup", "other",
]
EXPECTED_PRODUCTS = [
    "washing_machine", "refrigerator", "dishwasher", "oven_or_cooktop",
    "air_conditioner", "water_heater", "microwave",
]

# Hand-written complaints that are NOT produced by the generator.
UNSEEN = [
    ("refrigerator", "The inside of my fridge is warm and the milk went sour overnight", "not_cooling"),
    ("washing_machine", "Water is left sitting in the drum after every wash and will not go down the drain", "drainage_or_blockage"),
    ("air_conditioner", "There is a pool of water under the indoor unit and it drips onto my sofa", "water_leak"),
    ("microwave", "The turntable plate does not move when I run it", "motor_or_drive"),
    ("dishwasher", "The door will not click shut so the machine refuses to start", "door_seal_or_latch"),
    ("oven_or_cooktop", "Smell of burning and I saw a spark behind the oven", "electrical_or_control_board"),
    ("water_heater", "No hot water at all, only cold comes out of the tap", "not_heating"),
    ("washing_machine", "Terrible banging and shaking when it spins, it travels across the floor", "noise_or_vibration"),
    ("refrigerator", "Display on the door is blank and none of the buttons respond", "control_panel_or_display"),
    ("dishwasher", "Newly delivered and it rocks from side to side, needs levelling", "installation_or_setup"),
    ("microwave", "Thanks a lot for the quick repair, everything works perfectly", "other"),
    ("oven_or_cooktop", "The unit is completely dead, no lights and nothing happens when I switch on", "no_power"),
]


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def guard(*args, **kwargs):
        raise RuntimeError("network access attempted during a test")
    monkeypatch.setattr(socket.socket, "connect", guard)


@pytest.fixture(scope="module")
def rows():
    return generate_dataset()


@pytest.fixture(scope="module")
def fitted(rows):
    """Train-only model (never sees dev/test) plus its fit time."""
    t0 = time.time()
    model = train_model(rows, splits=("train",))
    return {"model": model, "seconds": time.time() - t0}


# ---------------------------------------------------------------- constants
def test_label_and_product_constants_match_the_brief():
    assert LABELS == EXPECTED_LABELS
    assert PRODUCTS == EXPECTED_PRODUCTS


# --------------------------------------------------------------------- data
def test_all_labels_and_categories_in_training_data(rows):
    train = [r for r in rows if r["split"] == "train"]
    assert {r["fault_category"] for r in train} == set(LABELS)
    assert {r["product_category"] for r in train} == set(PRODUCTS)


def test_every_label_present_in_dev_and_test(rows):
    for split in ("dev", "test"):
        got = {r["fault_category"] for r in rows if r["split"] == split}
        assert got == set(LABELS), f"{split} is missing {set(LABELS) - got}"


def test_special_kinds_are_evaluable(rows):
    # code / overlap / ambiguous scenarios must exist outside the training split
    for kind in ("code", "overlap", "ambiguous"):
        for split in ("dev", "test"):
            assert any(r["kind"] == kind and r["split"] == split for r in rows), (kind, split)


def test_generation_is_deterministic():
    a, b = generate_dataset(seed=123), generate_dataset(seed=123)
    assert a == b
    assert generate_dataset(seed=124) != a  # the seed actually matters


def test_csv_roundtrip(rows, tmp_path):
    path = tmp_path / "data.csv"
    write_csv(rows, path)
    assert load_csv(path) == rows


def test_no_scenario_spans_splits_and_ids_are_opaque(rows):
    owner = {}
    for r in rows:
        assert owner.setdefault(r["scenario_id"], r["split"]) == r["split"]
        assert not any(label in r["scenario_id"] for label in LABELS)


def test_texts_are_unique_and_within_length_limits(rows):
    keys = [_key(r["complaint_text"]) for r in rows]
    assert len(keys) == len(set(keys))
    assert all(5 <= len(r["complaint_text"]) <= 2000 for r in rows)


def test_products_are_realistic_for_the_fault(rows):
    cooling = {"refrigerator", "air_conditioner"}
    drain = {"washing_machine", "dishwasher", "refrigerator", "air_conditioner"}
    for r in rows:
        if r["fault_category"] == "not_cooling":
            assert r["product_category"] in cooling
        if r["fault_category"] == "drainage_or_blockage":
            assert r["product_category"] in drain


# ----------------------------------------------------------------- contract
def test_training_stays_far_under_the_time_budget(fitted):
    assert fitted["seconds"] < 300  # the brief allows 5 minutes


@pytest.mark.parametrize("product", PRODUCTS)
@pytest.mark.parametrize("text", [
    "dead!",                                    # 5 characters
    "it is making a strange noise and shaking",
    "x" * 2000,                                 # maximum length
    "मशीन चालू नहीं हो रही",                      # non-Latin script
    "💥 it exploded 💥",
    "",                                         # outside the brief, must not crash
])
def test_predict_fault_returns_valid_label_and_bounded_confidence(fitted, product, text):
    label, conf = predict_fault(fitted["model"], product, text)
    assert label in EXPECTED_LABELS
    assert isinstance(conf, float)
    assert math.isfinite(conf) and 0.0 <= conf <= 1.0


def test_predict_proba_all_covers_all_12_labels(fitted):
    probs = predict_proba_all(fitted["model"], "refrigerator", "it is not cold inside")
    assert list(probs) == EXPECTED_LABELS
    assert all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in probs.values())
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)


def test_product_category_is_actually_used(fitted):
    text = "it is warm and not working properly"
    a = predict_proba_all(fitted["model"], "refrigerator", text)
    b = predict_proba_all(fitted["model"], "water_heater", text)
    assert a != b


def test_training_is_deterministic(rows, fitted):
    again = train_model(rows, splits=("train",))
    for product, text, _ in UNSEEN:
        p1 = predict_proba_all(fitted["model"], product, text)
        p2 = predict_proba_all(again, product, text)
        assert p1 == pytest.approx(p2)


def test_training_refuses_data_missing_a_label(rows):
    without_other = [r for r in rows if r["fault_category"] != "other"]
    with pytest.raises(ValueError):
        train_model(without_other, splits=("train",))


# -------------------------------------------------------- unseen paraphrases
def test_unseen_paraphrases_are_really_unseen(rows):
    seen = {_key(r["complaint_text"]) for r in rows}
    assert all(_key(text) not in seen for _, text, _ in UNSEEN)


def test_unseen_paraphrases_get_valid_outputs_and_beat_chance(fitted):
    correct = 0
    for product, text, expected in UNSEEN:
        label, conf = predict_fault(fitted["model"], product, text)
        assert label in EXPECTED_LABELS
        assert math.isfinite(conf) and 0.0 <= conf <= 1.0
        correct += int(label == expected)
    # Loose floor on purpose: chance is 1/12, not every case must be right.
    assert correct >= 5, f"only {correct}/{len(UNSEEN)} unseen paraphrases correct"


def test_typo_and_case_variants_still_produce_valid_outputs(fitted):
    variants = ["fridge is wrm insde n milk is spoiling", "FRIDGE NOT COOLING AT ALL!!!",
                "fridgee iss not coolng", "frdge   warm   inside"]
    for text in variants:
        label, conf = predict_fault(fitted["model"], "refrigerator", text)
        assert label in EXPECTED_LABELS and 0.0 <= conf <= 1.0


# --------------------------------------------------- held-out evaluation report
def test_heldout_report_has_per_class_results_and_macro_f1(rows, fitted, capsys):
    test_rows = [r for r in rows if r["split"] == "test"]
    X = [(r["product_category"], r["complaint_text"]) for r in test_rows]
    y = [r["fault_category"] for r in test_rows]
    pred = fitted["model"].predict(X)

    report = classification_report(y, pred, labels=LABELS, zero_division=0, output_dict=True)
    macro = f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)

    with capsys.disabled():  # visible with `-s`
        print(f"\nHELD-OUT (train-only model) Macro-F1 = {macro:.3f}")
        print(classification_report(y, pred, labels=LABELS, zero_division=0, digits=3))

    for label in LABELS:                       # per-class results exist for all 12
        assert report[label]["support"] > 0
        assert 0.0 <= report[label]["f1-score"] <= 1.0
    assert math.isfinite(macro) and 0.0 <= macro <= 1.0
    assert macro > 0.25                        # loose floor, ~3x chance, not a target


def test_evaluate_helper_returns_a_valid_macro_f1(rows, fitted, capsys):
    test_rows = [r for r in rows if r["split"] == "test"]
    value = evaluate(fitted["model"], test_rows, "test")
    capsys.readouterr()  # swallow the printed report
    assert math.isfinite(value) and 0.0 <= value <= 1.0