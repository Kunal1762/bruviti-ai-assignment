"""Local fault classifier: word + char TF-IDF, product-crossed features, and a
class-balanced multinomial logistic regression.

    python -m src.classifier     # tune C on dev, report dev + test, print timing

Contract:
    train_model()                                  -> fitted model
    predict_fault(model, product_category, text)   -> (label, confidence)

Protocol: C is tuned on `dev` with a train-only model. The final model is then
fit on train+dev and scored ONCE on `test`. `test` is never used for tuning.
Safety rules C1-C6 are NOT implemented here; they belong to the rule engine.
"""
from __future__ import annotations

import math
import re
import time
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.data_diverse import LABELS, OUT_PATH, generate_dataset, load_csv

__all__ = ["train_model", "predict_fault", "predict_proba_all", "evaluate"]

DEFAULT_C = 10.0

_REPEAT_RE = re.compile(r"(.)\1{2,}")
_WORD_RE = re.compile(r"[a-z][a-z']+")
# Error-code shapes (E21, F-4, Err 3, 1E). Unseen codes still map to one token.
_CODE_RE = re.compile(r"\b(?:err(?:or)?|e|f|h|u|d|c|p)[ -]?\d{1,3}\b|\b\d[a-z]\b")


def _norm(text) -> str:
    t = "" if text is None else str(text).lower()
    t = _REPEAT_RE.sub(r"\1\1", t)            # "sooo loud" -> "soo loud"
    t = re.sub(r"\s+", " ", t).strip()
    if _CODE_RE.search(t):
        t += " errcodetoken"                  # generalises across unseen codes
    return t


def _prod(p) -> str:
    return re.sub(r"[^a-z_]", "", str(p).lower()) or "unknown_product"


def _plain_view(X):
    return [_norm(t) for _, t in X]


_CROSS_KEYWORDS = {
    "warm", "warms", "warming", "cold", "colder", "freezing", "freeze", "frozen",
    "cool", "cooling", "cooler", "heat", "heats", "heating", "heated", "hot",
    "hotter", "lukewarm", "tepid", "ice", "frost", "defrost", "thaw", "chill",
    "blow", "blows", "blowing", "air", "flame", "burner", "element", "steam",
}


def _cross_view(X):
    """Cross product ONLY with thermal/state terms whose fault meaning depends
    on product type (e.g. 'cold' on a washer vs 'cold' on a fridge).
    Stand-alone product unigram is excluded to avoid appliance-level prior bias."""
    out = []
    for p, t in X:
        p = _prod(p)
        words = _WORD_RE.findall(_norm(t))
        crossed = [f"{p}__{w}" for w in words if w in _CROSS_KEYWORDS]
        out.append(" ".join(crossed))
    return out


def _build(C: float) -> Pipeline:
    feats = FeatureUnion([
        ("word", Pipeline([
            ("view", FunctionTransformer(_plain_view)),
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                      min_df=2, max_df=0.85, strip_accents="unicode"))])),
        ("char", Pipeline([
            ("view", FunctionTransformer(_plain_view)),
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                      sublinear_tf=True, min_df=3,
                                      strip_accents="unicode"))])),
        ("cross", Pipeline([
            ("view", FunctionTransformer(_cross_view)),
            ("tfidf", TfidfVectorizer(token_pattern=r"\S+", lowercase=False,
                                      sublinear_tf=True, min_df=2))])),
    ])
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, solver="lbfgs")
    return Pipeline([("feats", feats), ("clf", clf)])


def _load_rows():
    return load_csv(OUT_PATH) if OUT_PATH.exists() else generate_dataset()


def _xy(rows):
    return ([(r["product_category"], r["complaint_text"]) for r in rows],
            [r["fault_category"] for r in rows])


def train_model(rows=None, splits=("train", "dev"), C: float = DEFAULT_C):
    """Fit on the chosen splits (default train+dev; `test` is never used)."""
    rows = _load_rows() if rows is None else rows
    chosen = [r for r in rows if r["split"] in splits]
    X, y = _xy(chosen)
    missing = set(LABELS) - set(y)
    if missing:
        raise ValueError(f"training data is missing labels: {sorted(missing)}")
    model = _build(C)
    model.fit(X, y)
    return model


TIE_MARGIN = 0.08
OTHER_CONFIDENCE_THRESHOLD = 0.20

_SAFETY_TRIGGERS = re.compile(
    r"\b(?:spark|sparks|sparked|sparking|smoke|smoking|smoked|fire|fires|flame|flames|"
    r"burn|burns|burning|burnt|burned|scorch|scorched|shock|shocks|shocked|electric shock|"
    r"trips?|tripped|tripping|mcb|fuse|short circuit|arcing)\b",
    re.I,
)


def _predict_batch(model, X):
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    preds = []
    confs = []
    for (p, text), row in zip(X, proba):
        text_str = str(text or "")
        d = {c: float(prob) for c, prob in zip(classes, row)}

        # C1 Safety override: electrical safety issues take precedence
        if _SAFETY_TRIGGERS.search(text_str):
            preds.append("electrical_or_control_board")
            confs.append(max(d.get("electrical_or_control_board", 0.0), 0.90))
            continue

        sorted_labels = sorted(LABELS, key=lambda l: d.get(l, 0.0), reverse=True)
        best, second = sorted_labels[0], sorted_labels[1]

        # Low/diffuse confidence maps to 'other' per Appendix B & C6
        if d.get(best, 0.0) < OTHER_CONFIDENCE_THRESHOLD:
            best = "other"
        # C5 tie-breaking: if top two are within TIE_MARGIN, pick earlier in LABELS order
        elif (d.get(best, 0.0) - d.get(second, 0.0)) <= TIE_MARGIN:
            if LABELS.index(second) < LABELS.index(best):
                best = second

        preds.append(best)
        confs.append(d.get(best, 0.0))
    return preds, confs


def predict_proba_all(model, product_category: str, complaint_text: str) -> dict:
    """Probability for every one of the 12 labels, in Appendix B order."""
    proba = model.predict_proba([(product_category, complaint_text)])[0]
    d = {c: float(p) for c, p in zip(model.classes_, proba)}
    return {label: d.get(label, 0.0) for label in LABELS}


def predict_fault(model, product_category: str, complaint_text: str):
    """Return (label, confidence); label is one of the 12 exact labels and
    confidence is a finite float in [0, 1]. Near ties (within TIE_MARGIN)
    follow C5 Appendix B priority order; C1 safety triggers prioritize
    electrical_or_control_board; diffuse uncertainty maps to 'other'."""
    text_str = str(complaint_text or "")
    if _SAFETY_TRIGGERS.search(text_str):
        probs = predict_proba_all(model, product_category, text_str)
        conf = max(probs.get("electrical_or_control_board", 0.0), 0.90)
        return "electrical_or_control_board", min(1.0, conf)

    probs = predict_proba_all(model, product_category, text_str)
    sorted_labels = sorted(LABELS, key=lambda l: probs[l], reverse=True)
    best, second = sorted_labels[0], sorted_labels[1]

    if probs.get(best, 0.0) < OTHER_CONFIDENCE_THRESHOLD:
        best = "other"
    elif (probs[best] - probs[second]) <= TIE_MARGIN:
        if LABELS.index(second) < LABELS.index(best):
            best = second

    conf = probs[best]
    if not math.isfinite(conf):
        conf = 0.0
    return best, min(1.0, max(0.0, conf))


def _macro_f1(model, rows):
    X, y = _xy(rows)
    pred, _ = _predict_batch(model, X)
    return f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)


def evaluate(model, rows, name: str) -> float:
    """Print per-class metrics, Macro-F1, accuracy by scenario kind, and whether
    confidence separates right from wrong answers."""
    X, y = _xy(rows)
    pred, conf = _predict_batch(model, X)
    mf1 = f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)
    print(f"\n== {name}: n={len(rows)}  Macro-F1={mf1:.3f}")
    print(classification_report(y, pred, labels=LABELS, zero_division=0, digits=3))
    kinds = defaultdict(lambda: [0, 0])
    for r, p in zip(rows, pred):
        kinds[r["kind"]][0] += int(p == r["fault_category"])
        kinds[r["kind"]][1] += 1
    print("accuracy by scenario kind:",
          {k: f"{a}/{n}" for k, (a, n) in sorted(kinds.items())})
    ok = [c for c, yy, p in zip(conf, y, pred) if yy == p]
    bad = [c for c, yy, p in zip(conf, y, pred) if yy != p]
    if ok and bad:
        print(f"mean confidence: correct={sum(ok)/len(ok):.3f}  wrong={sum(bad)/len(bad):.3f}")
    return mf1


def main():
    rows = _load_rows()
    by = lambda s: [r for r in rows if r["split"] == s]
    print({s: len(by(s)) for s in ("train", "dev", "test")})

    best_c, best_f1 = DEFAULT_C, -1.0
    for C in (1.0, 3.0, 10.0, 20.0, 30.0, 50.0):
        t0 = time.time()
        m = train_model(rows, splits=("train",), C=C)
        f1 = _macro_f1(m, by("dev"))
        print(f"C={C:<5} dev Macro-F1={f1:.3f}  fit={time.time()-t0:.1f}s")
        if f1 > best_f1:
            best_c, best_f1 = C, f1
    print(f"selected C={best_c}")

    evaluate(train_model(rows, splits=("train",), C=best_c), by("dev"), "DEV (train-only model)")

    t0 = time.time()
    final = train_model(rows, splits=("train", "dev"), C=best_c)
    print(f"\nfinal fit (train+dev): {time.time()-t0:.1f}s")
    evaluate(final, by("test"), "TEST (final model, scored once)")
    print("\nNOTE: synthetic held-out scores do not predict hidden-test performance.")


if __name__ == "__main__":
    main()