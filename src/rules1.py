"""Deterministic business rules C1-C6, applied AFTER the ML model (rules1.py).

    probs (12 labels) + product + text + age  ->  (label, confidence, escalate, reason)

Only C1 and C2 can set `escalate`. Both are recall-biased: a missed safety
escalation is the costliest error, a false alarm only costs a person's time.

Matching is token based, on text that is lower-cased, accent-folded, with
apostrophes removed and long letter runs squashed. Tokens of 5+ letters are
fuzzy-matched to the trigger vocabulary (typos like "sparkk", "smokeee").
Negation ("no sparks", "did not see any smoke") suppresses a hit only when a
negator is reached through a short run of filler words directly before it.

Differences from rules.py:
- Imports LABELS from src.data_diverse (same 12 labels, same order).
- classifier1.predict_proba_all is used (already wired in rules.py too).
"""
from __future__ import annotations

import difflib
import functools
import re
import unicodedata
from dataclasses import dataclass, field

from src.classifier1 import predict_proba_all
from src.data_diverse import LABELS

__all__ = ["Decision", "apply_rules", "classify_ticket"]

TIE_MARGIN = 0.05            # C5: top-2 probabilities closer than this are a tie
C3_MODEL_THRESHOLD = 0.45    # C3: model P(installation) needed without a keyword
C4_CONFIDENCE = 0.65
C1_CONFIDENCE_FLOOR = 0.90
C6_CONFIDENCE_CAP = 0.29     # C6 says "below 0.3"
GAS_PRODUCTS = {"oven_or_cooktop", "water_heater"}

# ------------------------------------------------------------ vocabulary (C1)
SPARK = {"spark", "sparks", "sparked", "sparking", "sparky", "chingari", "chinghari",
         "etincelle", "etincelles", "chispa", "chispas", "funke", "funken"}
SMOKE = {"smoke", "smoking", "smoked", "smoky", "smokey", "dhuan", "dhua", "humo",
         "fumee", "fumees", "rauch"}
FIRE = {"fire", "fires", "aag", "fuego", "incendio", "incendie", "feu", "feuer"}
FLAME = {"flame", "flames", "flaming", "llamas", "flammes", "flammen"}
FLAME_CTX = {"caught", "catch", "catching", "burst", "bursting", "shooting", "shot",
             "leaping", "huge", "coming", "came", "jumping", "jumped", "behind"}
BURN = {"burn", "burns", "burning", "burnt", "burned", "scorch", "scorched", "scorching",
        "quemado", "quemada", "quemando", "brule", "brulee", "jalne", "jalta", "jalti",
        "jala", "jali", "verbrannt"}
EXPLODE = {"explode","exploded","explodes","exploding","explosion","blast"}
BURN_CTX = {"inside","board","pcb","motor","element","unit","machine","back","panel","wire","wiring","something"}
MELT = {"melt", "melting", "melted", "melts"}
PARTS = {"plastic", "rubber", "wire", "wiring", "wires", "cable", "cord", "insulation",
         "plug", "socket"}
SMELL = {"smell", "smells", "smelling", "smelly", "smelt", "smelled", "odor", "odour",
         "odeur", "stink", "stinks", "stinking", "stank", "fumes", "fume", "whiff",
         "scent", "acrid", "pungent", "huele", "olor", "boo", "badboo", "gandh",
         "geruch", "riecht"}
SHOCK_DIRECT = {"shock", "shocks", "electrocuted", "electrocute", "electrocution",
                "zapped", "calambre", "calambres", "stromschlag"}
SHOCK_CTX = {"shocked", "tingling", "tingle", "tingles", "jolt", "jolted"}
SHOCK_CONTEXT = {"electric", "electrical", "electricity", "current", "touch", "touched",
                 "touching", "metal", "body", "casing", "handle", "wet", "hands", "hand",
                 "tap", "door", "mild", "wire", "plug", "socket", "earth", "earthing"}
CURRENT_VERBS = {"laga", "lagta", "lagti", "lagi", "lagne", "lag", "mara"}
TRIP = {"trip", "trips", "tripped", "tripping", "tripd", "triped"}
TRIP_NOT = {"over", "on", "up", "hazard"}            # "tripped over the cable"
BLOW = {"blew", "blown", "blow", "blows", "blowing", "popped", "pops", "knocked",
        "knocks", "jumped", "jumps"}
SUPPLY_STRONG = {"breaker", "breakers", "mcb", "mccb", "rccb", "elcb", "gfci", "rcd",
                 "fuse", "fuses", "fusebox", "switchboard"}
SUPPLY_ALL = SUPPLY_STRONG | {"power", "mains", "supply", "electricity", "electrics",
                              "electric", "electrical", "earth", "gfi"}
BREAKER_ULTRA = {"mcb", "mccb", "rccb", "elcb", "gfci", "rcd", "breaker", "breakers",
                 "fuse", "fuses"}
BREAKER_VERBS = { "down", "popped", "pops",
                 "jumps", "jumped", "drops", "dropped", "knocks", "knocked", "goes",
                 "went", "tripped", "trips", "trip", "blew", "blown"}
# ----------------------------------------------------------- vocabulary (C2)
GAS = {"gas", "gass", "gaas", "lpg", "cng", "propane", "butane", "gaz", "gasgeruch"}
LEAK = {"leak", "leaks", "leaking", "leakage", "leaked",  "escaping",
        "escape", "escapes"}
LEAK_LINK = {"is", "was", "keeps", "has", "been", "kept"}

# ----------------------------------------------------------------- negation
_NEG = {"no", "not", "never", "without", "nothing", "nor", "neither", "isnt", "arent",
        "wasnt", "werent", "dont", "doesnt", "didnt", "hasnt", "havent", "hadnt",
        "cant", "cannot", "nahi", "nahin", "nai", "sin", "pas", "kein", "keine", "ni"}
_FILLER = ({"any", "a", "an", "the", "of", "sign", "signs", "trace", "traces", "hint",
            "real", "actual", "visible", "obvious", "strong", "bad", "noticeable", "see",
            "seen", "saw", "notice", "noticed", "hear", "heard", "even", "single", "kind",
            "sort", "type", "ever", "really", "at", "all", "or", "nor", "electric",
            "electrical", "electricity", "de", "d", "un", "una"}
           | SPARK | SMOKE | FIRE | BURN | SMELL)
_NEG_LOOKBACK = 6

_KNOWN = (SPARK | SMOKE | FIRE | FLAME | BURN | MELT | SMELL | SHOCK_DIRECT | SHOCK_CTX
          | TRIP | BLOW | SUPPLY_ALL | GAS | LEAK)
_FUZZY = sorted(w for w in _KNOWN if len(w) >= 5)
_FUZZY_BLOCK = {"sparkling", "sparkle", "sparkles", "sparkly", "shocking"}
_RUN3 = re.compile(r"(.)\1{2,}")

# ----------------------------------------------------------------- C3 / C4
_FIT_RE = re.compile(
    r"\b(?:un)?level(?:l?ing|l?ed)\b|\bnot (?:quite |properly |perfectly )?level\b"
    r"|\b(?:isnt|arent|wasnt) level\b|\bwobbl\w*|\binstall\w*|\bfit(?:s|ted|ting)?\b"
    r"|\bmisalign\w*|\balign\w*|\bfirst (?:use|time)\b|\bset(?:ting)? ?up\b|\bsetup\b"
    r"|\bconnect\w*|\bcrooked\b|\btilt\w*|\bnot straight\b|\buneven\b|\bplumb\w*")
_CODE_RE = re.compile(r"\b(?:err(?:or)?|e|f|h|u|d|c|p)[ -]?\d{1,3}\b|\b\d[a-z]\b")
_C4_STOP = set("""
a an the it is on in of at this that just only says say said saying shows show showing
shown display displayed displays screen panel error errors err code codes fault faults
message flashing flashes blinking reads reading what does mean means meaning getting get
got see seeing seen appears appeared there with and or for to now again still also same
number idea no know tell me us we our i my unit machine appliance product nothing else
other anything please pls help advise thanks thank you hi hello dear support team urgent
complaint issue problem query regards sir madam kindly why how can could explain fix
asap soon today yesterday since morning last night customer care good re
washer washing dishwasher dish oven cooker cooktop hob range ac air conditioner split
window heater geyser microwave fridge refrigerator freezer front loader dryer water
""".split())


# ------------------------------------------------------------ text handling
def _fold(text) -> str:
    s = "" if text is None else str(text).lower()
    out = []
    for ch in unicodedata.normalize("NFKD", s):
        if unicodedata.combining(ch) and out and out[-1].isascii():
            continue  # drop accents on Latin letters only
        out.append(ch)
    s = "".join(out).replace("'", "").replace("\u2019", "")
    return _RUN3.sub(r"\1\1", s)


@functools.lru_cache(maxsize=8192)
def _canon(tok: str) -> str:
    if tok in _KNOWN or tok in _FUZZY_BLOCK or len(tok) < 5 or not tok.isalpha():
        return tok
    m = difflib.get_close_matches(tok, _FUZZY, n=1, cutoff=0.85)
    return m[0] if m else tok


def _prepare(text):
    folded = _fold(text)
    return folded, [_canon(t) for t in re.findall(r"[a-z0-9]+", folded)]


def _negated(toks, i) -> bool:
    for k in range(i - 1, max(-1, i - 1 - _NEG_LOOKBACK), -1):
        if toks[k] in _NEG:
            return True
        if toks[k] not in _FILLER:
            return False
    return False


def _pairs(toks, a, b, window):
    pa = [i for i, t in enumerate(toks) if t in a]
    pb = [j for j, t in enumerate(toks) if t in b]
    return [min(i, j) for i in pa for j in pb if i != j and abs(i - j) <= window]


# ------------------------------------------------------------------ C1 / C2
def _c1_evidence(toks):
    hits, n = [], len(toks)
    for i, t in enumerate(toks):
        if t in SPARK:
            hits.append((i, "spark"))
        elif t in SMOKE:
            hits.append((i, "smoke"))
        elif t in FIRE:
            hits.append((i, "fire"))
        if t in SHOCK_DIRECT and not (i + 1 < n and toks[i + 1].startswith("absorb")):
            hits.append((i, "electric shock"))
        if t in TRIP and not (i + 1 < n and toks[i + 1] in TRIP_NOT):
            lo, hi = max(0, i - 4), min(n, i + 5)
            if any(j != i and toks[j] in SUPPLY_ALL for j in range(lo, hi)):
                hits.append((i, "tripped supply"))
    hits += [(i, "flames") for i in _pairs(toks, FLAME, FLAME_CTX, 3)]
    hits += [(i, "burning smell") for i in _pairs(toks, BURN, SMELL, 4)]
    hits += [(i, "burning wiring") for i in _pairs(toks, BURN, PARTS, 2)]
    hits += [(i, "melting plastic") for i in _pairs(toks, MELT, PARTS, 1)]
    hits += [(i, "electric shock") for i in _pairs(toks, SHOCK_CTX, SHOCK_CONTEXT, 4)]
    hits += [(i, "electric shock") for i in _pairs(toks, {"current"}, CURRENT_VERBS, 2)]
    hits += [(i, "electric shock") for i in _pairs(
        toks, {"descarga", "decharge"}, {"electrica", "electrique", "electric"}, 2)]
    hits += [(i, "electric shock") for i in _pairs(toks, {"choc"}, {"electrique"}, 1)]
    hits += [(i, "blown fuse") for i in _pairs(toks, BLOW, SUPPLY_STRONG, 4)]
    hits += [(i, "tripped breaker") for i in _pairs(toks, BREAKER_ULTRA, BREAKER_VERBS, 3)]
    
    hits += [(i, "explosion") for i, t in enumerate(toks) if t in EXPLODE]
    hits += [(i, "short circuit") for i in _pairs(toks, {"short","shorted"}, {"circuit","circuited"}, 1)]
    hits += [(i, "burning inside") for i in _pairs(toks, BURN, BURN_CTX, 3)]
    return [(i, lab) for i, lab in hits if not _negated(toks, i)]


def _c2_evidence(toks):
    hits, n = [], len(toks)
    for i, t in enumerate(toks):
        if t not in GAS:
            continue
        nxt = toks[i + 1] if i + 1 < n else ""
        nxt2 = toks[i + 2] if i + 2 < n else ""
        prv = toks[i - 1] if i > 0 else ""
        prv2 = toks[i - 2] if i > 1 else ""
        if (nxt in LEAK or (nxt in LEAK_LINK and nxt2 in LEAK) or prv in LEAK
                or (prv == "of" and prv2 in LEAK)):
            hits.append((i, "gas leak"))
    hits += [(i, "gas smell") for i in _pairs(toks, GAS, SMELL, 4)]
    hits += [(i, "rotten-egg smell") for i in _pairs(toks, {"rotten"}, {"egg", "eggs"}, 1)]
    return [(i, lab) for i, lab in hits if not _negated(toks, i)]


# ------------------------------------------------------------------ C3 / C4
def _valid_age(age) -> bool:
    return (isinstance(age, (int, float)) and not isinstance(age, bool)
            and age == age and age >= 0)


def _fit_complaint(folded, probs, top) -> bool:
    if top == "other" and probs["other"] >= 0.5:
        return False  # e.g. a sales enquiry about installation charges
    return bool(_FIT_RE.search(folded)) or probs["installation_or_setup"] >= C3_MODEL_THRESHOLD


def _code_only(folded) -> bool:
    if not _CODE_RE.search(folded):
        return False
    rest = _CODE_RE.sub(" ", folded)
    return not [w for w in re.findall(r"[a-z]+", rest) if w not in _C4_STOP]


# ------------------------------------------------------------------- engine
@dataclass
class Decision:
    fault_category: str
    confidence: float
    escalate: bool
    reason: str
    rules: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"fault_category": self.fault_category, "confidence": self.confidence,
                "escalate": self.escalate, "reason": self.reason}


def _uniq(evidence):
    seen = []
    for _, lab in evidence:
        if lab not in seen:
            seen.append(lab)
    return seen[:3]


def apply_rules(probs: dict, product_category: str, complaint_text: str,
                product_age_days=None, tie_margin: float = TIE_MARGIN) -> Decision:
    """probs: probability for each of the 12 labels. Precedence: C1 > C3 > C4 >
    C5 > C6; C2 only sets the escalate flag and never changes the category."""
    ranked = sorted(LABELS, key=lambda l: (-probs[l], LABELS.index(l)))
    top, second = ranked[0], ranked[1]
    folded, toks = _prepare(complaint_text)

    label, conf = top, probs[top]
    escalate, forced = False, False
    rules, parts = [], []

    c1 = _c1_evidence(toks)
    if c1:
        label, conf = "electrical_or_control_board", max(probs["electrical_or_control_board"],
                                                         C1_CONFIDENCE_FLOOR)
        escalate = forced = True
        rules.append("C1")
        parts.append(f"safety rule C1 ({', '.join(_uniq(c1))}) sets "
                     "electrical_or_control_board and escalates")

    if product_category in GAS_PRODUCTS:
        c2 = _c2_evidence(toks)
        if c2:
            escalate = True
            rules.append("C2")
            parts.append(f"{_uniq(c2)[0]} on a {product_category.replace('_', ' ')}: rule C2 "
                         "escalates, category comes from the rest of the text")

    if (not forced and _valid_age(product_age_days) and product_age_days <= 30
            and _fit_complaint(folded, probs, top)):
        label = "installation_or_setup"
        conf = min(0.90, max(probs[label], 0.80))
        forced = True
        rules.append("C3")
        parts.append(f"under 30 days old ({int(product_age_days)} days) with a fit, "
                     "levelling, connection or first-use complaint: rule C3")

    if not forced and _code_only(folded):
        label, conf, forced = "control_panel_or_display", C4_CONFIDENCE, True
        rules.append("C4")
        parts.append("error code with no other symptom described: rule C4")

    if not forced:
        if probs[top] - probs[second] <= tie_margin:
            earlier = top if LABELS.index(top) < LABELS.index(second) else second
            label, conf = earlier, max(0.05, probs[earlier] - 0.10)
            rules.append("C5")
            parts.append(f"{top} and {second} are almost equally supported, so rule C5 "
                         f"takes the earlier category ({label}) and lowers confidence")
        if label == "other":
            conf = min(conf, C6_CONFIDENCE_CAP)
            rules.append("C6")
            parts.append("no diagnosable fault in the text (not a complaint or too vague): "
                         "rule C6 keeps confidence below 0.3")

    if not set(rules) & {"C1", "C3", "C4", "C5", "C6"}:
        parts.append(f"category {label} from the text and product (model confidence {conf:.2f})")

    conf = round(min(1.0, max(0.0, float(conf))), 4)
    reason = "; ".join(parts)
    return Decision(label, conf, escalate, reason[:1].upper() + reason[1:] + ".", rules)


def classify_ticket(model, product_category: str, complaint_text: str,
                    product_age_days=None) -> dict:
    """Model + rules. Returns the four response fields of POST /classify."""
    probs = predict_proba_all(model, product_category, complaint_text)
    return apply_rules(probs, product_category, complaint_text, product_age_days).as_dict()