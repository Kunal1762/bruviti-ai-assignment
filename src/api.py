


"""FastAPI app exposing the classifier1 + rules1 pipeline.

Start: uvicorn src.api:app --host 0.0.0.0 --port 8000
Docs:  http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
import math
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from src.classifier1 import train_model, _SAFETY_TRIGGERS
from src.data_diverse import LABELS
from src.rules1 import classify_ticket

log = logging.getLogger("api")

# ------------------------------------------------------------------ lifecycle
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup; release on shutdown."""
    print("Loading classifier1 model...")
    _state["model"] = train_model()
    print("Model ready.")
    yield
    _state.clear()


app = FastAPI(
    title="Bruviti Fault Classifier",
    description="Classifies appliance service complaints into fault categories.",
    version="1.0.0",
    lifespan=lifespan,
)

_STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    """Single-page UI served by the same process."""
    return FileResponse(_STATIC / "index.html")


# ------------------------------------------------------------------ schema
# Validation is deliberately lenient: a rejected ticket (HTTP 422) can never be
# escalated, so every field is normalised instead of refused. Unknown product
# categories are still classified; the model just sees an unknown product.
class ClassifyRequest(BaseModel):
    ticket_id: str = ""
    product_category: str = "unknown_product"
    complaint_text: str = ""
    product_age_days: Optional[float] = None

    @field_validator("ticket_id", mode="before")
    @classmethod
    def _tid(cls, v: Any) -> str:
        return "" if v is None else str(v)

    @field_validator("product_category", mode="before")
    @classmethod
    def _prod(cls, v: Any) -> str:
        s = re.sub(r"[\s\-]+", "_", str(v or "").strip().lower())
        return s or "unknown_product"

    @field_validator("complaint_text", mode="before")
    @classmethod
    def _text(cls, v: Any) -> str:
        # Generous cap: truncating too aggressively could cut off a trailing
        # "sparks"/"smoke", and rule C1 must fire whatever else the text says.
        return str(v or "")[:10000]

    @field_validator("product_age_days", mode="before")
    @classmethod
    def _age(cls, v: Any) -> Optional[float]:
        try:
            if v is None or str(v).strip() == "":
                return None
            f = float(v)
            return None if (math.isnan(f) or math.isinf(f)) else f
        except (TypeError, ValueError):
            return None


class ClassifyResponse(BaseModel):
    ticket_id: str
    fault_category: str
    confidence: float
    escalate: bool
    reason: str


# ------------------------------------------------------------------ fallback
def _fallback(text: str) -> dict:
    """Used only if classify_ticket raises. Safety words still escalate."""
    if _SAFETY_TRIGGERS.search(text or ""):
        return {
            "fault_category": "electrical_or_control_board",
            "confidence": 0.9,
            "escalate": True,
            "reason": "Fallback path: safety keyword, rule C1.",
        }
    return {
        "fault_category": "other",
        "confidence": 0.1,
        "escalate": False,
        "reason": "Fallback path: classifier error, cannot determine.",
    }


# ------------------------------------------------------------------ endpoints
@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    """
    Classify an appliance service complaint.

    - **ticket_id**: your reference ID, echoed back in the response
    - **product_category**: appliance type (e.g. `washing_machine`)
    - **complaint_text**: free-text complaint from the customer
    - **product_age_days**: age of the product in days (optional, used for Rule C3)
    """
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        result = classify_ticket(
            model,
            req.product_category,
            req.complaint_text,
            req.product_age_days,
        )
    except Exception:
        log.exception("classify_ticket failed")
        result = _fallback(req.complaint_text)
    return ClassifyResponse(ticket_id=req.ticket_id, **result)


@app.get("/labels")
def labels():
    """Return the 12 supported fault categories in Appendix B order."""
    return {"labels": LABELS}