"""FastAPI app exposing the classifier1 + rules1 pipeline.

Start: uvicorn src.api:app --reload
Docs:  http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from src.classifier1 import train_model
from src.data_diverse import LABELS
from src.rules1 import classify_ticket

from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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
VALID_PRODUCTS = {
    "washing_machine", "refrigerator", "air_conditioner", "dishwasher",
    "microwave", "oven_or_cooktop", "water_heater", "dryer",
}


class ClassifyRequest(BaseModel):
    ticket_id: str
    product_category: str
    complaint_text: str
    product_age_days: Optional[float] = None

    @field_validator("product_category")
    @classmethod
    def check_product(cls, v: str) -> str:
        if v not in VALID_PRODUCTS:
            raise ValueError(
                f"Unknown product_category {v!r}. "
                f"Valid values: {sorted(VALID_PRODUCTS)}"
            )
        return v

    @field_validator("complaint_text")
    @classmethod
    def check_text(cls, v: str) -> str:
        return v[:3000]   # hard cap to avoid abuse


class ClassifyResponse(BaseModel):
    ticket_id: str
    fault_category: str
    confidence: float
    escalate: bool
    reason: str


# ------------------------------------------------------------------ endpoints
@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    """
    Classify an appliance service complaint.

    - **ticket_id**: your reference ID — echoed back in the response
    - **product_category**: appliance type (e.g. `washing_machine`)
    - **complaint_text**: free-text complaint from the customer
    - **product_age_days**: age of the product in days (optional, used for Rule C3)
    """
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    result = classify_ticket(
        model,
        req.product_category,
        req.complaint_text,
        req.product_age_days,
    )
    return ClassifyResponse(ticket_id=req.ticket_id, **result)


@app.get("/labels")
def labels():
    """Return the 12 supported fault categories in Appendix B order."""
    return {"labels": LABELS}