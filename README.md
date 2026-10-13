# Bruviti Fault Classifier (Use Case 1)

An automated service that classifies appliance service complaints into one of twelve fault categories, estimates an information confidence score, and enforces safety and business rules (C1–C6) to flag tickets that require human review.

---

## 1. System Requirements & Architecture

- **Stack:** Python 3.9+ with FastAPI, Uvicorn, scikit-learn, and Pydantic.
- **Constraints Met:** 
  - Runs completely offline (no LLMs or external API calls at request time).
  - Training completes in under 5 minutes on a laptop CPU.
  - Response time per request is under 1 second.
  - Single process serving both the REST API and the built-in UI at `/` on port `8000`.

---

## 2. Installation and Quick Start

From the repository root in PowerShell, create and activate a virtual environment, install dependencies, and start the development server:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

The server loads the classifier at startup. Open the **Interactive Single-Page UI** at `http://127.0.0.1:8000/`, the API docs at `http://127.0.0.1:8000/docs`, or the health check at `http://127.0.0.1:8000/health` in your browser.

### Linux and macOS

From the repository root, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

Then use the same UI, API docs, and health-check URLs above.

---

## 3. Running Tests

To run the offline test suite (fully functional without network access):

```bash
python -m pytest -q
```

---

## 4. API Contract (`POST /classify`)

- **Endpoint:** `POST /classify`
- **Request Body (JSON):**
  ```json
  {
    "ticket_id": "T-1001",
    "product_category": "washing_machine",
    "complaint_text": "machine fills with water then just sits there, drum never turns, display shows E21",
    "product_age_days": 410
  }
  ```
- **Response Body (JSON):**
  ```json
  {
    "ticket_id": "T-1001",
    "fault_category": "motor_or_drive",
    "confidence": 0.78,
    "escalate": false,
    "reason": "Drum not turning with fill complete; E21 on this product family is a drive fault."
  }
  ```

---

## 5. Repository Structure

```text
├── src/
│   ├── api.py           # FastAPI application, lifecycle, and endpoint routes
│   ├── classifier1.py   # TF-IDF vectorizer + Multinomial Logistic Regression model
│   ├── rules1.py        # Deterministic business rules (C1–C6 engine)
│   └── data_diverse.py  # Synthetic dataset generator and loader utilities
├── data/
│   └── synthetic_complaints1.csv # Training and evaluation data generated offline
├── tests/
│   ├── test_api.py      # End-to-end and schema validation tests
│   ├── test_classifier1.py# Classifier and metric verification tests
│   └── test_rules1.py   # Business rules unit tests
├── requirements.txt     # Python package dependencies
├── README.md            # System documentation and instructions
├── DATA.md              # Data sourcing, construction, and test design
├── NOTES.md             # Design decisions, trade-offs, and future work
└── AI.md                # List of AI tools and their utility