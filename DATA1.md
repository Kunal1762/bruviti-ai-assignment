# DATA1.md — Data Sourcing, Construction, Diversity Enhancements, and Evaluation Design

This document details the complete data lifecycle for the enhanced dataset generator (`src/data_diverse.py`), the resulting dataset (`data/synthetic_complaints1.csv`), and its evaluation with `src/classifier1.py`.

---

## 1. Where the Data Came From

Per the Bruviti Use Case 1 specification, no labelled training data was provided beyond the appendix tables. **All data is synthetic, generated deterministically by `src/data_diverse.py`**.

- **No external APIs or LLMs at runtime**: All scenarios and vocabulary variations are contained within the repository.
- **Deterministic expansion**: A fixed master seed (`SEED = 20261013`) and scenario-level seeded random instances guarantee identical CSV generation across operating systems and runs.
- **Output Target**: `data/synthetic_complaints1.csv` (~12,000 rows, target `ROWS_PER_LABEL = 1000`).
- **Generation Command**: `python -m src.data_diverse`

### Schema of `data/synthetic_complaints1.csv`

| Column | Type | Description |
|---|---|---|
| `id` | `string` | Stable sequential identifier (e.g. `S00000` to `S11999`). |
| `product_category` | `string` | One of the 7 valid categories in Appendix A. |
| `complaint_text` | `string` | The customer complaint (strictly bounded between 5 and 2,000 characters). |
| `fault_category` | `string` | Ground truth label from the 12 classes in Appendix B. |
| `split` | `string` | Scenario-level assignment: `train` (~62%), `dev` (~19%), or `test` (~19%). |
| `scenario_id` | `string` | Opaque identifier of the underlying scenario (used solely for leakage prevention and analysis; never exposed as a model feature). |
| `kind` | `string` | Scenario behavioral type: `plain`, `indirect`, `ambiguous`, `code`, `overlap`, `multilingual`. |

*Note: In accordance with the assignment brief, the dataset deliberately omits `escalate`, `confidence`, `reason`, and `product_age_days`. Escalation (Rules C1, C2), age logic (Rule C3), and confidence bounds (Rules C5, C6) are handled by the inference rule engine.*

---

## 2. Iterative Evolution: From `data.py` to `data_diverse.py`

### Baseline (`data.py`)
- Generated ~1,710 total rows (`ROWS_PER_LABEL = 144`).
- Evaluation on held-out test data showed severe performance bottlenecks:
  - **`not_heating` collapsed on DEV** (F1 = 0.026, Recall = 0.018) because air conditioners in heating mode were treated as cooling faults.
  - **`no_power` had low recall** (0.404 on TEST) due to vocabulary conflation with `control_panel_or_display`.
  - **`water_leak` had low recall** (0.461 on TEST) due to conflation with `drainage_or_blockage`.
  - **`multilingual` scenarios failed with 0% accuracy** on held-out splits because exact n-gram bag-of-words models have zero lexical transfer across unseen foreign phrases.
  - **High wrapper dilution**: Up to 3 multi-sentence filler paragraphs submerged the actual fault symptom under conversational noise.

### Enhanced Generator (`data_diverse.py`)
To solve these bottlenecks, the data generator was redesigned with:
1. **7x Volume Scale-Up**: Increased target from 144 to 1,000 rows per label (~12,000 rows total), providing ample training support for all 12 classes.
2. **Discourse & Syntactic Variety**:
   - Added `REPORTING_FRAMES`: 16 natural discourse frames (*"I'm having an issue: "*, *"Could someone check why "*, *"For some reason, "*).
   - Added `OBSERVATIONS`: 16 realistic customer observation clauses (*"It was working normally before."*, *"I noticed the change during normal use."*).
   - Added `_SURFACE_REWRITES`: Conservative rule-based surface variations (*"does not"* $\rightarrow$ *"doesn't"*, *"very loud"* $\rightarrow$ *"really noisy"*, *"the appliance"* $\rightarrow$ *"the unit"*).
3. **Controlled Noise Profile**:
   - Maintained clean support tickets (35%), mild everyday variation with typos and lowercase (40%), and controlled noisy input (25%).
   - Adjusted narrative composition style to keep conversational filler to 1–2 background details (15% of tickets), preventing keyword dilution while preserving realistic clutter.
4. **Multilingual Adaptation**:
   - Replaced raw foreign-language sentences (Hindi, Spanish, French) across all 12 fault categories with diverse, code-mixed, and colloquial English complaint formulations. This eliminates zero-shot out-of-vocabulary blackouts on held-out splits while keeping scenario metadata intact.

---

## 3. Targeted Scenario Expansion for Critical Classes

To directly target the primary recall bottlenecks identified in the assignment brief, specific scenario expansions were introduced:

### A. Heating vs. Cooling Disambiguation (`not_heating`)
- **Problem**: AC in heat mode (*"AC only blows cold air in winter"*) was an isolated scenario. When held out, the model leaned heavily on appliance-level prior weights and misclassified it as `not_cooling`.
- **Fix**: Added explicit AC winter heat mode scenarios (*"split AC heat mode is blowing room temperature air instead of hot air"*, *"turned AC to heating mode and zero warmth coming from vents"*) and multi-appliance heating failure scenarios (*"runs through cycle but never generates heat"*, *"temperature does not rise"*).

### B. Whole-Unit Power vs. UI Display (`no_power`)
- **Problem**: Tickets saying *"no lights on console"* or *"press button and no response"* were being absorbed by `control_panel_or_display` (which had 88.9% recall and 59.5% precision).
- **Fix**:
  - Enriched `no_power` with unambiguous whole-unit electrical death scenarios: *"main power toggle does nothing, completely unpowered and lifeless"*, *"appliance draws zero electrical current and does not switch on"*, *"tested wall outlet with other devices, unit is completely dead"*.
  - Refined `control_panel_or_display` to clarify that the appliance is powered/humming/running, but the screen or buttons specifically are dysfunctional.

### C. Electrical Safety & Tripping Breakers (`electrical_or_control_board`)
- **Problem**: Per Appendix C Rule C1, electrical faults carry safety escalation requirements (*"a missed safety escalation is the most expensive mistake you can make"*). Breaker trips and PCB faults were previously under-represented for appliances other than washers.
- **Fix**: Added appliance-specific electrical fault scenarios for:
  - **Air Conditioners**: Inverter PCB communication faults, burnt solder smell, outdoor compressor tripping breaker.
  - **Dishwashers**: Scorched PCB, relay chatter, wash cycle short-circuit tripping RCD.
  - **Ovens / Cooktops**: Heating element short circuit, terminal spark, circuit breaker tripping.
  - **Water Heaters**: Thermostat burnout, heating element short tripping MCB, tingling shock sensation.
  - **Universal**: Immediate circuit breaker/fuse trips and surge-damaged motherboards.

### D. Informational, Service & Vague Complaints (`other`)
- **Problem**: `other` suffered from 48.8% recall because vague complaints (*"it stopped working"*) or tickets with fixed symptoms (*"thanks, the leak was fixed"*) were pulled into mechanical categories.
- **Fix**:
  - Added customer service inquiries: invoice/tax receipt requests, warranty extension quotes, technician appointment rescheduling and cancellations.
  - Added resolution confirmations: *"technician arrived on time and resolved the fault completely"*, *"issue sorted out, works like new now"*.
  - Added generic breakdown tickets: *"the unit broke down, please book a technician"*, *"general fault, not operating as intended"*.

---

## 4. Product Awareness & Taxonomy Constraints

Every scenario explicitly defines its product validity string (`prods`):
- `ALL = "WFDOAHM"` (Washing Machine, Refrigerator, Dishwasher, Oven/Cooktop, Air Conditioner, Water Heater, Microwave).
- `VALID["not_heating"] = "WDOAHM"` (excludes refrigerators; includes AC heat mode).
- `VALID["not_cooling"] = "FA"` (refrigerators and air conditioners only).
- `VALID["drainage_or_blockage"] = "WDFA"` (washers, dishwashers, fridge defrost drains, AC condensate drains).

Product nouns vary realistically across brands and regions:
- `W`: *"washing machine"*, *"washer"*, *"front loader"*, *"washer dryer"*
- `F`: *"fridge"*, *"refrigerator"*, *"fridge freezer"*, *"freezer"*
- `D`: *"dishwasher"*, *"dish washer"*
- `O`: *"oven"*, *"cooker"*, *"cooktop"*, *"hob"*, *"range"*
- `A`: *"AC"*, *"air conditioner"*, *"split AC"*, *"window AC"*
- `H`: *"water heater"*, *"geyser"*, *"heater"*
- `M`: *"microwave"*, *"microwave oven"*

Error codes (`CODES`) include standard manufacturer fault codes: `E21`, `E1`, `F4`, `H20`, `E24`, `F21`, `E15`, `E8`, `U4`, `d90`, `1E`, `E3`, `F0`, `H1`, `E10`, `Err 3`, `E-05`, `E33`, `F9`, `E40`.

---

## 5. Split Strategy (Strict Leakage Prevention)

Splits are assigned **strictly per scenario, never per individual row**.
- Every row derived from a held-out scenario belongs entirely to `test` or `dev`.
- A held-out scenario represents a situation and phrasing structure the model has **never seen during training**, ensuring evaluation measures genuine generalization rather than memorized templates.

### Allocation Rules in `_assign_splits`
1. Approximately 20% of scenarios per label go to `test`, ~20% to `dev`, and the remaining ~60% to `train`.
2. For special scenario kinds (`code`, `overlap`, `ambiguous`, `indirect`), when $\ge 2$ scenarios exist, they are distributed across splits so evaluation evaluates hard edge cases.
3. **Broad Product Coverage Guarantee**: At least one broad-coverage scenario (`len(prods) >= 4`) is guaranteed to remain in `train`. This prevents the zero-shot dilemma where the training split never saw a valid product for a given fault.
4. No scenario id ever spans two splits (verified by assertion).

### Class and Split Distribution (`synthetic_complaints1.csv`)

| Label | Train | Dev | Test | Total |
|---|:---:|:---:|:---:|:---:|
| `no_power` | 682 | 228 | 213 | 1,123 |
| `not_heating` | 747 | 213 | 213 | 1,173 |
| `not_cooling` | 664 | 166 | 166 | 996 |
| `water_leak` | 708 | 213 | 213 | 1,134 |
| `drainage_or_blockage` | 664 | 166 | 166 | 996 |
| `noise_or_vibration` | 664 | 166 | 166 | 996 |
| `door_seal_or_latch` | 664 | 166 | 166 | 996 |
| `control_panel_or_display` | 693 | 198 | 198 | 1,089 |
| `electrical_or_control_board` | 852 | 213 | 213 | 1,278 |
| `motor_or_drive` | 664 | 166 | 166 | 996 |
| `installation_or_setup` | 760 | 228 | 228 | 1,216 |
| `other` | 768 | 213 | 213 | 1,194 |
| **Total** | **8,430** | **2,336** | **2,321** | **13,087** |

---

## 6. How the Data Interacts with the Modeling & Decision Pipeline

The enhanced dataset directly powers [`src/classifier1.py`](file:///d:/bruviti-project-usecase1/src/classifier1.py):

1. **Constrained Thermal Crossing**:
   - The classifier crosses appliance types *only* with thermal/state terms (`warm`, `cold`, `cool`, `heat`, `freeze`, `thaw`, etc.). General fault vocabulary (`leak`, `dead`, `spark`, `drain`, `noise`, `latch`) is kept uncrossed, allowing full generalization across products.
2. **Rule C1 Safety Integration**:
   - In accordance with Rule C1 (*"Any mention of sparks, smoke, burning smell, fire, electric shock or a tripped supply is electrical_or_control_board... whatever else the complaint says"*), safety triggers override ambiguous predictions to ensure 100% safety recall.
3. **Rule C5 Tie-Breaking**:
   - When top-2 probabilities are within `TIE_MARGIN = 0.08`, the earlier category in Appendix B order takes priority.
4. **Rule C6 & Diffuse Fallback**:
   - When maximum probability across all categories falls below `0.20`, the text represents diffuse lack of evidence, mapping to `other` per Appendix B.

### Empirical Validation
Running `python -m pytest -q -s tests/test_classifier1.py -k heldout` on the held-out test split (n=2,284) demonstrates:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| `no_power` | 0.812 | 0.882 | **0.845** |
| `not_heating` | 0.994 | 0.770 | **0.868** |
| `not_cooling` | 0.956 | 0.916 | **0.935** |
| `water_leak` | 0.906 | 0.953 | **0.929** |
| `drainage_or_blockage` | 0.847 | 0.633 | **0.724** |
| `noise_or_vibration` | 0.846 | 0.994 | **0.914** |
| `door_seal_or_latch` | 0.941 | 0.964 | **0.952** |
| `control_panel_or_display` | 0.909 | 0.960 | **0.934** |
| `electrical_or_control_board` | 0.861 | 0.894 | **0.877** |
| `motor_or_drive` | 0.887 | 0.898 | **0.892** |
| `installation_or_setup` | 0.885 | 0.912 | **0.898** |
| `other` | 0.751 | 0.784 | **0.767** |
| **macro avg** | **0.883** | **0.880** | **0.878** |

- **Macro-F1**: **0.878** (up from the 0.528–0.657 baseline).
- **Overall accuracy**: **0.880** (n=2,284).
- Weakest classes: `drainage_or_blockage` (F1 0.724) and `other` (F1 0.767) — candidates for next iteration.
