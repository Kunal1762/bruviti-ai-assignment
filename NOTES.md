<!-- # Data Calibration Notes

## Decisions made
- I started with synthetic, scenario-based complaint data because no real training set was supplied. I increased the generator target from **144 to ~1,000 rows per label** to test whether more volume helped, but treated the score—not row count—as the deciding signal.
- The first larger run was still weak (**Dev Macro-F1 0.657; Test 0.686**). The main failures were `not_heating` collapsing into `not_cooling`, weak `no_power` recall, confusion between `water_leak` and `drainage_or_blockage`, and poor `installation_or_setup` recall. More data alone did not solve these errors.
- I chose to **drop multilingual examples for now** rather than spend time on low-overlap Hindi/Spanish/French text. Antigravity replaced them with varied English complaint phrasing while preserving scenario metadata.
- I kept the split **by scenario, not by row**, and kept safety/business rules (C1–C6) in the deterministic rule engine rather than asking the classifier to learn escalation or age rules.
- I approved targeted generator and classifier changes: add AC heat-mode/multi-appliance heating cases; strengthen `no_power`, `water_leak`, and `installation_or_setup` scenarios; guarantee broad product coverage in training; reduce narrative filler to 1–2 background details; constrain product-cross features; filter frequent wrapper terms and tune word/character TF-IDF; implement C5 Appendix-B tie-breaking; and expand the regularization search beyond `C=30`.
- I asked Antigravity to modify the generator/classifier and rewrite tests **without running commands**, then separately an unseen-data evaluation.

## What I cut / avoided
- Full multilingual support for this iteration.
- Long filler narratives that bury the fault signal.
- Unconstrained product-only features that create misleading appliance priors across scenario-based splits.
- Treating synthetic held-out scores as proof of real-world performance: the generator was adjusted after reviewing errors, so the final test set is not fully independent of design decisions.
- I did **not** remove split-leakage validation. Scenario IDs are for grouping/auditing only, never model features.

## Improvements I Would Make Next

- **Strengthen evaluation validity:** keep scenario-level split isolation and avoid further tuning against the test set; reserve it for final evaluation.
- **Improve weak classes:** inspect confusion matrices and per-class recall, focusing on overlapping faults such as `not_heating` vs `motor_or_drive`, `water_leak` vs `drainage_or_blockage`, and vague `other` complaints.
- **Broaden hard-case coverage:** ensure error-code-only, overlapping-symptom, multilingual/code-mixed, and indirect complaints are represented in train, dev, and test without sharing scenarios across splits.
- **Test rules independently:** add focused tests for electrical-safety triggers, gas-smell escalation, the 30-day installation rule, bare error codes, tie-breaking, and non-complaints.
- **Reduce synthetic-data bias:** vary wording and product phrasing more naturally, then evaluate on a small independently written set of realistic complaints before trusting the score.
 -->


# NOTES

## Decisions
- **Synthetic data only.** No training data was supplied, so everything is generated from scenarios (details in DATA.md). Whole scenarios, not rows, are split into train/dev/test, so evaluation sees unseen situations and wording. Volume went from ~144 to ~1,000 rows per label, but the score, not row count, decided what to keep.
- **Rules stay out of the model.** C1–C6 live in a deterministic rule engine; the classifier only predicts the category. Escalation (C1/C2) is recall-biased because a missed safety ticket is the costliest error.
- **Honest evaluation.** C is tuned on dev, and the final model is scored once on test. `scenario_id` is for grouping and auditing only, never a feature.
- **Realistic inputs.** Training uses only what the real API receives (product, text, age) plus labels. Bookkeeping columns (rule, kind, scenario_id) are for slicing results, not training.
- **Generator safeguards.** Scenarios are product-aware, so there are no fridge-with-drum rows. An early validator failed generation whenever safety or gas wording appeared in a row not labelled for it, and caught a "burner" false match.
- **Fixes after error review.** The first larger run was weak (Dev 0.657, Test 0.686). I added AC heat-mode and multi-appliance heating cases, strengthened `no_power`, `water_leak` and `installation_or_setup` scenarios, constrained product-cross features, tuned TF-IDF, added C5 tie-breaking, and widened the C search.
- **Frontend.** One plain HTML page with a separate CSS file, served by the same FastAPI process at `/`. It uses no external assets, so it works with the network off.

## Avoided / cut
- Multilingual and code-mixed text (the Hinglish UI example was removed too). Overlap with real data is low and the time cost is high.
- Learning escalation or age rules in the model.
- Long filler narratives and unconstrained product-only features.
- Treating synthetic scores as real performance. The generator was adjusted after reviewing errors, so the test set is not fully independent.
- External data, LLM or API calls at request time, and JS/CSS frameworks.
- Claiming results before measuring them. DATA.md results stay blank until evaluation is done.

## Improvements I Would Make Next

- **Strengthen evaluation validity:** keep scenario-level split isolation and avoid further tuning against the test set; reserve it for final evaluation.
- **Improve weak classes:** inspect confusion matrices and per-class recall, focusing on overlapping faults such as `not_heating` vs `motor_or_drive`, `water_leak` vs `drainage_or_blockage`, and vague `other` complaints.
- **Broaden hard-case coverage:** ensure error-code-only, overlapping-symptom, multilingual/code-mixed, and indirect complaints are represented in train, dev, and test without sharing scenarios across splits.
- **Test rules independently:** add focused tests for electrical-safety triggers, gas-smell escalation, the 30-day installation rule, bare error codes, tie-breaking, and non-complaints.
- **Reduce synthetic-data bias:** vary wording and product phrasing more naturally, then evaluate on a small independently written set of realistic complaints before trusting the score.
 -->
