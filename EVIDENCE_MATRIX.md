# Evidence Matrix

This document is the project's final evidence boundary: what has been tested, what the result supports, and what it does not establish.

| Module | Final evidence | What it supports | What it does NOT prove |
|---|---|---|---|
| Data quality | 537 buildings; 196,005 education electricity records; 365 days/building; no duplicate building-day keys in the screened education subset | The public-data subset is sufficiently structured for modeling and screening | The source derivative is error-free; real campus data will have the same quality |
| Forecasting | ExtraTrees repeated rolling validation: mean R² ≈ 0.985, WAPE ≈ 6.2%; 3 rolling windows × 3 seeds | Strong predictive performance on the evaluated public-data setup | Universal accuracy; causal explanation of energy use |
| Model stability | Multiple time windows and seeds; ExtraTrees outperformed RF and HistGradientBoosting in the tested runs | Results are not based on one random split/model run | Robustness to every possible dataset shift |
| Leakage control | Chronological splits; lag/rolling features constructed from past observations | Evaluation better approximates deployment and reduces temporal leakage risk | Complete elimination of all possible dataset/feature leakage |
| Anomaly detection | Long-term peer-relative screening plus temporal anomaly detection; 132 buildings flagged in the diagnostic experiment | The system can prioritize unusual energy behavior using two complementary signals | That every flagged building has a real operational fault |
| Energy fingerprint | Fingerprints distinguish stable, intermittent, short-event, recurrent, persistent, gradual-drift and data-risk patterns | Anomalies can be translated into operationally interpretable temporal patterns | The fingerprint itself identifies the physical root cause |
| Intervention matching | Rule/library matching across schedule, occupancy, HVAC, maintenance and monitoring families | Different anomaly morphologies can be mapped to candidate actions | That a suggested action will achieve a specific saving rate |
| Counterfactual | Robust calibrated screening produced candidate potential signals; central estimates are model-derived | Counterfactual comparison can prioritize buildings for verification | Predicted potential equals realized savings |
| Calibration/placebo | Placebo and stratified calibration were used to stress-test counterfactual signals; 109 strict candidates under the stated screening rules | The candidate list can be made more conservative and auditable | Causal savings have been demonstrated without field treatment |
| Carbon scenario | Carbon results use an explicit scenario factor (base 0.5777 kgCO2e/kWh; low/high sensitivity factors) | The system can translate a verified energy quantity into scenario carbon impact | Actual emissions unless meter units and the emissions boundary are verified |
| Unit caveat | BDG2 lineage requires confirmation for the exact derivative's meter unit; carbon outputs are therefore conditional | The limitation is explicitly surfaced instead of hidden | That raw meter_reading can automatically be treated as verified kWh |
| Causal validation design | Field protocol uses treatment/control, DID, AI counterfactual and placebo; recommended baseline 8–12 weeks and validation 4–8 weeks | A defensible path exists from screening to real-world validation | Real-world causal savings have already been measured |
| Simulated validation | Simulated interventions recovered approximately 9.56%, 18.20%, and 29.92% for true 10%, 20%, and 30% effects respectively; DID was also evaluated | The validation pipeline can recover known synthetic effects under the simulation assumptions | These are real building savings |
| Productization | Streamlit workflow supports upload, diagnosis, counterfactual, intervention, carbon, report and field-validation modules; Railway deployment succeeded | The research workflow has been turned into a usable demo | Production-grade reliability, security, or campus integration |

## Claims discipline

Use these formulations in interviews and portfolio materials:

- Say **potential saving signal**, not AI saves X%.
- Say **conditional carbon scenario**, not verified carbon reduction.
- Say **public-data methodological validation**, not campus field validation.
- Say **candidate intervention**, not guaranteed intervention effect.
- The strongest result is the **end-to-end decision workflow**, not a single model metric.

## Final project boundary

The project demonstrates a reproducible AI workflow for energy forecasting, anomaly screening, energy fingerprinting, counterfactual prioritization, intervention recommendation, conditional carbon scenario analysis, and a field-validation path. It does not claim that the public-data model has already produced measured energy or carbon savings on a real campus.