# Final Project Brief

## Project
AI-driven Building Energy Anomaly Detection & Low-carbon Decision System

## One-line description
A reproducible AI workflow that turns building energy time series into anomaly screening, energy fingerprints, counterfactual prioritization, intervention suggestions, conditional carbon scenarios, and a field-validation workflow.

## Problem
Energy forecasting alone does not tell an operator what to inspect or what action to take. This project therefore connects prediction to operational decision-making.

## End-to-end pipeline
Data quality
→ time-aware forecasting
→ peer-relative + temporal anomaly detection
→ energy fingerprinting
→ counterfactual screening
→ intervention matching
→ conditional carbon scenario
→ treatment/control field validation

## Key evidence
- 537 buildings in the screened public-data subset.
- 196,005 education electricity records; 365 days per building in the screened subset.
- ExtraTrees repeated rolling validation: mean R² ≈ 0.985 and WAPE ≈ 6.2% across 3 rolling windows × 3 seeds.
- Historical energy features dominate short-term prediction in the tested setup; this is a modeling result, not a claim that weather is physically unimportant.
- Anomaly experiments combine long-term peer-relative behavior with temporal morphology.
- Counterfactual results are treated as potential signals, not measured savings.
- Simulated validation tests whether the validation pipeline can recover known synthetic intervention effects.
- Railway deployment is live; the Streamlit app exposes diagnosis, counterfactual, intervention, carbon, report, and field-validation workflows.

## What is novel in the portfolio context
The main contribution is the decision chain rather than a new neural-network architecture:
prediction → anomaly → fingerprint → counterfactual → intervention → carbon → validation.

## What the project does not claim
- It does not claim measured campus energy savings.
- It does not claim that a predicted 10–30% effect is a guaranteed saving.
- It does not treat conditional carbon scenarios as verified emissions reductions.
- It does not claim the public-data model is universally accurate.
- It does not claim that an anomaly automatically identifies a physical root cause.

## Interview positioning
Present this as an engineering + applied ML project demonstrating:
1. time-series modeling and leakage control;
2. anomaly detection and feature engineering;
3. model comparison and repeated validation;
4. decision-oriented counterfactual analysis;
5. causal-validation design;
6. deployment and reproducibility.

## 30-second answer
“I built an AI-driven building energy decision system using public building energy data. Instead of stopping at forecasting, I used time-aware prediction, peer-relative and temporal anomaly detection, energy fingerprints, counterfactual screening, and an intervention library to turn model outputs into operational candidates. I compared tree models with repeated rolling validation; ExtraTrees reached about 0.985 mean R² and 6.2% WAPE in the tested setup. I deliberately separate model-derived potential from real savings, so the final workflow includes treatment/control, DID and placebo validation for future campus deployment.”
