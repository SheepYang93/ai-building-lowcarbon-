# Resume Project Card

## Project
**AI-driven Building Energy Anomaly Diagnosis & Low-carbon Decision Support System**

## Resume-ready description
Built a Python/Streamlit prototype that converts building electricity time series into anomaly-screening results, behavioral fingerprints, intervention hypotheses and conditional Scope 2 CO₂ scenarios.

## Key contributions
- **Time-series ML:** engineered lag and rolling features with past-only windows; evaluated ExtraTrees using chronological holdout and compared it with naive-1-day, seasonal-7-day and rolling-median baselines on the same test rows.
- **Anomaly diagnosis:** implemented building-level rolling-median and robust-MAD screening, then summarized persistence, recurrence and trend into interpretable energy fingerprints.
- **Decision support:** connected diagnosis fingerprints to intervention families and scenario calculations; clearly separated screening signals from measured savings.
- **Field validation:** built a treatment/control DID workflow with pre-intervention slope diagnostics, a placebo DID using only pre-intervention observations, and optional building-cluster bootstrap intervals.
- **Productization:** deployed a bilingual Streamlit dashboard with CSV upload, data-quality checks, charts, report export and evidence-boundary messaging.

## Tech stack
Python · pandas · NumPy · scikit-learn · ExtraTrees · Streamlit · pytest · GitHub Actions · Railway

## Evidence and claim boundaries
- Public-data model metrics are specific to the documented dataset and split; do not present them as universal performance guarantees.
- Counterfactual and intervention values are screening estimates, not verified energy savings.
- Carbon outputs use configured electricity CO₂ factors and remain conditional until meter units, reporting boundary and factor applicability are verified.
- DID, pre-trend, placebo and bootstrap outputs are preliminary diagnostics, not proof of causality.

## Interview talking points
1. Why chronological holdout and past-only lag/rolling features are needed to reduce temporal leakage.
2. Why model evaluation should compare baselines on the same held-out rows.
3. Why robust MAD-based anomaly screening is used alongside prediction instead of treating prediction residuals alone as proof of waste.
4. How to move from public-data screening to a real-campus evaluation: verify meter units, define intervention and control groups, pre-register outcomes, account for weather/seasonality, and inspect placebo and uncertainty diagnostics.
