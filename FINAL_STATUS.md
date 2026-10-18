# Project Status

## Implemented
- Data loading and required-column validation
- Leakage-aware lag/rolling feature engineering
- Chronological ExtraTrees evaluation and same-holdout baseline benchmarks
- Building-level robust anomaly screening and energy fingerprints
- Counterfactual screening and intervention mapping
- Conditional Scope 2 electricity CO₂ scenario calculations with versioned factors
- Streamlit dashboard with CSV upload, quality checks, bilingual UI and report export
- Field-validation UI for treatment/control DID, pre-trend slope diagnostics, pre-intervention placebo DID and optional clustered bootstrap intervals
- Automated tests for pipeline guards, economics and validation helpers
- GitHub Actions workflow and Railway deployment configuration

## Not claimed
- No measured campus energy savings
- No measured CO₂ reduction
- No claim that every public meter reading is verified kWh
- No claim that simulated intervention rates are engineering measurements
- No claim that DID or diagnostic checks alone establish causality
- No unseen-building performance claim without a completed full-data evaluation

## Next validation priorities
1. Confirm the latest GitHub Actions pytest run is green; inspect the full failure log if not.
2. Run the reproducible pipeline on the actual dataset and record current metrics plus same-holdout baseline comparisons.
3. Run the held-out-building evaluation and document its performance separately from chronological holdout.
4. Verify meter units and factor applicability before presenting physical kWh or CO₂ estimates.
5. Validate a real intervention with treatment/control data and study-specific weather, seasonality and compliance controls.

## Evidence path
Public data → quality control → chronological prediction → anomaly screening → intervention hypothesis → conditional carbon scenario → field validation.
