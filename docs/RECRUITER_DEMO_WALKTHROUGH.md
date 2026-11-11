# Recruiter Demo Walkthrough

A concise, reproducible route through the prototype. The included data is synthetic and demonstrates software behavior—not measured building performance.

## 1. Start the app

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Use the one-click sample loader or upload `examples/sample_building_energy.csv`.

## 2. Follow the decision chain

1. **Overview** — identify the system boundary and the distinction between prediction, screening and field evidence.
2. **Upload & Diagnose** — inspect data validation, daily energy series and model predictions.
3. **Anomaly Diagnosis** — review flagged observations and their context; treat flags as candidates for operational review.
4. **Energy Fingerprint** — inspect persistence, recurrence and trend characteristics.
5. **Counterfactual Screening** — compare observed use with an expected-use baseline; interpret the gap as a screening signal, not guaranteed savings.
6. **Intervention Suggestions** — connect the diagnosed pattern to candidate operational actions; recommendations require engineering review.
7. **Scope 2 Scenario** — verify electricity units, emission factor and accounting boundary before interpreting the conditional carbon estimate.
8. **DID / Field Validation** — provide treatment/control data and inspect pre-trend, placebo and uncertainty diagnostics before drawing conclusions.
9. **Export** — generate a report that records analysis outputs and caveats.

## 3. Evidence labels to preserve in a portfolio

| Evidence | What it supports | What it does not support |
|---|---|---|
| Chronological holdout metrics | Forecast performance on the evaluated period | Intervention savings |
| Anomaly candidates | Data points worth investigating | Confirmed equipment faults |
| Counterfactual gap | Potential screening signal | Realized energy reduction |
| Scope 2 scenario | Conditional accounting estimate | Verified carbon reduction |
| DID diagnostics | Preliminary field-effect assessment | Causal certainty by itself |

## 4. Interview talking points

- Why time-series validation must preserve chronology and avoid future leakage.
- Why robust median/MAD can complement a forecasting model for anomaly screening.
- How a prediction residual becomes a diagnostic signal, then an intervention hypothesis.
- Why carbon accounting depends on units, emission-factor vintage and boundary.
- Why treatment/control design and field data are necessary to substantiate savings.

## 5. Online demo status

The public Streamlit deployment is **pending**. Do not describe this repository as a live production system or claim measured savings. Use the local demo until deployment is completed and smoke-tested.
